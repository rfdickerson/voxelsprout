#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "math/math.h"
#include "render/renderer_types.h"
#include "world/chunk.h"

namespace odai::render {

inline float celestialNightVisibility(float sunElevationDegrees) {
    const float t = std::clamp(-sunElevationDegrees / 6.0f, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

inline float authoredNightAmbientWeight(bool enabled, float lightingWeight,
                                        float sunElevationDegrees) {
    if (!enabled) return 0.0f;
    const float t = std::clamp(1.0f - sunElevationDegrees / 6.0f, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t) * std::clamp(lightingWeight, 0.0f, 1.0f);
}


// Homogeneous clip-plane rejection stays valid for boxes crossing the eye.
// Dividing by w and keeping every box with a behind-camera corner defeats
// culling for whole cells behind the viewer.
inline bool importedBoundsIntersectClip(const float low[3], const float high[3],
                                       const math::Matrix4& matrix, float margin) {
    for (int axis = 0; axis < 3; ++axis)
        if (!std::isfinite(low[axis]) || !std::isfinite(high[axis]) || low[axis] > high[axis]) return true;
    unsigned outsideAll = 63u;
    for (unsigned corner = 0; corner < 8; ++corner) {
        const auto clip = math::multiply(matrix, math::Vector4{
            (corner & 1u) ? high[0] : low[0], (corner & 2u) ? high[1] : low[1],
            (corner & 4u) ? high[2] : low[2], 1.0f});
        if (!std::isfinite(clip.x) || !std::isfinite(clip.y) ||
            !std::isfinite(clip.z) || !std::isfinite(clip.w)) return true;
        const float slack = std::max(margin, 0.0f) * std::abs(clip.w);
        unsigned outside = 0u;
        if (clip.x < -clip.w - slack) outside |= 1u;
        if (clip.x >  clip.w + slack) outside |= 2u;
        if (clip.y < -clip.w - slack) outside |= 4u;
        if (clip.y >  clip.w + slack) outside |= 8u;
        if (clip.z < -slack) outside |= 16u;
        if (clip.z > clip.w + slack) outside |= 32u;
        outsideAll &= outside;
    }
    return outsideAll == 0u;
}

inline constexpr std::uint32_t screenSpaceGiQuarterExtent(std::uint32_t fullExtent) {
    return std::max(1u, (fullExtent + 3u) / 4u);
}

inline bool screenSpaceGiHistorySampleAccepted(
    float storedDepth,
    float expectedDepth,
    float normalAgreement) {
    if (storedDepth <= 1e-4f || expectedDepth <= 1e-4f || normalAgreement < 0.5f) {
        return false;
    }
    return std::abs(storedDepth - expectedDepth) <=
        std::max(12.0f, expectedDepth * 0.02f);
}

inline float screenSpaceGiClampedLuminance(float indirectLuminance, float directLuminance) {
    return std::clamp(indirectLuminance, 0.0f,
                      0.35f * (std::max(directLuminance, 0.0f) + 0.05f));
}

struct CameraFrameDerived {
    math::Vector3 forward;
    int chunkX;
    int chunkY;
    int chunkZ;
};

struct VoxelGiComputeFlags {
    bool gridMoved;
    bool sunDirectionChanged;
    bool sunColorChanged;
    bool shChanged;
    bool computeSettingsChanged;
    bool lightingChanged;
    bool needsOccupancyUpload;
    bool needsComputeUpdate;
};

inline math::Vector3 computeCameraForward(float yawDegrees, float pitchDegrees) {
    const float yawRadians = math::radians(yawDegrees);
    const float pitchRadians = math::radians(pitchDegrees);
    const float cosPitch = std::cos(pitchRadians);
    return math::Vector3{
        std::cos(yawRadians) * cosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * cosPitch
    };
}

inline CameraFrameDerived computeCameraFrame(const CameraPose& camera) {
    return CameraFrameDerived{
        computeCameraForward(camera.yawDegrees, camera.pitchDegrees),
        static_cast<int>(std::floor(camera.x / static_cast<float>(world::Chunk::kSizeX))),
        static_cast<int>(std::floor(camera.y / static_cast<float>(world::Chunk::kSizeY))),
        static_cast<int>(std::floor(camera.z / static_cast<float>(world::Chunk::kSizeZ)))
    };
}

inline math::Matrix4 computeCameraView(const CameraPose& camera) {
    // Form the basis at the origin. Constructing target = eye + unitForward
    // first rounds the direction at Skyrim's large world coordinates; lookAt
    // then subtracts eye again and turns that rounding into camera rotation.
    auto view = math::lookAt(math::Vector3{0, 0, 0},
        computeCameraForward(camera.yawDegrees, camera.pitchDegrees),
        math::Vector3{0, 1, 0});
    for (int row = 0; row < 3; ++row) {
        view(row, 3) = -(view(row, 0) * camera.x + view(row, 1) * camera.y +
                         view(row, 2) * camera.z);
    }
    return view;
}

inline math::Vector3 computeSunDirection(float yawDegrees, float pitchDegrees) {
    const float yawRadians = math::radians(yawDegrees);
    const float pitchRadians = math::radians(pitchDegrees);
    const float sunCosPitch = std::cos(pitchRadians);
    math::Vector3 sunDirection{
        std::cos(yawRadians) * sunCosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * sunCosPitch
    };
    if (math::lengthSquared(sunDirection) <= 0.0001f) {
        sunDirection = math::Vector3{-0.58f, -0.42f, -0.24f};
    }
    return sunDirection;
}

inline float computeVoxelGiAxisOrigin(float cameraAxis, float halfSpan, float cellSize) {
    return std::floor((cameraAxis - halfSpan) / cellSize) * cellSize;
}

inline float computeVoxelGiStableOriginY(
    float desiredOriginY,
    float previousOriginY,
    bool hasPreviousFrameState,
    float verticalFollowThreshold
) {
    if (!hasPreviousFrameState) {
        return desiredOriginY;
    }
    if (std::abs(desiredOriginY - previousOriginY) < verticalFollowThreshold) {
        return previousOriginY;
    }
    return desiredOriginY;
}

inline VoxelGiComputeFlags computeVoxelGiFlags(
    const std::array<math::Vector3, 9>& shIrradiance,
    const std::array<std::array<float, 3>, 9>& previousShIrradiance,
    const std::array<float, 3>& gridOrigin,
    const std::array<float, 3>& previousGridOrigin,
    bool hasPreviousFrameState,
    bool worldDirty,
    bool occupancyInitialized,
    const math::Vector3& sunDirection,
    const math::Vector3& previousSunDirection,
    const math::Vector3& sunColor,
    const math::Vector3& previousSunColor,
    float bounceStrength,
    float previousBounceStrength,
    float diffusionSoftness,
    float previousDiffusionSoftness,
    float gridMoveThreshold,
    float lightingChangeThreshold,
    float tuningChangeThreshold
) {
    const bool gridMoved =
        !hasPreviousFrameState ||
        std::abs(gridOrigin[0] - previousGridOrigin[0]) > gridMoveThreshold ||
        std::abs(gridOrigin[1] - previousGridOrigin[1]) > gridMoveThreshold ||
        std::abs(gridOrigin[2] - previousGridOrigin[2]) > gridMoveThreshold;

    const bool sunDirectionChanged =
        !hasPreviousFrameState ||
        std::abs(sunDirection.x - previousSunDirection.x) > lightingChangeThreshold ||
        std::abs(sunDirection.y - previousSunDirection.y) > lightingChangeThreshold ||
        std::abs(sunDirection.z - previousSunDirection.z) > lightingChangeThreshold;

    const bool sunColorChanged =
        !hasPreviousFrameState ||
        std::abs(sunColor.x - previousSunColor.x) > lightingChangeThreshold ||
        std::abs(sunColor.y - previousSunColor.y) > lightingChangeThreshold ||
        std::abs(sunColor.z - previousSunColor.z) > lightingChangeThreshold;

    bool shChanged = !hasPreviousFrameState;
    if (!shChanged) {
        for (std::size_t coeffIndex = 0; coeffIndex < shIrradiance.size(); ++coeffIndex) {
            const std::array<float, 3>& previousCoeff = previousShIrradiance[coeffIndex];
            const math::Vector3& currentCoeff = shIrradiance[coeffIndex];
            if (std::abs(currentCoeff.x - previousCoeff[0]) > lightingChangeThreshold ||
                std::abs(currentCoeff.y - previousCoeff[1]) > lightingChangeThreshold ||
                std::abs(currentCoeff.z - previousCoeff[2]) > lightingChangeThreshold) {
                shChanged = true;
                break;
            }
        }
    }

    const bool computeSettingsChanged =
        !hasPreviousFrameState ||
        std::abs(bounceStrength - previousBounceStrength) > tuningChangeThreshold ||
        std::abs(diffusionSoftness - previousDiffusionSoftness) > tuningChangeThreshold;

    const bool lightingChanged = sunDirectionChanged || sunColorChanged || shChanged;
    const bool needsOccupancyUpload = worldDirty || gridMoved || !occupancyInitialized;
    const bool needsComputeUpdate =
        needsOccupancyUpload || lightingChanged || computeSettingsChanged || !hasPreviousFrameState;

    return VoxelGiComputeFlags{
        gridMoved,
        sunDirectionChanged,
        sunColorChanged,
        shChanged,
        computeSettingsChanged,
        lightingChanged,
        needsOccupancyUpload,
        needsComputeUpdate
    };
}

} // namespace odai::render
