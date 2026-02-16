#pragma once

#include <array>
#include <cmath>

#include "math/math.h"
#include "render/renderer_types.h"
#include "world/chunk.h"

namespace voxelsprout::render {

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

} // namespace voxelsprout::render
