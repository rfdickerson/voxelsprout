#pragma once

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

inline math::Vector3 ComputeCameraForward(float yawDegrees, float pitchDegrees) {
    const float yawRadians = math::radians(yawDegrees);
    const float pitchRadians = math::radians(pitchDegrees);
    const float cosPitch = std::cos(pitchRadians);
    return math::Vector3{
        std::cos(yawRadians) * cosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * cosPitch
    };
}

inline CameraFrameDerived ComputeCameraFrame(const CameraPose& camera) {
    return CameraFrameDerived{
        ComputeCameraForward(camera.yawDegrees, camera.pitchDegrees),
        static_cast<int>(std::floor(camera.x / static_cast<float>(world::Chunk::kSizeX))),
        static_cast<int>(std::floor(camera.y / static_cast<float>(world::Chunk::kSizeY))),
        static_cast<int>(std::floor(camera.z / static_cast<float>(world::Chunk::kSizeZ)))
    };
}

inline math::Vector3 ComputeSunDirection(float yawDegrees, float pitchDegrees) {
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

inline float ComputeVoxelGiAxisOrigin(float cameraAxis, float halfSpan, float cellSize) {
    return std::floor((cameraAxis - halfSpan) / cellSize) * cellSize;
}

inline float ComputeVoxelGiStableOriginY(
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

} // namespace voxelsprout::render
