#pragma once

#include "core/math.h"

namespace voxelsprout::core {

struct Camera {
    Vec3 position{0.0f, 0.0f, 3.0f};
    float yawDegrees = -90.0f;
    float pitchDegrees = 0.0f;
    float fovDegrees = 60.0f;

    [[nodiscard]] Vec3 forward() const;
    [[nodiscard]] Vec3 right() const;
};

} // namespace voxelsprout::core
