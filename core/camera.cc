#include "core/camera.h"

#include <algorithm>
#include <cmath>

namespace voxelsprout::core {

Vec3 Camera::forward() const {
    const float yaw = radians(yawDegrees);
    const float pitch = radians(std::clamp(pitchDegrees, -89.0f, 89.0f));
    const float cp = std::cos(pitch);
    return normalize(Vec3{std::cos(yaw) * cp, std::sin(pitch), std::sin(yaw) * cp});
}

Vec3 Camera::right() const {
    const Vec3 worldUp{0.0f, 1.0f, 0.0f};
    return normalize(cross(forward(), worldUp));
}

} // namespace voxelsprout::core
