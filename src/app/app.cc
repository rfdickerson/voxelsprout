#include "app/app.h"

#include <GLFW/glfw3.h>

#include "core/grid3.h"
#include "core/log.h"
#include "math/math.h"
#include "sim/network_procedural.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr float kMouseSensitivity = 0.1f;
constexpr float kMouseSmoothingSeconds = 0.035f;
constexpr float kMoveMaxSpeed = 5.0f;
constexpr float kMoveAcceleration = 14.0f;
constexpr float kMoveDeceleration = 18.0f;
constexpr float kJumpSpeed = 7.8f;
constexpr float kGravity = -24.0f;
constexpr float kMaxFallSpeed = -35.0f;
constexpr float kPitchMinDegrees = -89.0f;
constexpr float kPitchMaxDegrees = 89.0f;
[[maybe_unused]] constexpr float kVoxelSizeMeters = 0.25f;
constexpr float kBlockInteractMaxDistance = 6.0f;
constexpr float kRenderCullNearPlane = 0.1f;
constexpr float kRenderCullFarPlane = 500.0f;
constexpr float kRenderFrustumBoundsPadVoxels = 8.0f;
constexpr float kRenderFrustumPlaneSlackVoxels = 2.5f;
constexpr float kRenderAspectFallback = 16.0f / 9.0f;
// 1x voxel world scale: roughly Minecraft-like player proportions.
constexpr float kPlayerHeightVoxels = 1.8f;
constexpr float kPlayerDiameterVoxels = 0.8f;
constexpr float kPlayerEyeHeightVoxels = 1.62f;

constexpr float kPlayerRadius = kPlayerDiameterVoxels * 0.5f;
constexpr float kPlayerEyeHeight = kPlayerEyeHeightVoxels;
constexpr float kPlayerHeight = kPlayerHeightVoxels;
constexpr float kPlayerTopOffset = kPlayerHeight - kPlayerEyeHeight;
constexpr float kCollisionEpsilon = 0.001f;
constexpr float kHoverHeightAboveGround = 0.15f;
constexpr float kHoverResponse = 8.0f;
constexpr float kHoverMaxVerticalSpeed = 12.0f;
constexpr float kHoverManualVerticalSpeed = 8.0f;
constexpr int kHoverGroundSearchDepth = 96;
constexpr float kDayCycleSpeedCyclesPerSecond = 0.05f;
constexpr float kDayCycleLatitudeDegrees = 52.0f;
constexpr float kDayCycleWinterDeclinationDegrees = -23.0f;
constexpr float kDayCycleAzimuthOffsetDegrees = 0.0f;
constexpr float kTwoPi = 6.28318530718f;
constexpr float kGamepadTriggerPressedThreshold = 0.30f;
constexpr float kGamepadMoveDeadzone = 0.18f;
constexpr float kGamepadLookDeadzone = 0.14f;
constexpr float kGamepadLookDegreesPerSecond = 160.0f;
constexpr const char* kWorldFilePath = "world.vxw";
constexpr const char* kMagicaCastlePath = "assets/magicka/castle.vox";
constexpr const char* kMagicaTeapotPath = "assets/magicka/teapot.vox";
constexpr const char* kMagicaMonu2Path = "assets/magicka/monu2.vox";
constexpr float kWorldAutosaveDelaySeconds = 0.75f;

constexpr std::array<voxelsprout::world::VoxelType, 5> kPlaceableBlockTypes = {
    voxelsprout::world::VoxelType::Stone,
    voxelsprout::world::VoxelType::Dirt,
    voxelsprout::world::VoxelType::Grass,
    voxelsprout::world::VoxelType::Wood,
    voxelsprout::world::VoxelType::SolidRed
};
constexpr int kHotbarSlotBlock = 0;
constexpr int kHotbarSlotPipe = 1;
constexpr int kHotbarSlotConveyor = 2;
constexpr int kHotbarSlotTrack = 3;
constexpr int kHotbarSlotCount = 4;
constexpr float kDefaultPipeLength = 1.0f;
constexpr float kDefaultPipeRadius = 0.45f;
constexpr voxelsprout::math::Vector3 kDefaultPipeTint{0.95f, 0.95f, 0.95f};
constexpr float kConveyorCollisionRadius = 0.49f;
constexpr float kConveyorAlongHalfExtent = 0.5f;
constexpr float kConveyorCrossAxisScale = 2.0f;
constexpr float kConveyorVerticalScale = 0.25f;
constexpr double kSimulationFixedHz = 60.0;
constexpr double kSimulationFixedStepSeconds = 1.0 / kSimulationFixedHz;
constexpr double kFrameDeltaClampSeconds = 0.25;
constexpr int kMaxSimulationStepsPerFrame = 8;

struct Aabb3f {
    float minX = 0.0f;
    float maxX = 0.0f;
    float minY = 0.0f;
    float maxY = 0.0f;
    float minZ = 0.0f;
    float maxZ = 0.0f;
};

Aabb3f makePlayerCollisionAabb(float eyeX, float eyeY, float eyeZ) {
    Aabb3f bounds{};
    bounds.minX = eyeX - kPlayerRadius;
    bounds.maxX = eyeX + kPlayerRadius;
    bounds.minY = eyeY - kPlayerEyeHeight;
    bounds.maxY = eyeY + kPlayerTopOffset;
    bounds.minZ = eyeZ - kPlayerRadius;
    bounds.maxZ = eyeZ + kPlayerRadius;
    return bounds;
}

Aabb3f makeConveyorBeltAabb(const voxelsprout::sim::Belt& belt) {
    const float centerX = static_cast<float>(belt.x) + 0.5f;
    const float centerY = static_cast<float>(belt.y) + 0.5f;
    const float centerZ = static_cast<float>(belt.z) + 0.5f;
    const bool alongX = belt.direction == voxelsprout::sim::BeltDirection::East || belt.direction == voxelsprout::sim::BeltDirection::West;
    const float halfHeight = kConveyorVerticalScale * kConveyorCollisionRadius;
    const float halfCrossAxis = kConveyorCrossAxisScale * kConveyorCollisionRadius;
    const float halfExtentX = alongX ? kConveyorAlongHalfExtent : halfCrossAxis;
    const float halfExtentZ = alongX ? halfCrossAxis : kConveyorAlongHalfExtent;

    Aabb3f bounds{};
    bounds.minX = centerX - halfExtentX;
    bounds.maxX = centerX + halfExtentX;
    bounds.minY = centerY - halfHeight;
    bounds.maxY = centerY + halfHeight;
    bounds.minZ = centerZ - halfExtentZ;
    bounds.maxZ = centerZ + halfExtentZ;
    return bounds;
}

bool aabbOverlaps(const Aabb3f& lhs, const Aabb3f& rhs) {
    return
        lhs.maxX > (rhs.minX + kCollisionEpsilon) &&
        lhs.minX < (rhs.maxX - kCollisionEpsilon) &&
        lhs.maxY > (rhs.minY + kCollisionEpsilon) &&
        lhs.minY < (rhs.maxY - kCollisionEpsilon) &&
        lhs.maxZ > (rhs.minZ + kCollisionEpsilon) &&
        lhs.minZ < (rhs.maxZ - kCollisionEpsilon);
}

const char* placeableBlockLabel(voxelsprout::world::VoxelType type) {
    switch (type) {
    case voxelsprout::world::VoxelType::Solid:
        return "stone";
    case voxelsprout::world::VoxelType::Dirt:
        return "dirt";
    case voxelsprout::world::VoxelType::Grass:
        return "grass";
    case voxelsprout::world::VoxelType::Wood:
        return "wood";
    case voxelsprout::world::VoxelType::SolidRed:
        return "red";
    case voxelsprout::world::VoxelType::Empty:
    default:
        return "empty";
    }
}

struct FrustumPlane {
    voxelsprout::math::Vector3 normal{};
    float d = 0.0f;
};

struct CameraFrustum {
    std::array<FrustumPlane, 6> planes{};
    voxelsprout::core::CellAabb broadPhaseBounds{};
    bool valid = false;
};

FrustumPlane makePlaneFromPointNormal(const voxelsprout::math::Vector3& point, const voxelsprout::math::Vector3& normal) {
    const voxelsprout::math::Vector3 normalized = voxelsprout::math::normalize(normal);
    FrustumPlane plane{};
    plane.normal = normalized;
    plane.d = -voxelsprout::math::dot(normalized, point);
    return plane;
}

void orientPlaneTowardForward(FrustumPlane& plane, const voxelsprout::math::Vector3& forward) {
    if (voxelsprout::math::dot(plane.normal, forward) < 0.0f) {
        plane.normal = -plane.normal;
        plane.d = -plane.d;
    }
}

CameraFrustum buildCameraFrustum(
    const voxelsprout::math::Vector3& eye,
    float yawDegrees,
    float pitchDegrees,
    float fovDegrees,
    float aspectRatio
) {
    CameraFrustum frustum{};
    const float clampedAspect = std::max(aspectRatio, 0.1f);
    const float clampedFovDegrees = std::clamp(fovDegrees, 20.0f, 120.0f);
    const float yawRadians = voxelsprout::math::radians(yawDegrees);
    const float pitchRadians = voxelsprout::math::radians(pitchDegrees);
    const float cosPitch = std::cos(pitchRadians);
    voxelsprout::math::Vector3 forward{
        std::cos(yawRadians) * cosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * cosPitch
    };
    forward = voxelsprout::math::normalize(forward);
    if (voxelsprout::math::lengthSquared(forward) <= 0.0001f) {
        return frustum;
    }

    const voxelsprout::math::Vector3 worldUp{0.0f, 1.0f, 0.0f};
    voxelsprout::math::Vector3 right = voxelsprout::math::normalize(voxelsprout::math::cross(forward, worldUp));
    if (voxelsprout::math::lengthSquared(right) <= 0.0001f) {
        right = voxelsprout::math::Vector3{1.0f, 0.0f, 0.0f};
    }
    voxelsprout::math::Vector3 up = voxelsprout::math::normalize(voxelsprout::math::cross(right, forward));
    if (voxelsprout::math::lengthSquared(up) <= 0.0001f) {
        up = worldUp;
    }

    const float halfFovY = voxelsprout::math::radians(clampedFovDegrees) * 0.5f;
    const float tanHalfY = std::tan(halfFovY);
    const float tanHalfX = tanHalfY * clampedAspect;
    const float nearDistance = kRenderCullNearPlane;
    const float farDistance = kRenderCullFarPlane;

    const voxelsprout::math::Vector3 nearCenter = eye + (forward * nearDistance);
    const voxelsprout::math::Vector3 farCenter = eye + (forward * farDistance);
    const float nearHalfHeight = nearDistance * tanHalfY;
    const float nearHalfWidth = nearDistance * tanHalfX;
    const float farHalfHeight = farDistance * tanHalfY;
    const float farHalfWidth = farDistance * tanHalfX;

    const voxelsprout::math::Vector3 nearUp = up * nearHalfHeight;
    const voxelsprout::math::Vector3 nearRight = right * nearHalfWidth;
    const voxelsprout::math::Vector3 farUp = up * farHalfHeight;
    const voxelsprout::math::Vector3 farRight = right * farHalfWidth;

    const std::array<voxelsprout::math::Vector3, 8> corners = {
        nearCenter + nearUp - nearRight,
        nearCenter + nearUp + nearRight,
        nearCenter - nearUp - nearRight,
        nearCenter - nearUp + nearRight,
        farCenter + farUp - farRight,
        farCenter + farUp + farRight,
        farCenter - farUp - farRight,
        farCenter - farUp + farRight
    };

    float minX = corners[0].x;
    float minY = corners[0].y;
    float minZ = corners[0].z;
    float maxX = corners[0].x;
    float maxY = corners[0].y;
    float maxZ = corners[0].z;
    for (const voxelsprout::math::Vector3& corner : corners) {
        minX = std::min(minX, corner.x);
        minY = std::min(minY, corner.y);
        minZ = std::min(minZ, corner.z);
        maxX = std::max(maxX, corner.x);
        maxY = std::max(maxY, corner.y);
        maxZ = std::max(maxZ, corner.z);
    }

    voxelsprout::core::CellAabb broadPhaseBounds{};
    broadPhaseBounds.valid = true;
    broadPhaseBounds.minInclusive = voxelsprout::core::Cell3i{
        static_cast<int>(std::floor(minX - kRenderFrustumBoundsPadVoxels)),
        static_cast<int>(std::floor(minY - kRenderFrustumBoundsPadVoxels)),
        static_cast<int>(std::floor(minZ - kRenderFrustumBoundsPadVoxels))
    };
    broadPhaseBounds.maxExclusive = voxelsprout::core::Cell3i{
        static_cast<int>(std::floor(maxX + kRenderFrustumBoundsPadVoxels)) + 1,
        static_cast<int>(std::floor(maxY + kRenderFrustumBoundsPadVoxels)) + 1,
        static_cast<int>(std::floor(maxZ + kRenderFrustumBoundsPadVoxels)) + 1
    };

    const voxelsprout::math::Vector3 leftDir = voxelsprout::math::normalize(forward - (right * tanHalfX));
    const voxelsprout::math::Vector3 rightDir = voxelsprout::math::normalize(forward + (right * tanHalfX));
    const voxelsprout::math::Vector3 topDir = voxelsprout::math::normalize(forward + (up * tanHalfY));
    const voxelsprout::math::Vector3 bottomDir = voxelsprout::math::normalize(forward - (up * tanHalfY));

    std::array<FrustumPlane, 6> planes{};
    planes[0] = makePlaneFromPointNormal(nearCenter, forward);
    planes[1] = makePlaneFromPointNormal(farCenter, -forward);
    planes[2] = makePlaneFromPointNormal(eye, voxelsprout::math::cross(up, leftDir));
    planes[3] = makePlaneFromPointNormal(eye, voxelsprout::math::cross(rightDir, up));
    planes[4] = makePlaneFromPointNormal(eye, voxelsprout::math::cross(topDir, right));
    planes[5] = makePlaneFromPointNormal(eye, voxelsprout::math::cross(right, bottomDir));
    orientPlaneTowardForward(planes[2], forward);
    orientPlaneTowardForward(planes[3], forward);
    orientPlaneTowardForward(planes[4], forward);
    orientPlaneTowardForward(planes[5], forward);

    frustum.planes = planes;
    frustum.broadPhaseBounds = broadPhaseBounds;
    frustum.valid = true;
    return frustum;
}

bool chunkIntersectsFrustum(
    const voxelsprout::world::Chunk& chunk,
    const std::array<FrustumPlane, 6>& planes,
    float planeSlack
) {
    const float minX = static_cast<float>(chunk.chunkX() * voxelsprout::world::Chunk::kSizeX);
    const float minY = static_cast<float>(chunk.chunkY() * voxelsprout::world::Chunk::kSizeY);
    const float minZ = static_cast<float>(chunk.chunkZ() * voxelsprout::world::Chunk::kSizeZ);
    const float maxX = minX + static_cast<float>(voxelsprout::world::Chunk::kSizeX);
    const float maxY = minY + static_cast<float>(voxelsprout::world::Chunk::kSizeY);
    const float maxZ = minZ + static_cast<float>(voxelsprout::world::Chunk::kSizeZ);

    for (const FrustumPlane& plane : planes) {
        const float positiveX = (plane.normal.x >= 0.0f) ? maxX : minX;
        const float positiveY = (plane.normal.y >= 0.0f) ? maxY : minY;
        const float positiveZ = (plane.normal.z >= 0.0f) ? maxZ : minZ;
        const float distance =
            (plane.normal.x * positiveX) +
            (plane.normal.y * positiveY) +
            (plane.normal.z * positiveZ) +
            plane.d;
        if (distance < -planeSlack) {
            return false;
        }
    }
    return true;
}

void glfwErrorCallback(int errorCode, const char* description) {
    VOX_LOGE("glfw") << "error " << errorCode << ": "
                     << (description != nullptr ? description : "(no description)");
}

float approach(float current, float target, float maxDelta) {
    const float delta = target - current;
    if (delta > maxDelta) {
        return current + maxDelta;
    }
    if (delta < -maxDelta) {
        return current - maxDelta;
    }
    return target;
}

float applyStickDeadzone(float value, float deadzone) {
    const float clampedDeadzone = std::clamp(deadzone, 0.0f, 0.99f);
    const float magnitude = std::abs(value);
    if (magnitude <= clampedDeadzone) {
        return 0.0f;
    }
    const float normalized = (magnitude - clampedDeadzone) / (1.0f - clampedDeadzone);
    return std::copysign(normalized, value);
}

voxelsprout::core::Dir6 axisToDir6(const voxelsprout::math::Vector3& axis) {
    const voxelsprout::math::Vector3 normalized = voxelsprout::math::normalize(axis);
    const float absX = std::abs(normalized.x);
    const float absY = std::abs(normalized.y);
    const float absZ = std::abs(normalized.z);
    if (absX >= absY && absX >= absZ) {
        return normalized.x >= 0.0f ? voxelsprout::core::Dir6::PosX : voxelsprout::core::Dir6::NegX;
    }
    if (absY >= absX && absY >= absZ) {
        return normalized.y >= 0.0f ? voxelsprout::core::Dir6::PosY : voxelsprout::core::Dir6::NegY;
    }
    return normalized.z >= 0.0f ? voxelsprout::core::Dir6::PosZ : voxelsprout::core::Dir6::NegZ;
}

voxelsprout::core::Dir6 faceNormalToDir6(int nx, int ny, int nz) {
    if (nx > 0) {
        return voxelsprout::core::Dir6::PosX;
    }
    if (nx < 0) {
        return voxelsprout::core::Dir6::NegX;
    }
    if (ny > 0) {
        return voxelsprout::core::Dir6::PosY;
    }
    if (ny < 0) {
        return voxelsprout::core::Dir6::NegY;
    }
    if (nz > 0) {
        return voxelsprout::core::Dir6::PosZ;
    }
    return voxelsprout::core::Dir6::NegZ;
}

void dir6ToAxisInts(voxelsprout::core::Dir6 dir, int& outX, int& outY, int& outZ) {
    const voxelsprout::core::Cell3i offset = voxelsprout::core::dirToOffset(dir);
    outX = static_cast<int>(offset.x);
    outY = static_cast<int>(offset.y);
    outZ = static_cast<int>(offset.z);
}

bool dirSharesAxis(voxelsprout::core::Dir6 lhs, voxelsprout::core::Dir6 rhs) {
    return lhs == rhs || voxelsprout::core::areOpposite(lhs, rhs);
}

float wrapDegreesSigned(float degrees) {
    float wrapped = std::fmod(degrees, 360.0f);
    if (wrapped <= -180.0f) {
        wrapped += 360.0f;
    } else if (wrapped > 180.0f) {
        wrapped -= 360.0f;
    }
    return wrapped;
}

voxelsprout::core::Dir6 horizontalDirFromYaw(float yawDegrees) {
    const float yawRadians = voxelsprout::math::radians(yawDegrees);
    const float x = std::cos(yawRadians);
    const float z = std::sin(yawRadians);
    if (std::abs(x) >= std::abs(z)) {
        return x >= 0.0f ? voxelsprout::core::Dir6::PosX : voxelsprout::core::Dir6::NegX;
    }
    return z >= 0.0f ? voxelsprout::core::Dir6::PosZ : voxelsprout::core::Dir6::NegZ;
}

voxelsprout::sim::BeltDirection dir6ToBeltDirection(voxelsprout::core::Dir6 dir) {
    switch (dir) {
    case voxelsprout::core::Dir6::PosX:
        return voxelsprout::sim::BeltDirection::East;
    case voxelsprout::core::Dir6::NegX:
        return voxelsprout::sim::BeltDirection::West;
    case voxelsprout::core::Dir6::PosZ:
        return voxelsprout::sim::BeltDirection::South;
    case voxelsprout::core::Dir6::NegZ:
    default:
        return voxelsprout::sim::BeltDirection::North;
    }
}

voxelsprout::core::Dir6 beltDirectionToDir6(voxelsprout::sim::BeltDirection direction) {
    switch (direction) {
    case voxelsprout::sim::BeltDirection::East:
        return voxelsprout::core::Dir6::PosX;
    case voxelsprout::sim::BeltDirection::West:
        return voxelsprout::core::Dir6::NegX;
    case voxelsprout::sim::BeltDirection::South:
        return voxelsprout::core::Dir6::PosZ;
    case voxelsprout::sim::BeltDirection::North:
    default:
        return voxelsprout::core::Dir6::NegZ;
    }
}

voxelsprout::sim::TrackDirection dir6ToTrackDirection(voxelsprout::core::Dir6 dir) {
    switch (dir) {
    case voxelsprout::core::Dir6::PosX:
        return voxelsprout::sim::TrackDirection::East;
    case voxelsprout::core::Dir6::NegX:
        return voxelsprout::sim::TrackDirection::West;
    case voxelsprout::core::Dir6::PosZ:
        return voxelsprout::sim::TrackDirection::South;
    case voxelsprout::core::Dir6::NegZ:
    default:
        return voxelsprout::sim::TrackDirection::North;
    }
}

voxelsprout::core::Dir6 trackDirectionToDir6(voxelsprout::sim::TrackDirection direction) {
    switch (direction) {
    case voxelsprout::sim::TrackDirection::East:
        return voxelsprout::core::Dir6::PosX;
    case voxelsprout::sim::TrackDirection::West:
        return voxelsprout::core::Dir6::NegX;
    case voxelsprout::sim::TrackDirection::South:
        return voxelsprout::core::Dir6::PosZ;
    case voxelsprout::sim::TrackDirection::North:
    default:
        return voxelsprout::core::Dir6::NegZ;
    }
}

voxelsprout::core::Dir6 firstDirFromMask(std::uint8_t mask) {
    for (const voxelsprout::core::Dir6 dir : voxelsprout::core::kAllDir6) {
        if ((mask & voxelsprout::core::dirBit(dir)) != 0u) {
            return dir;
        }
    }
    return voxelsprout::core::Dir6::PosY;
}

voxelsprout::core::Dir6 resolveStraightAxisFromMask(std::uint8_t mask, voxelsprout::core::Dir6 preferredAxis) {
    for (const voxelsprout::core::Dir6 dir : voxelsprout::core::kAllDir6) {
        if ((mask & voxelsprout::core::dirBit(dir)) == 0u) {
            continue;
        }
        const voxelsprout::core::Dir6 opposite = voxelsprout::core::oppositeDir(dir);
        if ((mask & voxelsprout::core::dirBit(opposite)) == 0u) {
            continue;
        }
        if (dirSharesAxis(preferredAxis, dir)) {
            return preferredAxis;
        }
        return dir;
    }
    return preferredAxis;
}

} // namespace

namespace voxelsprout::app {

bool App::init() {
    using Clock = std::chrono::steady_clock;
    const auto initStart = Clock::now();
    auto elapsedMs = [](const Clock::time_point& start) -> std::int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    };

    VOX_LOGI("app") << "init begin";
    glfwSetErrorCallback(glfwErrorCallback);

    const auto glfwStart = Clock::now();
    if (glfwInit() == GLFW_FALSE) {
        VOX_LOGE("app") << "glfwInit failed";
        return false;
    }
    VOX_LOGI("app") << "init step glfwInit took " << elapsedMs(glfwStart) << " ms";

    // Vulkan renderer path requires no OpenGL context.
    const auto windowStart = Clock::now();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(1280, 720, "voxel_factory_toy", nullptr, nullptr);
    if (m_window == nullptr) {
        VOX_LOGE("app") << "glfwCreateWindow failed";
        glfwTerminate();
        return false;
    }
    VOX_LOGI("app") << "init step createWindow took " << elapsedMs(windowStart) << " ms";

    // Relative mouse mode for camera look.
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    const auto worldLoadStart = Clock::now();
    const std::filesystem::path worldPath{kWorldFilePath};
    voxelsprout::world::World::LoadResult worldLoadResult{};
    if (m_world.loadOrInitialize(worldPath, &worldLoadResult)) {
        const auto worldLoadMs = elapsedMs(worldLoadStart);
        VOX_LOGI("app") << "loaded world from " << std::filesystem::absolute(worldPath).string()
                        << " in " << worldLoadMs << " ms";
    } else if (worldLoadResult.initializedFallback) {
        const auto worldLoadMs = elapsedMs(worldLoadStart);
        VOX_LOGW("app") << "world file missing/invalid at " << std::filesystem::absolute(worldPath).string()
                        << "; using empty world (press R to regenerate) in " << worldLoadMs << " ms";
    }

    const auto magicaStampStart = Clock::now();
    constexpr std::array<voxelsprout::world::World::MagicaStampSpec, 3> kMagicaLoadSpecs = {
        voxelsprout::world::World::MagicaStampSpec{kMagicaCastlePath, 0.0f, 0.0f, 0.0f, 1.0f},
        voxelsprout::world::World::MagicaStampSpec{kMagicaTeapotPath, 64.0f, 0.0f, 0.0f, 0.36f},
        voxelsprout::world::World::MagicaStampSpec{kMagicaMonu2Path, -72.0f, 0.0f, 16.0f, 0.25f},
    };
    const voxelsprout::world::World::MagicaStampResult stampResult = m_world.stampMagicaResources(kMagicaLoadSpecs);
    VOX_LOGI("app") << "stamped " << stampResult.stampedResourceCount << "/" << kMagicaLoadSpecs.size()
                    << " magica resources into world (voxels=" << stampResult.stampedVoxelCount
                    << ", clipped=" << stampResult.clippedVoxelCount
                    << ", paletteColors=" << static_cast<std::uint32_t>(stampResult.baseColorPaletteCount)
                    << ") in " << elapsedMs(magicaStampStart) << " ms";
    m_renderer.setVoxelBaseColorPalette(stampResult.baseColorPalette);

    const auto clipmapStart = Clock::now();
    m_appliedClipmapConfig = m_renderer.clipmapQueryConfig();
    m_hasAppliedClipmapConfig = true;
    m_chunkClipmapIndex.setConfig(m_appliedClipmapConfig);
    m_chunkClipmapIndex.rebuild(m_world.chunkGrid());
    VOX_LOGI("app") << "chunk clipmap index rebuilt (" << m_chunkClipmapIndex.chunkCount()
                    << " chunks) in " << elapsedMs(clipmapStart) << " ms";

    const auto simInitStart = Clock::now();
    m_simulation.initializeSingleBelt();
    VOX_LOGI("app") << "init step simulation initialize took " << elapsedMs(simInitStart) << " ms";

    const auto rendererInitStart = Clock::now();
    const bool rendererOk = m_renderer.init(m_window, m_world.chunkGrid());
    const auto rendererInitMs = elapsedMs(rendererInitStart);
    VOX_LOGI("app") << "init step renderer init took " << rendererInitMs << " ms";
    if (!rendererOk) {
        VOX_LOGE("app") << "renderer init failed";
        return false;
    }

    VOX_LOGI("app") << "init complete in " << elapsedMs(initStart) << " ms";
    return true;
}

void App::run() {
    VOX_LOGI("app") << "run begin";
    VOX_LOGI("app") << "voxel edit mode disabled by default (press V to toggle)";
    double previousTime = glfwGetTime();
    double simulationAccumulatorSeconds = 0.0;
    uint64_t frameCount = 0;

    while (m_window != nullptr && glfwWindowShouldClose(m_window) == GLFW_FALSE) {
        const double currentTime = glfwGetTime();
        const double rawFrameSeconds = std::max(0.0, currentTime - previousTime);
        previousTime = currentTime;
        const double frameSeconds = std::min(rawFrameSeconds, kFrameDeltaClampSeconds);
        const float dt = static_cast<float>(frameSeconds);
        simulationAccumulatorSeconds += frameSeconds;

        pollInput();
        if (m_input.quitRequested) {
            glfwSetWindowShouldClose(m_window, GLFW_TRUE);
            break;
        }
        if (glfwWindowShouldClose(m_window) == GLFW_TRUE) {
            break;
        }

        int simulationStepCount = 0;
        while (simulationAccumulatorSeconds >= kSimulationFixedStepSeconds &&
               simulationStepCount < kMaxSimulationStepsPerFrame) {
            m_simulation.update(static_cast<float>(kSimulationFixedStepSeconds));
            simulationAccumulatorSeconds -= kSimulationFixedStepSeconds;
            ++simulationStepCount;
        }
        if (simulationStepCount == kMaxSimulationStepsPerFrame &&
            simulationAccumulatorSeconds >= kSimulationFixedStepSeconds) {
            // Drop excess backlog to keep simulation responsive after long stalls.
            simulationAccumulatorSeconds = std::fmod(simulationAccumulatorSeconds, kSimulationFixedStepSeconds);
        }

        const float simulationAlpha = static_cast<float>(
            std::clamp(simulationAccumulatorSeconds / kSimulationFixedStepSeconds, 0.0, 1.0)
        );
        update(dt, simulationAlpha);
        ++frameCount;
    }

    VOX_LOGI("app") << "run exit after " << frameCount
                    << " frame(s), windowShouldClose="
                    << (m_window != nullptr ? glfwWindowShouldClose(m_window) : 1);
}

void App::update(float dt, float simulationAlpha) {
    updateCamera(dt);

    if (!m_debugUiVisible && m_input.toggleVoxelEditModeDown && !m_wasToggleVoxelEditModeDown) {
        m_voxelEditModeEnabled = !m_voxelEditModeEnabled;
        VOX_LOGI("app") << "voxel edit mode "
                        << (m_voxelEditModeEnabled ? "enabled" : "disabled")
                        << " (V)";
    }
    m_wasToggleVoxelEditModeDown = m_input.toggleVoxelEditModeDown;

    const bool regeneratePressedThisFrame =
        !m_debugUiVisible && m_input.regenerateWorldDown && !m_wasRegenerateWorldDown;
    m_wasRegenerateWorldDown = m_input.regenerateWorldDown;
    if (regeneratePressedThisFrame) {
        regenerateWorld();
    }

    const CameraRaycastResult raycast = raycastFromCamera();

    const bool blockInteractionEnabled = !m_debugUiVisible && m_voxelEditModeEnabled;
    const bool placePressedThisFrame = blockInteractionEnabled && m_input.placeBlockDown && !m_wasPlaceBlockDown;
    const bool removePressedThisFrame = blockInteractionEnabled && m_input.removeBlockDown && !m_wasRemoveBlockDown;
    m_wasPlaceBlockDown = m_input.placeBlockDown;
    m_wasRemoveBlockDown = m_input.removeBlockDown;

    bool voxelChunkEdited = false;
    std::vector<std::size_t> editedChunkIndices;
    if (isPipeHotbarSelected()) {
        if (placePressedThisFrame) {
            (void)tryPlacePipeFromCameraRay();
        }
        if (removePressedThisFrame) {
            (void)tryRemovePipeFromCameraRay();
        }
    } else if (isConveyorHotbarSelected()) {
        if (placePressedThisFrame) {
            (void)tryPlaceBeltFromCameraRay();
        }
        if (removePressedThisFrame) {
            (void)tryRemoveBeltFromCameraRay();
        }
    } else if (isTrackHotbarSelected()) {
        if (placePressedThisFrame) {
            (void)tryPlaceTrackFromCameraRay();
        }
        if (removePressedThisFrame) {
            (void)tryRemoveTrackFromCameraRay();
        }
    } else {
        if (placePressedThisFrame && tryPlaceVoxelFromCameraRay(editedChunkIndices)) {
            voxelChunkEdited = true;
        }
        if (removePressedThisFrame && tryRemoveVoxelFromCameraRay(editedChunkIndices)) {
            voxelChunkEdited = true;
        }
    }

    if (voxelChunkEdited) {
        if (!m_renderer.updateChunkMesh(m_world.chunkGrid(), std::span<const std::size_t>(editedChunkIndices))) {
            VOX_LOGE("app") << "chunk mesh update failed after voxel edit";
        }
        m_worldDirty = true;
        m_worldAutosaveElapsedSeconds = 0.0f;
    }

    if (m_worldDirty) {
        m_worldAutosaveElapsedSeconds += std::max(0.0f, dt);
        if (m_worldAutosaveElapsedSeconds >= kWorldAutosaveDelaySeconds) {
            const std::filesystem::path worldPath{kWorldFilePath};
            if (!m_world.save(worldPath)) {
                VOX_LOGE("app") << "failed to autosave world to " << worldPath.string();
            } else {
                VOX_LOGD("app") << "autosaved world to " << worldPath.string();
                m_worldDirty = false;
                m_worldAutosaveElapsedSeconds = 0.0f;
            }
        }
    }

    if (m_dayCycleEnabled) {
        m_dayCyclePhase += std::max(dt, 0.0f) * kDayCycleSpeedCyclesPerSecond;
        m_dayCyclePhase -= std::floor(m_dayCyclePhase);
        // Winter solar arc model:
        // - fixed latitude + declination
        // - hour angle advances through a full day
        // - yields low sun altitude and modest azimuth drift (SE -> S -> SW)
        const float latitudeRadians = voxelsprout::math::radians(kDayCycleLatitudeDegrees);
        const float declinationRadians = voxelsprout::math::radians(kDayCycleWinterDeclinationDegrees);
        const float hourAngleRadians = ((m_dayCyclePhase * 360.0f) - 180.0f) * (kTwoPi / 360.0f);

        const float sinLat = std::sin(latitudeRadians);
        const float cosLat = std::cos(latitudeRadians);
        const float sinDec = std::sin(declinationRadians);
        const float cosDec = std::cos(declinationRadians);
        const float sinHour = std::sin(hourAngleRadians);
        const float cosHour = std::cos(hourAngleRadians);

        // Local ENU components of sun direction.
        // Solar convention: negative hour angle = morning (east), positive = afternoon (west).
        const float sunEast = -cosDec * sinHour;
        const float sunNorth = (cosLat * sinDec) - (sinLat * cosDec * cosHour);
        const float sunUp = (sinLat * sinDec) + (cosLat * cosDec * cosHour);

        const float sunPitchDegrees = voxelsprout::math::degrees(std::asin(std::clamp(sunUp, -1.0f, 1.0f)));
        float sunAzimuthDegrees = voxelsprout::math::degrees(std::atan2(sunEast, sunNorth));
        if (sunAzimuthDegrees < 0.0f) {
            sunAzimuthDegrees += 360.0f;
        }

        // Convert azimuth (north=0, east=90, south=180) to engine yaw
        // where yaw 0 = +X (east), yaw 90 = +Z (south), yaw -90 = -Z (north).
        const float sunYawDegrees = wrapDegreesSigned((sunAzimuthDegrees - 90.0f) + kDayCycleAzimuthOffsetDegrees);
        m_renderer.setSunAngles(sunYawDegrees, sunPitchDegrees);
    }

    voxelsprout::render::VoxelPreview preview{};
    const bool pipeSelected = isPipeHotbarSelected();
    const bool conveyorSelected = isConveyorHotbarSelected();
    const bool trackSelected = isTrackHotbarSelected();
    if (!m_debugUiVisible && m_voxelEditModeEnabled) {
        const bool showRemovePreview = m_input.removeBlockDown;
        if (pipeSelected || conveyorSelected || trackSelected) {
            const InteractionRaycastResult pipeRaycast = raycastInteractionFromCamera(true);
            if (pipeRaycast.hit && pipeRaycast.hitDistance <= kBlockInteractMaxDistance) {
                preview.pipeStyle = true;
                if (pipeSelected) {
                    preview.pipeRadius = 0.45f;
                    preview.pipeStyleId = 0.0f;
                } else if (conveyorSelected) {
                    preview.pipeRadius = 0.49f;
                    preview.pipeStyleId = 1.0f;
                } else {
                    preview.pipeRadius = 0.38f;
                    preview.pipeStyleId = 2.0f;
                }
                if (showRemovePreview) {
                    if (pipeSelected && pipeRaycast.hitPipe) {
                        preview.visible = true;
                        preview.mode = voxelsprout::render::VoxelPreview::Mode::Remove;
                        preview.x = pipeRaycast.x;
                        preview.y = pipeRaycast.y;
                        preview.z = pipeRaycast.z;
                        preview.brushSize = 1;
                        std::size_t pipeIndex = 0;
                        if (isPipeAtWorld(pipeRaycast.x, pipeRaycast.y, pipeRaycast.z, &pipeIndex)) {
                            const voxelsprout::sim::Pipe& pipe = m_simulation.pipes()[pipeIndex];
                            preview.pipeAxisX = pipe.axis.x;
                            preview.pipeAxisY = pipe.axis.y;
                            preview.pipeAxisZ = pipe.axis.z;
                        }
                    } else if (conveyorSelected && pipeRaycast.hitBelt) {
                        preview.visible = true;
                        preview.mode = voxelsprout::render::VoxelPreview::Mode::Remove;
                        preview.x = pipeRaycast.x;
                        preview.y = pipeRaycast.y;
                        preview.z = pipeRaycast.z;
                        preview.brushSize = 1;
                        std::size_t beltIndex = 0;
                        if (isBeltAtWorld(pipeRaycast.x, pipeRaycast.y, pipeRaycast.z, &beltIndex)) {
                            const voxelsprout::sim::Belt& belt = m_simulation.belts()[beltIndex];
                            const voxelsprout::core::Dir6 beltDir = beltDirectionToDir6(belt.direction);
                            const voxelsprout::core::Cell3i axis = voxelsprout::core::dirToOffset(beltDir);
                            preview.pipeAxisX = static_cast<float>(axis.x);
                            preview.pipeAxisY = static_cast<float>(axis.y);
                            preview.pipeAxisZ = static_cast<float>(axis.z);
                        }
                    } else if (trackSelected && pipeRaycast.hitTrack) {
                        preview.visible = true;
                        preview.mode = voxelsprout::render::VoxelPreview::Mode::Remove;
                        preview.x = pipeRaycast.x;
                        preview.y = pipeRaycast.y;
                        preview.z = pipeRaycast.z;
                        preview.brushSize = 1;
                        std::size_t trackIndex = 0;
                        if (isTrackAtWorld(pipeRaycast.x, pipeRaycast.y, pipeRaycast.z, &trackIndex)) {
                            const voxelsprout::sim::Track& track = m_simulation.tracks()[trackIndex];
                            const voxelsprout::core::Dir6 trackDir = trackDirectionToDir6(track.direction);
                            const voxelsprout::core::Cell3i axis = voxelsprout::core::dirToOffset(trackDir);
                            preview.pipeAxisX = static_cast<float>(axis.x);
                            preview.pipeAxisY = static_cast<float>(axis.y);
                            preview.pipeAxisZ = static_cast<float>(axis.z);
                        }
                    }
                } else {
                    int targetX = 0;
                    int targetY = 0;
                    int targetZ = 0;
                    int axisX = 0;
                    int axisY = 1;
                    int axisZ = 0;
                    bool hasPlacement = false;
                    if (pipeSelected) {
                        hasPlacement = computePipePlacementFromInteractionRaycast(
                            pipeRaycast,
                            targetX,
                            targetY,
                            targetZ,
                            axisX,
                            axisY,
                            axisZ
                        );
                    } else if (conveyorSelected) {
                        hasPlacement = computeBeltPlacementFromInteractionRaycast(
                            pipeRaycast,
                            targetX,
                            targetY,
                            targetZ,
                            axisX,
                            axisY,
                            axisZ
                        );
                    } else if (trackSelected) {
                        hasPlacement = computeTrackPlacementFromInteractionRaycast(
                            pipeRaycast,
                            targetX,
                            targetY,
                            targetZ,
                            axisX,
                            axisY,
                            axisZ
                        );
                    }
                    if (hasPlacement) {
                        preview.visible = true;
                        preview.mode = voxelsprout::render::VoxelPreview::Mode::Add;
                        preview.x = targetX;
                        preview.y = targetY;
                        preview.z = targetZ;
                        preview.brushSize = 1;
                        preview.pipeAxisX = static_cast<float>(axisX);
                        preview.pipeAxisY = static_cast<float>(axisY);
                        preview.pipeAxisZ = static_cast<float>(axisZ);
                    }
                }
            }
        } else if (raycast.hitSolid && raycast.hitDistance <= kBlockInteractMaxDistance) {
            if (raycast.hasHitFaceNormal) {
                auto normalToFaceId = [](int nx, int ny, int nz) -> uint32_t {
                    if (nx > 0) { return 0u; }
                    if (nx < 0) { return 1u; }
                    if (ny > 0) { return 2u; }
                    if (ny < 0) { return 3u; }
                    if (nz > 0) { return 4u; }
                    return 5u;
                };
                preview.faceVisible = true;
                preview.faceX = raycast.solidX;
                preview.faceY = raycast.solidY;
                preview.faceZ = raycast.solidZ;
                preview.faceId = normalToFaceId(raycast.hitFaceNormalX, raycast.hitFaceNormalY, raycast.hitFaceNormalZ);
            }

            if (showRemovePreview) {
                preview.visible = true;
                preview.mode = voxelsprout::render::VoxelPreview::Mode::Remove;
                preview.x = raycast.solidX;
                preview.y = raycast.solidY;
                preview.z = raycast.solidZ;
                preview.brushSize = 1;
            } else {
                int targetX = 0;
                int targetY = 0;
                int targetZ = 0;
                if (computePlacementVoxelFromRaycast(raycast, targetX, targetY, targetZ)) {
                    if (isWorldVoxelInBounds(targetX, targetY, targetZ)) {
                        preview.visible = true;
                        preview.mode = voxelsprout::render::VoxelPreview::Mode::Add;
                        preview.x = targetX;
                        preview.y = targetY;
                        preview.z = targetZ;
                        preview.brushSize = 1;
                    }
                }
            }
        }
    }

    const voxelsprout::render::CameraPose cameraPose{
        m_camera.x,
        m_camera.y,
        m_camera.z,
        m_camera.yawDegrees,
        m_camera.pitchDegrees,
        m_camera.fovDegrees
    };

    const voxelsprout::world::ClipmapConfig requestedClipmapConfig = m_renderer.clipmapQueryConfig();
    if (!m_hasAppliedClipmapConfig ||
        requestedClipmapConfig.levelCount != m_appliedClipmapConfig.levelCount ||
        requestedClipmapConfig.gridResolution != m_appliedClipmapConfig.gridResolution ||
        requestedClipmapConfig.baseVoxelSize != m_appliedClipmapConfig.baseVoxelSize ||
        requestedClipmapConfig.brickResolution != m_appliedClipmapConfig.brickResolution) {
        m_appliedClipmapConfig = requestedClipmapConfig;
        m_hasAppliedClipmapConfig = true;
        m_chunkClipmapIndex.setConfig(m_appliedClipmapConfig);
        m_chunkClipmapIndex.rebuild(m_world.chunkGrid());
        VOX_LOGI("app") << "clipmap config changed, rebuilt clipmap index (levels="
                        << m_appliedClipmapConfig.levelCount
                        << ", grid=" << m_appliedClipmapConfig.gridResolution
                        << ", baseVoxel=" << m_appliedClipmapConfig.baseVoxelSize
                        << ", brick=" << m_appliedClipmapConfig.brickResolution
                        << ")";
    }

    m_visibleChunkIndices.clear();
    voxelsprout::world::SpatialQueryStats spatialQueryStats{};
    bool spatialQueriesUsed = false;
    if (m_renderer.useSpatialPartitioningQueries()) {
        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(m_window, &framebufferWidth, &framebufferHeight);
        const float aspectRatio =
            (framebufferWidth > 0 && framebufferHeight > 0)
                ? static_cast<float>(framebufferWidth) / static_cast<float>(framebufferHeight)
                : kRenderAspectFallback;
        const CameraFrustum cameraFrustum = buildCameraFrustum(
            voxelsprout::math::Vector3{m_camera.x, m_camera.y, m_camera.z},
            m_camera.yawDegrees,
            m_camera.pitchDegrees,
            m_camera.fovDegrees,
            aspectRatio
        );
        if (cameraFrustum.valid && m_chunkClipmapIndex.valid()) {
            m_chunkClipmapIndex.updateCamera(m_camera.x, m_camera.y, m_camera.z, &spatialQueryStats);
            std::vector<std::size_t> candidateChunkIndices =
                m_chunkClipmapIndex.queryChunksIntersecting(cameraFrustum.broadPhaseBounds, &spatialQueryStats);
            spatialQueriesUsed = true;
            m_visibleChunkIndices.reserve(candidateChunkIndices.size());
            const std::vector<voxelsprout::world::Chunk>& chunks = m_world.chunkGrid().chunks();
            for (std::size_t chunkIndex : candidateChunkIndices) {
                if (chunkIndex >= chunks.size()) {
                    continue;
                }
                if (chunkIntersectsFrustum(chunks[chunkIndex], cameraFrustum.planes, kRenderFrustumPlaneSlackVoxels)) {
                    m_visibleChunkIndices.push_back(chunkIndex);
                }
            }
            std::sort(m_visibleChunkIndices.begin(), m_visibleChunkIndices.end());
            m_visibleChunkIndices.erase(
                std::unique(m_visibleChunkIndices.begin(), m_visibleChunkIndices.end()),
                m_visibleChunkIndices.end()
            );
            spatialQueryStats.visibleChunkCount = static_cast<std::uint32_t>(m_visibleChunkIndices.size());
        }
    }

    if (m_visibleChunkIndices.empty() && (!spatialQueriesUsed || !m_chunkClipmapIndex.valid())) {
        const std::size_t chunkCount = m_world.chunkGrid().chunks().size();
        m_visibleChunkIndices.resize(chunkCount);
        for (std::size_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
            m_visibleChunkIndices[chunkIndex] = chunkIndex;
        }
    }
    m_renderer.setSpatialQueryStats(
        spatialQueriesUsed,
        spatialQueryStats,
        static_cast<std::uint32_t>(m_visibleChunkIndices.size())
    );

    m_renderer.renderFrame(
        m_world.chunkGrid(),
        m_simulation,
        cameraPose,
        preview,
        simulationAlpha,
        std::span<const std::size_t>(m_visibleChunkIndices)
    );
    m_camera.fovDegrees = m_renderer.cameraFovDegrees();
}

void App::shutdown() {
    VOX_LOGI("app") << "shutdown begin";

    if (m_worldDirty) {
        const std::filesystem::path worldPath{kWorldFilePath};
        if (!m_world.save(worldPath)) {
            VOX_LOGE("app") << "failed to save dirty world on shutdown to " << worldPath.string();
        } else {
            VOX_LOGI("app") << "saved dirty world on shutdown to " << worldPath.string();
            m_worldDirty = false;
            m_worldAutosaveElapsedSeconds = 0.0f;
        }
    }

    m_renderer.shutdown();

    if (m_window != nullptr) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }

    glfwTerminate();
    VOX_LOGI("app") << "shutdown complete";
}

void App::pollInput() {
    glfwPollEvents();

    bool uiVisibilityChanged = false;
    const bool toggleFrameStatsDown = glfwGetKey(m_window, GLFW_KEY_F) == GLFW_PRESS;
    if (toggleFrameStatsDown && !m_wasToggleFrameStatsDown) {
        m_renderer.setFrameStatsVisible(!m_renderer.isFrameStatsVisible());
    }
    m_wasToggleFrameStatsDown = toggleFrameStatsDown;

    const bool toggleConfigUiDown = glfwGetKey(m_window, GLFW_KEY_C) == GLFW_PRESS;
    if (toggleConfigUiDown && !m_wasToggleConfigUiDown) {
        m_debugUiVisible = !m_debugUiVisible;
        uiVisibilityChanged = true;
    }
    m_wasToggleConfigUiDown = toggleConfigUiDown;

    const bool toggleDayCycleDown = glfwGetKey(m_window, GLFW_KEY_T) == GLFW_PRESS;
    if (toggleDayCycleDown && !m_wasToggleDayCycleDown) {
        m_dayCycleEnabled = !m_dayCycleEnabled;
        VOX_LOGI("app") << "day cycle " << (m_dayCycleEnabled ? "enabled" : "disabled")
                        << " (T, winter arc lat=" << kDayCycleLatitudeDegrees
                        << " decl=" << kDayCycleWinterDeclinationDegrees << ")";
    }
    m_wasToggleDayCycleDown = toggleDayCycleDown;

    m_input.toggleVoxelEditModeDown = glfwGetKey(m_window, GLFW_KEY_V) == GLFW_PRESS;

    m_renderer.setDebugUiVisible(m_debugUiVisible);
    const bool rendererUiVisible = m_renderer.isDebugUiVisible();
    if (rendererUiVisible != m_debugUiVisible) {
        m_debugUiVisible = rendererUiVisible;
        uiVisibilityChanged = true;
    }
    if (uiVisibilityChanged) {
        glfwSetInputMode(m_window, GLFW_CURSOR, m_debugUiVisible ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
        m_hasMouseSample = false;
    }

    m_input.quitRequested = glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
    m_input.moveForward = glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS;
    m_input.moveBackward = glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS;
    m_input.moveLeft = glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS;
    m_input.moveRight = glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS;
    m_input.moveUp = glfwGetKey(m_window, GLFW_KEY_SPACE) == GLFW_PRESS;
    m_input.moveDown =
        glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(m_window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    m_input.toggleHoverDown = glfwGetKey(m_window, GLFW_KEY_H) == GLFW_PRESS;
    m_input.regenerateWorldDown = glfwGetKey(m_window, GLFW_KEY_R) == GLFW_PRESS;
    bool controllerPlaceDown = false;
    bool controllerRemoveDown = false;
    bool controllerPrevBlockDown = false;
    bool controllerNextBlockDown = false;
    bool controllerMoveUpDown = false;
    bool controllerMoveDownDown = false;
    float controllerMoveForward = 0.0f;
    float controllerMoveRight = 0.0f;
    float controllerLookX = 0.0f;
    float controllerLookY = 0.0f;
    GLFWgamepadstate gamepadState{};
    const bool hasGamepad =
        glfwJoystickIsGamepad(GLFW_JOYSTICK_1) == GLFW_TRUE &&
        glfwGetGamepadState(GLFW_JOYSTICK_1, &gamepadState) == GLFW_TRUE;
    if (hasGamepad != m_gamepadConnected) {
        m_gamepadConnected = hasGamepad;
        if (m_gamepadConnected) {
            VOX_LOGI("app") << "gamepad connected: RT place, LT remove, LB/RB hotbar";
        } else {
            VOX_LOGI("app") << "gamepad disconnected";
        }
    }
    if (hasGamepad) {
        controllerPlaceDown = gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] > kGamepadTriggerPressedThreshold;
        controllerRemoveDown = gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER] > kGamepadTriggerPressedThreshold;
        controllerPrevBlockDown = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] == GLFW_PRESS;
        controllerNextBlockDown = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] == GLFW_PRESS;
        controllerMoveUpDown = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_A] == GLFW_PRESS;
        controllerMoveDownDown = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_B] == GLFW_PRESS;
        controllerMoveForward = -applyStickDeadzone(gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_Y], kGamepadMoveDeadzone);
        controllerMoveRight = applyStickDeadzone(gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_X], kGamepadMoveDeadzone);
        controllerLookX = applyStickDeadzone(gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_X], kGamepadLookDeadzone);
        controllerLookY = -applyStickDeadzone(gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y], kGamepadLookDeadzone);
    }

    const bool prevHotbarDown = controllerPrevBlockDown;
    const bool nextHotbarDown = controllerNextBlockDown;
    if (!m_debugUiVisible && prevHotbarDown && !m_wasPrevBlockDown) {
        cycleSelectedHotbar(-1);
    }
    if (!m_debugUiVisible && nextHotbarDown && !m_wasNextBlockDown) {
        cycleSelectedHotbar(+1);
    }
    m_wasPrevBlockDown = prevHotbarDown;
    m_wasNextBlockDown = nextHotbarDown;

    if (!m_debugUiVisible) {
        if (glfwGetKey(m_window, GLFW_KEY_1) == GLFW_PRESS) {
            selectHotbarSlot(kHotbarSlotBlock);
        } else if (glfwGetKey(m_window, GLFW_KEY_2) == GLFW_PRESS) {
            selectHotbarSlot(kHotbarSlotPipe);
        } else if (glfwGetKey(m_window, GLFW_KEY_3) == GLFW_PRESS) {
            selectHotbarSlot(kHotbarSlotConveyor);
        } else if (glfwGetKey(m_window, GLFW_KEY_4) == GLFW_PRESS) {
            selectHotbarSlot(kHotbarSlotTrack);
        }
        if (m_selectedHotbarIndex == kHotbarSlotBlock) {
            if (glfwGetKey(m_window, GLFW_KEY_5) == GLFW_PRESS) {
                selectPlaceableBlock(0);
            } else if (glfwGetKey(m_window, GLFW_KEY_6) == GLFW_PRESS) {
                selectPlaceableBlock(1);
            } else if (glfwGetKey(m_window, GLFW_KEY_7) == GLFW_PRESS) {
                selectPlaceableBlock(2);
            } else if (glfwGetKey(m_window, GLFW_KEY_8) == GLFW_PRESS) {
                selectPlaceableBlock(3);
            } else if (glfwGetKey(m_window, GLFW_KEY_9) == GLFW_PRESS) {
                selectPlaceableBlock(4);
            }
        }
    }

    m_input.placeBlockDown =
        glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS ||
        controllerPlaceDown;
    m_input.removeBlockDown =
        glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS ||
        controllerRemoveDown;
    m_input.moveUp = m_input.moveUp || controllerMoveUpDown;
    m_input.moveDown = m_input.moveDown || controllerMoveDownDown;
    m_input.gamepadMoveForward = controllerMoveForward;
    m_input.gamepadMoveRight = controllerMoveRight;
    m_input.gamepadLookX = controllerLookX;
    m_input.gamepadLookY = controllerLookY;

    double mouseX = 0.0;
    double mouseY = 0.0;
    glfwGetCursorPos(m_window, &mouseX, &mouseY);

    if (!m_hasMouseSample) {
        m_lastMouseX = mouseX;
        m_lastMouseY = mouseY;
        m_hasMouseSample = true;
    }

    m_input.mouseDeltaX = static_cast<float>(mouseX - m_lastMouseX);
    m_input.mouseDeltaY = static_cast<float>(mouseY - m_lastMouseY);

    if (m_debugUiVisible) {
        m_input.mouseDeltaX = 0.0f;
        m_input.mouseDeltaY = 0.0f;
    }

    m_lastMouseX = mouseX;
    m_lastMouseY = mouseY;
}

void App::updateCamera(float dt) {
    if (m_input.toggleHoverDown && !m_wasToggleHoverDown) {
        m_hoverEnabled = !m_hoverEnabled;
        m_camera.velocityY = 0.0f;
        VOX_LOGI("app") << "hover " << (m_hoverEnabled ? "enabled" : "disabled") << " (H)";
    }
    m_wasToggleHoverDown = m_input.toggleHoverDown;

    const float mouseSmoothingAlpha = 1.0f - std::exp(-dt / kMouseSmoothingSeconds);
    m_camera.smoothedMouseDeltaX += (m_input.mouseDeltaX - m_camera.smoothedMouseDeltaX) * mouseSmoothingAlpha;
    m_camera.smoothedMouseDeltaY += (m_input.mouseDeltaY - m_camera.smoothedMouseDeltaY) * mouseSmoothingAlpha;

    m_camera.yawDegrees += m_camera.smoothedMouseDeltaX * kMouseSensitivity;
    m_camera.pitchDegrees += m_camera.smoothedMouseDeltaY * kMouseSensitivity;
    m_camera.yawDegrees += m_input.gamepadLookX * kGamepadLookDegreesPerSecond * dt;
    m_camera.pitchDegrees += m_input.gamepadLookY * kGamepadLookDegreesPerSecond * dt;
    m_camera.pitchDegrees = std::clamp(m_camera.pitchDegrees, kPitchMinDegrees, kPitchMaxDegrees);

    const float yawRadians = voxelsprout::math::radians(m_camera.yawDegrees);
    const voxelsprout::math::Vector3 forward{std::cos(yawRadians), 0.0f, std::sin(yawRadians)};
    const voxelsprout::math::Vector3 right{-forward.z, 0.0f, forward.x};
    voxelsprout::math::Vector3 moveDirection{};

    float moveForwardInput = m_input.gamepadMoveForward;
    float moveRightInput = m_input.gamepadMoveRight;
    if (m_input.moveForward) {
        moveForwardInput += 1.0f;
    }
    if (m_input.moveBackward) {
        moveForwardInput -= 1.0f;
    }
    if (m_input.moveRight) {
        moveRightInput += 1.0f;
    }
    if (m_input.moveLeft) {
        moveRightInput -= 1.0f;
    }
    moveForwardInput = std::clamp(moveForwardInput, -1.0f, 1.0f);
    moveRightInput = std::clamp(moveRightInput, -1.0f, 1.0f);

    moveDirection += forward * moveForwardInput;
    moveDirection += right * moveRightInput;

    const float moveLengthSq = voxelsprout::math::lengthSquared(moveDirection);
    const float moveLength = std::sqrt(moveLengthSq);
    float targetVelocityX = 0.0f;
    float targetVelocityZ = 0.0f;
    if (moveLength > 0.0f) {
        moveDirection /= moveLength;
        const voxelsprout::math::Vector3 targetVelocity = moveDirection * kMoveMaxSpeed;
        targetVelocityX = targetVelocity.x;
        targetVelocityZ = targetVelocity.z;
    }

    const float accelPerFrame = kMoveAcceleration * dt;
    const float decelPerFrame = kMoveDeceleration * dt;

    const float maxDeltaX = (std::fabs(targetVelocityX) > std::fabs(m_camera.velocityX)) ? accelPerFrame : decelPerFrame;
    const float maxDeltaZ = (std::fabs(targetVelocityZ) > std::fabs(m_camera.velocityZ)) ? accelPerFrame : decelPerFrame;

    m_camera.velocityX = approach(m_camera.velocityX, targetVelocityX, maxDeltaX);
    m_camera.velocityZ = approach(m_camera.velocityZ, targetVelocityZ, maxDeltaZ);

    if (m_hoverEnabled) {
        float hoverVerticalSpeed = 0.0f;
        int supportY = 0;
        if (findGroundSupportY(m_camera.x, m_camera.y, m_camera.z, supportY)) {
            const float targetEyeY = static_cast<float>(supportY + 1) + kPlayerEyeHeight + kHoverHeightAboveGround;
            const float yError = targetEyeY - m_camera.y;
            hoverVerticalSpeed = std::clamp(
                yError * kHoverResponse,
                -kHoverMaxVerticalSpeed,
                kHoverMaxVerticalSpeed
            );
        } else {
            hoverVerticalSpeed = 0.0f;
        }

        if (m_input.moveUp) {
            hoverVerticalSpeed = std::max(hoverVerticalSpeed, kHoverManualVerticalSpeed);
        }
        if (m_input.moveDown) {
            hoverVerticalSpeed = std::min(hoverVerticalSpeed, -kHoverManualVerticalSpeed);
        }

        m_camera.velocityY = hoverVerticalSpeed;
        m_camera.onGround = false;
    } else {
        if (m_input.moveUp && m_camera.onGround) {
            m_camera.velocityY = kJumpSpeed;
            m_camera.onGround = false;
        }
        m_camera.velocityY = std::max(m_camera.velocityY + (kGravity * dt), kMaxFallSpeed);
    }
    resolvePlayerCollisions(dt);
}

bool App::isSolidWorldVoxel(int worldX, int worldY, int worldZ) const {
    if (worldY < 0) {
        return true;
    }

    const voxelsprout::world::Chunk* chunk = nullptr;
    int localX = 0;
    int localY = 0;
    int localZ = 0;
    if (worldToChunkLocalConst(worldX, worldY, worldZ, chunk, localX, localY, localZ)) {
        return chunk->isSolid(localX, localY, localZ);
    }

    return false;
}

bool App::worldToChunkLocal(
    int worldX,
    int worldY,
    int worldZ,
    std::size_t& outChunkIndex,
    int& outLocalX,
    int& outLocalY,
    int& outLocalZ
) const {
    const std::vector<voxelsprout::world::Chunk>& chunks = m_world.chunkGrid().chunks();
    for (std::size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        const voxelsprout::world::Chunk& chunk = chunks[chunkIndex];
        const int chunkMinX = chunk.chunkX() * voxelsprout::world::Chunk::kSizeX;
        const int chunkMinY = chunk.chunkY() * voxelsprout::world::Chunk::kSizeY;
        const int chunkMinZ = chunk.chunkZ() * voxelsprout::world::Chunk::kSizeZ;
        const int localX = worldX - chunkMinX;
        const int localY = worldY - chunkMinY;
        const int localZ = worldZ - chunkMinZ;
        const bool insideChunk =
            localX >= 0 && localX < voxelsprout::world::Chunk::kSizeX &&
            localY >= 0 && localY < voxelsprout::world::Chunk::kSizeY &&
            localZ >= 0 && localZ < voxelsprout::world::Chunk::kSizeZ;
        if (!insideChunk) {
            continue;
        }

        outChunkIndex = chunkIndex;
        outLocalX = localX;
        outLocalY = localY;
        outLocalZ = localZ;
        return true;
    }

    return false;
}

bool App::worldToChunkLocalConst(
    int worldX,
    int worldY,
    int worldZ,
    const voxelsprout::world::Chunk*& outChunk,
    int& outLocalX,
    int& outLocalY,
    int& outLocalZ
) const {
    std::size_t chunkIndex = 0;
    if (!worldToChunkLocal(worldX, worldY, worldZ, chunkIndex, outLocalX, outLocalY, outLocalZ)) {
        return false;
    }

    outChunk = &m_world.chunkGrid().chunks()[chunkIndex];
    return true;
}

bool App::findGroundSupportY(float eyeX, float eyeY, float eyeZ, int& outSupportY) const {
    const int startX = static_cast<int>(std::floor(eyeX - kPlayerRadius));
    const int endX = static_cast<int>(std::floor(eyeX + kPlayerRadius - kCollisionEpsilon));
    const int startZ = static_cast<int>(std::floor(eyeZ - kPlayerRadius));
    const int endZ = static_cast<int>(std::floor(eyeZ + kPlayerRadius - kCollisionEpsilon));

    const float feetY = eyeY - kPlayerEyeHeight;
    const int topSupportY = static_cast<int>(std::floor(feetY - kCollisionEpsilon)) - 1;
    const int minSupportY = std::max(0, topSupportY - kHoverGroundSearchDepth);

    for (int supportY = topSupportY; supportY >= minSupportY; --supportY) {
        for (int z = startZ; z <= endZ; ++z) {
            for (int x = startX; x <= endX; ++x) {
                if (isSolidWorldVoxel(x, supportY, z)) {
                    outSupportY = supportY;
                    return true;
                }
            }
        }
    }

    return false;
}

bool App::doesPlayerOverlapSolid(float eyeX, float eyeY, float eyeZ) const {
    const Aabb3f playerBounds = makePlayerCollisionAabb(eyeX, eyeY, eyeZ);

    const int startX = static_cast<int>(std::floor(playerBounds.minX));
    const int endX = static_cast<int>(std::floor(playerBounds.maxX - kCollisionEpsilon));
    const int startY = static_cast<int>(std::floor(playerBounds.minY));
    const int endY = static_cast<int>(std::floor(playerBounds.maxY - kCollisionEpsilon));
    const int startZ = static_cast<int>(std::floor(playerBounds.minZ));
    const int endZ = static_cast<int>(std::floor(playerBounds.maxZ - kCollisionEpsilon));

    for (int y = startY; y <= endY; ++y) {
        for (int z = startZ; z <= endZ; ++z) {
            for (int x = startX; x <= endX; ++x) {
                if (isSolidWorldVoxel(x, y, z)) {
                    return true;
                }
            }
        }
    }
    return doesPlayerOverlapConveyorBelt(eyeX, eyeY, eyeZ);
}

bool App::doesPlayerOverlapConveyorBelt(float eyeX, float eyeY, float eyeZ) const {
    const Aabb3f playerBounds = makePlayerCollisionAabb(eyeX, eyeY, eyeZ);
    for (const voxelsprout::sim::Belt& belt : m_simulation.belts()) {
        const Aabb3f beltBounds = makeConveyorBeltAabb(belt);
        if (aabbOverlaps(playerBounds, beltBounds)) {
            return true;
        }
    }
    return false;
}

void App::resolvePlayerCollisions(float dt) {
    const float totalDx = m_camera.velocityX * dt;
    const float totalDy = m_camera.velocityY * dt;
    const float totalDz = m_camera.velocityZ * dt;
    const float maxDelta = std::max({std::fabs(totalDx), std::fabs(totalDy), std::fabs(totalDz)});
    const int steps = std::max(1, static_cast<int>(std::ceil(maxDelta / 0.45f)));
    const float stepDx = totalDx / static_cast<float>(steps);
    const float stepDy = totalDy / static_cast<float>(steps);
    const float stepDz = totalDz / static_cast<float>(steps);

    bool groundedThisFrame = false;

    auto resolveHorizontalX = [&](float deltaX) {
        if (deltaX == 0.0f) {
            return;
        }

        m_camera.x += deltaX;
        if (!doesPlayerOverlapSolid(m_camera.x, m_camera.y, m_camera.z)) {
            return;
        }

        const Aabb3f playerBounds = makePlayerCollisionAabb(m_camera.x, m_camera.y, m_camera.z);
        const int startY = static_cast<int>(std::floor(playerBounds.minY));
        const int endY = static_cast<int>(std::floor(playerBounds.maxY - kCollisionEpsilon));
        const int startZ = static_cast<int>(std::floor(playerBounds.minZ));
        const int endZ = static_cast<int>(std::floor(playerBounds.maxZ - kCollisionEpsilon));
        const int startX = static_cast<int>(std::floor(playerBounds.minX));
        const int endX = static_cast<int>(std::floor(playerBounds.maxX - kCollisionEpsilon));

        if (deltaX > 0.0f) {
            float blockingMinX = std::numeric_limits<float>::infinity();
            for (int y = startY; y <= endY; ++y) {
                for (int z = startZ; z <= endZ; ++z) {
                    for (int x = startX; x <= endX; ++x) {
                        if (isSolidWorldVoxel(x, y, z)) {
                            blockingMinX = std::min(blockingMinX, static_cast<float>(x));
                        }
                    }
                }
            }
            for (const voxelsprout::sim::Belt& belt : m_simulation.belts()) {
                const Aabb3f beltBounds = makeConveyorBeltAabb(belt);
                if (aabbOverlaps(playerBounds, beltBounds)) {
                    blockingMinX = std::min(blockingMinX, beltBounds.minX);
                }
            }
            if (std::isfinite(blockingMinX)) {
                m_camera.x = blockingMinX - kPlayerRadius - kCollisionEpsilon;
            }
        } else {
            float blockingMaxX = -std::numeric_limits<float>::infinity();
            for (int y = startY; y <= endY; ++y) {
                for (int z = startZ; z <= endZ; ++z) {
                    for (int x = startX; x <= endX; ++x) {
                        if (isSolidWorldVoxel(x, y, z)) {
                            blockingMaxX = std::max(blockingMaxX, static_cast<float>(x + 1));
                        }
                    }
                }
            }
            for (const voxelsprout::sim::Belt& belt : m_simulation.belts()) {
                const Aabb3f beltBounds = makeConveyorBeltAabb(belt);
                if (aabbOverlaps(playerBounds, beltBounds)) {
                    blockingMaxX = std::max(blockingMaxX, beltBounds.maxX);
                }
            }
            if (std::isfinite(blockingMaxX)) {
                m_camera.x = blockingMaxX + kPlayerRadius + kCollisionEpsilon;
            }
        }

        if (doesPlayerOverlapSolid(m_camera.x, m_camera.y, m_camera.z)) {
            m_camera.x -= deltaX;
        }
        m_camera.velocityX = 0.0f;
    };

    auto resolveHorizontalZ = [&](float deltaZ) {
        if (deltaZ == 0.0f) {
            return;
        }

        m_camera.z += deltaZ;
        if (!doesPlayerOverlapSolid(m_camera.x, m_camera.y, m_camera.z)) {
            return;
        }

        const Aabb3f playerBounds = makePlayerCollisionAabb(m_camera.x, m_camera.y, m_camera.z);
        const int startX = static_cast<int>(std::floor(playerBounds.minX));
        const int endX = static_cast<int>(std::floor(playerBounds.maxX - kCollisionEpsilon));
        const int startY = static_cast<int>(std::floor(playerBounds.minY));
        const int endY = static_cast<int>(std::floor(playerBounds.maxY - kCollisionEpsilon));
        const int startZ = static_cast<int>(std::floor(playerBounds.minZ));
        const int endZ = static_cast<int>(std::floor(playerBounds.maxZ - kCollisionEpsilon));

        if (deltaZ > 0.0f) {
            float blockingMinZ = std::numeric_limits<float>::infinity();
            for (int y = startY; y <= endY; ++y) {
                for (int z = startZ; z <= endZ; ++z) {
                    for (int x = startX; x <= endX; ++x) {
                        if (isSolidWorldVoxel(x, y, z)) {
                            blockingMinZ = std::min(blockingMinZ, static_cast<float>(z));
                        }
                    }
                }
            }
            for (const voxelsprout::sim::Belt& belt : m_simulation.belts()) {
                const Aabb3f beltBounds = makeConveyorBeltAabb(belt);
                if (aabbOverlaps(playerBounds, beltBounds)) {
                    blockingMinZ = std::min(blockingMinZ, beltBounds.minZ);
                }
            }
            if (std::isfinite(blockingMinZ)) {
                m_camera.z = blockingMinZ - kPlayerRadius - kCollisionEpsilon;
            }
        } else {
            float blockingMaxZ = -std::numeric_limits<float>::infinity();
            for (int y = startY; y <= endY; ++y) {
                for (int z = startZ; z <= endZ; ++z) {
                    for (int x = startX; x <= endX; ++x) {
                        if (isSolidWorldVoxel(x, y, z)) {
                            blockingMaxZ = std::max(blockingMaxZ, static_cast<float>(z + 1));
                        }
                    }
                }
            }
            for (const voxelsprout::sim::Belt& belt : m_simulation.belts()) {
                const Aabb3f beltBounds = makeConveyorBeltAabb(belt);
                if (aabbOverlaps(playerBounds, beltBounds)) {
                    blockingMaxZ = std::max(blockingMaxZ, beltBounds.maxZ);
                }
            }
            if (std::isfinite(blockingMaxZ)) {
                m_camera.z = blockingMaxZ + kPlayerRadius + kCollisionEpsilon;
            }
        }

        if (doesPlayerOverlapSolid(m_camera.x, m_camera.y, m_camera.z)) {
            m_camera.z -= deltaZ;
        }
        m_camera.velocityZ = 0.0f;
    };

    auto resolveVerticalY = [&](float deltaY) {
        if (deltaY == 0.0f) {
            return;
        }

        m_camera.y += deltaY;
        if (!doesPlayerOverlapSolid(m_camera.x, m_camera.y, m_camera.z)) {
            return;
        }

        const Aabb3f playerBounds = makePlayerCollisionAabb(m_camera.x, m_camera.y, m_camera.z);
        const int startX = static_cast<int>(std::floor(playerBounds.minX));
        const int endX = static_cast<int>(std::floor(playerBounds.maxX - kCollisionEpsilon));
        const int startZ = static_cast<int>(std::floor(playerBounds.minZ));
        const int endZ = static_cast<int>(std::floor(playerBounds.maxZ - kCollisionEpsilon));
        const int startY = static_cast<int>(std::floor(playerBounds.minY));
        const int endY = static_cast<int>(std::floor(playerBounds.maxY - kCollisionEpsilon));

        if (deltaY > 0.0f) {
            float blockingMinY = std::numeric_limits<float>::infinity();
            for (int y = startY; y <= endY; ++y) {
                for (int z = startZ; z <= endZ; ++z) {
                    for (int x = startX; x <= endX; ++x) {
                        if (isSolidWorldVoxel(x, y, z)) {
                            blockingMinY = std::min(blockingMinY, static_cast<float>(y));
                        }
                    }
                }
            }
            for (const voxelsprout::sim::Belt& belt : m_simulation.belts()) {
                const Aabb3f beltBounds = makeConveyorBeltAabb(belt);
                if (aabbOverlaps(playerBounds, beltBounds)) {
                    blockingMinY = std::min(blockingMinY, beltBounds.minY);
                }
            }
            if (std::isfinite(blockingMinY)) {
                m_camera.y = blockingMinY - kPlayerTopOffset - kCollisionEpsilon;
            }
        } else {
            float blockingMaxY = -std::numeric_limits<float>::infinity();
            for (int y = startY; y <= endY; ++y) {
                for (int z = startZ; z <= endZ; ++z) {
                    for (int x = startX; x <= endX; ++x) {
                        if (isSolidWorldVoxel(x, y, z)) {
                            blockingMaxY = std::max(blockingMaxY, static_cast<float>(y + 1));
                        }
                    }
                }
            }
            for (const voxelsprout::sim::Belt& belt : m_simulation.belts()) {
                const Aabb3f beltBounds = makeConveyorBeltAabb(belt);
                if (aabbOverlaps(playerBounds, beltBounds)) {
                    blockingMaxY = std::max(blockingMaxY, beltBounds.maxY);
                }
            }
            if (std::isfinite(blockingMaxY)) {
                m_camera.y = blockingMaxY + kPlayerEyeHeight + kCollisionEpsilon;
                groundedThisFrame = true;
            }
        }

        if (doesPlayerOverlapSolid(m_camera.x, m_camera.y, m_camera.z)) {
            m_camera.y -= deltaY;
        }
        m_camera.velocityY = 0.0f;
    };

    for (int step = 0; step < steps; ++step) {
        resolveHorizontalX(stepDx);
        resolveHorizontalZ(stepDz);
        resolveVerticalY(stepDy);
    }

    m_camera.onGround = groundedThisFrame;
}

App::CameraRaycastResult App::raycastFromCamera() const {
    CameraRaycastResult result{};
    if (m_world.chunkGrid().chunks().empty()) {
        return result;
    }

    const float yawRadians = voxelsprout::math::radians(m_camera.yawDegrees);
    const float pitchRadians = voxelsprout::math::radians(m_camera.pitchDegrees);
    const float cosPitch = std::cos(pitchRadians);
    const voxelsprout::math::Vector3 rayDirection = voxelsprout::math::normalize(voxelsprout::math::Vector3{
        std::cos(yawRadians) * cosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * cosPitch
    });
    if (voxelsprout::math::lengthSquared(rayDirection) <= 0.0f) {
        return result;
    }

    // Nudge origin slightly forward so close-surface targeting does not start inside solids.
    const voxelsprout::math::Vector3 rayOrigin =
        voxelsprout::math::Vector3{m_camera.x, m_camera.y, m_camera.z} + (rayDirection * 0.02f);
    constexpr float kRayMaxDistance = kBlockInteractMaxDistance + 1.0f;

    int vx = static_cast<int>(std::floor(rayOrigin.x));
    int vy = static_cast<int>(std::floor(rayOrigin.y));
    int vz = static_cast<int>(std::floor(rayOrigin.z));

    const float kInf = std::numeric_limits<float>::infinity();

    const int stepX = (rayDirection.x > 0.0f) ? 1 : (rayDirection.x < 0.0f ? -1 : 0);
    const int stepY = (rayDirection.y > 0.0f) ? 1 : (rayDirection.y < 0.0f ? -1 : 0);
    const int stepZ = (rayDirection.z > 0.0f) ? 1 : (rayDirection.z < 0.0f ? -1 : 0);

    const float invAbsDirX = (stepX != 0) ? (1.0f / std::abs(rayDirection.x)) : kInf;
    const float invAbsDirY = (stepY != 0) ? (1.0f / std::abs(rayDirection.y)) : kInf;
    const float invAbsDirZ = (stepZ != 0) ? (1.0f / std::abs(rayDirection.z)) : kInf;

    const float voxelBoundaryX = (stepX > 0) ? static_cast<float>(vx + 1) : static_cast<float>(vx);
    const float voxelBoundaryY = (stepY > 0) ? static_cast<float>(vy + 1) : static_cast<float>(vy);
    const float voxelBoundaryZ = (stepZ > 0) ? static_cast<float>(vz + 1) : static_cast<float>(vz);

    float tMaxX = (stepX != 0) ? ((voxelBoundaryX - rayOrigin.x) / rayDirection.x) : kInf;
    float tMaxY = (stepY != 0) ? ((voxelBoundaryY - rayOrigin.y) / rayDirection.y) : kInf;
    float tMaxZ = (stepZ != 0) ? ((voxelBoundaryZ - rayOrigin.z) / rayDirection.z) : kInf;
    float tDeltaX = invAbsDirX;
    float tDeltaY = invAbsDirY;
    float tDeltaZ = invAbsDirZ;

    int hitFaceNormalX = 0;
    int hitFaceNormalY = 0;
    int hitFaceNormalZ = 0;
    bool hasHitFaceNormal = false;

    float distance = 0.0f;
    while (distance <= kRayMaxDistance) {
        if (isSolidWorldVoxel(vx, vy, vz)) {
            if (!hasHitFaceNormal) {
                // If we start inside a solid voxel, take one DDA step first so we get a stable face.
                if (tMaxX <= tMaxY && tMaxX <= tMaxZ) {
                    vx += stepX;
                    distance = tMaxX;
                    tMaxX += tDeltaX;
                    hitFaceNormalX = -stepX;
                    hitFaceNormalY = 0;
                    hitFaceNormalZ = 0;
                    hasHitFaceNormal = (stepX != 0);
                } else if (tMaxY <= tMaxX && tMaxY <= tMaxZ) {
                    vy += stepY;
                    distance = tMaxY;
                    tMaxY += tDeltaY;
                    hitFaceNormalX = 0;
                    hitFaceNormalY = -stepY;
                    hitFaceNormalZ = 0;
                    hasHitFaceNormal = (stepY != 0);
                } else {
                    vz += stepZ;
                    distance = tMaxZ;
                    tMaxZ += tDeltaZ;
                    hitFaceNormalX = 0;
                    hitFaceNormalY = 0;
                    hitFaceNormalZ = -stepZ;
                    hasHitFaceNormal = (stepZ != 0);
                }
                continue;
            }

            result.hitSolid = true;
            result.solidX = vx;
            result.solidY = vy;
            result.solidZ = vz;
            result.hitDistance = distance;
            result.hasHitFaceNormal = hasHitFaceNormal;
            result.hitFaceNormalX = hitFaceNormalX;
            result.hitFaceNormalY = hitFaceNormalY;
            result.hitFaceNormalZ = hitFaceNormalZ;

            if (hasHitFaceNormal) {
                const int adjacentX = vx + hitFaceNormalX;
                const int adjacentY = vy + hitFaceNormalY;
                const int adjacentZ = vz + hitFaceNormalZ;
                if (isWorldVoxelInBounds(adjacentX, adjacentY, adjacentZ) &&
                    !isSolidWorldVoxel(adjacentX, adjacentY, adjacentZ)) {
                    result.hasAdjacentEmpty = true;
                    result.adjacentEmptyX = adjacentX;
                    result.adjacentEmptyY = adjacentY;
                    result.adjacentEmptyZ = adjacentZ;
                }
            }
            return result;
        }

        if (tMaxX <= tMaxY && tMaxX <= tMaxZ) {
            vx += stepX;
            distance = tMaxX;
            tMaxX += tDeltaX;
            hitFaceNormalX = -stepX;
            hitFaceNormalY = 0;
            hitFaceNormalZ = 0;
            hasHitFaceNormal = (stepX != 0);
        } else if (tMaxY <= tMaxX && tMaxY <= tMaxZ) {
            vy += stepY;
            distance = tMaxY;
            tMaxY += tDeltaY;
            hitFaceNormalX = 0;
            hitFaceNormalY = -stepY;
            hitFaceNormalZ = 0;
            hasHitFaceNormal = (stepY != 0);
        } else {
            vz += stepZ;
            distance = tMaxZ;
            tMaxZ += tDeltaZ;
            hitFaceNormalX = 0;
            hitFaceNormalY = 0;
            hitFaceNormalZ = -stepZ;
            hasHitFaceNormal = (stepZ != 0);
        }
    }

    return result;
}

App::InteractionRaycastResult App::raycastInteractionFromCamera(bool includePipes) const {
    InteractionRaycastResult result{};
    if (m_world.chunkGrid().chunks().empty()) {
        return result;
    }

    const float yawRadians = voxelsprout::math::radians(m_camera.yawDegrees);
    const float pitchRadians = voxelsprout::math::radians(m_camera.pitchDegrees);
    const float cosPitch = std::cos(pitchRadians);
    const voxelsprout::math::Vector3 rayDirection = voxelsprout::math::normalize(voxelsprout::math::Vector3{
        std::cos(yawRadians) * cosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * cosPitch
    });
    if (voxelsprout::math::lengthSquared(rayDirection) <= 0.0f) {
        return result;
    }

    const voxelsprout::math::Vector3 rayOrigin =
        voxelsprout::math::Vector3{m_camera.x, m_camera.y, m_camera.z} + (rayDirection * 0.02f);
    constexpr float kRayMaxDistance = kBlockInteractMaxDistance + 1.0f;

    int vx = static_cast<int>(std::floor(rayOrigin.x));
    int vy = static_cast<int>(std::floor(rayOrigin.y));
    int vz = static_cast<int>(std::floor(rayOrigin.z));

    const float kInf = std::numeric_limits<float>::infinity();
    const int stepX = (rayDirection.x > 0.0f) ? 1 : (rayDirection.x < 0.0f ? -1 : 0);
    const int stepY = (rayDirection.y > 0.0f) ? 1 : (rayDirection.y < 0.0f ? -1 : 0);
    const int stepZ = (rayDirection.z > 0.0f) ? 1 : (rayDirection.z < 0.0f ? -1 : 0);

    const float invAbsDirX = (stepX != 0) ? (1.0f / std::abs(rayDirection.x)) : kInf;
    const float invAbsDirY = (stepY != 0) ? (1.0f / std::abs(rayDirection.y)) : kInf;
    const float invAbsDirZ = (stepZ != 0) ? (1.0f / std::abs(rayDirection.z)) : kInf;

    const float voxelBoundaryX = (stepX > 0) ? static_cast<float>(vx + 1) : static_cast<float>(vx);
    const float voxelBoundaryY = (stepY > 0) ? static_cast<float>(vy + 1) : static_cast<float>(vy);
    const float voxelBoundaryZ = (stepZ > 0) ? static_cast<float>(vz + 1) : static_cast<float>(vz);

    float tMaxX = (stepX != 0) ? ((voxelBoundaryX - rayOrigin.x) / rayDirection.x) : kInf;
    float tMaxY = (stepY != 0) ? ((voxelBoundaryY - rayOrigin.y) / rayDirection.y) : kInf;
    float tMaxZ = (stepZ != 0) ? ((voxelBoundaryZ - rayOrigin.z) / rayDirection.z) : kInf;
    float tDeltaX = invAbsDirX;
    float tDeltaY = invAbsDirY;
    float tDeltaZ = invAbsDirZ;

    int hitFaceNormalX = 0;
    int hitFaceNormalY = 0;
    int hitFaceNormalZ = 0;
    bool hasHitFaceNormal = false;

    float distance = 0.0f;
    while (distance <= kRayMaxDistance) {
        const bool hitSolid = isSolidWorldVoxel(vx, vy, vz);
        const bool hitPipe = includePipes && isPipeAtWorld(vx, vy, vz, nullptr);
        const bool hitBelt = includePipes && isBeltAtWorld(vx, vy, vz, nullptr);
        const bool hitTrack = includePipes && isTrackAtWorld(vx, vy, vz, nullptr);
        if (hitSolid || hitPipe || hitBelt || hitTrack) {
            if (!hasHitFaceNormal) {
                if (tMaxX <= tMaxY && tMaxX <= tMaxZ) {
                    vx += stepX;
                    distance = tMaxX;
                    tMaxX += tDeltaX;
                    hitFaceNormalX = -stepX;
                    hitFaceNormalY = 0;
                    hitFaceNormalZ = 0;
                    hasHitFaceNormal = (stepX != 0);
                } else if (tMaxY <= tMaxX && tMaxY <= tMaxZ) {
                    vy += stepY;
                    distance = tMaxY;
                    tMaxY += tDeltaY;
                    hitFaceNormalX = 0;
                    hitFaceNormalY = -stepY;
                    hitFaceNormalZ = 0;
                    hasHitFaceNormal = (stepY != 0);
                } else {
                    vz += stepZ;
                    distance = tMaxZ;
                    tMaxZ += tDeltaZ;
                    hitFaceNormalX = 0;
                    hitFaceNormalY = 0;
                    hitFaceNormalZ = -stepZ;
                    hasHitFaceNormal = (stepZ != 0);
                }
                continue;
            }

            result.hit = true;
            result.hitPipe = hitPipe;
            result.hitBelt = hitBelt;
            result.hitTrack = hitTrack;
            result.hitSolidVoxel = hitSolid;
            result.x = vx;
            result.y = vy;
            result.z = vz;
            result.hitDistance = distance;
            result.hasHitFaceNormal = hasHitFaceNormal;
            result.hitFaceNormalX = hitFaceNormalX;
            result.hitFaceNormalY = hitFaceNormalY;
            result.hitFaceNormalZ = hitFaceNormalZ;
            return result;
        }

        if (tMaxX <= tMaxY && tMaxX <= tMaxZ) {
            vx += stepX;
            distance = tMaxX;
            tMaxX += tDeltaX;
            hitFaceNormalX = -stepX;
            hitFaceNormalY = 0;
            hitFaceNormalZ = 0;
            hasHitFaceNormal = (stepX != 0);
        } else if (tMaxY <= tMaxX && tMaxY <= tMaxZ) {
            vy += stepY;
            distance = tMaxY;
            tMaxY += tDeltaY;
            hitFaceNormalX = 0;
            hitFaceNormalY = -stepY;
            hitFaceNormalZ = 0;
            hasHitFaceNormal = (stepY != 0);
        } else {
            vz += stepZ;
            distance = tMaxZ;
            tMaxZ += tDeltaZ;
            hitFaceNormalX = 0;
            hitFaceNormalY = 0;
            hitFaceNormalZ = -stepZ;
            hasHitFaceNormal = (stepZ != 0);
        }
    }

    return result;
}

bool App::isWorldVoxelInBounds(int x, int y, int z) const {
    std::size_t chunkIndex = 0;
    int localX = 0;
    int localY = 0;
    int localZ = 0;
    return worldToChunkLocal(x, y, z, chunkIndex, localX, localY, localZ);
}

void App::cycleSelectedHotbar(int direction) {
    if (kHotbarSlotCount <= 0) {
        m_selectedHotbarIndex = 0;
        return;
    }
    const int next = (m_selectedHotbarIndex + direction) % kHotbarSlotCount;
    selectHotbarSlot(next < 0 ? next + kHotbarSlotCount : next);
}

void App::selectHotbarSlot(int hotbarIndex) {
    const int clampedIndex = std::clamp(hotbarIndex, 0, kHotbarSlotCount - 1);
    if (m_selectedHotbarIndex == clampedIndex) {
        return;
    }
    m_selectedHotbarIndex = clampedIndex;
    const char* label = "block";
    if (m_selectedHotbarIndex == kHotbarSlotPipe) {
        label = "pipe";
    } else if (m_selectedHotbarIndex == kHotbarSlotConveyor) {
        label = "conveyor";
    } else if (m_selectedHotbarIndex == kHotbarSlotTrack) {
        label = "track";
    }
    VOX_LOGI("app") << "selected hotbar: " << label;
}

void App::selectPlaceableBlock(int blockIndex) {
    if (kPlaceableBlockTypes.empty()) {
        m_selectedBlockIndex = 0;
        return;
    }
    const int clampedIndex = std::clamp(blockIndex, 0, static_cast<int>(kPlaceableBlockTypes.size()) - 1);
    if (m_selectedBlockIndex == clampedIndex) {
        return;
    }
    m_selectedBlockIndex = clampedIndex;
    VOX_LOGI("app") << "selected block material: "
                    << placeableBlockLabel(kPlaceableBlockTypes[static_cast<std::size_t>(m_selectedBlockIndex)]);
}

bool App::isPipeHotbarSelected() const {
    return m_selectedHotbarIndex == kHotbarSlotPipe;
}

bool App::isConveyorHotbarSelected() const {
    return m_selectedHotbarIndex == kHotbarSlotConveyor;
}

bool App::isTrackHotbarSelected() const {
    return m_selectedHotbarIndex == kHotbarSlotTrack;
}

voxelsprout::world::Voxel App::selectedPlaceVoxel() const {
    if (kPlaceableBlockTypes.empty()) {
        return voxelsprout::world::Voxel{voxelsprout::world::VoxelType::Stone};
    }
    const int clampedIndex = std::clamp(
        m_selectedBlockIndex,
        0,
        static_cast<int>(kPlaceableBlockTypes.size()) - 1
    );
    return voxelsprout::world::Voxel{kPlaceableBlockTypes[static_cast<std::size_t>(clampedIndex)]};
}

bool App::computePlacementVoxelFromRaycast(const CameraRaycastResult& raycast, int& outX, int& outY, int& outZ) const {
    if (!raycast.hitSolid || !raycast.hasHitFaceNormal) {
        return false;
    }

    const int candidateX = raycast.solidX + raycast.hitFaceNormalX;
    const int candidateY = raycast.solidY + raycast.hitFaceNormalY;
    const int candidateZ = raycast.solidZ + raycast.hitFaceNormalZ;
    if (!isWorldVoxelInBounds(candidateX, candidateY, candidateZ)) {
        return false;
    }

    outX = candidateX;
    outY = candidateY;
    outZ = candidateZ;
    return true;
}

bool App::computePipePlacementFromInteractionRaycast(
    const InteractionRaycastResult& raycast,
    int& outX,
    int& outY,
    int& outZ,
    int& outAxisX,
    int& outAxisY,
    int& outAxisZ
) const {
    if (!raycast.hit || !raycast.hasHitFaceNormal) {
        return false;
    }

    const voxelsprout::core::Dir6 faceDir = faceNormalToDir6(
        raycast.hitFaceNormalX,
        raycast.hitFaceNormalY,
        raycast.hitFaceNormalZ
    );
    voxelsprout::core::Dir6 selectedAxis = faceDir;
    int extensionSign = 1;
    voxelsprout::core::Cell3i extensionAnchor{raycast.x, raycast.y, raycast.z};

    if (raycast.hitPipe) {
        std::size_t pipeIndex = 0;
        if (!isPipeAtWorld(raycast.x, raycast.y, raycast.z, &pipeIndex)) {
            return false;
        }

        const std::vector<voxelsprout::sim::Pipe>& pipes = m_simulation.pipes();
        if (pipeIndex >= pipes.size()) {
            return false;
        }

        selectedAxis = axisToDir6(pipes[pipeIndex].axis);
        const voxelsprout::core::Cell3i axisOffset = voxelsprout::core::dirToOffset(selectedAxis);
        const int faceNormalDotAxis =
            (raycast.hitFaceNormalX * axisOffset.x) +
            (raycast.hitFaceNormalY * axisOffset.y) +
            (raycast.hitFaceNormalZ * axisOffset.z);
        const bool sideSplitPlacement = (faceNormalDotAxis == 0);
        if (sideSplitPlacement) {
            // Side hits place a perpendicular branch from the clicked pipe cell.
            selectedAxis = faceDir;
            extensionSign = 1;
        } else {
            if (faceNormalDotAxis > 0) {
                extensionSign = 1;
            } else {
                extensionSign = -1;
            }

            // If a chain already exists, extend from its far end instead of failing at an internal segment.
            const voxelsprout::core::Dir6 extensionDir =
                extensionSign >= 0 ? selectedAxis : voxelsprout::core::oppositeDir(selectedAxis);
            while (true) {
                const voxelsprout::core::Cell3i nextCell = voxelsprout::core::neighborCell(extensionAnchor, extensionDir);
                std::size_t nextPipeIndex = 0;
                if (!isPipeAtWorld(nextCell.x, nextCell.y, nextCell.z, &nextPipeIndex)) {
                    break;
                }
                if (nextPipeIndex >= pipes.size()) {
                    break;
                }
                const voxelsprout::core::Dir6 nextAxis = axisToDir6(pipes[nextPipeIndex].axis);
                if (!dirSharesAxis(nextAxis, selectedAxis)) {
                    break;
                }
                extensionAnchor = nextCell;
            }
        }
    }

    const voxelsprout::core::Dir6 extensionDir =
        extensionSign >= 0 ? selectedAxis : voxelsprout::core::oppositeDir(selectedAxis);
    const voxelsprout::core::Cell3i targetCell = voxelsprout::core::neighborCell(extensionAnchor, extensionDir);
    const int targetX = targetCell.x;
    const int targetY = targetCell.y;
    const int targetZ = targetCell.z;
    if (!isWorldVoxelInBounds(targetX, targetY, targetZ)) {
        return false;
    }
    if (isSolidWorldVoxel(targetX, targetY, targetZ)) {
        return false;
    }
    if (isPipeAtWorld(targetX, targetY, targetZ, nullptr) ||
        isBeltAtWorld(targetX, targetY, targetZ, nullptr) ||
        isTrackAtWorld(targetX, targetY, targetZ, nullptr)) {
        return false;
    }

    const std::uint8_t neighborMask = voxelsprout::sim::neighborMask6(targetCell, [this](const voxelsprout::core::Cell3i& cell) {
        return isPipeAtWorld(cell.x, cell.y, cell.z, nullptr);
    });
    const std::uint32_t neighborCount = voxelsprout::sim::connectionCount(neighborMask);
    const voxelsprout::sim::JoinPiece joinPiece = voxelsprout::sim::classifyJoinPiece(neighborMask);

    voxelsprout::core::Dir6 resolvedAxis = selectedAxis;
    if (neighborCount == 1u) {
        const voxelsprout::core::Dir6 neighborDir = firstDirFromMask(neighborMask);
        resolvedAxis = voxelsprout::core::oppositeDir(neighborDir);
    } else if (joinPiece == voxelsprout::sim::JoinPiece::Straight) {
        resolvedAxis = resolveStraightAxisFromMask(neighborMask, selectedAxis);
    }

    outX = targetX;
    outY = targetY;
    outZ = targetZ;
    dir6ToAxisInts(resolvedAxis, outAxisX, outAxisY, outAxisZ);
    return true;
}

bool App::computeBeltPlacementFromInteractionRaycast(
    const InteractionRaycastResult& raycast,
    int& outX,
    int& outY,
    int& outZ,
    int& outAxisX,
    int& outAxisY,
    int& outAxisZ
) const {
    if (!raycast.hit || !raycast.hasHitFaceNormal) {
        return false;
    }

    voxelsprout::core::Dir6 selectedAxis = faceNormalToDir6(
        raycast.hitFaceNormalX,
        raycast.hitFaceNormalY,
        raycast.hitFaceNormalZ
    );
    if (selectedAxis == voxelsprout::core::Dir6::PosY || selectedAxis == voxelsprout::core::Dir6::NegY) {
        selectedAxis = horizontalDirFromYaw(m_camera.yawDegrees);
    }
    int extensionSign = 1;
    voxelsprout::core::Cell3i extensionAnchor{raycast.x, raycast.y, raycast.z};

    if (raycast.hitBelt) {
        std::size_t beltIndex = 0;
        if (!isBeltAtWorld(raycast.x, raycast.y, raycast.z, &beltIndex)) {
            return false;
        }
        const std::vector<voxelsprout::sim::Belt>& belts = m_simulation.belts();
        if (beltIndex >= belts.size()) {
            return false;
        }

        selectedAxis = beltDirectionToDir6(belts[beltIndex].direction);
        const voxelsprout::core::Cell3i axisOffset = voxelsprout::core::dirToOffset(selectedAxis);
        const int faceNormalDotAxis =
            (raycast.hitFaceNormalX * axisOffset.x) +
            (raycast.hitFaceNormalY * axisOffset.y) +
            (raycast.hitFaceNormalZ * axisOffset.z);
        if (faceNormalDotAxis == 0) {
            voxelsprout::core::Dir6 faceDir = faceNormalToDir6(
                raycast.hitFaceNormalX,
                raycast.hitFaceNormalY,
                raycast.hitFaceNormalZ
            );
            if (faceDir == voxelsprout::core::Dir6::PosY || faceDir == voxelsprout::core::Dir6::NegY) {
                faceDir = horizontalDirFromYaw(m_camera.yawDegrees);
            }
            selectedAxis = faceDir;
            extensionSign = 1;
        } else {
            extensionSign = (faceNormalDotAxis > 0) ? 1 : -1;
        }
    }

    const voxelsprout::core::Dir6 extensionDir =
        extensionSign >= 0 ? selectedAxis : voxelsprout::core::oppositeDir(selectedAxis);
    const voxelsprout::core::Cell3i targetCell = voxelsprout::core::neighborCell(extensionAnchor, extensionDir);
    const int targetX = targetCell.x;
    const int targetY = targetCell.y;
    const int targetZ = targetCell.z;
    if (!isWorldVoxelInBounds(targetX, targetY, targetZ) ||
        isSolidWorldVoxel(targetX, targetY, targetZ)) {
        return false;
    }
    if (isPipeAtWorld(targetX, targetY, targetZ, nullptr) ||
        isBeltAtWorld(targetX, targetY, targetZ, nullptr) ||
        isTrackAtWorld(targetX, targetY, targetZ, nullptr)) {
        return false;
    }

    outX = targetX;
    outY = targetY;
    outZ = targetZ;
    dir6ToAxisInts(selectedAxis, outAxisX, outAxisY, outAxisZ);
    return true;
}

bool App::computeTrackPlacementFromInteractionRaycast(
    const InteractionRaycastResult& raycast,
    int& outX,
    int& outY,
    int& outZ,
    int& outAxisX,
    int& outAxisY,
    int& outAxisZ
) const {
    if (!raycast.hit || !raycast.hasHitFaceNormal) {
        return false;
    }

    voxelsprout::core::Dir6 selectedAxis = faceNormalToDir6(
        raycast.hitFaceNormalX,
        raycast.hitFaceNormalY,
        raycast.hitFaceNormalZ
    );
    if (selectedAxis == voxelsprout::core::Dir6::PosY || selectedAxis == voxelsprout::core::Dir6::NegY) {
        selectedAxis = horizontalDirFromYaw(m_camera.yawDegrees);
    }
    int extensionSign = 1;
    voxelsprout::core::Cell3i extensionAnchor{raycast.x, raycast.y, raycast.z};

    if (raycast.hitTrack) {
        std::size_t trackIndex = 0;
        if (!isTrackAtWorld(raycast.x, raycast.y, raycast.z, &trackIndex)) {
            return false;
        }
        const std::vector<voxelsprout::sim::Track>& tracks = m_simulation.tracks();
        if (trackIndex >= tracks.size()) {
            return false;
        }

        selectedAxis = trackDirectionToDir6(tracks[trackIndex].direction);
        const voxelsprout::core::Cell3i axisOffset = voxelsprout::core::dirToOffset(selectedAxis);
        const int faceNormalDotAxis =
            (raycast.hitFaceNormalX * axisOffset.x) +
            (raycast.hitFaceNormalY * axisOffset.y) +
            (raycast.hitFaceNormalZ * axisOffset.z);
        if (faceNormalDotAxis == 0) {
            voxelsprout::core::Dir6 faceDir = faceNormalToDir6(
                raycast.hitFaceNormalX,
                raycast.hitFaceNormalY,
                raycast.hitFaceNormalZ
            );
            if (faceDir == voxelsprout::core::Dir6::PosY || faceDir == voxelsprout::core::Dir6::NegY) {
                faceDir = horizontalDirFromYaw(m_camera.yawDegrees);
            }
            selectedAxis = faceDir;
            extensionSign = 1;
        } else {
            extensionSign = (faceNormalDotAxis > 0) ? 1 : -1;
        }
    }

    const voxelsprout::core::Dir6 extensionDir =
        extensionSign >= 0 ? selectedAxis : voxelsprout::core::oppositeDir(selectedAxis);
    const voxelsprout::core::Cell3i targetCell = voxelsprout::core::neighborCell(extensionAnchor, extensionDir);
    const int targetX = targetCell.x;
    const int targetY = targetCell.y;
    const int targetZ = targetCell.z;
    if (!isWorldVoxelInBounds(targetX, targetY, targetZ) ||
        isSolidWorldVoxel(targetX, targetY, targetZ)) {
        return false;
    }
    if (isPipeAtWorld(targetX, targetY, targetZ, nullptr) ||
        isBeltAtWorld(targetX, targetY, targetZ, nullptr) ||
        isTrackAtWorld(targetX, targetY, targetZ, nullptr)) {
        return false;
    }

    outX = targetX;
    outY = targetY;
    outZ = targetZ;
    dir6ToAxisInts(selectedAxis, outAxisX, outAxisY, outAxisZ);
    return true;
}

bool App::applyVoxelEdit(
    int targetX,
    int targetY,
    int targetZ,
    voxelsprout::world::Voxel voxel,
    std::vector<std::size_t>& outDirtyChunkIndices
) {
    int localX = 0;
    int localY = 0;
    int localZ = 0;
    std::size_t editedChunkIndex = 0;
    if (!worldToChunkLocal(targetX, targetY, targetZ, editedChunkIndex, localX, localY, localZ)) {
        return false;
    }

    voxelsprout::world::Chunk& chunk = m_world.chunkGrid().chunks()[editedChunkIndex];
    if (chunk.voxelAt(localX, localY, localZ).type == voxel.type) {
        return false;
    }
    chunk.setVoxel(localX, localY, localZ, voxel);

    auto appendUniqueChunkIndex = [&outDirtyChunkIndices](std::size_t chunkIndex) {
        if (std::find(outDirtyChunkIndices.begin(), outDirtyChunkIndices.end(), chunkIndex) != outDirtyChunkIndices.end()) {
            return;
        }
        outDirtyChunkIndices.push_back(chunkIndex);
    };
    appendUniqueChunkIndex(editedChunkIndex);

    auto appendNeighborChunkForWorldVoxel = [this, &appendUniqueChunkIndex](int worldX, int worldY, int worldZ) {
        std::size_t chunkIndex = 0;
        int neighborLocalX = 0;
        int neighborLocalY = 0;
        int neighborLocalZ = 0;
        if (worldToChunkLocal(worldX, worldY, worldZ, chunkIndex, neighborLocalX, neighborLocalY, neighborLocalZ)) {
            appendUniqueChunkIndex(chunkIndex);
        }
    };

    if (localX == 0) {
        appendNeighborChunkForWorldVoxel(targetX - 1, targetY, targetZ);
    }
    if (localX == (voxelsprout::world::Chunk::kSizeX - 1)) {
        appendNeighborChunkForWorldVoxel(targetX + 1, targetY, targetZ);
    }
    if (localY == 0) {
        appendNeighborChunkForWorldVoxel(targetX, targetY - 1, targetZ);
    }
    if (localY == (voxelsprout::world::Chunk::kSizeY - 1)) {
        appendNeighborChunkForWorldVoxel(targetX, targetY + 1, targetZ);
    }
    if (localZ == 0) {
        appendNeighborChunkForWorldVoxel(targetX, targetY, targetZ - 1);
    }
    if (localZ == (voxelsprout::world::Chunk::kSizeZ - 1)) {
        appendNeighborChunkForWorldVoxel(targetX, targetY, targetZ + 1);
    }

    return true;
}

bool App::isPipeAtWorld(int worldX, int worldY, int worldZ, std::size_t* outPipeIndex) const {
    const std::vector<voxelsprout::sim::Pipe>& pipes = m_simulation.pipes();
    for (std::size_t pipeIndex = 0; pipeIndex < pipes.size(); ++pipeIndex) {
        const voxelsprout::sim::Pipe& pipe = pipes[pipeIndex];
        if (pipe.x == worldX && pipe.y == worldY && pipe.z == worldZ) {
            if (outPipeIndex != nullptr) {
                *outPipeIndex = pipeIndex;
            }
            return true;
        }
    }
    return false;
}

bool App::isBeltAtWorld(int worldX, int worldY, int worldZ, std::size_t* outBeltIndex) const {
    const std::vector<voxelsprout::sim::Belt>& belts = m_simulation.belts();
    for (std::size_t beltIndex = 0; beltIndex < belts.size(); ++beltIndex) {
        const voxelsprout::sim::Belt& belt = belts[beltIndex];
        if (belt.x == worldX && belt.y == worldY && belt.z == worldZ) {
            if (outBeltIndex != nullptr) {
                *outBeltIndex = beltIndex;
            }
            return true;
        }
    }
    return false;
}

bool App::isTrackAtWorld(int worldX, int worldY, int worldZ, std::size_t* outTrackIndex) const {
    const std::vector<voxelsprout::sim::Track>& tracks = m_simulation.tracks();
    for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex) {
        const voxelsprout::sim::Track& track = tracks[trackIndex];
        if (track.x == worldX && track.y == worldY && track.z == worldZ) {
            if (outTrackIndex != nullptr) {
                *outTrackIndex = trackIndex;
            }
            return true;
        }
    }
    return false;
}

void App::regenerateWorld() {
    m_world.regenerateFlatWorld();
    const voxelsprout::world::ClipmapConfig requestedClipmapConfig = m_renderer.clipmapQueryConfig();
    m_chunkClipmapIndex.setConfig(requestedClipmapConfig);
    m_appliedClipmapConfig = requestedClipmapConfig;
    m_hasAppliedClipmapConfig = true;
    m_chunkClipmapIndex.rebuild(m_world.chunkGrid());
    if (!m_renderer.updateChunkMesh(m_world.chunkGrid())) {
        VOX_LOGE("app") << "world regenerate failed to update chunk meshes";
    }

    const std::filesystem::path worldPath{kWorldFilePath};
    if (m_world.save(worldPath)) {
        VOX_LOGI("app") << "world regenerated and saved to " << worldPath.string() << " (R)";
        m_worldDirty = false;
        m_worldAutosaveElapsedSeconds = 0.0f;
    } else {
        VOX_LOGW("app") << "world regenerated, but failed to save " << worldPath.string();
    }
}

bool App::tryPlaceVoxelFromCameraRay(std::vector<std::size_t>& outDirtyChunkIndices) {
    if (m_world.chunkGrid().chunks().empty()) {
        return false;
    }

    const CameraRaycastResult raycast = raycastFromCamera();
    if (!raycast.hitSolid || raycast.hitDistance > kBlockInteractMaxDistance) {
        return false;
    }

    int targetX = 0;
    int targetY = 0;
    int targetZ = 0;
    if (!computePlacementVoxelFromRaycast(raycast, targetX, targetY, targetZ)) {
        return false;
    }

    return applyVoxelEdit(targetX, targetY, targetZ, selectedPlaceVoxel(), outDirtyChunkIndices);
}

bool App::tryRemoveVoxelFromCameraRay(std::vector<std::size_t>& outDirtyChunkIndices) {
    if (m_world.chunkGrid().chunks().empty()) {
        return false;
    }

    const CameraRaycastResult raycast = raycastFromCamera();
    if (!raycast.hitSolid || raycast.hitDistance > kBlockInteractMaxDistance) {
        return false;
    }

    return applyVoxelEdit(
        raycast.solidX,
        raycast.solidY,
        raycast.solidZ,
        voxelsprout::world::Voxel{voxelsprout::world::VoxelType::Empty},
        outDirtyChunkIndices
    );
}

bool App::tryPlacePipeFromCameraRay() {
    const InteractionRaycastResult raycast = raycastInteractionFromCamera(true);
    if (!raycast.hit || raycast.hitDistance > kBlockInteractMaxDistance) {
        return false;
    }

    int targetX = 0;
    int targetY = 0;
    int targetZ = 0;
    int axisX = 0;
    int axisY = 1;
    int axisZ = 0;
    if (!computePipePlacementFromInteractionRaycast(
            raycast,
            targetX,
            targetY,
            targetZ,
            axisX,
            axisY,
            axisZ
        )) {
        return false;
    }

    const voxelsprout::math::Vector3 axis{
        static_cast<float>(axisX),
        static_cast<float>(axisY),
        static_cast<float>(axisZ)
    };
    m_simulation.pipes().emplace_back(
        targetX,
        targetY,
        targetZ,
        axis,
        kDefaultPipeLength,
        kDefaultPipeRadius,
        kDefaultPipeTint
    );
    return true;
}

bool App::tryRemovePipeFromCameraRay() {
    const InteractionRaycastResult raycast = raycastInteractionFromCamera(true);
    if (!raycast.hit || !raycast.hitPipe || raycast.hitDistance > kBlockInteractMaxDistance) {
        return false;
    }

    std::size_t pipeIndex = 0;
    if (!isPipeAtWorld(raycast.x, raycast.y, raycast.z, &pipeIndex)) {
        return false;
    }

    std::vector<voxelsprout::sim::Pipe>& pipes = m_simulation.pipes();
    if (pipeIndex >= pipes.size()) {
        return false;
    }
    pipes.erase(pipes.begin() + static_cast<std::ptrdiff_t>(pipeIndex));
    return true;
}

bool App::tryPlaceBeltFromCameraRay() {
    const InteractionRaycastResult raycast = raycastInteractionFromCamera(true);
    if (!raycast.hit || raycast.hitDistance > kBlockInteractMaxDistance) {
        return false;
    }

    int targetX = 0;
    int targetY = 0;
    int targetZ = 0;
    int axisX = 0;
    int axisY = 0;
    int axisZ = 1;
    if (!computeBeltPlacementFromInteractionRaycast(
            raycast,
            targetX,
            targetY,
            targetZ,
            axisX,
            axisY,
            axisZ
        )) {
        return false;
    }

    voxelsprout::core::Dir6 axisDir = faceNormalToDir6(axisX, axisY, axisZ);
    if (axisDir == voxelsprout::core::Dir6::PosY || axisDir == voxelsprout::core::Dir6::NegY) {
        axisDir = horizontalDirFromYaw(m_camera.yawDegrees);
    }
    m_simulation.belts().emplace_back(targetX, targetY, targetZ, dir6ToBeltDirection(axisDir));
    return true;
}

bool App::tryRemoveBeltFromCameraRay() {
    const InteractionRaycastResult raycast = raycastInteractionFromCamera(true);
    if (!raycast.hit || !raycast.hitBelt || raycast.hitDistance > kBlockInteractMaxDistance) {
        return false;
    }

    std::size_t beltIndex = 0;
    if (!isBeltAtWorld(raycast.x, raycast.y, raycast.z, &beltIndex)) {
        return false;
    }

    std::vector<voxelsprout::sim::Belt>& belts = m_simulation.belts();
    if (beltIndex >= belts.size()) {
        return false;
    }
    belts.erase(belts.begin() + static_cast<std::ptrdiff_t>(beltIndex));
    return true;
}

bool App::tryPlaceTrackFromCameraRay() {
    const InteractionRaycastResult raycast = raycastInteractionFromCamera(true);
    if (!raycast.hit || raycast.hitDistance > kBlockInteractMaxDistance) {
        return false;
    }

    int targetX = 0;
    int targetY = 0;
    int targetZ = 0;
    int axisX = 0;
    int axisY = 0;
    int axisZ = 1;
    if (!computeTrackPlacementFromInteractionRaycast(
            raycast,
            targetX,
            targetY,
            targetZ,
            axisX,
            axisY,
            axisZ
        )) {
        return false;
    }

    voxelsprout::core::Dir6 axisDir = faceNormalToDir6(axisX, axisY, axisZ);
    if (axisDir == voxelsprout::core::Dir6::PosY || axisDir == voxelsprout::core::Dir6::NegY) {
        axisDir = horizontalDirFromYaw(m_camera.yawDegrees);
    }
    m_simulation.tracks().emplace_back(targetX, targetY, targetZ, dir6ToTrackDirection(axisDir));
    return true;
}

bool App::tryRemoveTrackFromCameraRay() {
    const InteractionRaycastResult raycast = raycastInteractionFromCamera(true);
    if (!raycast.hit || !raycast.hitTrack || raycast.hitDistance > kBlockInteractMaxDistance) {
        return false;
    }

    std::size_t trackIndex = 0;
    if (!isTrackAtWorld(raycast.x, raycast.y, raycast.z, &trackIndex)) {
        return false;
    }

    std::vector<voxelsprout::sim::Track>& tracks = m_simulation.tracks();
    if (trackIndex >= tracks.size()) {
        return false;
    }
    tracks.erase(tracks.begin() + static_cast<std::ptrdiff_t>(trackIndex));
    return true;
}

} // namespace voxelsprout::app
