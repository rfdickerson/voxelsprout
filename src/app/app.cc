#include "app/app.h"

#include <GLFW/glfw3.h>

#include "core/grid3.h"
#include "game/buildable.h"
#include "game/strategy_hex_terrain.h"
#include "game/strategy_map.h"
#include "game/strategy_map_io.h"
#include "game/strategy_map_mesh.h"
#include "core/log.h"
#include "math/math.h"
#include "sim/network_procedural.h"
#include "ui/icon_atlas.h"
#include "ui/widgets/button.h"
#include "ui/widgets/donut_chart.h"
#include "ui/widgets/image.h"
#include "ui/widgets/label.h"
#include "ui/widgets/line_chart.h"
#include "ui/widgets/icon_button.h"
#include "ui/widgets/panel.h"
#include "ui/widgets/production_panel.h"
#include "ui/widgets/rich_text_view.h"
#include "ui/widgets/stat_badge.h"
#include "ui/widgets/toolbar.h"
#include "ui/widgets/window.h"

#include <stb_image.h>

#include <memory>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

#ifndef ODAI_APP_VERSION
#define ODAI_APP_VERSION "dev"
#endif

#ifndef ODAI_RELEASE_PROFILE
#define ODAI_RELEASE_PROFILE "dev_runtime"
#endif

constexpr float kMouseSensitivity = 0.1f;
constexpr float kMouseSmoothingSeconds = 0.035f;
constexpr float kMoveMaxSpeed = 5.0f;
constexpr float kSprintSpeedMultiplier = 1.35f;
constexpr float kImportedSceneMoveSpeed = 1800.0f;
constexpr float kImportedSceneSprintSpeedMultiplier = 3.0f;
constexpr float kImportedSceneVerticalMoveSpeed = 1400.0f;
constexpr float kImportedSceneSunYawDegrees = 62.0f;
constexpr float kImportedSceneSunPitchDegrees = -61.0f;
// Fixed orientation/FOV for the strategy map's 3D relief camera.
constexpr float kMap3DYawDegrees = -90.0f;
constexpr float kMap3DFovDegrees = 50.0f;
// Time constant for low-passing per-frame pan displacement. Filters the beat
// between the mouse report rate (125/500/1000 Hz) and the frame rate, which
// makes per-frame deltas uneven, without adding perceptible grab lag.
constexpr float kStrategyPanSmoothingSeconds = 0.022f;
constexpr float kSneakSpeedMultiplier = 0.35f;
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
constexpr float kRenderFrustumKeepAlivePlaneSlackVoxels = 10.0f;
constexpr std::uint8_t kSpatialQueryVisibilityGraceFrames = 2;
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
constexpr int kVoxelBreakClicksRequired = 2;
constexpr const char* kWorldFilePath = "world.vxw";
constexpr const char* kConfigFilePath = "odai.cfg";
constexpr const char* kMagicaCastlePath = "assets/magicka/castle.vox";
constexpr const char* kMagicaTeapotPath = "assets/magicka/teapot.vox";
constexpr const char* kMagicaMonu2Path = "assets/magicka/monu2.vox";
constexpr float kWorldAutosaveDelaySeconds = 0.75f;
constexpr const char* kStrategyMapEnvVar = "ODAI_STRATEGY_MAP";
constexpr float kImportedInspectRayMaxDistance = 4096.0f;
constexpr std::uint32_t kImportedSceneMaterialFlagAlphaTest = 1u;

constexpr float kDefaultPipeLength = 1.0f;
constexpr float kDefaultPipeRadius = 0.45f;
constexpr odai::math::Vector3 kDefaultPipeTint{0.95f, 0.95f, 0.95f};
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

Aabb3f makeConveyorBeltAabb(const odai::sim::Belt& belt) {
    const float centerX = static_cast<float>(belt.x) + 0.5f;
    const float centerY = static_cast<float>(belt.y) + 0.5f;
    const float centerZ = static_cast<float>(belt.z) + 0.5f;
    const bool alongX = belt.direction == odai::sim::BeltDirection::East || belt.direction == odai::sim::BeltDirection::West;
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

const char* inventoryItemLabel(odai::render::InventoryItemId itemId) {
    switch (itemId) {
    case odai::render::InventoryItemId::Stone: return "stone";
    case odai::render::InventoryItemId::Dirt: return "dirt";
    case odai::render::InventoryItemId::Grass: return "grass";
    case odai::render::InventoryItemId::Wood: return "wood";
    case odai::render::InventoryItemId::Red: return "red";
    case odai::render::InventoryItemId::Empty:
    default:
        return "empty";
    }
}

odai::world::Voxel itemToVoxel(odai::render::InventoryItemId itemId) {
    using odai::render::InventoryItemId;
    using odai::world::Voxel;
    using odai::world::VoxelType;
    switch (itemId) {
    case InventoryItemId::Stone: return Voxel{VoxelType::Stone};
    case InventoryItemId::Dirt: return Voxel{VoxelType::Dirt};
    case InventoryItemId::Grass: return Voxel{VoxelType::Grass};
    case InventoryItemId::Wood: return Voxel{VoxelType::Wood};
    case InventoryItemId::Red: return Voxel{VoxelType::SolidRed};
    case InventoryItemId::Empty:
    default:
        return Voxel{VoxelType::Empty};
    }
}

const char* shadowModeConfigName(odai::render::ShadowMode mode) {
    switch (mode) {
    case odai::render::ShadowMode::ShadowMaps: return "shadow_maps";
    case odai::render::ShadowMode::RayTraced: return "ray_traced";
    case odai::render::ShadowMode::Auto: return "auto";
    }
    return "auto";
}

bool parseShadowModeConfigValue(const std::string& value, odai::render::ShadowMode& outMode) {
    if (value == "shadow_maps") {
        outMode = odai::render::ShadowMode::ShadowMaps;
        return true;
    }
    if (value == "ray_traced") {
        outMode = odai::render::ShadowMode::RayTraced;
        return true;
    }
    if (value == "auto") {
        outMode = odai::render::ShadowMode::Auto;
        return true;
    }
    return false;
}

const char* boolConfigName(bool value) {
    return value ? "true" : "false";
}

bool parseBoolConfigValue(const std::string& value, bool& outValue) {
    if (value == "true" || value == "1" || value == "yes" || value == "on") {
        outValue = true;
        return true;
    }
    if (value == "false" || value == "0" || value == "no" || value == "off") {
        outValue = false;
        return true;
    }
    return false;
}

std::string trimConfigString(const std::string& value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return value.substr(begin, end - begin);
}

struct FrustumPlane {
    odai::math::Vector3 normal{};
    float d = 0.0f;
};

struct CameraFrustum {
    std::array<FrustumPlane, 6> planes{};
    odai::core::CellAabb broadPhaseBounds{};
    bool valid = false;
};

FrustumPlane makePlaneFromPointNormal(const odai::math::Vector3& point, const odai::math::Vector3& normal) {
    const odai::math::Vector3 normalized = odai::math::normalize(normal);
    FrustumPlane plane{};
    plane.normal = normalized;
    plane.d = -odai::math::dot(normalized, point);
    return plane;
}

void orientPlaneTowardForward(FrustumPlane& plane, const odai::math::Vector3& forward) {
    if (odai::math::dot(plane.normal, forward) < 0.0f) {
        plane.normal = -plane.normal;
        plane.d = -plane.d;
    }
}

[[maybe_unused]] CameraFrustum buildCameraFrustum(
    const odai::math::Vector3& eye,
    float yawDegrees,
    float pitchDegrees,
    float fovDegrees,
    float aspectRatio
) {
    CameraFrustum frustum{};
    const float clampedAspect = std::max(aspectRatio, 0.1f);
    const float clampedFovDegrees = std::clamp(fovDegrees, 20.0f, 120.0f);
    const float yawRadians = odai::math::radians(yawDegrees);
    const float pitchRadians = odai::math::radians(pitchDegrees);
    const float cosPitch = std::cos(pitchRadians);
    odai::math::Vector3 forward{
        std::cos(yawRadians) * cosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * cosPitch
    };
    forward = odai::math::normalize(forward);
    if (odai::math::lengthSquared(forward) <= 0.0001f) {
        return frustum;
    }

    const odai::math::Vector3 worldUp{0.0f, 1.0f, 0.0f};
    odai::math::Vector3 right = odai::math::normalize(odai::math::cross(forward, worldUp));
    if (odai::math::lengthSquared(right) <= 0.0001f) {
        right = odai::math::Vector3{1.0f, 0.0f, 0.0f};
    }
    odai::math::Vector3 up = odai::math::normalize(odai::math::cross(right, forward));
    if (odai::math::lengthSquared(up) <= 0.0001f) {
        up = worldUp;
    }

    const float halfFovY = odai::math::radians(clampedFovDegrees) * 0.5f;
    const float tanHalfY = std::tan(halfFovY);
    const float tanHalfX = tanHalfY * clampedAspect;
    const float nearDistance = kRenderCullNearPlane;
    const float farDistance = kRenderCullFarPlane;

    const odai::math::Vector3 nearCenter = eye + (forward * nearDistance);
    const odai::math::Vector3 farCenter = eye + (forward * farDistance);
    const float nearHalfHeight = nearDistance * tanHalfY;
    const float nearHalfWidth = nearDistance * tanHalfX;
    const float farHalfHeight = farDistance * tanHalfY;
    const float farHalfWidth = farDistance * tanHalfX;

    const odai::math::Vector3 nearUp = up * nearHalfHeight;
    const odai::math::Vector3 nearRight = right * nearHalfWidth;
    const odai::math::Vector3 farUp = up * farHalfHeight;
    const odai::math::Vector3 farRight = right * farHalfWidth;

    const std::array<odai::math::Vector3, 8> corners = {
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
    for (const odai::math::Vector3& corner : corners) {
        minX = std::min(minX, corner.x);
        minY = std::min(minY, corner.y);
        minZ = std::min(minZ, corner.z);
        maxX = std::max(maxX, corner.x);
        maxY = std::max(maxY, corner.y);
        maxZ = std::max(maxZ, corner.z);
    }

    odai::core::CellAabb broadPhaseBounds{};
    broadPhaseBounds.valid = true;
    broadPhaseBounds.minInclusive = odai::core::Cell3i{
        static_cast<int>(std::floor(minX - kRenderFrustumBoundsPadVoxels)),
        static_cast<int>(std::floor(minY - kRenderFrustumBoundsPadVoxels)),
        static_cast<int>(std::floor(minZ - kRenderFrustumBoundsPadVoxels))
    };
    broadPhaseBounds.maxExclusive = odai::core::Cell3i{
        static_cast<int>(std::floor(maxX + kRenderFrustumBoundsPadVoxels)) + 1,
        static_cast<int>(std::floor(maxY + kRenderFrustumBoundsPadVoxels)) + 1,
        static_cast<int>(std::floor(maxZ + kRenderFrustumBoundsPadVoxels)) + 1
    };

    const odai::math::Vector3 leftDir = odai::math::normalize(forward - (right * tanHalfX));
    const odai::math::Vector3 rightDir = odai::math::normalize(forward + (right * tanHalfX));
    const odai::math::Vector3 topDir = odai::math::normalize(forward + (up * tanHalfY));
    const odai::math::Vector3 bottomDir = odai::math::normalize(forward - (up * tanHalfY));

    std::array<FrustumPlane, 6> planes{};
    planes[0] = makePlaneFromPointNormal(nearCenter, forward);
    planes[1] = makePlaneFromPointNormal(farCenter, -forward);
    planes[2] = makePlaneFromPointNormal(eye, odai::math::cross(up, leftDir));
    planes[3] = makePlaneFromPointNormal(eye, odai::math::cross(rightDir, up));
    planes[4] = makePlaneFromPointNormal(eye, odai::math::cross(topDir, right));
    planes[5] = makePlaneFromPointNormal(eye, odai::math::cross(right, bottomDir));
    orientPlaneTowardForward(planes[2], forward);
    orientPlaneTowardForward(planes[3], forward);
    orientPlaneTowardForward(planes[4], forward);
    orientPlaneTowardForward(planes[5], forward);

    frustum.planes = planes;
    frustum.broadPhaseBounds = broadPhaseBounds;
    frustum.valid = true;
    return frustum;
}

[[maybe_unused]] bool chunkIntersectsFrustum(
    const odai::world::Chunk& chunk,
    const std::array<FrustumPlane, 6>& planes,
    float planeSlack
) {
    const float minX = static_cast<float>(chunk.chunkX() * odai::world::Chunk::kSizeX);
    const float minY = static_cast<float>(chunk.chunkY() * odai::world::Chunk::kSizeY);
    const float minZ = static_cast<float>(chunk.chunkZ() * odai::world::Chunk::kSizeZ);
    const float maxX = minX + static_cast<float>(odai::world::Chunk::kSizeX);
    const float maxY = minY + static_cast<float>(odai::world::Chunk::kSizeY);
    const float maxZ = minZ + static_cast<float>(odai::world::Chunk::kSizeZ);

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

std::optional<std::string> readEnvironmentString(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t valueLength = 0;
    if (_dupenv_s(&value, &valueLength, name) != 0 || value == nullptr || valueLength == 0) {
        if (value != nullptr) {
            std::free(value);
        }
        return std::nullopt;
    }
    std::string result(value);
    std::free(value);
    return result;
#else
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return std::nullopt;
    }
    return std::string(value);
#endif
}

// Resolve a project-relative asset path. Mirrors the renderer's resolver: prefer
// the compiled-in source dir, then walk up from the working directory (the app is
// commonly launched from a build subdirectory). Returns the relative path as-is if
// nothing matches, so callers can still probe std::filesystem::exists.
std::filesystem::path resolveAssetPath(const std::filesystem::path& relativePath) {
    std::vector<std::filesystem::path> baseCandidates;
#if defined(ODAI_PROJECT_SOURCE_DIR)
    baseCandidates.emplace_back(std::filesystem::path{ODAI_PROJECT_SOURCE_DIR});
#endif
    std::error_code cwdError;
    const std::filesystem::path cwd = std::filesystem::current_path(cwdError);
    if (!cwdError) {
        baseCandidates.push_back(cwd);
        baseCandidates.push_back(cwd / "..");
        baseCandidates.push_back(cwd / ".." / "..");
        baseCandidates.push_back(cwd / ".." / ".." / "..");
    }
    for (const std::filesystem::path& base : baseCandidates) {
        const std::filesystem::path candidate = base / relativePath;
        std::error_code existsError;
        if (std::filesystem::exists(candidate, existsError) && !existsError) {
            return candidate;
        }
    }
    return relativePath;
}

std::optional<std::filesystem::path> findStrategyMapPath() {
    if (const std::optional<std::string> envPathValue = readEnvironmentString(kStrategyMapEnvVar);
        envPathValue.has_value() && !envPathValue->empty()) {
        return std::filesystem::path(*envPathValue);
    }

    const std::filesystem::path fallback("strategy_map.smap");
    if (std::filesystem::exists(fallback)) {
        return fallback;
    }
    return std::nullopt;
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

odai::core::Dir6 axisToDir6(const odai::math::Vector3& axis) {
    const odai::math::Vector3 normalized = odai::math::normalize(axis);
    const float absX = std::abs(normalized.x);
    const float absY = std::abs(normalized.y);
    const float absZ = std::abs(normalized.z);
    if (absX >= absY && absX >= absZ) {
        return normalized.x >= 0.0f ? odai::core::Dir6::PosX : odai::core::Dir6::NegX;
    }
    if (absY >= absX && absY >= absZ) {
        return normalized.y >= 0.0f ? odai::core::Dir6::PosY : odai::core::Dir6::NegY;
    }
    return normalized.z >= 0.0f ? odai::core::Dir6::PosZ : odai::core::Dir6::NegZ;
}

odai::core::Dir6 faceNormalToDir6(int nx, int ny, int nz) {
    if (nx > 0) {
        return odai::core::Dir6::PosX;
    }
    if (nx < 0) {
        return odai::core::Dir6::NegX;
    }
    if (ny > 0) {
        return odai::core::Dir6::PosY;
    }
    if (ny < 0) {
        return odai::core::Dir6::NegY;
    }
    if (nz > 0) {
        return odai::core::Dir6::PosZ;
    }
    return odai::core::Dir6::NegZ;
}

void dir6ToAxisInts(odai::core::Dir6 dir, int& outX, int& outY, int& outZ) {
    const odai::core::Cell3i offset = odai::core::dirToOffset(dir);
    outX = static_cast<int>(offset.x);
    outY = static_cast<int>(offset.y);
    outZ = static_cast<int>(offset.z);
}

bool dirSharesAxis(odai::core::Dir6 lhs, odai::core::Dir6 rhs) {
    return lhs == rhs || odai::core::areOpposite(lhs, rhs);
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

float lerpWrappedDegrees(float fromDegrees, float toDegrees, float alpha) {
    const float delta = wrapDegreesSigned(toDegrees - fromDegrees);
    return wrapDegreesSigned(fromDegrees + (delta * alpha));
}

odai::core::Dir6 horizontalDirFromYaw(float yawDegrees) {
    const float yawRadians = odai::math::radians(yawDegrees);
    const float x = std::cos(yawRadians);
    const float z = std::sin(yawRadians);
    if (std::abs(x) >= std::abs(z)) {
        return x >= 0.0f ? odai::core::Dir6::PosX : odai::core::Dir6::NegX;
    }
    return z >= 0.0f ? odai::core::Dir6::PosZ : odai::core::Dir6::NegZ;
}

odai::sim::BeltDirection dir6ToBeltDirection(odai::core::Dir6 dir) {
    switch (dir) {
    case odai::core::Dir6::PosX:
        return odai::sim::BeltDirection::East;
    case odai::core::Dir6::NegX:
        return odai::sim::BeltDirection::West;
    case odai::core::Dir6::PosZ:
        return odai::sim::BeltDirection::South;
    case odai::core::Dir6::NegZ:
    default:
        return odai::sim::BeltDirection::North;
    }
}

odai::core::Dir6 beltDirectionToDir6(odai::sim::BeltDirection direction) {
    switch (direction) {
    case odai::sim::BeltDirection::East:
        return odai::core::Dir6::PosX;
    case odai::sim::BeltDirection::West:
        return odai::core::Dir6::NegX;
    case odai::sim::BeltDirection::South:
        return odai::core::Dir6::PosZ;
    case odai::sim::BeltDirection::North:
    default:
        return odai::core::Dir6::NegZ;
    }
}

odai::sim::TrackDirection dir6ToTrackDirection(odai::core::Dir6 dir) {
    switch (dir) {
    case odai::core::Dir6::PosX:
        return odai::sim::TrackDirection::East;
    case odai::core::Dir6::NegX:
        return odai::sim::TrackDirection::West;
    case odai::core::Dir6::PosZ:
        return odai::sim::TrackDirection::South;
    case odai::core::Dir6::NegZ:
    default:
        return odai::sim::TrackDirection::North;
    }
}

odai::core::Dir6 trackDirectionToDir6(odai::sim::TrackDirection direction) {
    switch (direction) {
    case odai::sim::TrackDirection::East:
        return odai::core::Dir6::PosX;
    case odai::sim::TrackDirection::West:
        return odai::core::Dir6::NegX;
    case odai::sim::TrackDirection::South:
        return odai::core::Dir6::PosZ;
    case odai::sim::TrackDirection::North:
    default:
        return odai::core::Dir6::NegZ;
    }
}

odai::core::Dir6 firstDirFromMask(std::uint8_t mask) {
    for (const odai::core::Dir6 dir : odai::core::kAllDir6) {
        if ((mask & odai::core::dirBit(dir)) != 0u) {
            return dir;
        }
    }
    return odai::core::Dir6::PosY;
}

odai::core::Dir6 resolveStraightAxisFromMask(std::uint8_t mask, odai::core::Dir6 preferredAxis) {
    for (const odai::core::Dir6 dir : odai::core::kAllDir6) {
        if ((mask & odai::core::dirBit(dir)) == 0u) {
            continue;
        }
        const odai::core::Dir6 opposite = odai::core::oppositeDir(dir);
        if ((mask & odai::core::dirBit(opposite)) == 0u) {
            continue;
        }
        if (dirSharesAxis(preferredAxis, dir)) {
            return preferredAxis;
        }
        return dir;
    }
    return preferredAxis;
}

// Load a PNG as an ImportedSceneTexture with a full box-filter mip chain.
// Returns an empty texture (width==0) if the file is missing or unreadable.
odai::importer::ImportedSceneTexture loadTerrainTextureWithMips(
    const std::filesystem::path& path)
{
    odai::importer::ImportedSceneTexture tex{};
    tex.sourcePath = path.string();
    int w = 0, h = 0;
    stbi_uc* pixels = stbi_load(path.string().c_str(), &w, &h, nullptr, 4);
    if (!pixels || w <= 0 || h <= 0) {
        if (pixels) stbi_image_free(pixels);
        return tex;
    }
    tex.width  = static_cast<std::uint32_t>(w);
    tex.height = static_cast<std::uint32_t>(h);

    const std::size_t baseBytes =
        static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4u;
    tex.rgba8.assign(pixels, pixels + baseBytes);
    stbi_image_free(pixels);

    // Box-filter mip chain; the renderer validates size by inferring mip count
    // from rgba8.size(), so the chain must be packed in full down to 1x1.
    std::vector<std::uint8_t> cur(tex.rgba8);
    std::uint32_t mw = tex.width;
    std::uint32_t mh = tex.height;
    tex.mipLevelCount = 1u;
    while (mw > 1u || mh > 1u) {
        const std::uint32_t pw = mw;
        const std::uint32_t ph = mh;
        mw = std::max(1u, mw >> 1u);
        mh = std::max(1u, mh >> 1u);
        std::vector<std::uint8_t> next(
            static_cast<std::size_t>(mw) * static_cast<std::size_t>(mh) * 4u);
        for (std::uint32_t y = 0; y < mh; ++y) {
            for (std::uint32_t x = 0; x < mw; ++x) {
                const std::uint32_t sx0 = x * 2u;
                const std::uint32_t sy0 = y * 2u;
                const std::uint32_t sx1 = std::min(sx0 + 1u, pw - 1u);
                const std::uint32_t sy1 = std::min(sy0 + 1u, ph - 1u);
                for (int c = 0; c < 4; ++c) {
                    const int s00 = cur[(sy0 * pw + sx0) * 4u + c];
                    const int s10 = cur[(sy0 * pw + sx1) * 4u + c];
                    const int s01 = cur[(sy1 * pw + sx0) * 4u + c];
                    const int s11 = cur[(sy1 * pw + sx1) * 4u + c];
                    next[(y * mw + x) * 4u + c] =
                        static_cast<std::uint8_t>((s00 + s10 + s01 + s11 + 2) / 4);
                }
            }
        }
        tex.rgba8.insert(tex.rgba8.end(), next.begin(), next.end());
        cur = std::move(next);
        ++tex.mipLevelCount;
    }
    return tex;
}

// Build the strategy-map mesh options for the given relief mode. Terrain textures are
// passed in from the App's cached copies (loaded once at init) to avoid repeated disk
// I/O and mip-chain generation on every re-mesh (M toggle, end-turn, unit moves).
odai::game::StrategyMapMeshOptions makeStrategyMapMeshOptions(
    bool extruded,
    const std::vector<odai::importer::ImportedSceneTexture>& terrainTextures)
{
    odai::game::StrategyMapMeshOptions meshOptions{};
    meshOptions.extruded = extruded;
    meshOptions.drawSettlements = false;
    meshOptions.terrainTextures = terrainTextures;
    return meshOptions;
}

} // namespace

namespace odai::app {

bool App::loadConfig(const std::filesystem::path& configPath) {
    std::ifstream file(configPath);
    if (!file) {
        VOX_LOGI("app") << "config file missing at " << configPath.string()
                        << "; using defaults";
        return true;
    }

    AppConfig loadedConfig = m_config;
    std::string line;
    std::uint32_t parsedLineCount = 0;
    while (std::getline(file, line)) {
        ++parsedLineCount;
        const std::size_t commentPos = line.find('#');
        if (commentPos != std::string::npos) {
            line.erase(commentPos);
        }
        const std::size_t equalsPos = line.find('=');
        if (equalsPos == std::string::npos) {
            continue;
        }
        const std::string key = trimConfigString(line.substr(0, equalsPos));
        const std::string value = trimConfigString(line.substr(equalsPos + 1));
        if (key == "shadow_mode") {
            odai::render::ShadowMode parsedMode = loadedConfig.shadowMode;
            if (!parseShadowModeConfigValue(value, parsedMode)) {
                VOX_LOGW("app") << "invalid config shadow_mode='" << value
                                << "' at " << configPath.string() << "; keeping "
                                << shadowModeConfigName(loadedConfig.shadowMode);
                continue;
            }
            loadedConfig.shadowMode = parsedMode;
            continue;
        }
        if (key == "enable_vertex_ao") {
            continue;
        }
        if (key == "enable_ssao") {
            bool parsedValue = loadedConfig.enableSsao;
            if (!parseBoolConfigValue(value, parsedValue)) {
                VOX_LOGW("app") << "invalid config enable_ssao='" << value
                                << "' at " << configPath.string() << "; keeping "
                                << boolConfigName(loadedConfig.enableSsao);
                continue;
            }
            loadedConfig.enableSsao = parsedValue;
        }
    }
    m_config = loadedConfig;
    VOX_LOGI("app") << "config loaded from " << configPath.string()
                    << " (shadow_mode=" << shadowModeConfigName(m_config.shadowMode)
                    << ", enable_ssao=" << boolConfigName(m_config.enableSsao)
                    << ", parsedLines=" << parsedLineCount << ")";
    return true;
}

bool App::saveConfig(const std::filesystem::path& configPath) const {
    std::ofstream file(configPath, std::ios::trunc);
    if (!file) {
        return false;
    }
    file << "# Renderer runtime config\n";
    file << "shadow_mode=" << shadowModeConfigName(m_config.shadowMode) << "\n";
    file << "enable_ssao=" << boolConfigName(m_config.enableSsao) << "\n";
    return true;
}

bool App::init() {
    using Clock = std::chrono::steady_clock;
    const auto initStart = Clock::now();
    auto elapsedMs = [](const Clock::time_point& start) -> std::int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    };

    VOX_LOGI("app") << "init begin"
                    << " version=" << ODAI_APP_VERSION
                    << " profile=" << ODAI_RELEASE_PROFILE;
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

    // Size the window to the primary monitor's logical (DPI-scaled) resolution so
    // it fills the screen regardless of the OS scaling factor (125 %, 150 %, 200 % …).
    // On a 4K display at 200 % DPI: physical=3840×2160, logical=1920×1080 → fills screen.
    // On a 4K display at 100 % DPI: physical=3840×2160, logical=3840×2160 → fills screen.
    int winW = 1920, winH = 1080;
    if (GLFWmonitor* mon = glfwGetPrimaryMonitor()) {
        float xs = 1.0f, ys = 1.0f;
        glfwGetMonitorContentScale(mon, &xs, &ys);
        if (const GLFWvidmode* mode = glfwGetVideoMode(mon)) {
            winW = static_cast<int>(std::round(mode->width  / std::max(xs, 1.0f)));
            winH = static_cast<int>(std::round(mode->height / std::max(ys, 1.0f)));
        }
    }
    m_window = glfwCreateWindow(winW, winH, "odai", nullptr, nullptr);
    if (m_window == nullptr) {
        VOX_LOGE("app") << "glfwCreateWindow failed";
        glfwTerminate();
        return false;
    }
    VOX_LOGI("app") << "init step createWindow took " << elapsedMs(windowStart) << " ms";

    // Relative mouse mode for camera look.
    glfwSetWindowUserPointer(m_window, this);
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Character input for editable UI text fields. GLFW delivers fully-composed
    // Unicode codepoints here; they are drained into the UI input each frame.
    glfwSetCharCallback(m_window, [](GLFWwindow* win, unsigned int codepoint) {
        if (auto* self = static_cast<App*>(glfwGetWindowUserPointer(win))) {
            self->m_pendingTextInput.push_back(codepoint);
        }
    });

    // Scroll wheel input â€” accumulated between frames and drained into UiInput.
    glfwSetScrollCallback(m_window, [](GLFWwindow* win, double /*xoffset*/, double yoffset) {
        if (auto* self = static_cast<App*>(glfwGetWindowUserPointer(win))) {
            self->m_pendingScrollDelta += static_cast<float>(yoffset);
        }
    });

    const std::filesystem::path configPath{kConfigFilePath};
    if (!loadConfig(configPath)) {
        VOX_LOGE("app") << "failed to load config from " << configPath.string();
        glfwDestroyWindow(m_window);
        m_window = nullptr;
        glfwTerminate();
        return false;
    }
    m_renderer.setShadowSettings(odai::render::ShadowSettings{m_config.shadowMode});
    m_renderer.setSsaoEnabled(m_config.enableSsao);
    m_gameplayUiState.selectedHotbarSlot = 0;
    m_gameplayUiState.hotbarItems = {
        odai::render::InventoryItemId::Stone,
        odai::render::InventoryItemId::Dirt,
        odai::render::InventoryItemId::Grass,
        odai::render::InventoryItemId::Wood,
        odai::render::InventoryItemId::Red,
        odai::render::InventoryItemId::Empty,
        odai::render::InventoryItemId::Empty,
        odai::render::InventoryItemId::Empty,
        odai::render::InventoryItemId::Empty,
    };

    // Load the hex strategy map. If no .smap file is found via the ODAI_STRATEGY_MAP
    // env-var or the default "strategy_map.smap" path, log a hint and exit.
    m_strategyMap = {};
    const std::optional<std::filesystem::path> strategyMapPath = findStrategyMapPath();
    if (strategyMapPath.has_value()) {
        m_importedScenePath = *strategyMapPath;
        if (!odai::game::loadStrategyMap(*strategyMapPath, m_strategyMap)) {
            VOX_LOGE("app") << "failed to load strategy map from "
                            << std::filesystem::absolute(*strategyMapPath).string()
                            << ": " << odai::game::getStrategyMapLastError();
            return false;
        }
        VOX_LOGI("app") << "strategy map loaded from "
                        << std::filesystem::absolute(*strategyMapPath).string()
                        << " (" << m_strategyMap.width << "x" << m_strategyMap.height
                        << ", settlements=" << m_strategyMap.settlements.size() << ")";
    } else {
        VOX_LOGE("app") << "no strategy map found; run odai_strategy_map_gen.exe to generate one, "
                           "then set ODAI_STRATEGY_MAP=strategy_map.smap or place strategy_map.smap "
                           "in the working directory";
        return false;
    }

    // Load terrain textures once. Forest/Jungle share grassland; Hills/Mountains share
    // rock. These are copied into StrategyMapMeshOptions on each re-mesh so the mesh
    // builder can move them into the scene without needing another disk read.
    {
        using T = odai::game::TerrainType;
        m_terrainTextures.resize(static_cast<std::size_t>(T::Count));
        std::unordered_map<std::string, std::size_t> pathToSlot;
        auto load = [&](T t, const char* rel) {
            const std::size_t slot = static_cast<std::size_t>(t);
            const std::string key = resolveAssetPath(rel).string();
            const auto it = pathToSlot.find(key);
            if (it == pathToSlot.end()) {
                pathToSlot.emplace(key, slot);
                m_terrainTextures[slot] = loadTerrainTextureWithMips(resolveAssetPath(rel));
            } else {
                m_terrainTextures[slot] = m_terrainTextures[it->second];
            }
        };
        load(T::Grassland, "assets/textures/terrain_grassland.png");
        load(T::Plains,    "assets/textures/terrain_plains.png");
        load(T::Forest,    "assets/textures/terrain_grassland.png");
        load(T::Jungle,    "assets/textures/terrain_grassland.png");
        load(T::Hills,     "assets/textures/terrain_rock.png");
        load(T::Mountains, "assets/textures/terrain_rock.png");
        load(T::Desert,    "assets/textures/terrain_desert.png");
        load(T::Tundra,    "assets/textures/terrain_tundra.png");
        load(T::Snow,      "assets/textures/terrain_snow.png");
    }

    // Seed a few playable units on friendly settlements so the supply mechanic has
    // something to drive; meshed below so they appear on the board from the start.
    spawnDemoUnits();

    // Start in 3D relief mode; toggling (M) re-meshes flat for the 2D board view.
    m_strategyMap3D = true;
    odai::game::StrategyMapMeshOptions meshOptions = makeStrategyMapMeshOptions(/*extruded=*/true, m_terrainTextures);
    m_importedScene = odai::game::buildStrategyMapScene(m_strategyMap, m_gameState.units, std::move(meshOptions));
    m_importedSceneDemoEnabled = true;
    m_hoverEnabled = true;

    // Compute map world extents for camera centering and pan clamping.
    constexpr float kStratMapSqrt3 = 1.7320508075688772f;
    const float mapWorldW = m_strategyMap.hexSize * kStratMapSqrt3 *
        (static_cast<float>(m_strategyMap.width) + 0.5f);
    const float mapWorldH = m_strategyMap.hexSize * 1.5f *
        static_cast<float>(m_strategyMap.height);
    m_mapMinX = 0.0f;
    m_mapMaxX = mapWorldW;
    m_mapMinZ = 0.0f;
    m_mapMaxZ = mapWorldH;
    // Start zoomed in so the map is bigger than the viewport; the player pans to explore.
    m_mapOrthoHalfHeight = m_strategyMap.hexSize * 8.0f;

    // 3D relief view orbits this focus point; seed it on the map center.
    m_map3DFocusX = mapWorldW * 0.5f;
    m_map3DFocusZ = mapWorldH * 0.5f;
    m_map3DDistance = std::max(mapWorldH, mapWorldW) * 0.30f;

    // Default to the tilted perspective 3D camera (M toggles to 2D top-down ortho).
    m_camera.yawDegrees = kMap3DYawDegrees;
    m_camera.pitchDegrees = m_map3DPitchDeg;
    m_camera.fovDegrees = kMap3DFovDegrees;
    {
        const float yawR = odai::math::radians(kMap3DYawDegrees);
        const float pitchR = odai::math::radians(m_map3DPitchDeg);
        const float cpv = std::cos(pitchR);
        const odai::math::Vector3 fwd{cpv * std::cos(yawR), std::sin(pitchR), cpv * std::sin(yawR)};
        m_camera.x = m_map3DFocusX - fwd.x * m_map3DDistance;
        m_camera.y = -fwd.y * m_map3DDistance;  // focus sits on the y=0 plane
        m_camera.z = m_map3DFocusZ - fwd.z * m_map3DDistance;
    }
    m_cameraPrevious = m_camera;

    m_renderer.setStrategyMapMode(true);
    // Strategy map is cursor-driven: show the OS cursor and stop mouselook so the
    // player can click HUD elements.
    m_strategyMapMode = true;
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    const auto rendererInitStart = Clock::now();
    if (!m_renderer.init(m_window, m_world.chunkGrid())) {
        VOX_LOGE("app") << "renderer init failed";
        return false;
    }
    VOX_LOGI("app") << "init step renderer init took " << elapsedMs(rendererInitStart) << " ms";
    if (!m_renderer.uploadImportedScene(m_importedScene)) {
        VOX_LOGE("app") << "strategy map scene upload failed";
        return false;
    }
    m_renderer.setImportedSceneInteriorMode(false);

    // GPU hex land: when the device created the tessellation pipeline, upload the
    // instanced, height-displaced hex terrain and re-mesh so the imported-static path
    // drops the land tops (the hex pipeline owns them). Falls back to flat land
    // otherwise. Active in 3D relief only; the flat 2D board keeps the imported land.
    m_useHexTerrain = m_renderer.hexTerrainReady();
    if (m_useHexTerrain) {
        m_hexTerrain = odai::game::buildHexTerrain(m_strategyMap, odai::game::HexTerrainOptions{});
        if (m_renderer.uploadHexTerrain(m_hexTerrain)) {
            m_renderer.setHexTerrainEnabled(m_strategyMap3D);
            rebuildStrategyMapScene();
        } else {
            m_useHexTerrain = false;
            VOX_LOGW("app") << "hex terrain upload failed; keeping flat land";
        }
    }

    // SSAO deepens land-against-water seams in 3D relief; we default to 3D, so on.
    // The flat 2D board has no occlusion to compute and turns it off on the M toggle.
    m_renderer.setSsaoEnabled(true);
    m_renderer.setSunAngles(kImportedSceneSunYawDegrees, kImportedSceneSunPitchDegrees);
    m_renderer.setImportedSceneDebugState(
        m_importedShowTerrain,
        m_importedShowStatics,
        m_importedShowTextures,
        m_importedFlatShading,
        m_importedWaterDebug);

    VOX_LOGI("app") << "init complete in " << elapsedMs(initStart) << " ms";
    return true;
}

void App::run() {
    VOX_LOGI("app") << "run begin";
    VOX_LOGI("app") << "strategy map ready (scroll/drag to pan, F/C renderer panel)";
    double previousTime = glfwGetTime();
    double simulationAccumulatorSeconds = 0.0;
    m_cameraPrevious = m_camera;
    uint64_t frameCount = 0;

    while (m_window != nullptr && glfwWindowShouldClose(m_window) == GLFW_FALSE) {
        const double currentTime = glfwGetTime();
        const double rawFrameSeconds = std::max(0.0, currentTime - previousTime);
        previousTime = currentTime;
        const double frameSeconds = std::min(rawFrameSeconds, kFrameDeltaClampSeconds);
        const float dt = static_cast<float>(frameSeconds);
        simulationAccumulatorSeconds += frameSeconds;

        pollInput();
        if (glfwWindowShouldClose(m_window) == GLFW_TRUE) {
            break;
        }

        float simulationAlpha = 0.0f;
        if (m_strategyMapMode) {
            // The strategy-map camera is purely kinematic (pan + inertia + zoom) with
            // no dependency on the fixed-step voxel physics. Driving it through the
            // fixed accumulator would sample the per-frame mouse delta at 0/1/2 steps
            // per rendered frame (a beat against vsync) and then render-interpolate the
            // result, smearing input-driven motion into micro-stutter. Update it once
            // per rendered frame with the real dt and render it directly (prev==current
            // makes interpolation a no-op) for stutter-free 1:1 mouse tracking.
            updateCamera(dt);
            m_cameraPrevious = m_camera;
            simulationAccumulatorSeconds = 0.0;
        } else {
            int simulationStepCount = 0;
            while (simulationAccumulatorSeconds >= kSimulationFixedStepSeconds &&
                   simulationStepCount < kMaxSimulationStepsPerFrame) {
                m_cameraPrevious = m_camera;
                updateCamera(static_cast<float>(kSimulationFixedStepSeconds));
                simulationAccumulatorSeconds -= kSimulationFixedStepSeconds;
                ++simulationStepCount;
            }
            if (simulationStepCount == kMaxSimulationStepsPerFrame &&
                simulationAccumulatorSeconds >= kSimulationFixedStepSeconds) {
                // Drop excess backlog to keep simulation responsive after long stalls.
                simulationAccumulatorSeconds = std::fmod(simulationAccumulatorSeconds, kSimulationFixedStepSeconds);
            }
            simulationAlpha = static_cast<float>(
                std::clamp(simulationAccumulatorSeconds / kSimulationFixedStepSeconds, 0.0, 1.0)
            );
        }
        update(dt, simulationAlpha);
        ++frameCount;
    }

    VOX_LOGI("app") << "run exit after " << frameCount
                    << " frame(s), windowShouldClose="
                    << (m_window != nullptr ? glfwWindowShouldClose(m_window) : 1);
}

void App::toggleDebugUi() {
    m_debugUiVisible = !m_debugUiVisible;
}

bool App::isAnyUiVisible() const {
    return m_debugUiVisible;
}

void App::syncGameplayUiState() {
}

void App::resetVoxelBreakProgress() {
    m_voxelBreakTargetValid = false;
    m_voxelBreakTargetX = 0;
    m_voxelBreakTargetY = 0;
    m_voxelBreakTargetZ = 0;
    m_voxelBreakClicks = 0;
}


void App::assignInventoryItemToSelectedHotbar(odai::render::InventoryItemId itemId) {
    const std::size_t hotbarIndex = std::min<std::size_t>(
        m_gameplayUiState.selectedHotbarSlot,
        m_gameplayUiState.hotbarItems.size() - 1
    );
    if (m_gameplayUiState.hotbarItems[hotbarIndex] == itemId) {
        return;
    }
    m_gameplayUiState.hotbarItems[hotbarIndex] = itemId;
    VOX_LOGI("app") << "assigned hotbar " << (hotbarIndex + 1) << ": " << inventoryItemLabel(itemId);
    syncGameplayUiState();
}

void App::handleInventoryClick(float mouseX, float mouseY, float displayWidth, float displayHeight) {
    const odai::render::GameplayUiLayout layout =
        odai::render::buildGameplayUiLayout(displayWidth, displayHeight);
    for (std::size_t slotIndex = 0; slotIndex < layout.hotbarSlots.size(); ++slotIndex) {
        if (!layout.hotbarSlots[slotIndex].contains(mouseX, mouseY)) {
            continue;
        }
        selectHotbarSlot(static_cast<int>(slotIndex));
        return;
    }
    for (std::size_t itemIndex = 0; itemIndex < layout.inventorySlots.size(); ++itemIndex) {
        if (!layout.inventorySlots[itemIndex].contains(mouseX, mouseY)) {
            continue;
        }
        assignInventoryItemToSelectedHotbar(m_gameplayUiState.creativeInventoryItems[itemIndex]);
        return;
    }
}

void App::update([[maybe_unused]] float dt, float simulationAlpha) {
    const float renderAlpha = std::clamp(simulationAlpha, 0.0f, 1.0f);
    odai::render::CameraPose cameraPose{
        m_cameraPrevious.x + ((m_camera.x - m_cameraPrevious.x) * renderAlpha),
        m_cameraPrevious.y + ((m_camera.y - m_cameraPrevious.y) * renderAlpha),
        m_cameraPrevious.z + ((m_camera.z - m_cameraPrevious.z) * renderAlpha),
        lerpWrappedDegrees(m_cameraPrevious.yawDegrees, m_camera.yawDegrees, renderAlpha),
        m_cameraPrevious.pitchDegrees + ((m_camera.pitchDegrees - m_cameraPrevious.pitchDegrees) * renderAlpha),
        m_camera.fovDegrees
    };
    cameraPose.orthographic = m_strategyMapMode && !m_strategyMap3D;
    cameraPose.orthoHalfHeight = m_mapOrthoHalfHeight;
    m_visibleChunkIndices.clear();
    m_renderer.setSpatialQueryStats(false, odai::world::SpatialQueryStats{}, 0u);
    m_renderCameraPose = cameraPose;
    updateUiOverlay(dt);
    m_renderer.renderFrame(
        m_world.chunkGrid(),
        m_simulation,
        cameraPose,
        odai::render::VoxelPreview{},
        simulationAlpha,
        std::span<const std::size_t>(m_visibleChunkIndices)
    );
    m_camera.fovDegrees = m_renderer.cameraFovDegrees();
}

void App::shutdown() {
    VOX_LOGI("app") << "shutdown begin";

    m_config.shadowMode = m_renderer.shadowSettings().mode;
    m_config.enableSsao = m_renderer.isSsaoEnabled();
    const std::filesystem::path configPath{kConfigFilePath};
    if (!saveConfig(configPath)) {
        VOX_LOGE("app") << "failed to save config to " << configPath.string();
    } else {
        VOX_LOGI("app") << "saved config to " << configPath.string()
                        << " (shadow_mode=" << shadowModeConfigName(m_config.shadowMode)
                        << ", enable_ssao=" << boolConfigName(m_config.enableSsao) << ")";
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
        toggleDebugUi();
        uiVisibilityChanged = true;
    }
    m_wasToggleFrameStatsDown = toggleFrameStatsDown;

    const bool toggleConfigUiDown = glfwGetKey(m_window, GLFW_KEY_C) == GLFW_PRESS;
    if (toggleConfigUiDown && !m_wasToggleConfigUiDown) {
        toggleDebugUi();
        uiVisibilityChanged = true;
    }
    m_wasToggleConfigUiDown = toggleConfigUiDown;

    const bool escapeKeyDown = glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
    if (escapeKeyDown && !m_wasEscapeKeyDown) {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }
    m_wasEscapeKeyDown = escapeKeyDown;

    const bool toggleDayCycleDown = glfwGetKey(m_window, GLFW_KEY_T) == GLFW_PRESS;
    if (toggleDayCycleDown && !m_wasToggleDayCycleDown) {
        m_dayCycleEnabled = !m_dayCycleEnabled;
        VOX_LOGI("app") << "day cycle " << (m_dayCycleEnabled ? "enabled" : "disabled")
                        << " (T, winter arc lat=" << kDayCycleLatitudeDegrees
                        << " decl=" << kDayCycleWinterDeclinationDegrees << ")";
    }
    m_wasToggleDayCycleDown = toggleDayCycleDown;

    // M toggles the strategy map between 2D flat board and 3D relief. Handled here
    // (before updateCamera) so the camera reframes the same frame the mode changes.
    const bool toggleMapViewDown = glfwGetKey(m_window, GLFW_KEY_M) == GLFW_PRESS;
    if (m_strategyMapMode && toggleMapViewDown && !m_wasToggleMapViewDown) {
        setStrategyMap3D(!m_strategyMap3D);
    }
    m_wasToggleMapViewDown = toggleMapViewDown;

    const bool toggleImportedTerrainDown = glfwGetKey(m_window, GLFW_KEY_F5) == GLFW_PRESS;
    if (m_importedSceneDemoEnabled && toggleImportedTerrainDown && !m_wasToggleImportedTerrainDown) {
        m_renderer.importedSceneDebugState(
            m_importedShowTerrain,
            m_importedShowStatics,
            m_importedShowTextures,
            m_importedFlatShading,
            m_importedWaterDebug);
        m_importedShowTerrain = !m_importedShowTerrain;
        m_renderer.setImportedSceneDebugState(
            m_importedShowTerrain,
            m_importedShowStatics,
            m_importedShowTextures,
            m_importedFlatShading,
            m_importedWaterDebug);
        VOX_LOGI("app") << "imported terrain " << (m_importedShowTerrain ? "visible" : "hidden") << " (F5)";
    }
    m_wasToggleImportedTerrainDown = toggleImportedTerrainDown;

    const bool toggleImportedStaticsDown = glfwGetKey(m_window, GLFW_KEY_F6) == GLFW_PRESS;
    if (m_importedSceneDemoEnabled && toggleImportedStaticsDown && !m_wasToggleImportedStaticsDown) {
        m_renderer.importedSceneDebugState(
            m_importedShowTerrain,
            m_importedShowStatics,
            m_importedShowTextures,
            m_importedFlatShading,
            m_importedWaterDebug);
        m_importedShowStatics = !m_importedShowStatics;
        m_renderer.setImportedSceneDebugState(
            m_importedShowTerrain,
            m_importedShowStatics,
            m_importedShowTextures,
            m_importedFlatShading,
            m_importedWaterDebug);
        VOX_LOGI("app") << "imported statics " << (m_importedShowStatics ? "visible" : "hidden") << " (F6)";
    }
    m_wasToggleImportedStaticsDown = toggleImportedStaticsDown;

    const bool toggleImportedTexturesDown = glfwGetKey(m_window, GLFW_KEY_K) == GLFW_PRESS;
    if (m_importedSceneDemoEnabled && toggleImportedTexturesDown && !m_wasToggleImportedTexturesDown) {
        m_renderer.importedSceneDebugState(
            m_importedShowTerrain,
            m_importedShowStatics,
            m_importedShowTextures,
            m_importedFlatShading,
            m_importedWaterDebug);
        m_importedShowTextures = !m_importedShowTextures;
        m_renderer.setImportedSceneDebugState(
            m_importedShowTerrain,
            m_importedShowStatics,
            m_importedShowTextures,
            m_importedFlatShading,
            m_importedWaterDebug);
        VOX_LOGI("app") << "imported textures " << (m_importedShowTextures ? "enabled" : "disabled") << " (K)";
    }
    m_wasToggleImportedTexturesDown = toggleImportedTexturesDown;

    const bool toggleImportedFlatShadingDown = glfwGetKey(m_window, GLFW_KEY_F7) == GLFW_PRESS;
    if (m_importedSceneDemoEnabled && toggleImportedFlatShadingDown && !m_wasToggleImportedFlatShadingDown) {
        m_renderer.importedSceneDebugState(
            m_importedShowTerrain,
            m_importedShowStatics,
            m_importedShowTextures,
            m_importedFlatShading,
            m_importedWaterDebug);
        m_importedFlatShading = !m_importedFlatShading;
        m_renderer.setImportedSceneDebugState(
            m_importedShowTerrain,
            m_importedShowStatics,
            m_importedShowTextures,
            m_importedFlatShading,
            m_importedWaterDebug);
        VOX_LOGI("app") << "imported flat shading " << (m_importedFlatShading ? "enabled" : "disabled") << " (F7)";
    }
    m_wasToggleImportedFlatShadingDown = toggleImportedFlatShadingDown;

    const bool toggleImportedWaterDebugDown = glfwGetKey(m_window, GLFW_KEY_F8) == GLFW_PRESS;
    if (m_importedSceneDemoEnabled && toggleImportedWaterDebugDown && !m_wasToggleImportedWaterDebugDown) {
        m_renderer.importedSceneDebugState(
            m_importedShowTerrain,
            m_importedShowStatics,
            m_importedShowTextures,
            m_importedFlatShading,
            m_importedWaterDebug);
        m_importedWaterDebug = !m_importedWaterDebug;
        m_renderer.setImportedSceneDebugState(
            m_importedShowTerrain,
            m_importedShowStatics,
            m_importedShowTextures,
            m_importedFlatShading,
            m_importedWaterDebug);
        VOX_LOGI("app") << "imported water debug " << (m_importedWaterDebug ? "enabled" : "disabled") << " (F8)";
    }
    m_wasToggleImportedWaterDebugDown = toggleImportedWaterDebugDown;

    const bool inspectImportedSceneDown = glfwGetKey(m_window, GLFW_KEY_I) == GLFW_PRESS;
    if (m_importedSceneDemoEnabled && inspectImportedSceneDown && !m_wasInspectImportedSceneDown) {
        inspectImportedSceneFromCamera();
    }
    m_wasInspectImportedSceneDown = inspectImportedSceneDown;

    m_renderer.setDebugUiVisible(m_debugUiVisible);
    const bool rendererUiVisible = m_renderer.isDebugUiVisible();
    if (rendererUiVisible != m_debugUiVisible) {
        m_debugUiVisible = rendererUiVisible;
        uiVisibilityChanged = true;
    }
    if (uiVisibilityChanged) {
        const bool showCursor = m_strategyMapMode || isAnyUiVisible();
        glfwSetInputMode(m_window, GLFW_CURSOR, showCursor ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
        m_hasMouseSample = false;
    }

    m_input.quitRequested = false;
    m_input.moveForward = glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS;
    m_input.moveBackward = glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS;
    m_input.moveLeft = glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS;
    m_input.moveRight = glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS;
    m_input.moveUp = glfwGetKey(m_window, GLFW_KEY_SPACE) == GLFW_PRESS;
    m_input.sneakDown =
        glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(m_window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    m_input.moveDown = m_importedSceneDemoEnabled
        ? m_input.sneakDown
        : (m_hoverEnabled && m_input.sneakDown);
    m_input.sprintDown =
        glfwGetKey(m_window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
        glfwGetKey(m_window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
    m_input.toggleHoverDown = glfwGetKey(m_window, GLFW_KEY_H) == GLFW_PRESS;
    m_input.regenerateWorldDown = glfwGetKey(m_window, GLFW_KEY_R) == GLFW_PRESS;
    bool controllerPlaceDown = false;
    bool controllerRemoveDown = false;
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
            VOX_LOGI("app") << "gamepad connected: RT place, LT remove";
        } else {
            VOX_LOGI("app") << "gamepad disconnected";
        }
    }
    if (hasGamepad) {
        controllerPlaceDown = gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] > kGamepadTriggerPressedThreshold;
        controllerRemoveDown = gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER] > kGamepadTriggerPressedThreshold;
        controllerMoveUpDown = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_A] == GLFW_PRESS;
        controllerMoveDownDown = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_B] == GLFW_PRESS;
        controllerMoveForward = -applyStickDeadzone(gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_Y], kGamepadMoveDeadzone);
        controllerMoveRight = applyStickDeadzone(gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_X], kGamepadMoveDeadzone);
        controllerLookX = applyStickDeadzone(gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_X], kGamepadLookDeadzone);
        controllerLookY = -applyStickDeadzone(gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y], kGamepadLookDeadzone);
    }

    m_input.placeBlockDown =
        glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS ||
        controllerPlaceDown;
    m_input.removeBlockDown =
        glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS ||
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

    if (isAnyUiVisible()) {
        m_input.mouseDeltaX = 0.0f;
        m_input.mouseDeltaY = 0.0f;
    }

    m_pendingMouseDeltaX += m_input.mouseDeltaX;
    m_pendingMouseDeltaY += m_input.mouseDeltaY;

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

    float mouseDeltaX = m_pendingMouseDeltaX;
    float mouseDeltaY = m_pendingMouseDeltaY;
    m_pendingMouseDeltaX = 0.0f;
    m_pendingMouseDeltaY = 0.0f;
    // Strategy map is cursor-driven: left-drag pans, scroll zooms, no mouselook.
    if (m_strategyMapMode) {
        int fbW = 0, fbH = 0;
        glfwGetFramebufferSize(m_window, &fbW, &fbH);
        const bool overUi = m_uiContext.wantsMouse() || isAnyUiVisible();

        // Scroll zoom (consumed here, before the UI overlay reads m_pendingScrollDelta).
        if (!overUi && m_pendingScrollDelta != 0.0f) {
            const float zoomFactor = std::pow(0.88f, m_pendingScrollDelta);  // wheel up zooms in
            if (m_strategyMap3D) {
                const float minDist = m_strategyMap.hexSize * 4.0f;
                const float maxDist = std::max(m_mapMaxX - m_mapMinX, m_mapMaxZ - m_mapMinZ) * 2.0f;
                m_map3DDistance = std::clamp(m_map3DDistance * zoomFactor, minDist, maxDist);
            } else {
                const float minHalf = m_strategyMap.hexSize * 2.0f;
                const float maxHalf = std::max((m_mapMaxZ - m_mapMinZ) * 0.6f, minHalf);
                m_mapOrthoHalfHeight = std::clamp(m_mapOrthoHalfHeight * zoomFactor, minHalf, maxHalf);
            }
            m_pendingScrollDelta = 0.0f;
        }

        if (m_strategyMap3D) {
            // Grab-point pan: keep the world point under the cursor pinned.
            const bool mapDragging3D = m_input.removeBlockDown && !overUi && fbW > 0 && fbH > 0;
            if (mapDragging3D && !m_wasStrategyMapDragging) {
                m_mapFocusVelocityX = 0.0f;
                m_mapFocusVelocityZ = 0.0f;
            }
            if (mapDragging3D) {
                const double curX = m_lastMouseX, curY = m_lastMouseY;
                const double prevX = curX - static_cast<double>(mouseDeltaX);
                const double prevY = curY - static_cast<double>(mouseDeltaY);
                float pX = 0.0f, pZ = 0.0f, cX = 0.0f, cZ = 0.0f;
                if (rayToGroundPlane(prevX, prevY, fbW, fbH, pX, pZ) &&
                    rayToGroundPlane(curX, curY, fbW, fbH, cX, cZ)) {
                    const float rawDispX = pX - cX;
                    const float rawDispZ = pZ - cZ;
                    // Low-pass the per-frame displacement to even out mouse-report jitter.
                    // Seed with the raw value on the first drag frame for an immediate 1:1
                    // grab, then smooth thereafter.
                    const float smoothAlpha = m_wasStrategyMapDragging
                        ? (1.0f - std::exp(-dt / kStrategyPanSmoothingSeconds)) : 1.0f;
                    m_panDispSmoothX += (rawDispX - m_panDispSmoothX) * smoothAlpha;
                    m_panDispSmoothZ += (rawDispZ - m_panDispSmoothZ) * smoothAlpha;
                    m_map3DFocusX += m_panDispSmoothX;
                    m_map3DFocusZ += m_panDispSmoothZ;
                    if (dt > 1e-5f) {
                        const float velAlpha = std::min(1.0f, dt * 20.0f);
                        m_mapFocusVelocityX += velAlpha * (m_panDispSmoothX / dt - m_mapFocusVelocityX);
                        m_mapFocusVelocityZ += velAlpha * (m_panDispSmoothZ / dt - m_mapFocusVelocityZ);
                    }
                }
            } else {
                m_map3DFocusX += m_mapFocusVelocityX * dt;
                m_map3DFocusZ += m_mapFocusVelocityZ * dt;
                const float drag = std::exp(-4.0f * dt);
                m_mapFocusVelocityX *= drag;
                m_mapFocusVelocityZ *= drag;
            }
            m_wasStrategyMapDragging = mapDragging3D;
            // Clamp to map bounds; kill velocity on the hit axis so inertia stops cleanly.
            const float clampedFocusX = std::clamp(m_map3DFocusX, m_mapMinX, m_mapMaxX);
            if (clampedFocusX != m_map3DFocusX) { m_mapFocusVelocityX = 0.0f; }
            m_map3DFocusX = clampedFocusX;
            const float clampedFocusZ = std::clamp(m_map3DFocusZ, m_mapMinZ, m_mapMaxZ);
            if (clampedFocusZ != m_map3DFocusZ) { m_mapFocusVelocityZ = 0.0f; }
            m_map3DFocusZ = clampedFocusZ;

            // Derive the eye from focus + tilt + distance each frame.
            m_camera.yawDegrees = kMap3DYawDegrees;
            m_camera.pitchDegrees = m_map3DPitchDeg;
            m_camera.fovDegrees = kMap3DFovDegrees;
            const float yawR = odai::math::radians(kMap3DYawDegrees);
            const float pitchR = odai::math::radians(m_map3DPitchDeg);
            const float cp = std::cos(pitchR);
            const odai::math::Vector3 fwd{cp * std::cos(yawR), std::sin(pitchR), cp * std::sin(yawR)};
            m_camera.x = m_map3DFocusX - fwd.x * m_map3DDistance;
            m_camera.y = -fwd.y * m_map3DDistance;  // focus sits on the y=0 plane
            m_camera.z = m_map3DFocusZ - fwd.z * m_map3DDistance;
        } else {
            // 2D top-down ortho pan: left-drag translates the camera in world XZ.
            const bool mapDragging2D = m_input.removeBlockDown && !overUi && fbH > 0;
            if (mapDragging2D && !m_wasStrategyMapDragging) {
                m_mapFocusVelocityX = 0.0f;
                m_mapFocusVelocityZ = 0.0f;
            }
            if (mapDragging2D) {
                const float worldPerPixel = (2.0f * m_mapOrthoHalfHeight) / static_cast<float>(fbH);
                const float rawDispX = -mouseDeltaX * worldPerPixel;
                const float rawDispZ = -mouseDeltaY * worldPerPixel;
                // Low-pass the per-frame displacement to even out mouse-report jitter.
                // Seed with the raw value on the first drag frame for an immediate 1:1
                // grab, then smooth thereafter.
                const float smoothAlpha = m_wasStrategyMapDragging
                    ? (1.0f - std::exp(-dt / kStrategyPanSmoothingSeconds)) : 1.0f;
                m_panDispSmoothX += (rawDispX - m_panDispSmoothX) * smoothAlpha;
                m_panDispSmoothZ += (rawDispZ - m_panDispSmoothZ) * smoothAlpha;
                m_camera.x += m_panDispSmoothX;
                m_camera.z += m_panDispSmoothZ;
                if (dt > 1e-5f) {
                    const float velAlpha = std::min(1.0f, dt * 20.0f);
                    m_mapFocusVelocityX += velAlpha * (m_panDispSmoothX / dt - m_mapFocusVelocityX);
                    m_mapFocusVelocityZ += velAlpha * (m_panDispSmoothZ / dt - m_mapFocusVelocityZ);
                }
            } else {
                m_camera.x += m_mapFocusVelocityX * dt;
                m_camera.z += m_mapFocusVelocityZ * dt;
                const float drag = std::exp(-4.0f * dt);
                m_mapFocusVelocityX *= drag;
                m_mapFocusVelocityZ *= drag;
            }
            m_wasStrategyMapDragging = mapDragging2D;
            // Clamp to map edges; kill velocity on the hit axis so inertia stops at the boundary.
            if (fbW > 0 && fbH > 0) {
                const float halfH = m_mapOrthoHalfHeight;
                const float halfW = halfH * (static_cast<float>(fbW) / static_cast<float>(fbH));
                const float cxMin = m_mapMinX + halfW, cxMax = m_mapMaxX - halfW;
                const float czMin = m_mapMinZ + halfH, czMax = m_mapMaxZ - halfH;
                const float clampedX = (cxMin <= cxMax)
                    ? std::clamp(m_camera.x, cxMin, cxMax) : (m_mapMinX + m_mapMaxX) * 0.5f;
                const float clampedZ = (czMin <= czMax)
                    ? std::clamp(m_camera.z, czMin, czMax) : (m_mapMinZ + m_mapMaxZ) * 0.5f;
                if (clampedX != m_camera.x) { m_mapFocusVelocityX = 0.0f; }
                if (clampedZ != m_camera.z) { m_mapFocusVelocityZ = 0.0f; }
                m_camera.x = clampedX;
                m_camera.z = clampedZ;
            }
        }

        mouseDeltaX = 0.0f;
        mouseDeltaY = 0.0f;
    }

    const float mouseSmoothingAlpha = 1.0f - std::exp(-dt / kMouseSmoothingSeconds);
    m_camera.smoothedMouseDeltaX += (mouseDeltaX - m_camera.smoothedMouseDeltaX) * mouseSmoothingAlpha;
    m_camera.smoothedMouseDeltaY += (mouseDeltaY - m_camera.smoothedMouseDeltaY) * mouseSmoothingAlpha;

    m_camera.yawDegrees += m_camera.smoothedMouseDeltaX * kMouseSensitivity;
    m_camera.pitchDegrees += m_camera.smoothedMouseDeltaY * kMouseSensitivity;
    m_camera.yawDegrees += m_input.gamepadLookX * kGamepadLookDegreesPerSecond * dt;
    m_camera.pitchDegrees += m_input.gamepadLookY * kGamepadLookDegreesPerSecond * dt;
    m_camera.pitchDegrees = std::clamp(m_camera.pitchDegrees, kPitchMinDegrees, kPitchMaxDegrees);

    const float yawRadians = odai::math::radians(m_camera.yawDegrees);
    const odai::math::Vector3 forward{std::cos(yawRadians), 0.0f, std::sin(yawRadians)};
    const odai::math::Vector3 right{-forward.z, 0.0f, forward.x};
    odai::math::Vector3 moveDirection{};

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

    // In 3D relief the eye is derived from the focus point every frame, so free-fly
    // WASD would be overwritten (and jitter); skip it. 2D keeps WASD as an alt-pan.
    if (m_importedSceneDemoEnabled && !(m_strategyMapMode && m_strategyMap3D)) {
        moveDirection += forward * moveForwardInput;
        moveDirection += right * moveRightInput;
        if (m_input.moveUp) {
            moveDirection.y += 1.0f;
        }
        if (m_input.moveDown) {
            moveDirection.y -= 1.0f;
        }

        const float moveLengthSq = odai::math::lengthSquared(moveDirection);
        if (moveLengthSq > 0.0f) {
            moveDirection /= std::sqrt(moveLengthSq);
        }

        float moveSpeed = kImportedSceneMoveSpeed;
        if (m_input.sprintDown) {
            moveSpeed *= kImportedSceneSprintSpeedMultiplier;
        }
        if (std::fabs(moveDirection.y) > 0.0f) {
            moveDirection.y *= (kImportedSceneVerticalMoveSpeed / kImportedSceneMoveSpeed);
        }

        m_camera.velocityX = moveDirection.x * moveSpeed;
        m_camera.velocityY = moveDirection.y * moveSpeed;
        m_camera.velocityZ = moveDirection.z * moveSpeed;
        m_camera.x += m_camera.velocityX * dt;
        m_camera.y += m_camera.velocityY * dt;
        m_camera.z += m_camera.velocityZ * dt;
        m_camera.onGround = false;
        return;
    }

    moveDirection += forward * moveForwardInput;
    moveDirection += right * moveRightInput;

    const float moveLengthSq = odai::math::lengthSquared(moveDirection);
    const float moveLength = std::sqrt(moveLengthSq);
    float targetVelocityX = 0.0f;
    float targetVelocityZ = 0.0f;
    if (moveLength > 0.0f) {
        moveDirection /= moveLength;
        float moveSpeed = kMoveMaxSpeed;
        if (m_input.sprintDown && !m_hoverEnabled && moveForwardInput > 0.0f) {
            moveSpeed *= kSprintSpeedMultiplier;
        }
        if (m_input.sneakDown && !m_hoverEnabled) {
            moveSpeed *= kSneakSpeedMultiplier;
        }
        const odai::math::Vector3 targetVelocity = moveDirection * moveSpeed;
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

    const odai::world::Chunk* chunk = nullptr;
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
    const std::vector<odai::world::Chunk>& chunks = m_world.chunkGrid().chunks();
    for (std::size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        const odai::world::Chunk& chunk = chunks[chunkIndex];
        const int chunkMinX = chunk.chunkX() * odai::world::Chunk::kSizeX;
        const int chunkMinY = chunk.chunkY() * odai::world::Chunk::kSizeY;
        const int chunkMinZ = chunk.chunkZ() * odai::world::Chunk::kSizeZ;
        const int localX = worldX - chunkMinX;
        const int localY = worldY - chunkMinY;
        const int localZ = worldZ - chunkMinZ;
        const bool insideChunk =
            localX >= 0 && localX < odai::world::Chunk::kSizeX &&
            localY >= 0 && localY < odai::world::Chunk::kSizeY &&
            localZ >= 0 && localZ < odai::world::Chunk::kSizeZ;
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
    const odai::world::Chunk*& outChunk,
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
    for (const odai::sim::Belt& belt : m_simulation.belts()) {
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
            for (const odai::sim::Belt& belt : m_simulation.belts()) {
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
            for (const odai::sim::Belt& belt : m_simulation.belts()) {
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
            for (const odai::sim::Belt& belt : m_simulation.belts()) {
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
            for (const odai::sim::Belt& belt : m_simulation.belts()) {
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
            for (const odai::sim::Belt& belt : m_simulation.belts()) {
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
            for (const odai::sim::Belt& belt : m_simulation.belts()) {
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

    const float yawRadians = odai::math::radians(m_camera.yawDegrees);
    const float pitchRadians = odai::math::radians(m_camera.pitchDegrees);
    const float cosPitch = std::cos(pitchRadians);
    const odai::math::Vector3 rayDirection = odai::math::normalize(odai::math::Vector3{
        std::cos(yawRadians) * cosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * cosPitch
    });
    if (odai::math::lengthSquared(rayDirection) <= 0.0f) {
        return result;
    }

    // Nudge origin slightly forward so close-surface targeting does not start inside solids.
    const odai::math::Vector3 rayOrigin =
        odai::math::Vector3{m_camera.x, m_camera.y, m_camera.z} + (rayDirection * 0.02f);
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

    const float yawRadians = odai::math::radians(m_camera.yawDegrees);
    const float pitchRadians = odai::math::radians(m_camera.pitchDegrees);
    const float cosPitch = std::cos(pitchRadians);
    const odai::math::Vector3 rayDirection = odai::math::normalize(odai::math::Vector3{
        std::cos(yawRadians) * cosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * cosPitch
    });
    if (odai::math::lengthSquared(rayDirection) <= 0.0f) {
        return result;
    }

    const odai::math::Vector3 rayOrigin =
        odai::math::Vector3{m_camera.x, m_camera.y, m_camera.z} + (rayDirection * 0.02f);
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

App::ImportedSceneInspectHit App::raycastImportedSceneFromCamera() const {
    ImportedSceneInspectHit result{};
    if (!m_importedSceneDemoEnabled ||
        m_importedScene.packedVertices.empty() ||
        m_importedScene.packedIndices.empty() ||
        m_importedScene.packedDraws.empty()) {
        return result;
    }

    const float yawRadians = odai::math::radians(m_camera.yawDegrees);
    const float pitchRadians = odai::math::radians(m_camera.pitchDegrees);
    const float cosPitch = std::cos(pitchRadians);
    const odai::math::Vector3 rayDirection = odai::math::normalize(odai::math::Vector3{
        std::cos(yawRadians) * cosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * cosPitch
    });
    if (odai::math::lengthSquared(rayDirection) <= 0.0f) {
        return result;
    }
    const odai::math::Vector3 rayOrigin =
        odai::math::Vector3{m_camera.x, m_camera.y, m_camera.z} + (rayDirection * 0.02f);

    const auto packedPosition = [](const odai::importer::ImportedScenePackedVertex& vertex) {
        return odai::math::Vector3{vertex.position[0], vertex.position[1], vertex.position[2]};
    };

    float bestDistance = kImportedInspectRayMaxDistance;
    constexpr float kRayEpsilon = 1e-4f;
    for (std::uint32_t drawIndex = 0; drawIndex < m_importedScene.packedDraws.size(); ++drawIndex) {
        const odai::importer::ImportedScenePackedDraw& draw = m_importedScene.packedDraws[drawIndex];
        const std::uint32_t lastIndex = std::min<std::uint32_t>(
            draw.firstIndex + draw.indexCount,
            static_cast<std::uint32_t>(m_importedScene.packedIndices.size()));
        for (std::uint32_t index = draw.firstIndex; index + 2u < lastIndex; index += 3u) {
            const std::uint32_t i0 = m_importedScene.packedIndices[index];
            const std::uint32_t i1 = m_importedScene.packedIndices[index + 1u];
            const std::uint32_t i2 = m_importedScene.packedIndices[index + 2u];
            if (i0 >= m_importedScene.packedVertices.size() ||
                i1 >= m_importedScene.packedVertices.size() ||
                i2 >= m_importedScene.packedVertices.size()) {
                continue;
            }

            const odai::math::Vector3 p0 = packedPosition(m_importedScene.packedVertices[i0]);
            const odai::math::Vector3 p1 = packedPosition(m_importedScene.packedVertices[i1]);
            const odai::math::Vector3 p2 = packedPosition(m_importedScene.packedVertices[i2]);
            const odai::math::Vector3 edge1 = p1 - p0;
            const odai::math::Vector3 edge2 = p2 - p0;
            const odai::math::Vector3 pvec = odai::math::cross(rayDirection, edge2);
            const float det = odai::math::dot(edge1, pvec);
            if (std::fabs(det) <= kRayEpsilon) {
                continue;
            }

            const float invDet = 1.0f / det;
            const odai::math::Vector3 tvec = rayOrigin - p0;
            const float u = odai::math::dot(tvec, pvec) * invDet;
            if (u < 0.0f || u > 1.0f) {
                continue;
            }
            const odai::math::Vector3 qvec = odai::math::cross(tvec, edge1);
            const float v = odai::math::dot(rayDirection, qvec) * invDet;
            if (v < 0.0f || (u + v) > 1.0f) {
                continue;
            }
            const float distance = odai::math::dot(edge2, qvec) * invDet;
            if (distance <= kRayEpsilon || distance >= bestDistance) {
                continue;
            }

            const odai::importer::ImportedScenePackedVertex& hitVertex = m_importedScene.packedVertices[i0];
            bestDistance = distance;
            result.hit = true;
            result.distance = distance;
            result.position = rayOrigin + (rayDirection * distance);
            result.drawIndex = drawIndex;
            result.triangleIndex = (index - draw.firstIndex) / 3u;
            result.textureIndex = hitVertex.textureIndex;
            result.flags = hitVertex.flags;
        }
    }

    return result;
}

void App::inspectImportedSceneFromCamera() const {
    const ImportedSceneInspectHit hit = raycastImportedSceneFromCamera();
    if (!hit.hit) {
        VOX_LOGI("app") << "imported inspect: no hit (I)";
        return;
    }

    std::uint32_t drawCursor = 0u;
    const bool importedSceneIsInterior = m_importedScene.sourceTag == "morrowind_interior";
    const std::uint32_t terrainDrawCount = importedSceneIsInterior
        ? 0u
        : std::min<std::uint32_t>(
            m_importedScene.sourceLandscapeCellCount,
            static_cast<std::uint32_t>(m_importedScene.packedDraws.size()));
    const odai::importer::ImportedSceneInstance* hitInstance = nullptr;
    if (hit.drawIndex < terrainDrawCount) {
        drawCursor = terrainDrawCount;
    } else {
        drawCursor = terrainDrawCount;
        for (const odai::importer::ImportedSceneInstance& instance : m_importedScene.instances) {
            if (instance.meshIndex >= m_importedScene.meshes.size()) {
                continue;
            }
            const odai::importer::ImportedSceneMesh& mesh = m_importedScene.meshes[instance.meshIndex];
            if (mesh.vertices.empty() || mesh.indices.empty()) {
                continue;
            }

            std::uint32_t emittedIndexCount = 0u;
            if (mesh.parts.empty()) {
                emittedIndexCount = static_cast<std::uint32_t>(mesh.indices.size());
            } else {
                for (const odai::importer::ImportedSceneMeshPart& part : mesh.parts) {
                    if (part.indexCount == 0u || part.firstIndex >= mesh.indices.size()) {
                        continue;
                    }
                    const std::uint32_t lastPartIndex = std::min<std::uint32_t>(
                        part.firstIndex + part.indexCount,
                        static_cast<std::uint32_t>(mesh.indices.size()));
                    emittedIndexCount += lastPartIndex - part.firstIndex;
                }
            }
            if (emittedIndexCount == 0u) {
                continue;
            }
            if (drawCursor == hit.drawIndex) {
                hitInstance = &instance;
                break;
            }
            ++drawCursor;
        }
    }

    const char* kind = (hit.drawIndex < terrainDrawCount) ? "terrain" : "static";
    const std::string texturePath =
        (hit.textureIndex < m_importedScene.textures.size())
            ? m_importedScene.textures[hit.textureIndex].sourcePath
            : std::string("<none>");
    VOX_LOGI("app") << "imported inspect (I): kind=" << kind
                    << ", draw=" << hit.drawIndex
                    << ", tri=" << hit.triangleIndex
                    << ", dist=" << hit.distance
                    << ", pos=(" << hit.position.x << "," << hit.position.y << "," << hit.position.z << ")"
                    << ", texture=" << texturePath
                    << ", alphaTest=" << (((hit.flags & kImportedSceneMaterialFlagAlphaTest) != 0u) ? "yes" : "no");
    if (hitInstance != nullptr) {
        VOX_LOGI("app") << "imported inspect source: refId=" << hitInstance->sourceId
                        << ", model=" << hitInstance->modelPath
                        << ", meshIndex=" << hitInstance->meshIndex;
    }
}

bool App::isWorldVoxelInBounds(int x, int y, int z) const {
    std::size_t chunkIndex = 0;
    int localX = 0;
    int localY = 0;
    int localZ = 0;
    return worldToChunkLocal(x, y, z, chunkIndex, localX, localY, localZ);
}

void App::cycleSelectedHotbar(int direction) {
    const int hotbarSlotCount = static_cast<int>(odai::render::kGameplayHotbarSlotCount);
    if (hotbarSlotCount <= 0) {
        m_gameplayUiState.selectedHotbarSlot = 0;
        return;
    }
    const int currentSlot = static_cast<int>(m_gameplayUiState.selectedHotbarSlot);
    const int next = (currentSlot + direction) % hotbarSlotCount;
    selectHotbarSlot(next < 0 ? next + hotbarSlotCount : next);
}

void App::selectHotbarSlot(int hotbarIndex) {
    const int clampedIndex = std::clamp(
        hotbarIndex,
        0,
        static_cast<int>(odai::render::kGameplayHotbarSlotCount) - 1
    );
    if (m_gameplayUiState.selectedHotbarSlot == static_cast<std::uint32_t>(clampedIndex)) {
        return;
    }
    m_gameplayUiState.selectedHotbarSlot = static_cast<std::uint32_t>(clampedIndex);
    syncGameplayUiState();
    const odai::render::InventoryItemId itemId =
        m_gameplayUiState.hotbarItems[static_cast<std::size_t>(clampedIndex)];
    VOX_LOGI("app") << "selected hotbar " << (clampedIndex + 1)
                    << ": " << inventoryItemLabel(itemId);
}

odai::world::Voxel App::selectedPlaceVoxel() const {
    const std::size_t hotbarIndex = std::min<std::size_t>(
        m_gameplayUiState.selectedHotbarSlot,
        m_gameplayUiState.hotbarItems.size() - 1
    );
    return itemToVoxel(m_gameplayUiState.hotbarItems[hotbarIndex]);
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

    const odai::core::Dir6 faceDir = faceNormalToDir6(
        raycast.hitFaceNormalX,
        raycast.hitFaceNormalY,
        raycast.hitFaceNormalZ
    );
    odai::core::Dir6 selectedAxis = faceDir;
    int extensionSign = 1;
    odai::core::Cell3i extensionAnchor{raycast.x, raycast.y, raycast.z};

    if (raycast.hitPipe) {
        std::size_t pipeIndex = 0;
        if (!isPipeAtWorld(raycast.x, raycast.y, raycast.z, &pipeIndex)) {
            return false;
        }

        const std::vector<odai::sim::Pipe>& pipes = m_simulation.pipes();
        if (pipeIndex >= pipes.size()) {
            return false;
        }

        selectedAxis = axisToDir6(pipes[pipeIndex].axis);
        const odai::core::Cell3i axisOffset = odai::core::dirToOffset(selectedAxis);
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
            const odai::core::Dir6 extensionDir =
                extensionSign >= 0 ? selectedAxis : odai::core::oppositeDir(selectedAxis);
            while (true) {
                const odai::core::Cell3i nextCell = odai::core::neighborCell(extensionAnchor, extensionDir);
                std::size_t nextPipeIndex = 0;
                if (!isPipeAtWorld(nextCell.x, nextCell.y, nextCell.z, &nextPipeIndex)) {
                    break;
                }
                if (nextPipeIndex >= pipes.size()) {
                    break;
                }
                const odai::core::Dir6 nextAxis = axisToDir6(pipes[nextPipeIndex].axis);
                if (!dirSharesAxis(nextAxis, selectedAxis)) {
                    break;
                }
                extensionAnchor = nextCell;
            }
        }
    }

    const odai::core::Dir6 extensionDir =
        extensionSign >= 0 ? selectedAxis : odai::core::oppositeDir(selectedAxis);
    const odai::core::Cell3i targetCell = odai::core::neighborCell(extensionAnchor, extensionDir);
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

    const std::uint8_t neighborMask = odai::sim::neighborMask6(targetCell, [this](const odai::core::Cell3i& cell) {
        return isPipeAtWorld(cell.x, cell.y, cell.z, nullptr);
    });
    const std::uint32_t neighborCount = odai::sim::connectionCount(neighborMask);
    const odai::sim::JoinPiece joinPiece = odai::sim::classifyJoinPiece(neighborMask);

    odai::core::Dir6 resolvedAxis = selectedAxis;
    if (neighborCount == 1u) {
        const odai::core::Dir6 neighborDir = firstDirFromMask(neighborMask);
        resolvedAxis = odai::core::oppositeDir(neighborDir);
    } else if (joinPiece == odai::sim::JoinPiece::Straight) {
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

    odai::core::Dir6 selectedAxis = faceNormalToDir6(
        raycast.hitFaceNormalX,
        raycast.hitFaceNormalY,
        raycast.hitFaceNormalZ
    );
    if (selectedAxis == odai::core::Dir6::PosY || selectedAxis == odai::core::Dir6::NegY) {
        selectedAxis = horizontalDirFromYaw(m_camera.yawDegrees);
    }
    int extensionSign = 1;
    odai::core::Cell3i extensionAnchor{raycast.x, raycast.y, raycast.z};

    if (raycast.hitBelt) {
        std::size_t beltIndex = 0;
        if (!isBeltAtWorld(raycast.x, raycast.y, raycast.z, &beltIndex)) {
            return false;
        }
        const std::vector<odai::sim::Belt>& belts = m_simulation.belts();
        if (beltIndex >= belts.size()) {
            return false;
        }

        selectedAxis = beltDirectionToDir6(belts[beltIndex].direction);
        const odai::core::Cell3i axisOffset = odai::core::dirToOffset(selectedAxis);
        const int faceNormalDotAxis =
            (raycast.hitFaceNormalX * axisOffset.x) +
            (raycast.hitFaceNormalY * axisOffset.y) +
            (raycast.hitFaceNormalZ * axisOffset.z);
        if (faceNormalDotAxis == 0) {
            odai::core::Dir6 faceDir = faceNormalToDir6(
                raycast.hitFaceNormalX,
                raycast.hitFaceNormalY,
                raycast.hitFaceNormalZ
            );
            if (faceDir == odai::core::Dir6::PosY || faceDir == odai::core::Dir6::NegY) {
                faceDir = horizontalDirFromYaw(m_camera.yawDegrees);
            }
            selectedAxis = faceDir;
            extensionSign = 1;
        } else {
            extensionSign = (faceNormalDotAxis > 0) ? 1 : -1;
        }
    }

    const odai::core::Dir6 extensionDir =
        extensionSign >= 0 ? selectedAxis : odai::core::oppositeDir(selectedAxis);
    const odai::core::Cell3i targetCell = odai::core::neighborCell(extensionAnchor, extensionDir);
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

    odai::core::Dir6 selectedAxis = faceNormalToDir6(
        raycast.hitFaceNormalX,
        raycast.hitFaceNormalY,
        raycast.hitFaceNormalZ
    );
    if (selectedAxis == odai::core::Dir6::PosY || selectedAxis == odai::core::Dir6::NegY) {
        selectedAxis = horizontalDirFromYaw(m_camera.yawDegrees);
    }
    int extensionSign = 1;
    odai::core::Cell3i extensionAnchor{raycast.x, raycast.y, raycast.z};

    if (raycast.hitTrack) {
        std::size_t trackIndex = 0;
        if (!isTrackAtWorld(raycast.x, raycast.y, raycast.z, &trackIndex)) {
            return false;
        }
        const std::vector<odai::sim::Track>& tracks = m_simulation.tracks();
        if (trackIndex >= tracks.size()) {
            return false;
        }

        selectedAxis = trackDirectionToDir6(tracks[trackIndex].direction);
        const odai::core::Cell3i axisOffset = odai::core::dirToOffset(selectedAxis);
        const int faceNormalDotAxis =
            (raycast.hitFaceNormalX * axisOffset.x) +
            (raycast.hitFaceNormalY * axisOffset.y) +
            (raycast.hitFaceNormalZ * axisOffset.z);
        if (faceNormalDotAxis == 0) {
            odai::core::Dir6 faceDir = faceNormalToDir6(
                raycast.hitFaceNormalX,
                raycast.hitFaceNormalY,
                raycast.hitFaceNormalZ
            );
            if (faceDir == odai::core::Dir6::PosY || faceDir == odai::core::Dir6::NegY) {
                faceDir = horizontalDirFromYaw(m_camera.yawDegrees);
            }
            selectedAxis = faceDir;
            extensionSign = 1;
        } else {
            extensionSign = (faceNormalDotAxis > 0) ? 1 : -1;
        }
    }

    const odai::core::Dir6 extensionDir =
        extensionSign >= 0 ? selectedAxis : odai::core::oppositeDir(selectedAxis);
    const odai::core::Cell3i targetCell = odai::core::neighborCell(extensionAnchor, extensionDir);
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
    odai::world::Voxel voxel,
    std::vector<std::size_t>& outDirtyChunkIndices
) {
    int localX = 0;
    int localY = 0;
    int localZ = 0;
    std::size_t editedChunkIndex = 0;
    if (!worldToChunkLocal(targetX, targetY, targetZ, editedChunkIndex, localX, localY, localZ)) {
        return false;
    }

    const odai::world::Voxel existingVoxel =
        m_world.chunkGrid().chunks()[editedChunkIndex].voxelAt(localX, localY, localZ);
    if (existingVoxel.type == voxel.type) {
        return false;
    }
    if (!m_world.setVoxelAtWorld(targetX, targetY, targetZ, voxel)) {
        return false;
    }

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
    if (localX == (odai::world::Chunk::kSizeX - 1)) {
        appendNeighborChunkForWorldVoxel(targetX + 1, targetY, targetZ);
    }
    if (localY == 0) {
        appendNeighborChunkForWorldVoxel(targetX, targetY - 1, targetZ);
    }
    if (localY == (odai::world::Chunk::kSizeY - 1)) {
        appendNeighborChunkForWorldVoxel(targetX, targetY + 1, targetZ);
    }
    if (localZ == 0) {
        appendNeighborChunkForWorldVoxel(targetX, targetY, targetZ - 1);
    }
    if (localZ == (odai::world::Chunk::kSizeZ - 1)) {
        appendNeighborChunkForWorldVoxel(targetX, targetY, targetZ + 1);
    }

    return true;
}

bool App::isPipeAtWorld(int worldX, int worldY, int worldZ, std::size_t* outPipeIndex) const {
    const std::vector<odai::sim::Pipe>& pipes = m_simulation.pipes();
    for (std::size_t pipeIndex = 0; pipeIndex < pipes.size(); ++pipeIndex) {
        const odai::sim::Pipe& pipe = pipes[pipeIndex];
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
    const std::vector<odai::sim::Belt>& belts = m_simulation.belts();
    for (std::size_t beltIndex = 0; beltIndex < belts.size(); ++beltIndex) {
        const odai::sim::Belt& belt = belts[beltIndex];
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
    const std::vector<odai::sim::Track>& tracks = m_simulation.tracks();
    for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex) {
        const odai::sim::Track& track = tracks[trackIndex];
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
    refreshStreamingWindow(true);

    const std::filesystem::path worldPath{kWorldFilePath};
    if (m_world.save(worldPath)) {
        VOX_LOGI("app") << "world regenerated and saved to " << worldPath.string() << " (R)";
        m_worldDirty = false;
        m_worldAutosaveElapsedSeconds = 0.0f;
    } else {
        VOX_LOGW("app") << "world regenerated, but failed to save " << worldPath.string();
    }
}

void App::refreshStreamingWindow(bool forceRendererUpload) {
    const odai::world::World::ChunkStreamingUpdate streamingUpdate =
        m_world.updateStreamingWindowForWorldPosition(m_camera.x, m_camera.z);
    const odai::world::World::ChunkStreamingStats& streamingStats = streamingUpdate.stats;
    if (!streamingStats.changed && !forceRendererUpload) {
        return;
    }

    const odai::world::ClipmapConfig requestedClipmapConfig = m_renderer.clipmapQueryConfig();
    const bool clipmapConfigChanged =
        !m_hasAppliedClipmapConfig ||
        requestedClipmapConfig.levelCount != m_appliedClipmapConfig.levelCount ||
        requestedClipmapConfig.gridResolution != m_appliedClipmapConfig.gridResolution ||
        requestedClipmapConfig.baseVoxelSize != m_appliedClipmapConfig.baseVoxelSize ||
        requestedClipmapConfig.brickResolution != m_appliedClipmapConfig.brickResolution;
    m_chunkClipmapIndex.setConfig(requestedClipmapConfig);
    m_appliedClipmapConfig = requestedClipmapConfig;
    m_hasAppliedClipmapConfig = true;
    if (clipmapConfigChanged || !m_chunkClipmapIndex.valid()) {
        m_chunkClipmapIndex.rebuild(m_world.chunkGrid());
    } else {
        m_chunkClipmapIndex.syncResidentChunks(m_world.chunkGrid());
    }
    m_visibleChunkIndices.clear();
    m_visibleChunkGraceFrames.assign(m_world.chunkGrid().chunkCount(), 0u);
    m_previousVisibleChunkMask.assign(m_world.chunkGrid().chunkCount(), 0u);
    m_currentVisibleChunkMask.assign(m_world.chunkGrid().chunkCount(), 0u);
    m_directlyVisibleChunkMask.assign(m_world.chunkGrid().chunkCount(), 0u);

    const bool uploadOk = streamingUpdate.requiresFullMeshUpload || forceRendererUpload
        ? m_renderer.updateChunkMesh(m_world.chunkGrid())
        : m_renderer.updateChunkMesh(
            m_world.chunkGrid(),
            std::span<const std::size_t>(streamingUpdate.residentChunkIndicesNeedingUpload)
        );
    if (!uploadOk) {
        VOX_LOGE("app") << "streaming update failed to refresh resident chunk meshes";
        return;
    }

    VOX_LOGI("app") << "chunk streaming update: center="
                    << streamingStats.centerChunkX << "," << streamingStats.centerChunkZ
                    << ", resident=" << streamingStats.residentChunkCount
                    << "/" << streamingStats.storedChunkCount
                    << ", generated=" << streamingUpdate.generatedChunkKeys.size()
                    << ", entered=" << streamingStats.enteredChunkCount
                    << ", exited=" << streamingStats.exitedChunkCount
                    << ", meshUpload=" << (streamingUpdate.requiresFullMeshUpload || forceRendererUpload ? "full" : "partial");
}

bool App::tryPlaceVoxelFromCameraRay(std::vector<std::size_t>& outDirtyChunkIndices) {
    if (m_world.chunkGrid().chunks().empty()) {
        return false;
    }
    const odai::world::Voxel placeVoxel = selectedPlaceVoxel();
    if (placeVoxel.type == odai::world::VoxelType::Empty) {
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

    return applyVoxelEdit(targetX, targetY, targetZ, placeVoxel, outDirtyChunkIndices);
}

bool App::tryRemoveVoxelFromCameraRay(std::vector<std::size_t>& outDirtyChunkIndices) {
    if (m_world.chunkGrid().chunks().empty()) {
        return false;
    }

    const CameraRaycastResult raycast = raycastFromCamera();
    if (!raycast.hitSolid || raycast.hitDistance > kBlockInteractMaxDistance) {
        return false;
    }

    const bool sameTarget =
        m_voxelBreakTargetValid &&
        m_voxelBreakTargetX == raycast.solidX &&
        m_voxelBreakTargetY == raycast.solidY &&
        m_voxelBreakTargetZ == raycast.solidZ;
    if (!sameTarget) {
        m_voxelBreakTargetValid = true;
        m_voxelBreakTargetX = raycast.solidX;
        m_voxelBreakTargetY = raycast.solidY;
        m_voxelBreakTargetZ = raycast.solidZ;
        m_voxelBreakClicks = 0;
    }

    ++m_voxelBreakClicks;
    if (m_voxelBreakClicks < kVoxelBreakClicksRequired) {
        return false;
    }

    resetVoxelBreakProgress();

    return applyVoxelEdit(
        raycast.solidX,
        raycast.solidY,
        raycast.solidZ,
        odai::world::Voxel{odai::world::VoxelType::Empty},
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

    const odai::math::Vector3 axis{
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

    std::vector<odai::sim::Pipe>& pipes = m_simulation.pipes();
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

    odai::core::Dir6 axisDir = faceNormalToDir6(axisX, axisY, axisZ);
    if (axisDir == odai::core::Dir6::PosY || axisDir == odai::core::Dir6::NegY) {
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

    std::vector<odai::sim::Belt>& belts = m_simulation.belts();
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

    odai::core::Dir6 axisDir = faceNormalToDir6(axisX, axisY, axisZ);
    if (axisDir == odai::core::Dir6::PosY || axisDir == odai::core::Dir6::NegY) {
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

    std::vector<odai::sim::Track>& tracks = m_simulation.tracks();
    if (trackIndex >= tracks.size()) {
        return false;
    }
    tracks.erase(tracks.begin() + static_cast<std::ptrdiff_t>(trackIndex));
    return true;
}

namespace {

// Procedurally generate a square RGBA window-frame texture for 9-slice rendering:
// a rounded-corner bronze border around a dark translucent interior. Authored in
// straight-alpha sRGB (the UI shader linearizes on output). The 9-slice keeps the
// rounded corners fixed while stretching the straight edges and centre.
std::vector<std::uint8_t> makeWindowFrameRgba(int size) {
    const float half = static_cast<float>(size) * 0.5f;
    const float radius = static_cast<float>(size) * 0.22f;   // corner radius (texels).
    const float borderW = static_cast<float>(size) * 0.05f;  // border line thickness (texels).
    const float border[3] = {0.74f, 0.60f, 0.34f};           // bronze.
    const float fill[3] = {0.045f, 0.065f, 0.10f};           // dark interior.
    const float fillA = 0.93f;
    const float borderA = 1.0f;

    const auto clamp01 = [](float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); };
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(size) * size * 4u, 0u);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            // Signed distance to a rounded box covering the whole texture.
            const float px = (static_cast<float>(x) + 0.5f) - half;
            const float py = (static_cast<float>(y) + 0.5f) - half;
            const float qx = std::abs(px) - (half - radius);
            const float qy = std::abs(py) - (half - radius);
            const float mx = std::max(qx, 0.0f);
            const float my = std::max(qy, 0.0f);
            const float d = std::sqrt(mx * mx + my * my) + std::min(std::max(qx, qy), 0.0f) - radius;

            const float coverage = clamp01(0.5f - d);                      // 1 inside, 0 outside (1px AA).
            const float borderAmt = clamp01((d + borderW + 0.5f) / borderW);  // 1 near edge, 0 deep inside.

            const float r = fill[0] + (border[0] - fill[0]) * borderAmt;
            const float g = fill[1] + (border[1] - fill[1]) * borderAmt;
            const float b = fill[2] + (border[2] - fill[2]) * borderAmt;
            const float a = (fillA + (borderA - fillA) * borderAmt) * coverage;

            const auto byte = [&](float v) { return static_cast<std::uint8_t>(clamp01(v) * 255.0f + 0.5f); };
            const std::size_t i = (static_cast<std::size_t>(y) * size + x) * 4u;
            pixels[i + 0] = byte(r);
            pixels[i + 1] = byte(g);
            pixels[i + 2] = byte(b);
            pixels[i + 3] = byte(a);
        }
    }
    return pixels;
}

const char* terrainName(odai::game::TerrainType t) {
    switch (t) {
        case odai::game::TerrainType::Ocean:     return "Ocean";
        case odai::game::TerrainType::Coast:     return "Coast";
        case odai::game::TerrainType::Grassland: return "Grassland";
        case odai::game::TerrainType::Plains:    return "Plains";
        case odai::game::TerrainType::Forest:    return "Forest";
        case odai::game::TerrainType::Jungle:    return "Jungle";
        case odai::game::TerrainType::Hills:     return "Hills";
        case odai::game::TerrainType::Mountains: return "Mountains";
        case odai::game::TerrainType::Desert:    return "Desert";
        case odai::game::TerrainType::Tundra:    return "Tundra";
        case odai::game::TerrainType::Snow:      return "Snow";
        default:                                 return "Unknown";
    }
}

}  // namespace

void App::setupDemoUi(float viewW, float viewH) {
    m_setupViewW = viewW;
    m_setupViewH = viewH;
    // DPI scale: framebuffer pixels per logical pixel. On a 125%-scaled display the
    // framebuffer is 1.25ï¿½ the GLFW window size, so all hardcoded pixel values must
    // be multiplied by this factor so the UI looks the same physical size on screen.
    // DPI scale: use glfwGetWindowContentScale (the OS-reported factor, e.g. 1.5× at
    // 150 % DPI) as the authoritative multiplier. Cross-check with the framebuffer-to-
    // window pixel ratio; take the larger so layout is correct on all driver configs.
    float xscale = 1.0f;
    glfwGetWindowContentScale(m_window, &xscale, nullptr);
    int windowW = 0;
    glfwGetWindowSize(m_window, &windowW, nullptr);
    const float pixelRatio = (windowW > 0) ? viewW / static_cast<float>(windowW) : 1.0f;
    const float s = std::max(pixelRatio, xscale);
    m_uiScale = s;
    VOX_LOGI("ui") << "HUD setup: framebuffer=" << viewW << "x" << viewH
                   << " window=" << windowW << " contentScale=" << xscale
                   << " pixelRatio=" << pixelRatio << " effective=" << s;

    // Base font size in logical pixels at 100 % DPI (multiplied by s at bake time).
    // 28 px gives comfortable readability; scales up cleanly on 4K/HiDPI displays.
    const float kBaseFontPx = 28.0f;

    // Load the regular face into the built-in atlas (kUiFontAtlas). Fall back to a
    // system font so the HUD still renders text if the bundled TTF is missing.
    const std::array<const char*, 3> kRegularCandidates = {
        "assets/fonts/EBGaramond-Regular.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    };
    for (const char* candidate : kRegularCandidates) {
        const std::filesystem::path resolved = resolveAssetPath(candidate);
        if (std::filesystem::exists(resolved) &&
            m_uiFont.loadFromFile(resolved.string(), kBaseFontPx * s)) {
            m_uiFontReady = m_uiFont.atlasWidth() > 0 &&
                            m_renderer.setUiFontAtlas(m_uiFont.atlasPixels().data(),
                                                      m_uiFont.atlasWidth(), m_uiFont.atlasHeight());
            VOX_LOGI("ui") << "UI regular font: " << resolved.string();
            break;
        }
    }
    m_uiFonts = {};
    if (m_uiFontReady) {
        m_uiFont.setTextureId(odai::ui::kUiFontAtlas);
        m_uiFonts.regular = &m_uiFont;

        // Bold and italic faces each get their own atlas texture so glyph quads
        // can bind the right coverage map per run.
        const auto loadVariant = [&](odai::ui::Font& font, const char* relPath) -> const odai::ui::Font* {
            const std::filesystem::path path = resolveAssetPath(relPath);
            if (!std::filesystem::exists(path) || !font.loadFromFile(path.string(), kBaseFontPx * s) ||
                font.atlasWidth() == 0) {
                return nullptr;
            }
            const odai::ui::UiTextureId id = m_renderer.registerUiFontAtlas(
                font.atlasPixels().data(), font.atlasWidth(), font.atlasHeight());
            if (id == odai::ui::kUiNoTexture) {
                return nullptr;
            }
            font.setTextureId(id);
            return &font;
        };
        m_uiFonts.bold = loadVariant(m_uiFontBold, "assets/fonts/EBGaramond-Bold.ttf");
        m_uiFonts.italic = loadVariant(m_uiFontItalic, "assets/fonts/EBGaramond-Italic.ttf");

        // Title font - same face as regular but baked ~40% larger for window headers.
        const float kTitleFontPx = kBaseFontPx * 1.4f;
        const std::array<const char*, 3> kTitleCandidates = {
            "assets/fonts/EBGaramond-Bold.ttf",
            "assets/fonts/EBGaramond-Regular.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        };
        for (const char* candidate : kTitleCandidates) {
            const std::filesystem::path tp = resolveAssetPath(candidate);
            if (std::filesystem::exists(tp) &&
                m_uiFontTitle.loadFromFile(tp.string(), kTitleFontPx * s) &&
                m_uiFontTitle.atlasWidth() > 0) {
                const odai::ui::UiTextureId tid = m_renderer.registerUiFontAtlas(
                    m_uiFontTitle.atlasPixels().data(), m_uiFontTitle.atlasWidth(), m_uiFontTitle.atlasHeight());
                if (tid != odai::ui::kUiNoTexture) {
                    m_uiFontTitle.setTextureId(tid);
                }
                break;
            }
        }

        VOX_LOGI("ui") << "fonts loaded: regular=yes bold=" << (m_uiFonts.bold ? "yes" : "no")
                       << " italic=" << (m_uiFonts.italic ? "yes" : "no")
                       << " title=" << (m_uiFontTitle.valid() ? "yes" : "no");
    }
    const odai::ui::FontSet& fonts = m_uiFonts;

    // Register the procedural 9-slice window-frame texture (kept for future use).
    constexpr int kFrameTexSize = 64;
    const std::vector<std::uint8_t> framePixels = makeWindowFrameRgba(kFrameTexSize);
    m_windowFrameTexture = m_renderer.registerUiTextureRgba8(framePixels.data(), kFrameTexSize, kFrameTexSize);

    // Load religion icon with mipmaps; registered globally for [icon=religion] in rich text.
    {
        const std::filesystem::path iconPath = resolveAssetPath("assets/icons/religion.png");
        int iw = 0, ih = 0;
        stbi_uc* ipx = stbi_load(iconPath.string().c_str(), &iw, &ih, nullptr, 4);
        if (ipx && iw > 0 && ih > 0) {
            const odai::ui::UiTextureId iconTex = m_renderer.registerUiTextureRgba8Mipmapped(
                ipx, static_cast<std::uint32_t>(iw), static_cast<std::uint32_t>(ih));
            stbi_image_free(ipx);
            if (iconTex != odai::ui::kUiNoTexture) {
                const std::string meta =
                    "{\"iconSize\":" + std::to_string(std::max(iw, ih)) +
                    ",\"icons\":{\"religion\":[0,0]}}";
                odai::ui::UiIconRegistry::global().registerAtlas(
                    iconTex, static_cast<std::uint32_t>(iw), static_cast<std::uint32_t>(ih), meta.c_str());
            }
        }
    }
    // Load top-toolbar resource icons. The toolbar sizes these from its own band height.
    {
        const std::filesystem::path iconPath = resolveAssetPath("assets/icons/toolbar_yields.png");
        int iw = 0, ih = 0;
        stbi_uc* ipx = stbi_load(iconPath.string().c_str(), &iw, &ih, nullptr, 4);
        if (ipx && iw > 0 && ih > 0) {
            const odai::ui::UiTextureId iconTex = m_renderer.registerUiTextureRgba8Mipmapped(
                ipx, static_cast<std::uint32_t>(iw), static_cast<std::uint32_t>(ih));
            stbi_image_free(ipx);
            if (iconTex != odai::ui::kUiNoTexture) {
                constexpr int kToolbarIconSize = 128;
                const std::string meta =
                    "{\"iconSize\":" + std::to_string(kToolbarIconSize) +
                    ",\"icons\":{\"science\":[0,0],\"culture\":[1,0],\"gold\":[2,0],"
                    "\"faith\":[0,1],\"food\":[1,1],\"production\":[2,1]}}";
                odai::ui::UiIconRegistry::global().registerAtlas(
                    iconTex, static_cast<std::uint32_t>(iw), static_cast<std::uint32_t>(ih), meta.c_str());
            }
        } else if (ipx != nullptr) {
            stbi_image_free(ipx);
        }
    }

    // Load civilization empire icons atlas (12 civs, 4ï¿½3, 128px).
    {
        const std::filesystem::path imgPath = resolveAssetPath("assets/icons/civ_empires.png");
        const std::filesystem::path jsonPath = resolveAssetPath("assets/icons/civ_empires.json");
        int iw = 0, ih = 0;
        stbi_uc* ipx = stbi_load(imgPath.string().c_str(), &iw, &ih, nullptr, 4);
        if (ipx && iw > 0 && ih > 0) {
            const odai::ui::UiTextureId tex = m_renderer.registerUiTextureRgba8Mipmapped(
                ipx, static_cast<std::uint32_t>(iw), static_cast<std::uint32_t>(ih));
            stbi_image_free(ipx);
            if (tex != odai::ui::kUiNoTexture) {
                std::ifstream f(jsonPath);
                std::string meta((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                odai::ui::UiIconRegistry::global().registerAtlas(
                    tex, static_cast<std::uint32_t>(iw), static_cast<std::uint32_t>(ih), meta.c_str());
            }
        } else if (ipx) {
            stbi_image_free(ipx);
        }
    }
    // Load unit icons atlas (9 units, 3ï¿½3 grid, 128px per icon).
    {
        const std::filesystem::path imgPath = resolveAssetPath("assets/icons/unit_icons.png");
        const std::filesystem::path jsonPath = resolveAssetPath("assets/icons/unit_icons.json");
        int iw = 0, ih = 0;
        stbi_uc* ipx = stbi_load(imgPath.string().c_str(), &iw, &ih, nullptr, 4);
        if (ipx && iw > 0 && ih > 0) {
            const odai::ui::UiTextureId tex = m_renderer.registerUiTextureRgba8Mipmapped(
                ipx, static_cast<std::uint32_t>(iw), static_cast<std::uint32_t>(ih));
            stbi_image_free(ipx);
            if (tex != odai::ui::kUiNoTexture) {
                std::ifstream f(jsonPath);
                std::string meta((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                odai::ui::UiIconRegistry::global().registerAtlas(
                    tex, static_cast<std::uint32_t>(iw), static_cast<std::uint32_t>(ih), meta.c_str());
            }
        } else if (ipx) {
            stbi_image_free(ipx);
        }
    }

    // Load unit action icons atlas (12 actions, 4ï¿½3 grid, 128px per icon).
    {
        const std::filesystem::path imgPath = resolveAssetPath("assets/icons/unit_action_icons.png");
        const std::filesystem::path jsonPath = resolveAssetPath("assets/icons/unit_action_icons.json");
        int iw = 0, ih = 0;
        stbi_uc* ipx = stbi_load(imgPath.string().c_str(), &iw, &ih, nullptr, 4);
        if (ipx && iw > 0 && ih > 0) {
            const odai::ui::UiTextureId tex = m_renderer.registerUiTextureRgba8Mipmapped(
                ipx, static_cast<std::uint32_t>(iw), static_cast<std::uint32_t>(ih));
            stbi_image_free(ipx);
            if (tex != odai::ui::kUiNoTexture) {
                std::ifstream f(jsonPath);
                std::string meta((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                odai::ui::UiIconRegistry::global().registerAtlas(
                    tex, static_cast<std::uint32_t>(iw), static_cast<std::uint32_t>(ih), meta.c_str());
            }
        } else if (ipx) {
            stbi_image_free(ipx);
        }
    }

    // Load civilization leader portraits atlas (12 leaders, 4ï¿½3, 256px).
    odai::ui::UiTextureId civLeaderTex = odai::ui::kUiNoTexture;
    {
        const std::filesystem::path imgPath = resolveAssetPath("assets/leaders/civ_leaders.png");
        const std::filesystem::path jsonPath = resolveAssetPath("assets/leaders/civ_leaders.json");
        int iw = 0, ih = 0;
        stbi_uc* ipx = stbi_load(imgPath.string().c_str(), &iw, &ih, nullptr, 4);
        if (ipx && iw > 0 && ih > 0) {
            civLeaderTex = m_renderer.registerUiTextureRgba8Mipmapped(
                ipx, static_cast<std::uint32_t>(iw), static_cast<std::uint32_t>(ih));
            stbi_image_free(ipx);
            if (civLeaderTex != odai::ui::kUiNoTexture) {
                std::ifstream f(jsonPath);
                std::string meta((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                odai::ui::UiIconRegistry::global().registerAtlas(
                    civLeaderTex, static_cast<std::uint32_t>(iw), static_cast<std::uint32_t>(ih), meta.c_str());
            }
        } else if (ipx) {
            stbi_image_free(ipx);
        }
    }

    // Load the UI sprite sheet and build 9-slice descriptors for the dark ornate frame
    // and the parchment frame. Both live in the same 1024ï¿½1536 texture atlas.
    odai::ui::UiNineSlice parchmentSlice{};
    bool parchmentSliceReady = false;
    {
        const std::filesystem::path sheetPath = resolveAssetPath("assets/ui/images/uisheet.png");
        int sw = 0, sh = 0;
        stbi_uc* spx = stbi_load(sheetPath.string().c_str(), &sw, &sh, nullptr, 4);
        if (spx && sw > 0 && sh > 0) {
            m_uiSheetTexture = m_renderer.registerUiTextureRgba8Mipmapped(
                spx, static_cast<std::uint32_t>(sw), static_cast<std::uint32_t>(sh));
            stbi_image_free(spx);
            if (m_uiSheetTexture != odai::ui::kUiNoTexture) {
                const float shW = static_cast<float>(sw);
                const float shH = static_cast<float>(sh);
                // Build a 9-slice from a sprite sub-rect of the sheet. Sprite bounds
                // and border insets are measured in absolute sheet pixels; the
                // uvBorder* values are fractions of the *sub-rect*, not the sheet.
                // Borders are sized to fully contain the ornate corner artwork so the
                // stretched middle edges only ever sample plain frame/interior.
                const auto makeSlice = [&](float x0, float y0, float x1, float y1,
                                           float bL, float bT, float bR, float bB) {
                    const float sprW = x1 - x0;
                    const float sprH = y1 - y0;
                    odai::ui::UiNineSlice ns{};
                    ns.textureId      = m_uiSheetTexture;
                    ns.uv             = {x0 / shW, y0 / shH, x1 / shW, y1 / shH};
                    ns.borderLeftPx   = bL * s;
                    ns.borderTopPx    = bT * s;
                    ns.borderRightPx  = bR * s;
                    ns.borderBottomPx = bB * s;
                    ns.uvBorderLeft   = bL / sprW;
                    ns.uvBorderTop    = bT / sprH;
                    ns.uvBorderRight  = bR / sprW;
                    ns.uvBorderBottom = bB / sprH;
                    return ns;
                };
                // Parchment panel: tight body bounds x=731..993, y=52..411. The body
                // starts at x=731 - a small decorative finial sits detached at
                // x=700..709 with a transparent gap, so it is excluded from the slice
                // (including it left a magenta/empty strip down the left frame).
                parchmentSlice = makeSlice(731.0f, 52.0f, 993.0f, 411.0f,
                                           46.0f, 52.0f, 42.0f, 40.0f);
                parchmentSliceReady = true;
            }
        } else if (spx) {
            stbi_image_free(spx);
        }
    }

    auto root = std::make_unique<odai::ui::Widget>();
    root->mousePassthrough = true;  // Container only; children claim the mouse, not the root.

    // Top resource toolbar - a full-width strip of icon+value badges (gold,
    // science, culture, faith, food, production) in the style of a 4X game's
    // status bar. Other top-anchored panels are pushed down by its height.
    const float kToolbarH = 40.0f * s;
    {
        using TB = odai::ui::Toolbar;
        auto tb = std::make_unique<TB>(fonts.regular);
        tb->setRect(odai::ui::UiRect::fromXYWH(0.0f, 0.0f, viewW, kToolbarH));
        tb->paddingXPx = 54.0f * s;  // extra left margin for the civ-panel toggle icon
        tb->itemGapPx  = 26.0f * s;
        tb->iconGapPx  = 8.0f * s;
        tb->iconScale  = 0.68f;
        const odai::ui::UiColor textCol{0.92f, 0.95f, 0.97f, 1.0f};
        m_tbScienceItem = tb->addItem(TB::IconKind::Science, {0.31f, 0.64f, 0.82f, 1.0f}, "+14", textCol);
        tb->addItem(TB::IconKind::Culture, {0.69f, 0.44f, 0.84f, 1.0f}, "+8", textCol);
        m_tbGoldItem = tb->addItem(TB::IconKind::Coin, {0.94f, 0.75f, 0.25f, 1.0f}, "240 (+39)", textCol);
        m_tbFaithItem = tb->addItem(TB::IconKind::Faith, {0.90f, 0.88f, 0.96f, 1.0f}, "+12", textCol);
        tb->addItem(TB::IconKind::Food, {0.49f, 0.78f, 0.31f, 1.0f}, "+6", textCol);
        tb->addItem(TB::IconKind::Production, {0.85f, 0.54f, 0.23f, 1.0f}, "+7", textCol);
        // Hover breakdown tooltips for each yield badge.
        tb->setTooltip(0,
            "<b><color=#5bb0d8>Science  +14 / turn</color></b>\n"
            "Capital city       <color=#5bb0d8>+8</color>\n"
            "Library            <color=#5bb0d8>+4</color>\n"
            "Academy tile       <color=#5bb0d8>+2</color>");
        tb->setTooltip(1,
            "<b><color=#b070e0>Culture  +8 / turn</color></b>\n"
            "Capital city       <color=#b070e0>+4</color>\n"
            "Monument           <color=#b070e0>+2</color>\n"
            "Amphitheatre       <color=#b070e0>+2</color>");
        tb->setTooltip(2,
            "<b><color=#f0c040>Gold  240 in treasury</color></b>\n"
            "Income / turn      <color=#f0c040>+39</color>\n"
            "  Capital city     <color=#f0c040>+20</color>\n"
            "  Trade routes     <color=#f0c040>+15</color>\n"
            "  Market           <color=#f0c040>+4</color>");
        tb->setTooltip(3,
            "<b><color=#d8d0f0>Faith  +12 / turn</color></b>\n"
            "Capital city       <color=#d8d0f0>+6</color>\n"
            "Shrine             <color=#d8d0f0>+3</color>\n"
            "Holy site tile     <color=#d8d0f0>+3</color>");
        tb->setTooltip(4,
            "<b><color=#7dca4f>Food  +6 / turn</color></b>\n"
            "Capital city       <color=#7dca4f>+4</color>\n"
            "Farms              <color=#7dca4f>+2</color>");
        tb->setTooltip(5,
            "<b><color=#da8a3a>Production  +7 / turn</color></b>\n"
            "Capital city       <color=#da8a3a>+4</color>\n"
            "Mine               <color=#da8a3a>+2</color>\n"
            "Forest harvest     <color=#da8a3a>+1</color>");
        m_toolbar = static_cast<TB*>(root->addChild(std::move(tb)));

        // Civilizations-panel toggle icon — always visible in the toolbar so the
        // panel can be re-opened after it is collapsed.
        {
            auto civBtn = std::make_unique<odai::ui::Button>(
                fonts.regular, "Civ",
                [this]() {
                    m_civExpanded = !m_civExpanded;
                    m_civExpandTween.setTarget(m_civExpanded ? 1.0f : 0.0f);
                });
            const float btnSz = kToolbarH - 6.0f * s;
            civBtn->setRect(odai::ui::UiRect::fromXYWH(4.0f * s, 3.0f * s, btnSz, btnSz));
            civBtn->colorNormal   = odai::ui::UiColor{0.14f, 0.20f, 0.28f, 0.85f};
            civBtn->colorHover    = odai::ui::UiColor{0.24f, 0.32f, 0.42f, 0.95f};
            civBtn->colorPressed  = odai::ui::UiColor{0.10f, 0.14f, 0.20f, 1.00f};
            civBtn->borderColor   = odai::ui::UiColor{0.75f, 0.62f, 0.34f, 0.55f};
            civBtn->labelColor    = odai::ui::UiColor{0.91f, 0.80f, 0.48f, 1.0f};
            civBtn->glowSizePx    = 8.0f * s;
            civBtn->cornerRadiusPx = 3.0f * s;
            m_toolbar->addChild(std::move(civBtn));
        }

        // Right-aligned turn readout on the toolbar.
        auto turn = std::make_unique<odai::ui::Label>(fonts, "");
        turn->align = odai::ui::UiTextAlign::Right;
        turn->setRect(odai::ui::UiRect::fromXYWH(viewW - 240.0f * s, (kToolbarH - 24.0f * s) * 0.5f,
                                                 224.0f * s, 24.0f * s));
        m_toolbarTurnLabel = static_cast<odai::ui::Label*>(m_toolbar->addChild(std::move(turn)));
    }

    // Bottom HUD bar - full-width strip at the bottom of the screen.
    const float kBarH = 64.0f * s;
    const float barY = viewH - kBarH;
    auto bar = std::make_unique<odai::ui::Panel>();
    bar->setRect(odai::ui::UiRect::fromXYWH(0.0f, barY, viewW, kBarH));
    bar->background  = odai::ui::UiColor{0.04f, 0.06f, 0.09f, 0.92f};
    bar->borderColor = odai::ui::UiColor{0.75f, 0.62f, 0.34f, 0.40f};

    auto statsLabel = std::make_unique<odai::ui::Label>(fonts, "");
    statsLabel->setRect(odai::ui::UiRect::fromXYWH(viewW * 0.5f - 160.0f * s, barY + 12.0f * s, 320.0f * s, 40.0f * s));
    statsLabel->align = odai::ui::UiTextAlign::Center;
    m_hudStatsLabel = statsLabel.get();
    bar->addChild(std::move(statsLabel));

    auto endTurnBtn = std::make_unique<odai::ui::Button>(fonts.regular, "End Turn", [this]() {
        if (idleCount() > 0) {
            cycleToNextIdle();
            return;
        }
        ++m_currentTurn;
        m_visitedUnitIds.clear();
        // Resolve supply/attrition/regen for every unit, then re-mesh so the unit
        // tokens reflect new health (wounded units redden, the dead disappear).
        odai::game::advanceTurn(m_gameState, m_strategyMap);
        rebuildStrategyMapScene();
        // Close any open production panel — the city's state may have changed.
        if (m_productionPanel != nullptr) m_productionPanel->visible = false;
        m_selectedCityCol = -1;
        m_selectedCityRow = -1;
        VOX_LOGI("ui") << "turn advanced to " << m_currentTurn;
    });
    // Apply styling before moving into the tree.
    endTurnBtn->cornerRadiusPx    = 12.0f * s;
    endTurnBtn->colorNormal       = odai::ui::UiColor{0.28f, 0.20f, 0.06f, 0.95f};
    endTurnBtn->colorHover        = odai::ui::UiColor{0.42f, 0.30f, 0.08f, 1.00f};
    endTurnBtn->colorPressed      = odai::ui::UiColor{0.20f, 0.14f, 0.04f, 1.00f};
    endTurnBtn->borderColor       = odai::ui::UiColor{0.88f, 0.70f, 0.28f, 0.90f};
    endTurnBtn->borderThicknessPx = 2.0f * s;
    endTurnBtn->glowColor         = odai::ui::UiColor{0.95f, 0.75f, 0.30f, 0.70f};
    endTurnBtn->glowSizePx        = 22.0f * s;
    m_endTurnBtn = static_cast<odai::ui::Button*>(bar->addChild(std::move(endTurnBtn)));
    root->addChild(std::move(bar));


    // Tile info window (bottom-right, hidden until a tile is hovered).
    const float tileWinW = 250.0f * s;
    const float tileWinH = 130.0f * s;
    const float tileWinX = viewW - tileWinW - 16.0f * s;
    const float tileWinY = viewH - kBarH - tileWinH - 12.0f * s;
    auto tileWin = std::make_unique<odai::ui::Window>(fonts.regular, "Tile Info");
    tileWin->setRect(odai::ui::UiRect::fromXYWH(tileWinX, tileWinY, tileWinW, tileWinH));
    if (m_uiFontTitle.valid()) {
        tileWin->titleFont = &m_uiFontTitle;
        tileWin->titleBarH = m_uiFontTitle.lineHeightPx() + 12.0f * s;
    } else {
        tileWin->titleBarH = odai::ui::Window::kDefaultTitleBarH * s;
    }
    tileWin->showCloseButton = false;
    tileWin->draggable = false;
    tileWin->visible = false;

    const float titleBarH = tileWin->titleBarH;
    auto tileInfoLabel = std::make_unique<odai::ui::Label>(fonts, "");
    tileInfoLabel->setRect(odai::ui::UiRect::fromXYWH(
        tileWinX + 10.0f * s,
        tileWinY + titleBarH + 8.0f * s,
        tileWinW - 20.0f * s,
        tileWinH - titleBarH - 16.0f * s));
    m_hudTileInfoLabel = tileInfoLabel.get();
    tileWin->addChild(std::move(tileInfoLabel));

    m_hudTileInfoWindow = root->addChild(std::move(tileWin));

    // Civilizations panel (left side): empire icon grid + selected leader portrait.
    {
        const float civPadX    = 12.0f * s;
        const float civIconSz  = 52.0f * s;
        const float civPortSz  = 160.0f * s;
        const float civW       = civPadX + 4.0f * civIconSz + civPadX;
        const float civX       = 0.0f;
        const float civY       = kToolbarH;
        const float civH       = 14.0f * s         // top padding
                               + 22.0f * s         // title label
                               + 8.0f  * s         // gap
                               + 3.0f * civIconSz  // 3 rows of icons
                               + 12.0f * s         // gap
                               + civPortSz          // leader portrait
                               + 6.0f  * s         // gap
                               + 18.0f * s         // name label
                               + 14.0f * s;        // bottom padding

        auto civPanel = std::make_unique<odai::ui::Panel>();
        civPanel->setRect(odai::ui::UiRect::fromXYWH(civX, civY, civW, civH));
        civPanel->cornerRadiusPx    = 0.0f;  // flush to screen edges
        civPanel->background        = odai::ui::UiColor{0.08f, 0.10f, 0.13f, 0.95f};
        civPanel->borderColor       = odai::ui::UiColor{0.75f, 0.62f, 0.34f, 0.55f};
        civPanel->borderThicknessPx = 1.5f * s;
        civPanel->showShadow        = true;
        civPanel->shadowBlurPx      = 5.0f * s;
        civPanel->shadowOffsetX     = 3.0f * s;
        civPanel->shadowOffsetY     = 0.0f;
        civPanel->shadowColor       = odai::ui::UiColor{0.0f, 0.0f, 0.0f, 0.25f};
        civPanel->clipContents      = true;

        float cy = civY + 14.0f * s;

        // "Civilizations" title label.
        auto civTitle = std::make_unique<odai::ui::Label>(fonts.bold, "Civilizations");
        civTitle->color = odai::ui::UiColor{0.91f, 0.80f, 0.48f, 1.0f};
        civTitle->setRect(odai::ui::UiRect::fromXYWH(civX, cy, civW, 22.0f * s));
        civTitle->align = odai::ui::UiTextAlign::Center;
        civPanel->addChild(std::move(civTitle));
        cy += 22.0f * s + 8.0f * s;

        // 4ï¿½3 grid of empire icons - 12 civilizations.
        static const std::array<const char*, 12> kCivIds = {{
            "babylon", "aztec",    "british", "egypt",
            "rome",    "greece",   "china",   "japan",
            "france",  "mongolia", "persia",  "india",
        }};
        for (std::size_t ci = 0; ci < kCivIds.size(); ++ci) {
            odai::ui::UiIconEntry entry{};
            if (odai::ui::UiIconRegistry::global().resolve(kCivIds[ci], entry)) {
                const float ix = civX + civPadX + static_cast<float>(ci % 4) * civIconSz;
                const float iy = cy  + static_cast<float>(ci / 4) * civIconSz;
                auto img = std::make_unique<odai::ui::Image>(entry.textureId);
                img->uvRect = entry.uv;
                img->setRect(odai::ui::UiRect::fromXYWH(ix, iy, civIconSz, civIconSz));
                civPanel->addChild(std::move(img));
            }
        }
        cy += 3.0f * civIconSz + 12.0f * s;

        // Leader portrait for the currently selected civilization (default: caesar/rome).
        const float portX = civX + (civW - civPortSz) * 0.5f;
        auto portrait = std::make_unique<odai::ui::Image>();
        {
            odai::ui::UiIconEntry pe{};
            if (odai::ui::UiIconRegistry::global().resolve("caesar", pe)) {
                portrait->textureId = pe.textureId;
                portrait->uvRect = pe.uv;
            }
        }
        portrait->setRect(odai::ui::UiRect::fromXYWH(portX, cy, civPortSz, civPortSz));
        m_civPortraitImage = portrait.get();
        civPanel->addChild(std::move(portrait));
        cy += civPortSz + 6.0f * s;

        // Leader name label.
        auto nameLabel = std::make_unique<odai::ui::Label>(fonts.regular, "Caesar â€” Rome");
        nameLabel->color = odai::ui::UiColor{0.90f, 0.85f, 0.72f, 1.0f};
        nameLabel->setRect(odai::ui::UiRect::fromXYWH(civX, cy, civW, 18.0f * s));
        nameLabel->align = odai::ui::UiTextAlign::Center;
        m_civLeaderNameLabel = nameLabel.get();
        civPanel->addChild(std::move(nameLabel));

        // Collapse/expand chevron button in the header row (right-aligned).
        {
            auto colBtn = std::make_unique<odai::ui::Button>(
                fonts.regular, "^",
                [this]() {
                    m_civExpanded = !m_civExpanded;
                    m_civExpandTween.setTarget(m_civExpanded ? 1.0f : 0.0f);
                });
            const float btnW = 22.0f * s;
            const float btnH = 22.0f * s;
            colBtn->setRect(odai::ui::UiRect::fromXYWH(
                civX + civW - btnW - 5.0f * s,
                civY + (14.0f - 1.0f) * s,
                btnW, btnH));
            colBtn->colorNormal   = odai::ui::UiColor{0.0f, 0.0f, 0.0f, 0.0f};
            colBtn->colorHover    = odai::ui::UiColor{1.0f, 1.0f, 1.0f, 0.08f};
            colBtn->colorPressed  = odai::ui::UiColor{0.0f, 0.0f, 0.0f, 0.18f};
            colBtn->borderColor   = odai::ui::UiColor{0.0f, 0.0f, 0.0f, 0.0f};
            colBtn->labelColor    = odai::ui::UiColor{0.91f, 0.80f, 0.48f, 0.80f};
            colBtn->glowSizePx    = 0.0f;
            colBtn->cornerRadiusPx = 3.0f * s;
            m_civCollapseBtn = static_cast<odai::ui::Button*>(civPanel->addChild(std::move(colBtn)));
        }

        // Store dimensions for accordion animation.
        m_civFullH   = civH;
        m_civHeaderH = (14.0f + 22.0f + 8.0f) * s;  // top pad + title label + gap

        // Initialize tween in the fully-expanded state.
        m_civExpandTween.durationSec = 0.25f;
        m_civExpandTween.easing      = odai::ui::Easing::EaseInOut;
        m_civExpandTween.value       = 1.0f;
        m_civExpandTween.setTarget(1.0f);

        m_civPanel = static_cast<odai::ui::Panel*>(root->addChild(std::move(civPanel)));
    }

    // City production panel (right column, Civ 6 style).
    // Buildings first, then Units, each in a labelled section.
    // Clicking a row selects it as the city's active build.
    // Clicking the info (i) button opens CivPedia for that specific item.
    {
        const float prodW   = 320.0f * s;
        const float prodX   = viewW - prodW - 10.0f * s;
        const float prodTop = kToolbarH + 8.0f * s;
        const float prodH   = barY - prodTop - 8.0f * s;

        // City production values (mock city: 8 accumulated, +3/turn).
        const int kAccumulated = 8;
        const int kPerTurn     = 3;

        if (m_cityProductionId.empty()) {
            m_cityProductionId = odai::game::defaultBuildables().front().id;
        }

        auto makeRow = [&](const odai::game::BuildableItem& item,
                           const std::string& section) {
            odai::ui::ProductionPanel::Row row;
            row.id             = item.id;
            row.name           = item.name;
            row.iconName       = item.iconName;
            row.productionCost = item.productionCost;
            row.turns          = odai::game::turnsToBuild(item.productionCost, kAccumulated, kPerTurn);
            row.selected       = (item.id == m_cityProductionId);
            row.section        = section;

            const std::string itemId = item.id;
            row.onSelect = [this, itemId]() {
                m_cityProductionId = itemId;
                if (m_productionPanel != nullptr) m_productionPanel->setSelected(itemId);
                VOX_LOGI("ui") << "city production set to " << itemId;
            };

            // Capture item fields by value so the lambda owns them.
            const std::string pName    = item.name;
            const std::string pIcon    = item.iconName;
            const bool        pIsUnit  = (item.kind == odai::game::BuildableKind::Unit);
            const std::string pArticle = odai::game::getPediaArticle(item.id);
            row.onOpenPedia = [this, pName, pIcon, pIsUnit, pArticle]() {
                if (m_civpediaWindow != nullptr) m_civpediaWindow->visible = true;

                // Update portrait icon.
                if (m_civpediaPortrait != nullptr) {
                    odai::ui::UiIconEntry pe{};
                    if (odai::ui::UiIconRegistry::global().resolve(pIcon, pe)) {
                        m_civpediaPortrait->textureId = pe.textureId;
                        m_civpediaPortrait->uvRect    = pe.uv;
                    }
                }

                // Update name + class label.
                const std::string kindStr = pIsUnit ? "Ancient Era Unit" : "City Building";
                if (m_civpediaNameLabel != nullptr) {
                    m_civpediaNameLabel->setText(
                        "<b><color=#c06820>" + pName + "</color></b>\n"
                        "<color=#9fa8a8><i>" + kindStr + "</i></color>");
                }

                // Update body text with the item's article.
                if (m_civpediaView != nullptr) {
                    m_civpediaView->setText(pArticle);
                    m_civpediaView->scrollOffsetY = 0.0f;
                }
            };
            return row;
        };

        // Buildings first ("Districts & Buildings"), then Units.
        std::vector<odai::ui::ProductionPanel::Row> rows;
        for (const odai::game::BuildableItem& item : odai::game::defaultBuildables()) {
            if (item.kind == odai::game::BuildableKind::Building)
                rows.push_back(makeRow(item, "Districts & Buildings"));
        }
        for (const odai::game::BuildableItem& item : odai::game::defaultBuildables()) {
            if (item.kind == odai::game::BuildableKind::Unit)
                rows.push_back(makeRow(item, "Units"));
        }

        odai::ui::ProductionPanel::CityInfo cityInfo;
        cityInfo.name         = "Rome";
        cityInfo.food         = 4;
        cityInfo.production   = 3;
        cityInfo.gold         = 7;
        cityInfo.science      = 3;
        cityInfo.faith        = 2;
        cityInfo.culture      = 4;
        cityInfo.governorName = "Magnus";

        auto prodPanel = std::make_unique<odai::ui::ProductionPanel>(fonts);
        prodPanel->setItems(odai::ui::UiRect::fromXYWH(prodX, prodTop, prodW, prodH),
                            s, "Choose Production", rows, cityInfo);
        m_productionPanel = static_cast<odai::ui::ProductionPanel*>(root->addChild(std::move(prodPanel)));
        m_productionPanel->visible = false;  // hidden until the player clicks a city
    }

    // CivPedia window (top-right): a draggable framed window. When the parchment
    // 9-slice is loaded, it replaces the solid fill and the title bar / body colors
    // are made transparent so the parchment texture shows through. Text colors shift
    // to dark-brown so they read on the light background.
    const float pediaW = parchmentSliceReady ? 520.0f * s : 440.0f * s;
    const float pediaH = parchmentSliceReady ? 500.0f * s : 500.0f * s;
    const float pediaX = (viewW - pediaW) * 0.5f;
    const float pediaY = (viewH - pediaH) * 0.5f;
    // Title bar height - tall enough for the title font.  The parchment frame's top
    // border is ornate art (86px), so that case is fixed; otherwise derive from font.
    const odai::ui::Font* pediaTitleFontPtr = m_uiFontTitle.valid() ? &m_uiFontTitle : fonts.bold;
    const float kTitlePad = 12.0f * s;
    const float pediaTitleH = parchmentSliceReady
        ? 86.0f * s
        : (pediaTitleFontPtr ? pediaTitleFontPtr->lineHeightPx() + kTitlePad * 2.0f
                             : odai::ui::Window::kDefaultTitleBarH * s);
    auto pediaWin = std::make_unique<odai::ui::Window>(
        fonts.bold, "CivPedia",
        [this]() {
            if (m_civpediaWindow != nullptr) {
                m_civpediaWindow->visible = false;
            }
        });
    pediaWin->setRect(odai::ui::UiRect::fromXYWH(pediaX, pediaY, pediaW, pediaH));
    pediaWin->titleBarH = pediaTitleH;
    if (m_uiFontTitle.valid()) pediaWin->titleFont = &m_uiFontTitle;
    pediaWin->margin = 0.0f;
    pediaWin->showCloseButton = true;
    pediaWin->draggable = true;
    if (parchmentSliceReady) {
        pediaWin->frame        = parchmentSlice;
        pediaWin->padding      = {56.0f * s, 8.0f * s};
        pediaWin->titleBarColor = odai::ui::UiColor{0.0f, 0.0f, 0.0f, 0.0f};
        pediaWin->bodyColor     = odai::ui::UiColor{0.0f, 0.0f, 0.0f, 0.0f};
        pediaWin->borderColor   = odai::ui::UiColor{0.0f, 0.0f, 0.0f, 0.0f};
        pediaWin->titleColor    = odai::ui::UiColor{0.25f, 0.14f, 0.05f, 1.0f};
        pediaWin->closeColor    = odai::ui::UiColor{0.55f, 0.18f, 0.08f, 0.90f};
        pediaWin->closeHoverColor = odai::ui::UiColor{0.80f, 0.10f, 0.05f, 1.0f};
    } else {
        // Solid fill path - use corner radius, no 9-slice frame (avoids heavy rounding).
        pediaWin->padding       = {14.0f * s, 10.0f * s};
        pediaWin->titleBarColor = odai::ui::UiColor{0.22f, 0.26f, 0.32f, 1.0f};
        pediaWin->bodyColor     = odai::ui::UiColor{0.07f, 0.10f, 0.14f, 0.96f};
        pediaWin->borderColor   = odai::ui::UiColor{0.65f, 0.52f, 0.28f, 0.70f};
        pediaWin->cornerRadiusPx = 4.0f * s;
        pediaWin->opacity       = 0.95f;
    }

    const float pBodyPadX   = parchmentSliceReady ? 56.0f * s : 18.0f * s;
    const float pBodyPadY   = parchmentSliceReady ? 8.0f * s  : 12.0f * s;
    const float pBodyBotPad = parchmentSliceReady ? 48.0f * s : 24.0f * s;

    // Unit portrait header: 54-px icon on the left, unit name + class on the right.
    const float headerIconSz = 54.0f * s;
    const float headerGap    = 8.0f * s;
    const float headerH      = headerIconSz + headerGap;
    const float headerY      = pediaY + pediaTitleH + pBodyPadY;
    {
        auto portrait = std::make_unique<odai::ui::Image>();
        odai::ui::UiIconEntry pe{};
        if (odai::ui::UiIconRegistry::global().resolve("spearman", pe)) {
            portrait->textureId = pe.textureId;
            portrait->uvRect    = pe.uv;
        }
        portrait->setRect(odai::ui::UiRect::fromXYWH(
            pediaX + pBodyPadX, headerY, headerIconSz, headerIconSz));
        m_civpediaPortrait = static_cast<odai::ui::Image*>(pediaWin->addChild(std::move(portrait)));

        const float nameLabelX = pediaX + pBodyPadX + headerIconSz + 8.0f * s;
        const float nameLabelW = pediaW - pBodyPadX - headerIconSz - 8.0f * s - pBodyPadX;
        auto nameLabel = std::make_unique<odai::ui::Label>(
            fonts, "<b><color=#c06820>Spearman</color></b>\n<color=#9fa8a8><i>Ancient Melee Unit</i></color>");
        nameLabel->setRect(odai::ui::UiRect::fromXYWH(nameLabelX, headerY, nameLabelW, headerIconSz));
        m_civpediaNameLabel = static_cast<odai::ui::Label*>(pediaWin->addChild(std::move(nameLabel)));
    }

    // Rich-text body -- populated from the article registry; shown when the user
    // clicks (i) on a production row. Starts with the Spearman article so the
    // window has real content if it is ever made visible without a selection.
    auto pediaBody = std::make_unique<odai::ui::RichTextView>(
        fonts, odai::game::getPediaArticle("spearman"));
    pediaBody->padding = {6.0f * s, 4.0f * s};
    pediaBody->scrollBarThumbColor = odai::ui::UiColor{0.70f, 0.56f, 0.28f, 0.65f};
    pediaBody->scrollBarTrackColor = odai::ui::UiColor{0.0f, 0.0f, 0.0f, 0.15f};
    pediaBody->setRect(odai::ui::UiRect::fromXYWH(
        pediaX + pBodyPadX,
        headerY + headerH,
        pediaW - pBodyPadX * 2.0f,
        pediaH - pediaTitleH - pBodyPadY - headerH - pBodyBotPad));
    m_civpediaView = pediaBody.get();

    // Wire hyperlink navigation: <tip=link:ID>text</tip> spans in articles.
    m_civpediaView->onLinkClick = [this](std::string id) {
        const std::string& article = odai::game::getPediaArticle(id);
        if (article.empty()) return;
        for (const auto& item : odai::game::defaultBuildables()) {
            if (item.id != id) continue;
            if (m_civpediaNameLabel) m_civpediaNameLabel->setText("<b>" + item.name + "</b>");
            if (m_civpediaPortrait) {
                odai::ui::UiIconEntry pe{};
                if (odai::ui::UiIconRegistry::global().resolve(item.iconName, pe)) {
                    m_civpediaPortrait->textureId = pe.textureId;
                    m_civpediaPortrait->uvRect    = pe.uv;
                }
            }
            break;
        }
        m_civpediaView->setText(article);
        m_civpediaView->scrollOffsetY = 0.0f;
    };

    pediaWin->addChild(std::move(pediaBody));
    m_civpediaWindow = root->addChild(std::move(pediaWin));
    m_civpediaWindow->visible = false;  // Hidden until user clicks the info button on a row.

    // Unit action bar - floats above the bottom bar at the left, shown when a unit
    // is selected. Layout: [unit portrait] [unit name] | [action btn 0..5]
    // Slots are pre-built and hidden; updateUiOverlay sets their icons and visibility.
    {
        // Actions available per unit type - ordered for display left-to-right.
        // Defined as static data; lookup in updateUiOverlay by m_selectedUnitType.

        const float portSz   = 48.0f * s;
        const float btnSz    = 48.0f * s;
        const float btnGap   = 6.0f  * s;
        const float padX     = 10.0f * s;
        const float padY     = 8.0f  * s;
        const float nameW    = 96.0f * s;
        const float sepW     = 10.0f * s;
        const float panelH   = portSz + padY * 2.0f;
        const float panelW   = padX + portSz + padX + nameW + sepW
                             + kMaxActionSlots * (btnSz + btnGap) + padX;
        const float panelX   = 0.0f;
        const float panelY   = barY - panelH - 6.0f * s;

        auto ap = std::make_unique<odai::ui::Panel>();
        ap->setRect(odai::ui::UiRect::fromXYWH(panelX, panelY, panelW, panelH));
        ap->background        = odai::ui::UiColor{0.06f, 0.07f, 0.10f, 0.94f};
        ap->borderColor       = odai::ui::UiColor{0.75f, 0.62f, 0.34f, 0.55f};
        ap->borderThicknessPx = 1.5f * s;
        ap->visible           = false;  // hidden until a unit is selected

        // Unit portrait image.
        const float portX = panelX + padX;
        const float portY = panelY + padY;
        auto portrait = std::make_unique<odai::ui::Image>();
        portrait->setRect(odai::ui::UiRect::fromXYWH(portX, portY, portSz, portSz));
        m_unitActionPortrait = portrait.get();
        ap->addChild(std::move(portrait));

        // Unit name label.
        const float nameX = portX + portSz + padX * 0.5f;
        const float nameY = panelY + (panelH - 16.0f * s) * 0.5f;
        auto nameLabel = std::make_unique<odai::ui::Label>(fonts.bold, "");
        nameLabel->color = odai::ui::UiColor{0.91f, 0.80f, 0.48f, 1.0f};
        nameLabel->setRect(odai::ui::UiRect::fromXYWH(nameX, nameY, nameW, 20.0f * s));
        m_unitActionName = nameLabel.get();
        ap->addChild(std::move(nameLabel));

        // Pre-built action slot buttons (icon-only, hidden until assigned).
        const float slotsX = nameX + nameW + sepW;
        for (int si = 0; si < kMaxActionSlots; ++si) {
            const float bx = slotsX + static_cast<float>(si) * (btnSz + btnGap);
            const float by = panelY + padY;
            auto btn = std::make_unique<odai::ui::IconButton>([]() {});
            btn->setRect(odai::ui::UiRect::fromXYWH(bx, by, btnSz, btnSz));
            btn->cornerRadiusPx = 8.0f * s;
            btn->glowSizePx     = 12.0f * s;
            btn->iconPaddingPx  = 5.0f * s;
            btn->visible        = false;
            m_actionSlotBtns[si] = btn.get();
            ap->addChild(std::move(btn));
        }

        m_unitActionPanel = root->addChild(std::move(ap));
    }

    // Tooltip fade: ease-in-out over ~0.16s.
    m_tooltipFade.durationSec = 0.16f;
    m_tooltipFade.easing = odai::ui::Easing::EaseInOut;

    m_uiContext.setRoot(std::move(root));
}

void App::drawStrategyMapLabels(float fbW, float fbH) {
    if (!m_strategyMapMode || !m_uiFontReady) return;
    if (m_strategyMap.settlements.empty() && m_gameState.units.empty()) return;

    const auto& cam = m_renderCameraPose;
    const float aspectRatio = fbW / fbH;
    const float nearPlane = 1.0f;
    const float farPlane = 50000.0f;

    // Build VP matrix matching frame_run.cc.
    const float yawR   = odai::math::radians(cam.yawDegrees);
    const float pitchR = odai::math::radians(cam.pitchDegrees);
    const float cosPitch = std::cos(pitchR);
    const odai::math::Vector3 eye{cam.x, cam.y, cam.z};
    const odai::math::Vector3 fwd{std::cos(yawR) * cosPitch, std::sin(pitchR), std::sin(yawR) * cosPitch};
    const odai::math::Matrix4 view = odai::math::lookAt(eye, eye + fwd, {0.f, 1.f, 0.f});

    odai::math::Matrix4 proj;
    if (cam.orthographic) {
        const float halfH = cam.orthoHalfHeight;
        const float halfW = halfH * aspectRatio;
        proj = odai::math::orthographicVulkan(-halfW, halfW, -halfH, halfH, nearPlane, farPlane);
    } else {
        proj = odai::math::perspectiveVulkan(
            odai::math::radians(cam.fovDegrees), aspectRatio, nearPlane, farPlane);
    }
    const odai::math::Matrix4 vp = proj * view;

    const odai::ui::UiColor shadowColor{0.f, 0.f, 0.f, 0.65f};
    const odai::ui::UiColor textColor{1.f, 0.95f, 0.82f, 1.f};
    const odai::ui::Font& font = m_uiFont;
    const float halfLineH = font.lineHeightPx() * 0.5f;

    for (const auto& s : m_strategyMap.settlements) {
        const float rowOffset = (s.row % 2 == 0) ? 0.0f : 0.5f;
        const float wx = m_strategyMap.hexSize * std::sqrt(3.0f)
                         * (static_cast<float>(s.col) + rowOffset);
        const float wz = m_strategyMap.hexSize * 1.5f * static_cast<float>(s.row);
        float wy = 0.0f;
        if (s.col < m_strategyMap.width && s.row < m_strategyMap.height) {
            wy = static_cast<float>(
                     m_strategyMap.tiles[s.row * m_strategyMap.width + s.col].elevation)
                 * m_strategyMap.elevationStep;
        }
        wy += m_strategyMap.hexSize * 0.5f;

        const odai::math::Vector4 clip = vp * odai::math::Vector4{wx, wy, wz, 1.0f};
        if (clip.w <= 0.0f) continue;

        const float ndcX = clip.x / clip.w;
        const float ndcY = clip.y / clip.w;
        if (ndcX < -1.1f || ndcX > 1.1f || ndcY < -1.1f || ndcY > 1.1f) continue;

        // Vulkan NDC: -1 = top-left, +1 = bottom-right.
        const float sx = (ndcX + 1.0f) * 0.5f * fbW;
        const float sy = (ndcY + 1.0f) * 0.5f * fbH;

        const float textW = font.measureText(s.name);
        const float tx = sx - textW * 0.5f;
        const float ty = sy - halfLineH;

        m_uiDrawList.addText(font, s.name, {tx + 1.0f, ty + 1.0f}, shadowColor);
        m_uiDrawList.addText(font, s.name, {tx, ty}, textColor);
    }

    // Floating "HP/maxHP  supply/maxSupply" over each unit token; reddened when
    // starving, bracketed + bright when selected.
    const odai::ui::UiColor starveColor{1.0f, 0.40f, 0.36f, 1.0f};
    const odai::ui::UiColor selColor{0.72f, 0.86f, 1.0f, 1.0f};
    for (const odai::game::Unit& u : m_gameState.units) {
        if (!u.alive() || u.col >= m_strategyMap.width || u.row >= m_strategyMap.height) continue;
        const float rowOffset = (u.row % 2 == 0) ? 0.0f : 0.5f;
        const float wx = m_strategyMap.hexSize * std::sqrt(3.0f)
                         * (static_cast<float>(u.col) + rowOffset);
        const float wz = m_strategyMap.hexSize * 1.5f * static_cast<float>(u.row);
        float wy = static_cast<float>(
                       m_strategyMap.tiles[u.row * m_strategyMap.width + u.col].elevation)
                   * m_strategyMap.elevationStep;
        wy += m_strategyMap.hexSize * 1.3f;  // float above the token box

        const odai::math::Vector4 clip = vp * odai::math::Vector4{wx, wy, wz, 1.0f};
        if (clip.w <= 0.0f) continue;
        const float ndcX = clip.x / clip.w;
        const float ndcY = clip.y / clip.w;
        if (ndcX < -1.1f || ndcX > 1.1f || ndcY < -1.1f || ndcY > 1.1f) continue;
        const float sx = (ndcX + 1.0f) * 0.5f * fbW;
        const float sy = (ndcY + 1.0f) * 0.5f * fbH;

        std::string label = std::to_string(u.hp) + "/" + std::to_string(u.maxHp) + "  "
                          + std::to_string(u.supply) + "/" + std::to_string(u.maxSupply);
        if (u.id == m_selectedUnitId) label = "[" + label + "]";
        const odai::ui::UiColor col = (u.supply == 0)
            ? starveColor
            : (u.id == m_selectedUnitId ? selColor : textColor);

        const float textW = font.measureText(label);
        const float tx = sx - textW * 0.5f;
        const float ty = sy - halfLineH;
        m_uiDrawList.addText(font, label, {tx + 1.0f, ty + 1.0f}, shadowColor);
        m_uiDrawList.addText(font, label, {tx, ty}, col);
    }

    // Planned-route preview: a small dot at each remaining waypoint of the selected
    // unit's A* move order, so the player can see where it is headed.
    if (m_selectedUnitId != 0) {
        if (const odai::game::Unit* sel = m_gameState.findUnit(m_selectedUnitId)) {
            const odai::ui::UiColor dotColor{0.95f, 0.85f, 0.35f, 0.85f};
            for (const auto& wp : sel->path) {
                if (wp[0] >= m_strategyMap.width || wp[1] >= m_strategyMap.height) continue;
                const float rowOffset = (wp[1] % 2 == 0) ? 0.0f : 0.5f;
                const float wx = m_strategyMap.hexSize * std::sqrt(3.0f)
                                 * (static_cast<float>(wp[0]) + rowOffset);
                const float wz = m_strategyMap.hexSize * 1.5f * static_cast<float>(wp[1]);
                float wy = static_cast<float>(
                               m_strategyMap.tiles[wp[1] * m_strategyMap.width + wp[0]].elevation)
                           * m_strategyMap.elevationStep;
                wy += m_strategyMap.hexSize * 0.55f;

                const odai::math::Vector4 clip = vp * odai::math::Vector4{wx, wy, wz, 1.0f};
                if (clip.w <= 0.0f) continue;
                const float ndcX = clip.x / clip.w;
                const float ndcY = clip.y / clip.w;
                if (ndcX < -1.1f || ndcX > 1.1f || ndcY < -1.1f || ndcY > 1.1f) continue;
                const float sx = (ndcX + 1.0f) * 0.5f * fbW;
                const float sy = (ndcY + 1.0f) * 0.5f * fbH;
                constexpr float kDotR = 4.0f;
                m_uiDrawList.addRectFilled(
                    odai::ui::UiRect::fromXYWH(sx - kDotR, sy - kDotR, 2.0f * kDotR, 2.0f * kDotR),
                    dotColor);
            }
        }
    }

    // Live right-drag preview: cyan dots along the candidate march route, or a red
    // marker over an enemy the selected unit would attack on release.
    if (m_previewActive) {
        const odai::ui::UiColor previewDot{0.45f, 0.85f, 0.95f, 0.90f};
        const odai::ui::UiColor attackMark{0.95f, 0.30f, 0.25f, 0.95f};
        auto projectAndDraw = [&](std::uint32_t col, std::uint32_t row,
                                  const odai::ui::UiColor& color, float radius) {
            if (col >= m_strategyMap.width || row >= m_strategyMap.height) return;
            const float rowOffset = (row % 2 == 0) ? 0.0f : 0.5f;
            const float wx = m_strategyMap.hexSize * std::sqrt(3.0f)
                             * (static_cast<float>(col) + rowOffset);
            const float wz = m_strategyMap.hexSize * 1.5f * static_cast<float>(row);
            float wy = static_cast<float>(
                           m_strategyMap.tiles[row * m_strategyMap.width + col].elevation)
                       * m_strategyMap.elevationStep;
            wy += m_strategyMap.hexSize * 0.55f;
            const odai::math::Vector4 clip = vp * odai::math::Vector4{wx, wy, wz, 1.0f};
            if (clip.w <= 0.0f) return;
            const float ndcX = clip.x / clip.w;
            const float ndcY = clip.y / clip.w;
            if (ndcX < -1.1f || ndcX > 1.1f || ndcY < -1.1f || ndcY > 1.1f) return;
            const float sx = (ndcX + 1.0f) * 0.5f * fbW;
            const float sy = (ndcY + 1.0f) * 0.5f * fbH;
            m_uiDrawList.addRectFilled(
                odai::ui::UiRect::fromXYWH(sx - radius, sy - radius, 2.0f * radius, 2.0f * radius), color);
        };
        if (m_previewIsAttack) {
            projectAndDraw(m_previewTargetCol, m_previewTargetRow, attackMark, 8.0f);
        } else {
            for (const auto& wp : m_previewPath) {
                projectAndDraw(wp[0], wp[1], previewDot, 4.0f);
            }
        }
    }
}

void App::updateUiOverlay(float dt) {
    if (m_window == nullptr) {
        return;
    }

    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize(m_window, &fbW, &fbH);
    if (fbW <= 0 || fbH <= 0) {
        return;
    }
    // Keep the cached viewport current so openProductionPanelForCity always
    // computes its rect from the live framebuffer size, not the setup-time size.
    m_setupViewW = static_cast<float>(fbW);
    m_setupViewH = static_cast<float>(fbH);

    if (m_uiContext.root() == nullptr) {
        setupDemoUi(static_cast<float>(fbW), static_cast<float>(fbH));
    }

    double mouseX = 0.0, mouseY = 0.0;
    glfwGetCursorPos(m_window, &mouseX, &mouseY);
    const bool leftDown = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

    m_uiInput.beginFrame();
    m_uiInput.mousePx = {static_cast<float>(mouseX), static_cast<float>(mouseY)};
    m_uiInput.setButton(odai::ui::UiMouseButton::Left, leftDown);
    m_uiInput.scrollDelta = m_pendingScrollDelta;
    m_pendingScrollDelta = 0.0f;

    // Feed typed characters into the UI, plus synthetic codepoints for the editing
    // keys GLFW reports as keys rather than characters (8 = backspace, 13 = enter).
    for (std::uint32_t cp : m_pendingTextInput) {
        m_uiInput.textInput.push_back(cp);
    }
    m_pendingTextInput.clear();
    const bool backspaceDown = glfwGetKey(m_window, GLFW_KEY_BACKSPACE) == GLFW_PRESS;
    if (backspaceDown && !m_backspacePrev) {
        m_uiInput.textInput.push_back(8u);
    }
    m_backspacePrev = backspaceDown;
    const bool enterDown = glfwGetKey(m_window, GLFW_KEY_ENTER) == GLFW_PRESS;
    if (enterDown && !m_enterPrev) {
        m_uiInput.textInput.push_back(13u);
    }
    m_enterPrev = enterDown;

    // Time the UI update + geometry build (the CPU cost that grows with a
    // text-heavy UI). With layout/geometry caching this stays near-flat per frame.
    const auto uiBuildStart = std::chrono::steady_clock::now();

    m_uiContext.setViewport({static_cast<float>(fbW), static_cast<float>(fbH)});
    m_uiContext.update(m_uiInput);

    // Unit action bar: rebuild slots when selection changes.
    if (m_selectedUnitType != m_prevSelectedUnitType) {
        m_prevSelectedUnitType = m_selectedUnitType;

        // Actions per unit type - ordered for display.
        static const std::unordered_map<std::string, std::vector<std::string>> kUnitActions = {
            {"warrior",  {"move", "attack",        "fortify", "wait", "sleep"}},
            {"archer",   {"move", "ranged_attack",  "fortify", "wait", "sleep"}},
            {"spearman", {"move", "attack",         "fortify", "wait", "sleep"}},
            {"scout",    {"move", "explore",        "wait",    "sleep"}},
            {"settler",  {"move", "found_city",     "wait"}},
            {"builder",  {"move", "build",          "wait"}},
            {"cavalry",  {"move", "attack",         "fortify", "wait", "sleep"}},
            {"ship",     {"move", "attack",         "embark",  "wait"}},
            {"siege",    {"move", "attack",         "wait"}},
        };

        // Display name for each unit type.
        static const std::unordered_map<std::string, std::string> kUnitNames = {
            {"warrior",  "Warrior"},  {"archer",   "Archer"},
            {"spearman", "Spearman"}, {"scout",    "Scout"},
            {"settler",  "Settler"},  {"builder",  "Builder"},
            {"cavalry",  "Cavalry"},  {"ship",     "Ship"},
            {"siege",    "Siege"},
        };

        const bool hasSelection = !m_selectedUnitType.empty();
        if (m_unitActionPanel != nullptr) {
            m_unitActionPanel->visible = hasSelection;
        }

        if (hasSelection) {
            // Update portrait.
            if (m_unitActionPortrait != nullptr) {
                odai::ui::UiIconEntry pe{};
                if (odai::ui::UiIconRegistry::global().resolve(m_selectedUnitType, pe)) {
                    m_unitActionPortrait->textureId = pe.textureId;
                    m_unitActionPortrait->uvRect    = pe.uv;
                }
            }
            // Update name label.
            if (m_unitActionName != nullptr) {
                const auto nit = kUnitNames.find(m_selectedUnitType);
                m_unitActionName->setText(nit != kUnitNames.end() ? nit->second : m_selectedUnitType);
            }
            // Update action slots.
            const auto ait = kUnitActions.find(m_selectedUnitType);
            const std::vector<std::string>* actions =
                (ait != kUnitActions.end()) ? &ait->second : nullptr;
            for (int si = 0; si < kMaxActionSlots; ++si) {
                if (m_actionSlotBtns[si] == nullptr) continue;
                const bool slotUsed = actions && si < static_cast<int>(actions->size());
                m_actionSlotBtns[si]->visible = slotUsed;
                if (slotUsed) {
                    odai::ui::UiIconEntry ae{};
                    if (odai::ui::UiIconRegistry::global().resolve((*actions)[si], ae)) {
                        m_actionSlotBtns[si]->textureId = ae.textureId;
                        m_actionSlotBtns[si]->uvRect    = ae.uv;
                    }
                    // Re-bind click handler to log the action name.
                    const std::string actionName = (*actions)[si];
                    // (Action callbacks can be wired to game logic here.)
                }
            }
        }
    }

    // Update map stats label.
    if (m_hudStatsLabel != nullptr && m_strategyMap.width > 0) {
        const auto cnt = static_cast<int>(m_strategyMap.settlements.size());
        m_hudStatsLabel->setText(
            std::to_string(m_strategyMap.width) + " \xc3\x97 " +
            std::to_string(m_strategyMap.height) + "  \xc2\xb7  " +
            std::to_string(cnt) + (cnt == 1 ? " settlement" : " settlements"));
    }

    // Top toolbar: accumulate gold and reflect the current turn.
    if (m_toolbar != nullptr) {
        m_tbGold += dt * 39.0f / 60.0f;  // ~+39/turn at ~60 fps
        const int goldInt = static_cast<int>(m_tbGold);
        m_toolbar->setValue(m_tbGoldItem, std::to_string(goldInt) + " (+39)");
        m_toolbar->setTooltip(2,
            "<b><color=#f0c040>Gold  " + std::to_string(goldInt) + " in treasury</color></b>\n"
            "Income / turn      <color=#f0c040>+39</color>\n"
            "  Capital city     <color=#f0c040>+20</color>\n"
            "  Trade routes     <color=#f0c040>+15</color>\n"
            "  Market           <color=#f0c040>+4</color>");
    }
    if (m_toolbarTurnLabel != nullptr) {
        m_toolbarTurnLabel->setText(
            "<color=#c9b896>Turn</color> <b><color=#ecd39a>" +
            std::to_string(m_currentTurn) + "</color></b>");
    }

    // End Turn / Next Idle: label changes to show idle count while there are
    // cities without production or unvisited units waiting for orders.
    if (m_endTurnBtn != nullptr && m_strategyMapMode) {
        const int idle = idleCount();
        if (idle > 0) {
            m_endTurnBtn->setLabel("Next Idle (" + std::to_string(idle) + ")");
        } else {
            m_endTurnBtn->setLabel("End Turn");
        }
    }

    // Hex hover: pick the tile under the mouse and update the tile info panel.
    if (m_hudTileInfoWindow != nullptr && m_hudTileInfoLabel != nullptr) {
        int hCol = -1, hRow = -1;
        if (!m_uiContext.wantsMouse() &&
            pickHexFromMouse(mouseX, mouseY, fbW, fbH, hCol, hRow)) {
            m_hoveredHexCol = hCol;
            m_hoveredHexRow = hRow;
            m_hudTileInfoWindow->visible = true;

            const odai::game::MapTile& tile = m_strategyMap.at(
                static_cast<std::uint32_t>(hCol),
                static_cast<std::uint32_t>(hRow));

            std::string info = "<b><color=#ecd39a>";
            info += terrainName(tile.terrain);
            info += "</color></b>";
            if (tile.elevation != 0) {
                info += "\nElev. " + std::to_string(static_cast<int>(tile.elevation));
            }
            // Find settlement at this tile.
            for (const auto& s : m_strategyMap.settlements) {
                if (static_cast<int>(s.col) == hCol && static_cast<int>(s.row) == hRow) {
                    info += "\n<color=#c0e8c0>" + s.name + "</color>";
                    break;
                }
            }
            if (tile.flags & odai::game::TileFlag_River) info += "\nRiver";
            if (tile.flags & odai::game::TileFlag_Road)  info += "\nRoad";

            // City production: what this settlement is building, plus its buildings.
            if (const odai::game::CityState* city = m_gameState.cityAt(
                    static_cast<std::uint32_t>(hCol), static_cast<std::uint32_t>(hRow))) {
                if (!city->producing.empty()) {
                    const odai::game::BuildableItem* bi = odai::game::findBuildable(city->producing);
                    const std::string pname = (bi != nullptr) ? bi->name : city->producing;
                    const std::string cost = (bi != nullptr) ? std::to_string(bi->productionCost) : "?";
                    info += "\n<color=#cfe0ff>Building " + pname + " (" +
                            std::to_string(city->accumulated) + "/" + cost + ")</color>";
                }
                if (!city->buildings.empty()) {
                    std::string blds;
                    for (std::size_t i = 0; i < city->buildings.size(); ++i) {
                        if (i != 0) blds += ", ";
                        blds += city->buildings[i];
                    }
                    info += "\n<color=#9fb0c8>" + blds + "</color>";
                }
            }

            // Unit standing here: health, provisions, combat stats, warnings.
            if (const odai::game::Unit* u = m_gameState.unitAt(
                    static_cast<std::uint32_t>(hCol), static_cast<std::uint32_t>(hRow))) {
                std::string uname = u->typeId;
                if (!uname.empty()) uname[0] = static_cast<char>(std::toupper(uname[0]));
                info += "\n<b><color=#ffe0a0>" + uname + "</color></b>";
                if (u->id == m_selectedUnitId) info += " <color=#a0d0ff>[selected]</color>";
                info += "\nHP " + std::to_string(u->hp) + "/" + std::to_string(u->maxHp);
                info += "\nSupply " + std::to_string(u->supply) + "/" + std::to_string(u->maxSupply);

                const odai::game::UnitStats& us = odai::game::unitStatsFor(u->typeId);
                std::string combat = "\nAtk " + std::to_string(us.attack);
                if (us.rangedAttack > 0) {
                    combat += "  Ranged " + std::to_string(us.rangedAttack) +
                              " (rng " + std::to_string(us.range) + ")";
                }
                if (u->armor > 0) combat += "  Armor " + std::to_string(u->armor);
                info += combat;
                if (u->supply == 0) info += "\n<color=#ff5a5a>STARVING</color>";
            }

            m_hudTileInfoLabel->setText(info);
        } else {
            m_hoveredHexCol = -1;
            m_hoveredHexRow = -1;
            m_hudTileInfoWindow->visible = false;
        }
    }

    // Strategy-map mouse: left-click (press+release, negligible drag) selects the
    // unit under the cursor — a real left drag stays a map pan. The right button is
    // a hold gesture: press and hold to preview a move route or attack for the
    // selected unit, then release to commit it.
    if (m_strategyMapMode) {
        const bool rightDown = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        const bool overUiNow = m_uiContext.wantsMouse() || isAnyUiVisible();
        constexpr float kClickSlopPx = 6.0f;

        // Left click: select.
        if (leftDown && !m_mapLeftPrevDown) {
            m_mapPressX = static_cast<float>(mouseX);
            m_mapPressY = static_cast<float>(mouseY);
            m_mapPressOverUi = overUiNow;
        }
        if (!leftDown && m_mapLeftPrevDown && !m_mapPressOverUi && !overUiNow) {
            const float dx = static_cast<float>(mouseX) - m_mapPressX;
            const float dy = static_cast<float>(mouseY) - m_mapPressY;
            if ((dx * dx + dy * dy) <= (kClickSlopPx * kClickSlopPx)) {
                int cCol = -1, cRow = -1;
                if (pickHexFromMouse(mouseX, mouseY, fbW, fbH, cCol, cRow)) {
                    selectUnitAtHex(cCol, cRow);
                    if (m_selectedUnitId != 0) {
                        // A unit was clicked: close any open city panel.
                        if (m_productionPanel != nullptr) m_productionPanel->visible = false;
                        m_selectedCityCol = -1;
                        m_selectedCityRow = -1;
                    } else {
                        // No unit here: check for a city and open its production panel.
                        if (m_gameState.cityAt(static_cast<std::uint32_t>(cCol),
                                               static_cast<std::uint32_t>(cRow)) != nullptr) {
                            openProductionPanelForCity(cCol, cRow);
                        } else {
                            if (m_productionPanel != nullptr) m_productionPanel->visible = false;
                            m_selectedCityCol = -1;
                            m_selectedCityRow = -1;
                        }
                    }
                }
            }
        }
        m_mapLeftPrevDown = leftDown;

        // Right button: remember whether the press began over UI, refresh the
        // move/attack preview while held, and commit it on release.
        if (rightDown && !m_mapRightPrevDown) {
            m_mapRightPressOverUi = overUiNow;
        }
        if (rightDown && !m_mapRightPressOverUi) {
            int tc = -1, tr = -1;
            if (!overUiNow && pickHexFromMouse(mouseX, mouseY, fbW, fbH, tc, tr)) {
                updateMoveAttackPreview(tc, tr);
            } else {
                clearMoveAttackPreview();
            }
        }
        if (!rightDown && m_mapRightPrevDown) {
            if (!m_mapRightPressOverUi) {
                commitMoveAttackPreview();
            }
            clearMoveAttackPreview();
        }
        m_mapRightPrevDown = rightDown;
    }

    // Animate the civ-panel accordion expand/collapse.
    m_civExpandTween.update(dt);
    if (m_civPanel != nullptr && m_civFullH > 0.0f) {
        const float frac = m_civExpandTween.eased();
        const float h = m_civHeaderH + frac * (m_civFullH - m_civHeaderH);
        const odai::ui::UiRect cr = m_civPanel->rect();
        m_civPanel->setRect(odai::ui::UiRect::fromXYWH(cr.minX, cr.minY, cr.width(), h));
    }
    // Sync the chevron label to match expand state.
    if (m_civCollapseBtn != nullptr) {
        m_civCollapseBtn->setLabel(m_civExpandTween.value > 0.5f ? "^" : "v");
    }

    // Reset the draw list here so map labels can be drawn first (behind the UI).
    m_uiDrawList.reset({static_cast<float>(fbW), static_cast<float>(fbH)});
    drawStrategyMapLabels(static_cast<float>(fbW), static_cast<float>(fbH));
    m_uiContext.buildAppend(m_uiDrawList);

    // Tooltip overlay: drawn after the widget tree (so it sits on top, unclipped),
    // with an ease-in-out fade. Latch the text/anchor while hovering so the tooltip
    // can keep fading out for a moment after the cursor leaves the term.
    bool tipActive = m_uiFontReady && m_civpediaView != nullptr && m_civpediaView->hasTooltip();
    if (tipActive) {
        m_lastTooltipText = m_civpediaView->tooltipText();
        m_lastTooltipAnchor = m_civpediaView->tooltipAnchor();
    } else if (m_uiFontReady && m_toolbar != nullptr && m_toolbar->hasHoveredTooltip()) {
        m_lastTooltipText = m_toolbar->hoveredTooltipText();
        m_lastTooltipAnchor = m_toolbar->hoveredTooltipAnchor();
        tipActive = true;
    }
    m_tooltipFade.setTarget(tipActive ? 1.0f : 0.0f);
    m_tooltipFade.update(dt);
    const float tipAlpha = m_tooltipFade.eased();

    if (tipAlpha > 0.001f && !m_lastTooltipText.empty()) {
        const float s = m_uiScale;
        const odai::ui::UiVec2 anchor = m_lastTooltipAnchor;
        const odai::ui::FontSet& fonts = m_uiFonts;

        const float padX = 14.0f * s;
        const float padY = 10.0f * s;
        const float maxW = 300.0f * s;
        const odai::ui::RichTextLayout tip = odai::ui::layoutRichText(
            m_lastTooltipText, odai::ui::UiColor{0.92f, 0.93f, 0.95f, 1.0f},
            fonts, maxW - padX * 2.0f, odai::ui::UiTextAlign::Left);

        const float boxW = tip.width + padX * 2.0f;
        const float boxH = tip.height + padY * 2.0f;
        float boxX = anchor.x + 16.0f * s;
        float boxY = anchor.y + 18.0f * s;
        // Clamp inside the framebuffer.
        boxX = std::min(boxX, static_cast<float>(fbW) - boxW - 4.0f * s);
        boxY = std::min(boxY, static_cast<float>(fbH) - boxH - 4.0f * s);
        boxX = std::max(boxX, 4.0f * s);
        boxY = std::max(boxY, 4.0f * s);

        const odai::ui::UiRect box{boxX, boxY, boxX + boxW, boxY + boxH};
        m_uiDrawList.pushOpacity(tipAlpha);
        m_uiDrawList.addRectFilled(box, odai::ui::UiColor{0.05f, 0.07f, 0.10f, 0.96f});
        m_uiDrawList.addRect(box, odai::ui::UiColor{0.85f, 0.62f, 0.30f, 0.75f}, 1.0f);
        odai::ui::drawRichText(m_uiDrawList, tip, fonts, odai::ui::UiVec2{boxX + padX, boxY + padY});
        m_uiDrawList.popOpacity();
    }

    m_renderer.setUiDrawData(m_uiDrawList.data());

    const auto uiBuildEnd = std::chrono::steady_clock::now();
    const float uiBuildMs =
        std::chrono::duration<float, std::milli>(uiBuildEnd - uiBuildStart).count();
    m_uiBuildMsEma = (m_uiBuildMsEma <= 0.0f) ? uiBuildMs : (m_uiBuildMsEma * 0.95f + uiBuildMs * 0.05f);
    if (++m_uiBuildLogCounter >= 120) {
        m_uiBuildLogCounter = 0;
        VOX_LOGD("ui") << "UI build CPU: " << m_uiBuildMsEma << " ms (ema), "
                       << m_uiDrawList.data().vertices.size() << " verts, "
                       << m_uiDrawList.data().commands.size() << " draws";
    }
}

bool App::pickHexFromMouse(double mouseX, double mouseY, int fbW, int fbH,
                           int& outCol, int& outRow) const {
    if (!m_strategyMap.width || !m_strategyMap.height) return false;
    if (fbW <= 0 || fbH <= 0) return false;

    // Shared hex lookup: world XZ ? nearest hex cell.
    const float hexSize = m_strategyMap.hexSize;
    constexpr float kSqrt3 = 1.7320508075688772f;
    auto findHex = [&](float wx, float wz) -> bool {
        const int approxRow = static_cast<int>(std::round(wz / (hexSize * 1.5f)));
        float bestDist2 = 1e30f;
        int bestCol = -1, bestRow = -1;
        for (int dr = -1; dr <= 1; ++dr) {
            const int r = approxRow + dr;
            if (r < 0 || r >= static_cast<int>(m_strategyMap.height)) continue;
            const float ro = (r & 1) ? 0.5f : 0.0f;
            const int cBase = static_cast<int>(std::round(wx / (hexSize * kSqrt3) - ro));
            for (int dc = -1; dc <= 1; ++dc) {
                const int c = cBase + dc;
                if (c < 0 || c >= static_cast<int>(m_strategyMap.width)) continue;
                const auto ctr = odai::game::tileCenterWorld(
                    m_strategyMap, static_cast<std::uint32_t>(c), static_cast<std::uint32_t>(r));
                const float dx = ctr.x - wx, dz = ctr.z - wz;
                const float d2 = dx*dx + dz*dz;
                if (d2 < bestDist2) { bestDist2 = d2; bestCol = c; bestRow = r; }
            }
        }
        if (bestCol < 0) return false;
        outCol = bestCol;
        outRow = bestRow;
        return true;
    };

    float wx = 0.0f, wz = 0.0f;
    if (!rayToGroundPlane(mouseX, mouseY, fbW, fbH, wx, wz)) {
        return false;
    }
    return findHex(wx, wz);
}

bool App::rayToGroundPlane(double mouseX, double mouseY, int fbW, int fbH,
                           float& outX, float& outZ) const {
    if (fbW <= 0 || fbH <= 0) return false;

    // Build camera basis from yaw/pitch (same convention as computeCameraForward).
    const float yaw   = odai::math::radians(m_camera.yawDegrees);
    const float pitch = odai::math::radians(m_camera.pitchDegrees);
    const float cp = std::cos(pitch);
    const float sp = std::sin(pitch);
    const float cy = std::cos(yaw);
    const float sy = std::sin(yaw);

    // Forward, right, up in world space.
    const float fwdX =  cy * cp,  fwdY = sp,   fwdZ = sy * cp;
    const float rgtX = -sy,        rgtY = 0.0f, rgtZ = cy;
    const float upX  = -cy * sp,  upY  = cp,   upZ  = -sy * sp;

    // NDC of the mouse pixel (+Y up).
    const float ndcX = static_cast<float>(mouseX) / static_cast<float>(fbW) * 2.0f - 1.0f;
    const float ndcY = 1.0f - static_cast<float>(mouseY) / static_cast<float>(fbH) * 2.0f;
    const float aspect = static_cast<float>(fbW) / static_cast<float>(fbH);

    if (m_strategyMapMode && !m_strategyMap3D) {
        // Orthographic: ray origin offset per pixel, direction = forward.
        const float halfH = m_mapOrthoHalfHeight;
        const float halfW = halfH * aspect;
        const float roX = m_camera.x + rgtX * ndcX * halfW + upX * ndcY * halfH;
        const float roY = m_camera.y + rgtY * ndcX * halfW + upY * ndcY * halfH;
        const float roZ = m_camera.z + rgtZ * ndcX * halfW + upZ * ndcY * halfH;
        if (std::abs(fwdY) < 1e-4f) return false;
        const float t = -roY / fwdY;
        if (t < 0.0f) return false;
        outX = roX + t * fwdX;
        outZ = roZ + t * fwdZ;
        return true;
    }

    // Perspective: build ray through pixel and intersect with y = 0.
    const float tanHalfFov = std::tan(odai::math::radians(m_camera.fovDegrees) * 0.5f);
    const float scaleX = ndcX * tanHalfFov * aspect;
    const float scaleY = ndcY * tanHalfFov;
    float rdX = fwdX + rgtX * scaleX + upX * scaleY;
    float rdY = fwdY + rgtY * scaleX + upY * scaleY;
    float rdZ = fwdZ + rgtZ * scaleX + upZ * scaleY;
    const float rdLen = std::sqrt(rdX*rdX + rdY*rdY + rdZ*rdZ);
    if (rdLen < 1e-6f) return false;
    rdX /= rdLen; rdY /= rdLen; rdZ /= rdLen;
    if (std::abs(rdY) < 1e-4f) return false;
    const float t = -m_camera.y / rdY;
    if (t < 0.0f) return false;
    outX = m_camera.x + t * rdX;
    outZ = m_camera.z + t * rdZ;
    return true;
}

void App::rebuildStrategyMapScene() {
    odai::game::StrategyMapMeshOptions meshOptions = makeStrategyMapMeshOptions(m_strategyMap3D, m_terrainTextures);
    // The GPU hex pipeline owns the land surface in 3D relief; skip the imported-static
    // land tops/skirts there so it is not drawn twice. The flat 2D board keeps them.
    meshOptions.emitLandSurface = !(m_useHexTerrain && m_strategyMap3D);
    m_importedScene = odai::game::buildStrategyMapScene(m_strategyMap, m_gameState.units, std::move(meshOptions));
    if (!m_renderer.uploadImportedScene(m_importedScene)) {
        VOX_LOGE("app") << "strategy map scene re-upload failed";
    }
}

void App::spawnDemoUnits() {
    m_gameState = {};
    m_selectedUnitId = 0;
    clearMoveAttackPreview();
    m_gameState.initCities(m_strategyMap);
    if (m_strategyMap.settlements.empty()) {
        return;
    }

    // Seed demo buildings so cities can immediately pump out units and the build
    // prerequisites are visible: the capital gets the full kit (and a smithy for
    // armor); other cities get a single military building so they specialize.
    for (std::size_t i = 0; i < m_gameState.cities.size(); ++i) {
        odai::game::CityState& city = m_gameState.cities[i];
        city.perTurn = 20;  // demo pace: a unit every few End Turns
        if (i == 0) {
            city.buildings = {"barracks", "fletcher", "smithy"};
        } else if (i % 3 == 1) {
            city.buildings = {"barracks"};  // melee only
        } else if (i % 3 == 2) {
            city.buildings = {"fletcher"};  // archers only
        }
        // Remaining cities have no military building and stay idle.
    }

    // A warrior garrisons each of the first few settlements (starts fully supplied).
    const std::size_t garrisons = std::min<std::size_t>(m_strategyMap.settlements.size(), 3u);
    for (std::size_t i = 0; i < garrisons; ++i) {
        const odai::game::Settlement& s = m_strategyMap.settlements[i];
        m_gameState.spawnUnit("warrior", s.col, s.row, s.owner);
    }

    // Give the capital a scout on a free neighbor so the player has a longer-range
    // unit to march into the wilderness immediately.
    const odai::game::Settlement& home = m_strategyMap.settlements.front();
    const odai::game::FreeTile scoutTile = odai::game::findFreeNeighbor(m_strategyMap, m_gameState, home.col, home.row);
    if (scoutTile.found) {
        m_gameState.spawnUnit("scout", scoutTile.col, scoutTile.row, home.owner);
    }
    VOX_LOGI("app") << "spawned " << m_gameState.units.size() << " demo units across "
                    << m_gameState.cities.size() << " cities";
}

void App::selectUnitAtHex(int col, int row) {
    if (!m_strategyMap.inBounds(col, row)) {
        return;
    }
    odai::game::Unit* onTile = m_gameState.unitAt(static_cast<std::uint32_t>(col), static_cast<std::uint32_t>(row));
    if (onTile != nullptr) {
        // Select the unit here (and drive the existing unit action bar).
        m_selectedUnitId = onTile->id;
        m_selectedUnitType = onTile->typeId;
    } else {
        // Clicking empty ground clears the selection.
        m_selectedUnitId = 0;
    }
}

void App::clearMoveAttackPreview() {
    m_previewActive = false;
    m_previewIsAttack = false;
    m_previewPath.clear();
}

void App::updateMoveAttackPreview(int targetCol, int targetRow) {
    clearMoveAttackPreview();
    odai::game::Unit* sel = m_gameState.findUnit(m_selectedUnitId);
    if (sel == nullptr || !m_strategyMap.inBounds(targetCol, targetRow)) {
        return;
    }
    const std::uint32_t tc = static_cast<std::uint32_t>(targetCol);
    const std::uint32_t tr = static_cast<std::uint32_t>(targetRow);
    m_previewActive = true;
    m_previewTargetCol = tc;
    m_previewTargetRow = tr;

    const odai::game::Unit* enemy = m_gameState.unitAt(tc, tr);
    if (enemy != nullptr && enemy->owner != sel->owner &&
        odai::game::canAttack(m_gameState, *sel, *enemy)) {
        m_previewIsAttack = true;  // release will fire/charge
    } else {
        // Show the A* route the unit would march on release.
        m_previewPath = odai::game::findHexPath(m_strategyMap, m_gameState, sel->col, sel->row, tc, tr);
    }
}

void App::commitMoveAttackPreview() {
    odai::game::Unit* sel = m_gameState.findUnit(m_selectedUnitId);
    if (sel == nullptr || !m_previewActive) {
        return;
    }
    if (m_previewIsAttack) {
        const odai::game::Unit* target = m_gameState.unitAt(m_previewTargetCol, m_previewTargetRow);
        if (target != nullptr) {
            odai::game::resolveAttack(m_gameState, m_strategyMap, sel->id, target->id);
            rebuildStrategyMapScene();
        }
    } else if (sel->col != m_previewTargetCol || sel->row != m_previewTargetRow) {
        // Plan an A* path and begin marching; the order continues across turns
        // until the unit arrives (advanceTurn keeps following the path).
        odai::game::issueMoveOrder(m_gameState, m_strategyMap, *sel, m_previewTargetCol, m_previewTargetRow);
        rebuildStrategyMapScene();
    }
}

void App::setStrategyMap3D(bool enabled) {
    if (!m_strategyMapMode || enabled == m_strategyMap3D) {
        return;
    }
    m_strategyMap3D = enabled;
    m_mapFocusVelocityX = 0.0f;
    m_mapFocusVelocityZ = 0.0f;
    m_panDispSmoothX = 0.0f;
    m_panDispSmoothZ = 0.0f;
    m_wasStrategyMapDragging = false;

    // Re-mesh with the matching geometry (3D extrudes prisms + skirts, 2D is flat)
    // and re-upload. m_strategyMap3D is already set to `enabled` above, so the
    // shared rebuild path picks the right relief mode.
    rebuildStrategyMapScene();
    // The displaced hex land is 3D-only; the flat 2D board uses the imported-static land.
    m_renderer.setHexTerrainEnabled(m_useHexTerrain && enabled);
    // SSAO deepens land-against-water seams in 3D relief; off for the flat board.
    m_renderer.setSsaoEnabled(enabled);

    if (enabled) {
        // Center the 3D focus on the point the 2D view was looking at, then derive
        // the tilted eye now so there's no interpolation streak this frame.
        m_map3DFocusX = std::clamp(m_camera.x, m_mapMinX, m_mapMaxX);
        m_map3DFocusZ = std::clamp(m_camera.z, m_mapMinZ, m_mapMaxZ);
        m_camera.yawDegrees = kMap3DYawDegrees;
        m_camera.pitchDegrees = m_map3DPitchDeg;
        m_camera.fovDegrees = kMap3DFovDegrees;
        const float yawR = odai::math::radians(kMap3DYawDegrees);
        const float pitchR = odai::math::radians(m_map3DPitchDeg);
        const float cpv = std::cos(pitchR);
        const odai::math::Vector3 fwd{cpv * std::cos(yawR), std::sin(pitchR), cpv * std::sin(yawR)};
        m_camera.x = m_map3DFocusX - fwd.x * m_map3DDistance;
        m_camera.y = -fwd.y * m_map3DDistance;
        m_camera.z = m_map3DFocusZ - fwd.z * m_map3DDistance;
    } else {
        // Restore the top-down orthographic camera over the 3D focus point.
        m_camera.x = m_map3DFocusX;
        m_camera.z = m_map3DFocusZ;
        m_camera.y = 5000.0f;
        m_camera.yawDegrees = 0.0f;
        m_camera.pitchDegrees = -89.0f;
        m_camera.fovDegrees = kMap3DFovDegrees;
    }
    // Avoid a one-frame interpolation streak across the discontinuous jump.
    m_cameraPrevious = m_camera;

    VOX_LOGI("app") << "strategy map view: " << (enabled ? "3D relief" : "2D flat") << " (M)";
}

void App::focusOnHex(std::uint32_t col, std::uint32_t row) {
    if (!m_strategyMap.inBounds(static_cast<int>(col), static_cast<int>(row))) {
        return;
    }
    const odai::math::Vector3 ctr = odai::game::tileCenterWorld(m_strategyMap, col, row);
    m_mapFocusVelocityX = 0.0f;
    m_mapFocusVelocityZ = 0.0f;
    if (m_strategyMap3D) {
        m_map3DFocusX = ctr.x;
        m_map3DFocusZ = ctr.z;
        // Reposition eye so the focus is centred with no interpolation streak.
        const float yawR   = odai::math::radians(kMap3DYawDegrees);
        const float pitchR = odai::math::radians(m_map3DPitchDeg);
        const float cpv = std::cos(pitchR);
        const odai::math::Vector3 fwd{cpv * std::cos(yawR), std::sin(pitchR), cpv * std::sin(yawR)};
        m_camera.x = m_map3DFocusX - fwd.x * m_map3DDistance;
        m_camera.y = -fwd.y * m_map3DDistance;
        m_camera.z = m_map3DFocusZ - fwd.z * m_map3DDistance;
    } else {
        m_camera.x = ctr.x;
        m_camera.z = ctr.z;
    }
    m_cameraPrevious = m_camera;
}

int App::idleCount() const {
    if (!m_strategyMapMode) {
        return 0;
    }
    int count = 0;
    for (const auto& city : m_gameState.cities) {
        if (city.producing.empty()) {
            count++;
        }
    }
    for (const auto& unit : m_gameState.units) {
        if (unit.alive() && unit.path.empty() && unit.movementLeft > 0 &&
            m_visitedUnitIds.find(unit.id) == m_visitedUnitIds.end()) {
            count++;
        }
    }
    return count;
}

void App::cycleToNextIdle() {
    // Idle cities take priority: open the production panel for the first one found.
    for (const auto& city : m_gameState.cities) {
        if (city.producing.empty()) {
            focusOnHex(city.col, city.row);
            openProductionPanelForCity(static_cast<int>(city.col), static_cast<int>(city.row));
            return;
        }
    }
    // Then idle units (have moves, no path, not yet visited this turn).
    for (const auto& unit : m_gameState.units) {
        if (unit.alive() && unit.path.empty() && unit.movementLeft > 0 &&
            m_visitedUnitIds.find(unit.id) == m_visitedUnitIds.end()) {
            m_visitedUnitIds.insert(unit.id);
            focusOnHex(unit.col, unit.row);
            m_selectedUnitId = unit.id;
            m_selectedUnitType = unit.typeId;
            if (m_productionPanel != nullptr) m_productionPanel->visible = false;
            m_selectedCityCol = -1;
            m_selectedCityRow = -1;
            return;
        }
    }
}

void App::openProductionPanelForCity(int col, int row) {
    if (m_productionPanel == nullptr || m_setupViewW <= 0.0f) {
        return;
    }
    odai::game::CityState* city = m_gameState.cityAt(
        static_cast<std::uint32_t>(col), static_cast<std::uint32_t>(row));
    if (city == nullptr) {
        return;
    }
    m_selectedCityCol = col;
    m_selectedCityRow = row;

    // Find the settlement name.
    std::string cityName;
    for (const auto& s : m_strategyMap.settlements) {
        if (static_cast<int>(s.col) == col && static_cast<int>(s.row) == row) {
            cityName = s.name;
            break;
        }
    }
    if (cityName.empty()) {
        cityName = "City";
    }

    const float s      = m_uiScale;
    const float kToolbarH = 40.0f * s;
    const float kBarH  = 64.0f * s;
    const float barY   = m_setupViewH - kBarH;
    const float prodW  = 320.0f * s;
    const float prodX  = m_setupViewW - prodW - 10.0f * s;
    const float prodTop = kToolbarH + 8.0f * s;
    const float prodH  = barY - prodTop - 8.0f * s;

    const odai::ui::UiRect rect = odai::ui::UiRect::fromXYWH(prodX, prodTop, prodW, prodH);

    auto makeRow = [&](const odai::game::BuildableItem& item, const std::string& section) {
        odai::ui::ProductionPanel::Row row;
        row.id             = item.id;
        row.name           = item.name;
        row.iconName       = item.iconName;
        row.productionCost = item.productionCost;
        row.turns          = odai::game::turnsToBuild(item.productionCost, city->accumulated, city->perTurn);
        row.selected       = (item.id == city->producing);
        row.section        = section;

        const std::string itemId = item.id;
        row.onSelect = [this, itemId]() {
            odai::game::CityState* c = m_gameState.cityAt(
                static_cast<std::uint32_t>(m_selectedCityCol),
                static_cast<std::uint32_t>(m_selectedCityRow));
            if (c == nullptr) {
                return;
            }
            if (c->producing != itemId) {
                c->producing    = itemId;
                c->accumulated  = 0;
            }
            if (m_productionPanel != nullptr) {
                m_productionPanel->setSelected(itemId);
            }
            VOX_LOGI("ui") << "city " << m_selectedCityCol << "," << m_selectedCityRow
                           << " production set to " << itemId;
        };

        const std::string pName    = item.name;
        const std::string pIcon    = item.iconName;
        const bool        pIsUnit  = (item.kind == odai::game::BuildableKind::Unit);
        const std::string pArticle = odai::game::getPediaArticle(item.id);
        row.onOpenPedia = [this, pName, pIcon, pIsUnit, pArticle]() {
            if (m_civpediaWindow != nullptr) m_civpediaWindow->visible = true;
            if (m_civpediaPortrait != nullptr) {
                odai::ui::UiIconEntry pe{};
                if (odai::ui::UiIconRegistry::global().resolve(pIcon, pe)) {
                    m_civpediaPortrait->textureId = pe.textureId;
                    m_civpediaPortrait->uvRect    = pe.uv;
                }
            }
            const std::string kindStr = pIsUnit ? "Ancient Era Unit" : "City Building";
            if (m_civpediaNameLabel != nullptr) {
                m_civpediaNameLabel->setText(
                    "<b><color=#c06820>" + pName + "</color></b>\n"
                    "<color=#9fa8a8><i>" + kindStr + "</i></color>");
            }
            if (m_civpediaView != nullptr) {
                m_civpediaView->setText(pArticle);
                m_civpediaView->scrollOffsetY = 0.0f;
            }
        };
        return row;
    };

    std::vector<odai::ui::ProductionPanel::Row> rows;
    for (const auto& item : odai::game::defaultBuildables()) {
        if (item.kind == odai::game::BuildableKind::Building && !city->hasBuilding(item.id)) {
            rows.push_back(makeRow(item, "Districts & Buildings"));
        }
    }
    for (const auto& item : odai::game::defaultBuildables()) {
        if (item.kind == odai::game::BuildableKind::Unit && odai::game::cityCanProduce(*city, item.id)) {
            rows.push_back(makeRow(item, "Units"));
        }
    }

    odai::ui::ProductionPanel::CityInfo cityInfo;
    cityInfo.name       = cityName;
    cityInfo.production = city->perTurn;

    m_productionPanel->setItems(rect, s, "Choose Production", rows, cityInfo);
    m_productionPanel->visible = true;
}

} // namespace odai::app
