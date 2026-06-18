#include "app/app.h"

#include <GLFW/glfw3.h>

#include "core/grid3.h"
#include "game/strategy_map.h"
#include "game/strategy_map_io.h"
#include "game/strategy_map_mesh.h"
#include "import/morrowind_nif.h"
#include "core/log.h"
#include "math/math.h"
#include "sim/network_procedural.h"
#include "ui/icon_atlas.h"
#include "ui/widgets/button.h"
#include "ui/widgets/image.h"
#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"
#include "ui/widgets/progress_bar.h"
#include "ui/widgets/rich_text_view.h"
#include "ui/widgets/text_box.h"
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
constexpr float kImportedSceneWalkSpeed = 650.0f;
constexpr float kImportedSceneWalkSprintSpeedMultiplier = 1.6f;
constexpr float kImportedSceneJumpSpeed = 950.0f;
constexpr float kImportedSceneGravity = -3600.0f;
constexpr float kImportedSceneMaxFallSpeed = -2600.0f;
constexpr float kImportedSceneSunYawDegrees = 62.0f;
constexpr float kImportedSceneSunPitchDegrees = -61.0f;
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
constexpr float kImportedPlayerHeight = 128.0f;
constexpr float kImportedPlayerEyeHeight = 112.0f;
constexpr float kImportedPlayerTopOffset = kImportedPlayerHeight - kImportedPlayerEyeHeight;
constexpr float kImportedPlayerRadius = 28.0f;
constexpr float kImportedPlayerStepHeight = 28.0f;
constexpr float kImportedPlayerGroundSnapDistance = 24.0f;
constexpr float kImportedPlayerMinStepHeight = 1.0f;
constexpr float kImportedPlayerWalkableNormalY = 0.65f;
constexpr float kImportedPlayerCollisionSkin = 0.5f;
constexpr float kImportedPlayerCollisionSubstepDistance = 24.0f;
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
constexpr const char* kImportedSceneEnvVar = "ODAI_IMPORTED_SCENE";
constexpr const char* kStrategyMapEnvVar = "ODAI_STRATEGY_MAP";
constexpr const char* kMorrowindDataFilesEnvVar = "ODAI_MORROWIND_DATA_FILES";
constexpr const char* kBalmoraGuardsEnvVar = "ODAI_ENABLE_BALMORA_GUARDS";
constexpr const char* kBalmoraInteriorCacheEnvVar = "ODAI_BALMORA_INTERIOR_CACHE";
constexpr float kBalmoraDoorActivationRadius = 180.0f;
constexpr float kMorrowindInteriorSpawnEyeHeight = 112.0f;
constexpr float kImportedInspectRayMaxDistance = 4096.0f;
constexpr float kBalmoraGuardRouteReachRadius = 54.0f;
constexpr float kBalmoraGuardNavmeshProbeHeight = 96.0f;
constexpr float kBalmoraGuardWalkCycleScale = 0.046f;
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

CameraFrustum buildCameraFrustum(
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

bool chunkIntersectsFrustum(
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

std::string lowerPathCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        if (ch == '\\') {
            return '/';
        }
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::uint64_t fnv1a64(std::string_view value) {
    std::uint64_t hash = 14695981039346656037ull;
    for (const char ch : value) {
        hash ^= static_cast<std::uint8_t>(ch);
        hash *= 1099511628211ull;
    }
    return hash;
}

std::filesystem::path balmoraInteriorCacheRoot() {
    if (const std::optional<std::string> envPath = readEnvironmentString(kBalmoraInteriorCacheEnvVar);
        envPath.has_value() && !envPath->empty()) {
        return std::filesystem::path(*envPath);
    }
#ifdef ODAI_PROJECT_SOURCE_DIR
    return std::filesystem::path(ODAI_PROJECT_SOURCE_DIR) / "balmora_interior_cache";
#else
    return std::filesystem::path("balmora_interior_cache");
#endif
}

std::filesystem::path balmoraInteriorCachePath(std::string_view normalizedCellName) {
    return balmoraInteriorCacheRoot() /
        ("interior_" + std::to_string(fnv1a64(normalizedCellName)) + ".bin");
}

std::filesystem::path balmoraDoorCachePath() {
    return balmoraInteriorCacheRoot() / "balmora_doors.bin";
}

std::optional<std::filesystem::path> findMorrowindDataFilesPath() {
    if (const std::optional<std::string> envPath = readEnvironmentString(kMorrowindDataFilesEnvVar);
        envPath.has_value() && !envPath->empty()) {
        const std::filesystem::path path(*envPath);
        if (std::filesystem::exists(path / "Morrowind.esm")) {
            return path;
        }
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> findImportedSceneDemoPath() {
    if (const std::optional<std::string> envPathValue = readEnvironmentString(kImportedSceneEnvVar);
        envPathValue.has_value() && !envPathValue->empty()) {
        return std::filesystem::path(*envPathValue);
    }

    constexpr std::array<const char*, 4> kFallbackPaths = {
        "balmora_scene.bin",
        "C:/temp/balmora_scene.bin",
        "/mnt/c/temp/balmora_scene.bin",
        "/tmp/balmora_scene.bin"
    };
    for (const char* candidate : kFallbackPaths) {
        const std::filesystem::path path(candidate);
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return std::nullopt;
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

bool balmoraGuardsEnabledByEnvironment() {
    const std::optional<std::string> value = readEnvironmentString(kBalmoraGuardsEnvVar);
    if (!value.has_value()) {
        return false;
    }
    const std::string normalized = lowerPathCopy(*value);
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

struct ImportedSceneCameraPose {
    float x = 0.0f;
    float y = 2200.0f;
    float z = -2200.0f;
    float yawDegrees = 45.0f;
    float pitchDegrees = -18.0f;
};

ImportedSceneCameraPose makeImportedSceneLookAtPose(
    float eyeX,
    float eyeY,
    float eyeZ,
    float targetX,
    float targetY,
    float targetZ
) {
    ImportedSceneCameraPose pose{};
    pose.x = eyeX;
    pose.y = eyeY;
    pose.z = eyeZ;
    const float dx = targetX - eyeX;
    const float dy = targetY - eyeY;
    const float dz = targetZ - eyeZ;
    const float horizontalDistance = std::sqrt((dx * dx) + (dz * dz));
    pose.yawDegrees = odai::math::degrees(std::atan2(dz, dx));
    pose.pitchDegrees = odai::math::degrees(std::atan2(dy, std::max(horizontalDistance, 0.001f)));
    return pose;
}

ImportedSceneCameraPose configureImportedSceneCamera(const odai::importer::ImportedScene& scene) {
    ImportedSceneCameraPose pose{};
    const bool haveBounds =
        scene.boundsMax[0] > scene.boundsMin[0] ||
        scene.boundsMax[1] > scene.boundsMin[1] ||
        scene.boundsMax[2] > scene.boundsMin[2];

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();
    if (haveBounds) {
        minX = scene.boundsMin[0];
        minY = scene.boundsMin[1];
        minZ = scene.boundsMin[2];
        maxX = scene.boundsMax[0];
        maxY = scene.boundsMax[1];
        maxZ = scene.boundsMax[2];
    } else {
        if (scene.landscapeCells.empty()) {
            return pose;
        }
        constexpr float kCellSizeUnits = 8192.0f;
        for (const odai::importer::ImportedSceneLandscapeCell& cell : scene.landscapeCells) {
            minX = std::min(minX, static_cast<float>(cell.gridX) * kCellSizeUnits);
            minZ = std::min(minZ, static_cast<float>(cell.gridY) * kCellSizeUnits);
            maxX = std::max(maxX, static_cast<float>(cell.gridX + 1) * kCellSizeUnits);
            maxZ = std::max(maxZ, static_cast<float>(cell.gridY + 1) * kCellSizeUnits);
            for (const float height : cell.heights) {
                minY = std::min(minY, height);
                maxY = std::max(maxY, height);
            }
        }
    }

    const float centerX = (minX + maxX) * 0.5f;
    const float centerZ = (minZ + maxZ) * 0.5f;
    const float centerY = (minY + maxY) * 0.5f;

    if (scene.sourceTag == "morrowind_balmora" && !scene.waterPatches.empty()) {
        const odai::importer::ImportedSceneWaterPatch* bridgePatch = nullptr;
        float bestScore = std::numeric_limits<float>::max();
        for (const odai::importer::ImportedSceneWaterPatch& patch : scene.waterPatches) {
            const float patchCenterX = patch.originX + (patch.sizeX * 0.5f);
            const float patchCenterZ = patch.originZ + (patch.sizeZ * 0.5f);
            const float dx = patchCenterX - centerX;
            const float dz = patchCenterZ - centerZ;
            const float score = (dx * dx) + (dz * dz);
            if (score < bestScore) {
                bestScore = score;
                bridgePatch = &patch;
            }
        }
        if (bridgePatch != nullptr) {
            const float bridgeCenterX = bridgePatch->originX + (bridgePatch->sizeX * 0.5f);
            const float bridgeCenterZ = bridgePatch->originZ + (bridgePatch->sizeZ * 0.5f);
            const float bridgeY = bridgePatch->waterLevel + 260.0f;
            const float eyeX = bridgeCenterX - 520.0f;
            const float eyeY = bridgeY;
            const float eyeZ = bridgeCenterZ - 180.0f;
            return makeImportedSceneLookAtPose(
                eyeX,
                eyeY,
                eyeZ,
                centerX + 420.0f,
                centerY + 260.0f,
                centerZ);
        }
    }

    const float spanX = maxX - minX;
    const float spanZ = maxZ - minZ;
    const float span = std::max(spanX, spanZ);
    // Civ 6 style: steep top-down angle (~60°), camera centered above the map
    const float height = span * 0.65f;
    const float tilt   = height * 0.58f; // ~60° pitch: atan2(height, tilt) ≈ 60°
    return makeImportedSceneLookAtPose(
        centerX,
        centerY + height,
        centerZ - tilt,
        centerX, centerY, centerZ
    );
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
    file << "# Morrowind renderer runtime config\n";
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
    m_window = glfwCreateWindow(1280, 720, "odai", nullptr, nullptr);
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

    // Map mesh build is disabled for fast startup — UI work only.
    // m_importedScene = odai::game::buildStrategyMapScene(m_strategyMap);
    m_importedSceneDemoEnabled = true;
    m_hoverEnabled = true;

    // Fixed overhead camera instead of scene-derived pose (scene is empty).
    m_camera.x = 0.0f;
    m_camera.y = 2000.0f;
    m_camera.z = 0.0f;
    m_camera.yawDegrees = 0.0f;
    m_camera.pitchDegrees = -70.0f;
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
    // Skip scene upload — 3D map rendering is disabled for fast startup.
    // if (!m_renderer.uploadImportedScene(m_importedScene)) { ... }
    m_renderer.setImportedSceneInteriorMode(false);
    m_renderer.setSunAngles(kImportedSceneSunYawDegrees, kImportedSceneSunPitchDegrees);
    m_renderer.setImportedSceneDebugState(
        m_importedShowTerrain,
        m_importedShowStatics,
        m_importedShowTextures,
        m_importedFlatShading,
        m_importedWaterDebug);

    VOX_LOGI("app") << "init complete in " << elapsedMs(initStart) << " ms (3D map disabled)";
    return true;
}

void App::run() {
    VOX_LOGI("app") << "run begin";
    VOX_LOGI("app") << "morrowind viewer ready (WASD move, R activate door, Space jump/up, Shift down, F/C renderer panel, F5 terrain, F6 statics, K textures, F7 flat shading, F8 water debug)";
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

void App::initializeBalmoraDoorActivation() {
    m_balmoraDoors.clear();
    m_balmoraInteriorCache.clear();
    if (m_importedScene.sourceTag != "morrowind_balmora") {
        return;
    }

    odai::importer::MorrowindDoorCache doorCache{};
    const auto doorLoadStart = std::chrono::steady_clock::now();
    const bool loadedDoorCache =
        odai::importer::loadMorrowindDoorCache(balmoraDoorCachePath(), doorCache);
    if (loadedDoorCache) {
        m_balmoraDoors = doorCache.exteriorDoors;
    } else {
        VOX_LOGW("app") << "Balmora doors disabled: missing cooked door manifest "
                        << balmoraDoorCachePath().string()
                        << " (" << odai::importer::getImportedSceneLastError()
                        << "); run odai_balmora_interior_cache";
        m_balmoraDoors.clear();
        return;
    }
    const auto doorLoadMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - doorLoadStart).count();

    VOX_LOGI("app") << "Balmora door activation ready (doors=" << m_balmoraDoors.size()
                    << ", source=cache"
                    << ", elapsedMs=" << doorLoadMs
                    << ", manifest=" << balmoraDoorCachePath().string() << ")";

    std::vector<std::string> destinationCells;
    destinationCells.reserve(m_balmoraDoors.size());
    for (const odai::importer::MorrowindDoorReference& door : m_balmoraDoors) {
        const std::string normalizedCell = lowerPathCopy(door.destination.destinationCell);
        if (normalizedCell.empty() ||
            std::find(destinationCells.begin(), destinationCells.end(), normalizedCell) != destinationCells.end()) {
            continue;
        }
        destinationCells.push_back(normalizedCell);
    }

    const auto cacheLoadStart = std::chrono::steady_clock::now();
    std::uint32_t cachedInteriorFileCount = 0;
    std::uint32_t preparedInteriorCount = 0;
    std::uint32_t missingDoorCount = 0;
    for (const std::string& destinationCell : destinationCells) {
        const std::filesystem::path cachePath = balmoraInteriorCachePath(destinationCell);
        if (!std::filesystem::exists(cachePath)) {
            continue;
        }
        ++cachedInteriorFileCount;

        MorrowindInteriorCacheEntry entry{};
        if (loadedDoorCache) {
            if (const auto doorIt = doorCache.interiorDoorsByCell.find(destinationCell);
                doorIt != doorCache.interiorDoorsByCell.end()) {
                entry.doors = doorIt->second;
            } else {
                ++missingDoorCount;
            }
        }
        m_balmoraInteriorCache.emplace(destinationCell, std::move(entry));
        ++preparedInteriorCount;
    }

    const auto cacheLoadMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - cacheLoadStart).count();
    VOX_LOGI("app") << "Balmora interior disk cache ready (prepared=" << preparedInteriorCount
                    << ", cachedFiles=" << cachedInteriorFileCount
                    << "/" << destinationCells.size()
                    << ", lazyLoad=true"
                    << ", missingDoorManifests=" << missingDoorCount
                    << ", elapsedMs=" << cacheLoadMs
                    << ", directory=" << balmoraInteriorCacheRoot().string() << ")";
}

bool App::tryActivateBalmoraDoor() {
    if (!m_importedSceneDemoEnabled) {
        return false;
    }

    if (m_importedScene.sourceTag == "morrowind_interior") {
        auto cacheIt = m_balmoraInteriorCache.find(m_currentMorrowindInteriorCell);
        if (cacheIt == m_balmoraInteriorCache.end() || cacheIt->second.doors.empty()) {
            return false;
        }

        const odai::importer::MorrowindDoorReference* bestDoor = nullptr;
        float bestDistanceSq = kBalmoraDoorActivationRadius * kBalmoraDoorActivationRadius;
        for (const odai::importer::MorrowindDoorReference& door : cacheIt->second.doors) {
            if (lowerPathCopy(door.destination.destinationCell) != "balmora") {
                continue;
            }
            const float dx = door.position[0] - m_camera.x;
            const float dy = door.position[1] - (m_camera.y - kImportedPlayerEyeHeight);
            const float dz = door.position[2] - m_camera.z;
            const float distanceSq = (dx * dx) + (dy * dy * 0.25f) + (dz * dz);
            if (distanceSq < bestDistanceSq) {
                bestDistanceSq = distanceSq;
                bestDoor = &door;
            }
        }
        if (bestDoor == nullptr) {
            return false;
        }
        return leaveMorrowindInterior(*bestDoor);
    }

    if (m_importedScene.sourceTag != "morrowind_balmora" || m_balmoraDoors.empty()) {
        return false;
    }

    const odai::importer::MorrowindDoorReference* bestDoor = nullptr;
    float bestDistanceSq = kBalmoraDoorActivationRadius * kBalmoraDoorActivationRadius;
    for (const odai::importer::MorrowindDoorReference& door : m_balmoraDoors) {
        const float dx = door.position[0] - m_camera.x;
        const float dy = door.position[1] - (m_camera.y - kImportedPlayerEyeHeight);
        const float dz = door.position[2] - m_camera.z;
        const float distanceSq = (dx * dx) + (dy * dy * 0.25f) + (dz * dz);
        if (distanceSq < bestDistanceSq) {
            bestDistanceSq = distanceSq;
            bestDoor = &door;
        }
    }

    if (bestDoor == nullptr) {
        return false;
    }
    return enterMorrowindInterior(*bestDoor);
}

bool App::enterMorrowindInterior(const odai::importer::MorrowindDoorReference& door) {
    const std::string destinationCell = lowerPathCopy(door.destination.destinationCell);
    auto cacheIt = m_balmoraInteriorCache.find(destinationCell);
    if (cacheIt == m_balmoraInteriorCache.end()) {
        VOX_LOGW("app") << "failed to enter " << door.destination.destinationCell
                        << ": cached interior is missing; run odai_balmora_interior_cache for "
                        << balmoraInteriorCacheRoot().string();
        return false;
    }
    MorrowindInteriorCacheEntry& cachedInterior = cacheIt->second;
    if (!cachedInterior.sceneLoaded) {
        const std::filesystem::path cachePath = balmoraInteriorCachePath(destinationCell);
        const auto interiorLoadStart = std::chrono::steady_clock::now();
        if (!odai::importer::loadImportedSceneRuntime(cachePath, cachedInterior.scene)) {
            VOX_LOGW("app") << "failed to load cached interior " << door.destination.destinationCell
                            << " from " << cachePath.string()
                            << ": " << odai::importer::getImportedSceneLastError();
            return false;
        }
        if (cachedInterior.scene.sourceTag != "morrowind_interior" ||
            cachedInterior.scene.sourceMeshCount != 0u ||
            cachedInterior.scene.sourceInstanceCount != 0u ||
            cachedInterior.scene.sourceLandscapeCellCount != 0u ||
            cachedInterior.scene.sourceWaterPatchCount != 0u ||
            cachedInterior.scene.packedVertices.empty() ||
            cachedInterior.scene.packedIndices.empty() ||
            cachedInterior.scene.packedDraws.empty()) {
            VOX_LOGW("app") << "stale cached interior " << cachePath.string()
                            << "; regenerate it with odai_balmora_interior_cache --force";
            cachedInterior.scene = {};
            return false;
        }
        if (!cachedInterior.collision.buildFromPackedScene(cachedInterior.scene)) {
            VOX_LOGW("app") << "failed to build collision for cached interior " << cachePath.string();
            cachedInterior.scene = {};
            return false;
        }
        if (cachedInterior.doors.empty()) {
            VOX_LOGW("app") << "cached interior " << door.destination.destinationCell
                            << " has no exit doors in " << balmoraDoorCachePath().string()
                            << "; regenerate the door manifest with odai_balmora_interior_cache";
        }
        cachedInterior.sceneLoaded = true;
        const auto interiorLoadMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - interiorLoadStart).count();
        VOX_LOGI("app") << "loaded cached interior '" << door.destination.destinationCell
                        << "' on demand in " << interiorLoadMs << " ms";
    }

    if (!m_renderer.uploadImportedScene(cachedInterior.scene)) {
        VOX_LOGW("app") << "failed to upload interior scene for " << door.destination.destinationCell;
        return false;
    }
    m_renderer.setImportedSceneInteriorMode(true);

    m_importedScene = cachedInterior.scene;
    m_gpuSceneAsset = {};
    m_gpuSceneRuntime = {};
    m_importedSceneCollision = cachedInterior.collision;
    m_currentMorrowindInteriorCell = destinationCell;
    m_balmoraGuards.clear();
    m_balmoraGuardPrototype = {};
    m_balmoraNavmesh.clear();
    m_balmoraGuardFrameVertices.clear();
    m_balmoraGuardFrameIndices.clear();
    m_balmoraGuardFrameDraws.clear();
    if (balmoraGuardsEnabledByEnvironment()) {
        initializeBalmoraGuards();
    }

    m_camera.x = door.destination.position[0];
    m_camera.y = door.destination.position[1] + kMorrowindInteriorSpawnEyeHeight;
    m_camera.z = door.destination.position[2];
    m_camera.yawDegrees = odai::math::degrees(door.destination.rotationRadians[2]);
    m_camera.pitchDegrees = 0.0f;
    m_camera.velocityX = 0.0f;
    m_camera.velocityY = 0.0f;
    m_camera.velocityZ = 0.0f;
    m_camera.onGround = false;
    m_cameraPrevious = m_camera;

    VOX_LOGI("app") << "entered interior '" << door.destination.destinationCell
                    << "' from door " << door.refId
                    << " (renderVertices=" << m_importedScene.packedVertices.size()
                    << ", renderIndices=" << m_importedScene.packedIndices.size()
                    << ", collisionTriangles=" << m_importedSceneCollision.stats().triangleCount << ")";
    return true;
}

bool App::leaveMorrowindInterior(const odai::importer::MorrowindDoorReference& door) {
    if (!m_balmoraExteriorCached) {
        VOX_LOGW("app") << "failed to leave interior: Balmora exterior scene is not cached";
        return false;
    }

    if (!m_renderer.uploadGpuScene(m_balmoraExteriorGpuSceneAsset)) {
        VOX_LOGW("app") << "failed to upload Balmora exterior scene while leaving interior";
        return false;
    }
    m_renderer.setImportedSceneInteriorMode(false);

    m_importedScene = m_balmoraExteriorScene;
    m_gpuSceneAsset = m_balmoraExteriorGpuSceneAsset;
    m_gpuSceneRuntime = odai::importer::createGpuSceneRuntime(m_gpuSceneAsset);
    odai::importer::rebuildGpuSceneWorldTransforms(m_gpuSceneRuntime);
    m_importedSceneCollision = m_balmoraExteriorCollision;
    m_currentMorrowindInteriorCell.clear();
    m_balmoraGuards.clear();
    m_balmoraGuardPrototype = {};
    m_balmoraNavmesh.clear();
    m_balmoraGuardFrameVertices.clear();
    m_balmoraGuardFrameIndices.clear();
    m_balmoraGuardFrameDraws.clear();

    m_camera.x = door.destination.position[0];
    m_camera.y = door.destination.position[1] + kImportedPlayerEyeHeight;
    m_camera.z = door.destination.position[2];
    m_camera.yawDegrees = odai::math::degrees(door.destination.rotationRadians[2]);
    m_camera.pitchDegrees = 0.0f;
    m_camera.velocityX = 0.0f;
    m_camera.velocityY = 0.0f;
    m_camera.velocityZ = 0.0f;
    m_camera.onGround = false;
    m_cameraPrevious = m_camera;

    VOX_LOGI("app") << "left interior '" << door.sourceCell
                    << "' through door " << door.refId
                    << " to " << door.destination.destinationCell;
    return true;
}

void App::initializeBalmoraGuards() {
    m_balmoraGuards.clear();
    m_balmoraGuardPrototype = {};
    m_balmoraNavmesh.clear();

    if (m_importedScene.sourceTag != "morrowind_balmora") {
        return;
    }

    const std::optional<std::filesystem::path> dataFilesPath = findMorrowindDataFilesPath();
    if (!dataFilesPath.has_value()) {
        VOX_LOGW("app") << "Balmora guards disabled: Morrowind Data Files not found; set "
                         << kMorrowindDataFilesEnvVar;
        return;
    }

    std::unordered_map<std::string, std::uint32_t> textureSlotByPath;
    for (std::uint32_t textureIndex = 0; textureIndex < m_importedScene.textures.size(); ++textureIndex) {
        textureSlotByPath.emplace(
            lowerPathCopy(m_importedScene.textures[textureIndex].sourcePath),
            textureIndex);
    }
    auto addActorTextureSlot = [&](const std::string& sourcePath) -> std::uint32_t {
        if (sourcePath.empty()) {
            return std::numeric_limits<std::uint32_t>::max();
        }
        const std::string key = lowerPathCopy(sourcePath);
        const auto existing = textureSlotByPath.find(key);
        if (existing != textureSlotByPath.end()) {
            return existing->second;
        }

        odai::importer::ImportedSceneTexture texture{};
        if (!odai::importer::loadMorrowindTexture(*dataFilesPath, sourcePath, texture)) {
            VOX_LOGW("app") << "Balmora guard texture skipped: " << sourcePath
                             << " (" << odai::importer::getImportedSceneLastError() << ")";
            return std::numeric_limits<std::uint32_t>::max();
        }
        const std::uint32_t textureIndex = static_cast<std::uint32_t>(m_importedScene.textures.size());
        textureSlotByPath.emplace(lowerPathCopy(texture.sourcePath), textureIndex);
        m_importedScene.textures.push_back(std::move(texture));
        return textureIndex;
    };

    auto appendNifToPrototype = [&](std::string_view relativeModelPath) {
        std::filesystem::path nifPath = *dataFilesPath / "Meshes" / std::filesystem::path(std::string(relativeModelPath));
        if (!std::filesystem::exists(nifPath)) {
            VOX_LOGW("app") << "Balmora guard mesh missing: " << nifPath.string();
            return;
        }

        odai::importer::ImportedNifResult nifResult{};
        std::string nifError;
        if (!odai::importer::loadMorrowindStaticNif(nifPath, nifResult, nifError)) {
            VOX_LOGW("app") << "Balmora guard mesh skipped: " << nifPath.string()
                             << " (" << nifError << ")";
            return;
        }

        if (nifResult.mesh.parts.empty()) {
            odai::importer::ImportedSceneMeshPart part{};
            part.firstIndex = 0u;
            part.indexCount = static_cast<std::uint32_t>(nifResult.mesh.indices.size());
            part.textureIndex = addActorTextureSlot(nifResult.diffuseTexturePath);
            part.alphaTest = nifResult.alphaTest;
            nifResult.mesh.parts.push_back(part);
        } else {
            for (std::size_t partIndex = 0; partIndex < nifResult.mesh.parts.size(); ++partIndex) {
                odai::importer::ImportedSceneMeshPart& part = nifResult.mesh.parts[partIndex];
                std::string texturePath = nifResult.diffuseTexturePath;
                if (partIndex < nifResult.partDiffuseTexturePaths.size() &&
                    !nifResult.partDiffuseTexturePaths[partIndex].empty()) {
                    texturePath = nifResult.partDiffuseTexturePaths[partIndex];
                }
                part.textureIndex = addActorTextureSlot(texturePath);
            }
        }

        for (const odai::importer::ImportedSceneMeshPart& part : nifResult.mesh.parts) {
            const std::uint32_t firstIndex = part.firstIndex;
            const std::uint32_t indexEnd = std::min<std::uint32_t>(
                firstIndex + part.indexCount,
                static_cast<std::uint32_t>(nifResult.mesh.indices.size()));
            if (firstIndex >= indexEnd) {
                continue;
            }

            const std::uint32_t drawFirstIndex =
                static_cast<std::uint32_t>(m_balmoraGuardPrototype.indices.size());
            std::vector<std::uint32_t> remap(
                nifResult.mesh.vertices.size(),
                std::numeric_limits<std::uint32_t>::max());
            for (std::uint32_t indexOffset = firstIndex; indexOffset < indexEnd; ++indexOffset) {
                const std::uint32_t sourceVertexIndex = nifResult.mesh.indices[indexOffset];
                if (sourceVertexIndex >= nifResult.mesh.vertices.size()) {
                    continue;
                }
                std::uint32_t& mappedIndex = remap[sourceVertexIndex];
                if (mappedIndex == std::numeric_limits<std::uint32_t>::max()) {
                    const odai::importer::ImportedSceneVertex& sourceVertex =
                        nifResult.mesh.vertices[sourceVertexIndex];
                    odai::importer::ImportedScenePackedVertex vertex{};
                    std::copy(std::begin(sourceVertex.position), std::end(sourceVertex.position), std::begin(vertex.position));
                    std::copy(std::begin(sourceVertex.normal), std::end(sourceVertex.normal), std::begin(vertex.normal));
                    std::copy(std::begin(sourceVertex.uv), std::end(sourceVertex.uv), std::begin(vertex.uv));
                    vertex.color[0] = 0.78f;
                    vertex.color[1] = 0.72f;
                    vertex.color[2] = 0.62f;
                    vertex.textureIndex = part.textureIndex;
                    vertex.flags = part.alphaTest ? 1u : 0u;
                    mappedIndex = static_cast<std::uint32_t>(m_balmoraGuardPrototype.vertices.size());
                    m_balmoraGuardPrototype.vertices.push_back(vertex);
                }
                m_balmoraGuardPrototype.indices.push_back(mappedIndex);
            }

            const std::uint32_t drawIndexCount =
                static_cast<std::uint32_t>(m_balmoraGuardPrototype.indices.size()) - drawFirstIndex;
            if (drawIndexCount == 0u) {
                continue;
            }
            odai::importer::ImportedScenePackedDraw draw{};
            draw.firstIndex = drawFirstIndex;
            draw.indexCount = drawIndexCount;
            m_balmoraGuardPrototype.draws.push_back(draw);
        }
    };

    constexpr std::array<std::string_view, 13> kGuardModelParts = {
        "b/B_N_Dark Elf_M_Neck.NIF",
        "b/B_N_Dark Elf_M_Head_01.NIF",
        "b/B_N_Dark Elf_M_Hair_17.nif",
        "a/A_Bonemold_Armun_An_helm.nif",
        "a/A_Bonemold_Cuirass_C.NIF",
        "a/A_Bonemold_Armun_An_UA.nif",
        "a/A_Bonemold_Armun_An_CL.nif",
        "a/A_Bonemold_Bracer_W.nif",
        "a/A_Bonemold_Greaves_G.nif",
        "a/A_Bonemold_Greaves_UL.nif",
        "a/A_Bonemold_Greaves_K.nif",
        "a/A_Bonemold_Boots_F.nif",
        "a/A_Bonemold_Boots_A.nif"
    };
    for (const std::string_view modelPart : kGuardModelParts) {
        appendNifToPrototype(modelPart);
    }
    if (m_balmoraGuardPrototype.vertices.empty() ||
        m_balmoraGuardPrototype.indices.empty() ||
        m_balmoraGuardPrototype.draws.empty()) {
        VOX_LOGW("app") << "Balmora guards disabled: no guard actor geometry imported";
        m_balmoraGuardPrototype = {};
        return;
    }

    odai::world::NavmeshSettings navmeshSettings{};
    navmeshSettings.agentRadius = 32.0f;
    navmeshSettings.agentHeight = 128.0f;
    navmeshSettings.maxClimb = 72.0f;
    navmeshSettings.maxSlopeDegrees = 52.0f;
    navmeshSettings.nearestPointMaxDistance = 1200.0f;
    navmeshSettings.edgeMergeEpsilon = 0.35f;
    if (!m_balmoraNavmesh.buildFromGpuSceneAsset(m_gpuSceneAsset, navmeshSettings)) {
        VOX_LOGW("app") << "Balmora guards disabled: navmesh build failed";
        m_balmoraGuardPrototype = {};
        return;
    }

    std::vector<odai::math::Vector3> guardSpawns;
    for (const odai::importer::ImportedSceneCellRef& ref : m_importedScene.unresolvedRefs) {
        if (lowerPathCopy(ref.refId) != "hlaalu guard_outside") {
            continue;
        }
        guardSpawns.push_back({ref.position[0], ref.position[2], ref.position[1]});
    }
    if (guardSpawns.empty()) {
        guardSpawns = {
            {-17051.3105f, 228.6063f, -14461.3164f},
            {-20190.7988f, 169.6947f, -13064.7422f},
            {-22458.5312f, 576.0219f, -14094.3115f},
            {-24606.0293f, 971.5856f, -12884.6289f}
        };
    }

    const auto snapToNavmesh = [&](const odai::math::Vector3& point, odai::math::Vector3& outPoint) {
        return m_balmoraNavmesh.findNearestPoint(
            {point.x, point.y + kBalmoraGuardNavmeshProbeHeight, point.z},
            outPoint);
    };

    std::vector<odai::math::Vector3> route;
    constexpr std::array<odai::math::Vector3, 8> kRawRoute = {
        odai::math::Vector3{-17051.3105f, 228.6063f, -14461.3164f},
        odai::math::Vector3{-18470.0f, 260.0f, -14440.0f},
        odai::math::Vector3{-20190.7988f, 169.6947f, -13064.7422f},
        odai::math::Vector3{-21340.0f, 330.0f, -13780.0f},
        odai::math::Vector3{-22458.5312f, 576.0219f, -14094.3115f},
        odai::math::Vector3{-23640.0f, 810.0f, -13520.0f},
        odai::math::Vector3{-24606.0293f, 971.5856f, -12884.6289f},
        odai::math::Vector3{-21760.0f, 640.0f, -15440.0f}
    };
    for (const odai::math::Vector3& rawPoint : kRawRoute) {
        odai::math::Vector3 snapped{};
        if (snapToNavmesh(rawPoint, snapped)) {
            route.push_back(snapped);
        }
    }

    for (std::size_t guardIndex = 0; guardIndex < guardSpawns.size(); ++guardIndex) {
        odai::math::Vector3 snappedSpawn{};
        if (!snapToNavmesh(guardSpawns[guardIndex], snappedSpawn)) {
            continue;
        }
        BalmoraGuardAgent guard{};
        guard.position = snappedSpawn;
        guard.previousPosition = snappedSpawn;
        guard.route = route;
        guard.routeIndex = route.empty() ? 0u : ((guardIndex * 2u) % route.size());
        guard.speed = 84.0f + (static_cast<float>(guardIndex % 3u) * 9.0f);
        guard.walkPhase = static_cast<float>(guardIndex) * 1.37f;
        guard.previousWalkPhase = guard.walkPhase;
        m_balmoraGuards.push_back(std::move(guard));
    }

    const odai::world::Navmesh::Stats navmeshStats = m_balmoraNavmesh.stats();
    VOX_LOGI("app") << "Balmora guards enabled"
                    << " guards=" << m_balmoraGuards.size()
                    << " actorVertices=" << m_balmoraGuardPrototype.vertices.size()
                    << " actorIndices=" << m_balmoraGuardPrototype.indices.size()
                    << " navmeshWalkable=" << navmeshStats.walkableTriangleCount
                    << " navmeshLinks=" << navmeshStats.linkCount;
}

void App::updateBalmoraGuards(float dt) {
    if (m_balmoraGuards.empty() || m_balmoraGuardPrototype.vertices.empty() || m_balmoraNavmesh.empty()) {
        return;
    }

    auto chooseNextPath = [&](BalmoraGuardAgent& guard) {
        guard.path.clear();
        guard.pathIndex = 0;
        if (guard.route.empty()) {
            return;
        }
        for (std::size_t attempt = 0; attempt < guard.route.size(); ++attempt) {
            const odai::math::Vector3 target = guard.route[guard.routeIndex];
            guard.routeIndex = (guard.routeIndex + 1u) % guard.route.size();
            const odai::math::Vector3 delta = target - guard.position;
            if ((delta.x * delta.x) + (delta.z * delta.z) <
                (kBalmoraGuardRouteReachRadius * kBalmoraGuardRouteReachRadius)) {
                continue;
            }
            guard.path.push_back({guard.position});
            guard.path.push_back({target});
            guard.pathIndex = 1u;
            return;
        }
    };

    for (BalmoraGuardAgent& guard : m_balmoraGuards) {
        guard.previousPosition = guard.position;
        guard.previousYawRadians = guard.yawRadians;
        guard.previousWalkPhase = guard.walkPhase;

        if (guard.path.empty() || guard.pathIndex >= guard.path.size()) {
            chooseNextPath(guard);
        }
        if (guard.path.empty() || guard.pathIndex >= guard.path.size()) {
            continue;
        }

        float remainingStep = std::max(guard.speed * std::max(dt, 0.0f), 0.0f);
        while (remainingStep > 0.0f && guard.pathIndex < guard.path.size()) {
            const odai::math::Vector3 target = guard.path[guard.pathIndex].position;
            const odai::math::Vector3 delta{target.x - guard.position.x, 0.0f, target.z - guard.position.z};
            const float distance = std::sqrt((delta.x * delta.x) + (delta.z * delta.z));
            if (distance <= kBalmoraGuardRouteReachRadius) {
                ++guard.pathIndex;
                if (guard.pathIndex >= guard.path.size()) {
                    chooseNextPath(guard);
                    break;
                }
                continue;
            }

            const float moveDistance = std::min(remainingStep, distance);
            const odai::math::Vector3 direction = delta / std::max(distance, 0.001f);
            odai::math::Vector3 moved{
                guard.position.x + (direction.x * moveDistance),
                guard.position.y,
                guard.position.z + (direction.z * moveDistance)
            };
            odai::math::Vector3 snapped{};
            if (m_balmoraNavmesh.findNearestPoint(
                    {moved.x, moved.y + kBalmoraGuardNavmeshProbeHeight, moved.z},
                    snapped)) {
                moved = snapped;
            }
            guard.position = moved;
            guard.yawRadians = std::atan2(direction.x, direction.z);
            guard.walkPhase += moveDistance * kBalmoraGuardWalkCycleScale;
            remainingStep -= moveDistance;
            if (moveDistance < 0.001f) {
                break;
            }
        }
    }
}

void App::rebuildBalmoraGuardRenderFrame(float simulationAlpha) {
    m_balmoraGuardFrameVertices.clear();
    m_balmoraGuardFrameIndices.clear();
    m_balmoraGuardFrameDraws.clear();
    if (m_balmoraGuards.empty() || m_balmoraGuardPrototype.vertices.empty()) {
        return;
    }

    const float alpha = std::clamp(simulationAlpha, 0.0f, 1.0f);
    m_balmoraGuardFrameVertices.reserve(m_balmoraGuardPrototype.vertices.size() * m_balmoraGuards.size());
    m_balmoraGuardFrameIndices.reserve(m_balmoraGuardPrototype.indices.size() * m_balmoraGuards.size());
    m_balmoraGuardFrameDraws.reserve(m_balmoraGuardPrototype.draws.size() * m_balmoraGuards.size());

    const auto lerpAngle = [](float from, float to, float t) {
        float delta = std::fmod(to - from, kTwoPi);
        if (delta <= -odai::math::kPi) {
            delta += kTwoPi;
        } else if (delta > odai::math::kPi) {
            delta -= kTwoPi;
        }
        return from + (delta * t);
    };
    const auto rotateX = [](odai::math::Vector3 value, float pivotY, float pivotZ, float radians) {
        const float c = std::cos(radians);
        const float s = std::sin(radians);
        const float y = value.y - pivotY;
        const float z = value.z - pivotZ;
        value.y = pivotY + (y * c) - (z * s);
        value.z = pivotZ + (y * s) + (z * c);
        return value;
    };
    const auto applyWalkSkin = [&](odai::math::Vector3& position,
                                   odai::math::Vector3& normal,
                                   float walkPhase) {
        const float side = position.x < 0.0f ? -1.0f : 1.0f;
        if (position.y < 82.0f) {
            const float legWeight = std::clamp((82.0f - position.y) / 82.0f, 0.0f, 1.0f) *
                std::clamp((std::abs(position.x) - 2.0f) / 18.0f, 0.0f, 1.0f);
            const float legSwing = std::sin(walkPhase + (side > 0.0f ? odai::math::kPi : 0.0f)) * 0.34f * legWeight;
            position = rotateX(position, 72.0f, 0.0f, legSwing);
            normal = rotateX(normal, 0.0f, 0.0f, legSwing * 0.65f);
        }
        if (position.y > 42.0f && position.y < 128.0f) {
            const float armWeight = std::clamp((std::abs(position.x) - 18.0f) / 18.0f, 0.0f, 1.0f) *
                std::clamp((128.0f - position.y) / 58.0f, 0.0f, 1.0f);
            const float armSwing = std::sin(walkPhase + (side > 0.0f ? 0.0f : odai::math::kPi)) * 0.28f * armWeight;
            position = rotateX(position, 104.0f, 0.0f, armSwing);
            normal = rotateX(normal, 0.0f, 0.0f, armSwing * 0.65f);
        }
    };

    for (const BalmoraGuardAgent& guard : m_balmoraGuards) {
        const odai::math::Vector3 position =
            guard.previousPosition + ((guard.position - guard.previousPosition) * alpha);
        const float yawRadians = lerpAngle(guard.previousYawRadians, guard.yawRadians, alpha);
        const float walkPhase = guard.previousWalkPhase + ((guard.walkPhase - guard.previousWalkPhase) * alpha);
        const float c = std::cos(yawRadians);
        const float s = std::sin(yawRadians);
        const float bob = std::abs(std::sin(walkPhase * 2.0f)) * 1.6f;

        const std::uint32_t baseVertex = static_cast<std::uint32_t>(m_balmoraGuardFrameVertices.size());
        for (const odai::importer::ImportedScenePackedVertex& sourceVertex : m_balmoraGuardPrototype.vertices) {
            odai::math::Vector3 localPosition{
                sourceVertex.position[0],
                sourceVertex.position[1],
                sourceVertex.position[2]
            };
            odai::math::Vector3 localNormal{
                sourceVertex.normal[0],
                sourceVertex.normal[1],
                sourceVertex.normal[2]
            };
            applyWalkSkin(localPosition, localNormal, walkPhase);

            odai::importer::ImportedScenePackedVertex vertex = sourceVertex;
            vertex.position[0] = position.x + (localPosition.x * c) + (localPosition.z * s);
            vertex.position[1] = position.y + localPosition.y + bob;
            vertex.position[2] = position.z + (-localPosition.x * s) + (localPosition.z * c);
            const odai::math::Vector3 worldNormal = odai::math::normalize({
                (localNormal.x * c) + (localNormal.z * s),
                localNormal.y,
                (-localNormal.x * s) + (localNormal.z * c)
            });
            vertex.normal[0] = worldNormal.x;
            vertex.normal[1] = worldNormal.y;
            vertex.normal[2] = worldNormal.z;
            m_balmoraGuardFrameVertices.push_back(vertex);
        }

        const std::uint32_t firstActorIndex = static_cast<std::uint32_t>(m_balmoraGuardFrameIndices.size());
        for (const std::uint32_t sourceIndex : m_balmoraGuardPrototype.indices) {
            m_balmoraGuardFrameIndices.push_back(baseVertex + sourceIndex);
        }
        for (const odai::importer::ImportedScenePackedDraw& sourceDraw : m_balmoraGuardPrototype.draws) {
            odai::importer::ImportedScenePackedDraw draw{};
            draw.firstIndex = firstActorIndex + sourceDraw.firstIndex;
            draw.indexCount = sourceDraw.indexCount;
            m_balmoraGuardFrameDraws.push_back(draw);
        }
    }
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
    const odai::render::CameraPose cameraPose{
        m_cameraPrevious.x + ((m_camera.x - m_cameraPrevious.x) * renderAlpha),
        m_cameraPrevious.y + ((m_camera.y - m_cameraPrevious.y) * renderAlpha),
        m_cameraPrevious.z + ((m_camera.z - m_cameraPrevious.z) * renderAlpha),
        lerpWrappedDegrees(m_cameraPrevious.yawDegrees, m_camera.yawDegrees, renderAlpha),
        m_cameraPrevious.pitchDegrees + ((m_camera.pitchDegrees - m_cameraPrevious.pitchDegrees) * renderAlpha),
        m_camera.fovDegrees
    };
    m_visibleChunkIndices.clear();
    m_renderer.setSpatialQueryStats(false, odai::world::SpatialQueryStats{}, 0u);
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

    const bool activateDoorPressedThisStep =
        m_importedSceneDemoEnabled &&
        !isAnyUiVisible() &&
        glfwGetKey(m_window, GLFW_KEY_R) == GLFW_PRESS &&
        !m_wasActivateDoorDown;
    m_wasActivateDoorDown = glfwGetKey(m_window, GLFW_KEY_R) == GLFW_PRESS;
    if (activateDoorPressedThisStep && tryActivateBalmoraDoor()) {
        return;
    }

    float mouseDeltaX = m_pendingMouseDeltaX;
    float mouseDeltaY = m_pendingMouseDeltaY;
    m_pendingMouseDeltaX = 0.0f;
    m_pendingMouseDeltaY = 0.0f;
    // Strategy map is cursor-driven: never rotate the camera from raw mouse motion.
    if (m_strategyMapMode) {
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

    if (m_importedSceneDemoEnabled && (m_hoverEnabled || m_importedSceneCollision.empty())) {
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

    if (m_importedSceneDemoEnabled) {
        moveDirection += forward * moveForwardInput;
        moveDirection += right * moveRightInput;

        const float moveLengthSq = odai::math::lengthSquared(moveDirection);
        if (moveLengthSq > 0.0f) {
            moveDirection /= std::sqrt(moveLengthSq);
        }

        float moveSpeed = kImportedSceneWalkSpeed;
        if (m_input.sprintDown) {
            moveSpeed *= kImportedSceneWalkSprintSpeedMultiplier;
        }

        m_camera.velocityX = moveDirection.x * moveSpeed;
        m_camera.velocityZ = moveDirection.z * moveSpeed;
        if (m_input.moveUp && m_camera.onGround) {
            m_camera.velocityY = kImportedSceneJumpSpeed;
            m_camera.onGround = false;
        }
        m_camera.velocityY = std::max(
            m_camera.velocityY + (kImportedSceneGravity * dt),
            kImportedSceneMaxFallSpeed);
        resolveImportedScenePlayerCollisions(dt);
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

void App::resolveImportedScenePlayerCollisions(float dt) {
    const float totalDx = m_camera.velocityX * dt;
    const float totalDy = m_camera.velocityY * dt;
    const float totalDz = m_camera.velocityZ * dt;
    const float maxDelta = std::max({std::fabs(totalDx), std::fabs(totalDy), std::fabs(totalDz)});
    const int steps = std::max(1, static_cast<int>(std::ceil(maxDelta / kImportedPlayerCollisionSubstepDistance)));
    const float stepDx = totalDx / static_cast<float>(steps);
    const float stepDy = totalDy / static_cast<float>(steps);
    const float stepDz = totalDz / static_cast<float>(steps);

    bool groundedThisFrame = false;

    auto feetY = [&]() {
        return m_camera.y - kImportedPlayerEyeHeight;
    };
    auto topY = [&]() {
        return m_camera.y + kImportedPlayerTopOffset;
    };

    auto hasCeilingAtCurrentPosition = [&]() {
        odai::world::ImportedSceneCollision::CeilingHit ceilingHit{};
        return m_importedSceneCollision.findCeiling(
            m_camera.x,
            topY(),
            m_camera.z,
            kImportedPlayerRadius,
            kImportedPlayerCollisionSkin,
            ceilingHit);
    };

    auto tryStepUp = [&]() {
        if (!m_camera.onGround && !groundedThisFrame) {
            return false;
        }

        odai::world::ImportedSceneCollision::GroundHit groundHit{};
        if (!m_importedSceneCollision.findGroundSupport(
                m_camera.x,
                feetY(),
                m_camera.z,
                kImportedPlayerRadius,
                kImportedPlayerCollisionSkin,
                kImportedPlayerStepHeight,
                kImportedPlayerWalkableNormalY,
                groundHit)) {
            return false;
        }

        const float currentFeetY = feetY();
        const float stepHeight = groundHit.y - currentFeetY;
        if (stepHeight < kImportedPlayerMinStepHeight ||
            stepHeight > kImportedPlayerStepHeight) {
            return false;
        }

        const float previousEyeY = m_camera.y;
        m_camera.y = groundHit.y + kImportedPlayerEyeHeight + kImportedPlayerCollisionSkin;
        if (hasCeilingAtCurrentPosition()) {
            m_camera.y = previousEyeY;
            return false;
        }

        groundedThisFrame = true;
        if (m_camera.velocityY < 0.0f) {
            m_camera.velocityY = 0.0f;
        }
        return true;
    };

    auto resolveHorizontal = [&]() {
        odai::math::Vector3 correction{};
        if (!m_importedSceneCollision.resolveHorizontalCylinder(
                m_camera.x,
                feetY(),
                m_camera.z,
                kImportedPlayerRadius,
                kImportedPlayerHeight,
                kImportedPlayerWalkableNormalY,
                correction)) {
            return;
        }

        if (tryStepUp() &&
            !m_importedSceneCollision.resolveHorizontalCylinder(
                m_camera.x,
                feetY(),
                m_camera.z,
                kImportedPlayerRadius,
                kImportedPlayerHeight,
                kImportedPlayerWalkableNormalY,
                correction)) {
            return;
        }

        m_camera.x += correction.x;
        m_camera.z += correction.z;
        if (std::fabs(correction.x) > kImportedPlayerCollisionSkin) {
            m_camera.velocityX = 0.0f;
        }
        if (std::fabs(correction.z) > kImportedPlayerCollisionSkin) {
            m_camera.velocityZ = 0.0f;
        }
    };

    auto snapToGround = [&](float maxDrop, float maxStepUp) {
        odai::world::ImportedSceneCollision::GroundHit groundHit{};
        if (!m_importedSceneCollision.findGroundSupport(
                m_camera.x,
                feetY(),
                m_camera.z,
                kImportedPlayerRadius,
                maxDrop,
                maxStepUp,
                kImportedPlayerWalkableNormalY,
                groundHit)) {
            return false;
        }

        m_camera.y = groundHit.y + kImportedPlayerEyeHeight + kImportedPlayerCollisionSkin;
        groundedThisFrame = true;
        if (m_camera.velocityY < 0.0f) {
            m_camera.velocityY = 0.0f;
        }
        return true;
    };

    for (int step = 0; step < steps; ++step) {
        if (stepDx != 0.0f) {
            m_camera.x += stepDx;
            resolveHorizontal();
        }
        if (stepDz != 0.0f) {
            m_camera.z += stepDz;
            resolveHorizontal();
        }

        if (m_camera.onGround || groundedThisFrame) {
            snapToGround(kImportedPlayerGroundSnapDistance, kImportedPlayerStepHeight);
        }

        if (stepDy != 0.0f) {
            m_camera.y += stepDy;
        }

        if (stepDy > 0.0f) {
            odai::world::ImportedSceneCollision::CeilingHit ceilingHit{};
            if (m_importedSceneCollision.findCeiling(
                    m_camera.x,
                    topY(),
                    m_camera.z,
                    kImportedPlayerRadius,
                    std::fabs(stepDy) + kImportedPlayerCollisionSkin,
                    ceilingHit)) {
                m_camera.y = ceilingHit.y - kImportedPlayerTopOffset - kImportedPlayerCollisionSkin;
                m_camera.velocityY = 0.0f;
            }
        } else {
            snapToGround(
                std::fabs(stepDy) + kImportedPlayerGroundSnapDistance,
                std::fabs(stepDy) + kImportedPlayerCollisionSkin);
        }
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
    // DPI scale: framebuffer pixels per logical pixel. On a 125%-scaled display the
    // framebuffer is 1.25× the GLFW window size, so all hardcoded pixel values must
    // be multiplied by this factor so the UI looks the same physical size on screen.
    int windowW = 0;
    glfwGetWindowSize(m_window, &windowW, nullptr);
    const float s = (windowW > 0) ? viewW / static_cast<float>(windowW) : 1.0f;
    m_uiScale = s;
    VOX_LOGI("ui") << "HUD setup: framebuffer=" << viewW << "x" << viewH
                   << " window=" << windowW << " dpiScale=" << s;

    const float kBaseFontPx = 22.0f;

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
        VOX_LOGI("ui") << "fonts loaded: regular=yes bold=" << (m_uiFonts.bold ? "yes" : "no")
                       << " italic=" << (m_uiFonts.italic ? "yes" : "no");
    }
    const odai::ui::FontSet& fonts = m_uiFonts;

    // Register the procedural 9-slice window-frame texture and build the slice
    // (16-texel corners of a 64-texel source, drawn at 16*dpi destination corners).
    constexpr int kFrameTexSize = 64;
    const std::vector<std::uint8_t> framePixels = makeWindowFrameRgba(kFrameTexSize);
    m_windowFrameTexture = m_renderer.registerUiTextureRgba8(framePixels.data(), kFrameTexSize, kFrameTexSize);
    odai::ui::UiNineSlice windowFrame{};
    windowFrame.textureId = m_windowFrameTexture;
    windowFrame.uv = odai::ui::UiRect{0.0f, 0.0f, 1.0f, 1.0f};
    const float frameCornerTex = 16.0f;
    const float frameCornerDst = frameCornerTex * s;
    windowFrame.borderLeftPx = windowFrame.borderRightPx = frameCornerDst;
    windowFrame.borderTopPx = windowFrame.borderBottomPx = frameCornerDst;
    const float frameUvBorder = frameCornerTex / static_cast<float>(kFrameTexSize);
    windowFrame.uvBorderLeft = windowFrame.uvBorderRight = frameUvBorder;
    windowFrame.uvBorderTop = windowFrame.uvBorderBottom = frameUvBorder;
    const bool windowFrameReady = m_windowFrameTexture != odai::ui::kUiNoTexture;

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

    // Load civilization empire icons atlas (12 civs, 4×3, 128px).
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
    // Load unit icons atlas (9 units, 3×3 grid, 128px per icon).
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

    // Load civilization leader portraits atlas (12 leaders, 4×3, 256px).
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
    // and the parchment frame. Both live in the same 1024×1536 texture atlas.
    odai::ui::UiNineSlice darkPanelSlice{};
    odai::ui::UiNineSlice parchmentSlice{};
    bool darkSliceReady = false;
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
                // Dark ornate panel: tight sprite bounds x=27..695, y=51..403.
                // (The old UVs started at 0,0 and ran to 670x450, pulling in the
                // transparent margin and the next sprite — corners floated inward
                // with transparent gaps along the top/right edges.) 74px corners
                // contain the top-left medallion and the bottom flourishes.
                darkPanelSlice = makeSlice(27.0f, 51.0f, 695.0f, 403.0f,
                                           74.0f, 74.0f, 74.0f, 74.0f);
                darkSliceReady = true;
                // Parchment panel: tight body bounds x=731..993, y=52..411. The body
                // starts at x=731 — a small decorative finial sits detached at
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

    // Top resource toolbar — a full-width strip of icon+value badges (gold,
    // science, culture, faith, food, production) in the style of a 4X game's
    // status bar. Other top-anchored panels are pushed down by its height.
    const float kToolbarH = 40.0f * s;
    {
        using TB = odai::ui::Toolbar;
        auto tb = std::make_unique<TB>(fonts.regular);
        tb->setRect(odai::ui::UiRect::fromXYWH(0.0f, 0.0f, viewW, kToolbarH));
        tb->paddingXPx = 16.0f * s;
        tb->itemGapPx  = 26.0f * s;
        tb->iconGapPx  = 8.0f * s;
        tb->iconScale  = 0.68f;
        const odai::ui::UiColor textCol{0.92f, 0.95f, 0.97f, 1.0f};
        m_tbScienceItem = tb->addItem(TB::IconKind::Science, {0.31f, 0.64f, 0.82f, 1.0f}, "+14.4", textCol);
        tb->addItem(TB::IconKind::Culture, {0.69f, 0.44f, 0.84f, 1.0f}, "+8.2", textCol);
        m_tbGoldItem = tb->addItem(TB::IconKind::Coin, {0.94f, 0.75f, 0.25f, 1.0f}, "240  +39", textCol);
        m_tbFaithItem = tb->addItem(TB::IconKind::Faith, {0.90f, 0.88f, 0.96f, 1.0f}, "+11.5", textCol);
        tb->addItem(TB::IconKind::Food, {0.49f, 0.78f, 0.31f, 1.0f}, "+5.6", textCol);
        tb->addItem(TB::IconKind::Production, {0.85f, 0.54f, 0.23f, 1.0f}, "+6.7", textCol);
        m_toolbar = static_cast<TB*>(root->addChild(std::move(tb)));

        // Right-aligned turn readout on the toolbar.
        auto turn = std::make_unique<odai::ui::Label>(fonts, "");
        turn->align = odai::ui::UiTextAlign::Right;
        turn->setRect(odai::ui::UiRect::fromXYWH(viewW - 240.0f * s, (kToolbarH - 24.0f * s) * 0.5f,
                                                 224.0f * s, 24.0f * s));
        m_toolbarTurnLabel = static_cast<odai::ui::Label*>(m_toolbar->addChild(std::move(turn)));
    }

    // Bottom HUD bar — full-width strip at the bottom of the screen.
    const float kBarH = 64.0f * s;
    const float barY = viewH - kBarH;
    auto bar = std::make_unique<odai::ui::Panel>();
    bar->setRect(odai::ui::UiRect::fromXYWH(0.0f, barY, viewW, kBarH));
    bar->background  = odai::ui::UiColor{0.04f, 0.06f, 0.09f, 0.92f};
    bar->borderColor = odai::ui::UiColor{0.75f, 0.62f, 0.34f, 0.40f};

    auto turnLabel = std::make_unique<odai::ui::Label>(fonts, "Turn 1");
    turnLabel->setRect(odai::ui::UiRect::fromXYWH(20.0f * s, barY + 12.0f * s, 200.0f * s, 40.0f * s));
    m_hudTurnLabel = turnLabel.get();
    bar->addChild(std::move(turnLabel));

    auto statsLabel = std::make_unique<odai::ui::Label>(fonts, "");
    statsLabel->setRect(odai::ui::UiRect::fromXYWH(viewW * 0.5f - 160.0f * s, barY + 12.0f * s, 320.0f * s, 40.0f * s));
    statsLabel->align = odai::ui::UiTextAlign::Center;
    m_hudStatsLabel = statsLabel.get();
    bar->addChild(std::move(statsLabel));

    auto endTurnBtn = std::make_unique<odai::ui::Button>(fonts.regular, "End Turn", [this]() {
        ++m_currentTurn;
        VOX_LOGI("ui") << "turn advanced to " << m_currentTurn;
    });
    endTurnBtn->setRect(odai::ui::UiRect::fromXYWH(viewW - 172.0f * s, barY + 10.0f * s, 152.0f * s, 44.0f * s));
    endTurnBtn->cornerRadiusPx = 10.0f * s;  // vector (SDF) rounded corners, DPI-scaled
    endTurnBtn->glowSizePx = 16.0f * s;      // mouse-over glow
    bar->addChild(std::move(endTurnBtn));
    root->addChild(std::move(bar));

    // Left command panel. When the dark ornate frame is available the panel is enlarged
    // to accommodate the 95px top and 85px bottom border artwork; content is inset
    // accordingly. Falls back to the solid slate-gray panel if the sheet is missing.
    const float kPanelW = darkSliceReady ? 380.0f * s : 310.0f * s;
    const float kPanelH = darkSliceReady ? 440.0f * s : 258.0f * s;
    const float kPanelX = 0.0f;
    const float kPanelY = kToolbarH;  // sit below the top toolbar
    const float kCntPadX = darkSliceReady ? 74.0f * s : 12.0f * s;
    const float kCntPadY = darkSliceReady ? 104.0f * s : 10.0f * s;
    const float kCntW    = kPanelW - kCntPadX * 2.0f;
    auto panel = std::make_unique<odai::ui::Panel>();
    panel->setRect(odai::ui::UiRect::fromXYWH(kPanelX, kPanelY, kPanelW, kPanelH));
    if (darkSliceReady) {
        panel->nineSlice  = darkPanelSlice;
        panel->background = odai::ui::UiColor{1.0f, 1.0f, 1.0f, 1.0f};
    } else {
        panel->background = odai::ui::UiColor{0.27f, 0.31f, 0.35f, 1.0f};
        panel->opacity    = 0.80f;
    }

    auto title = std::make_unique<odai::ui::Label>(fonts, "<b><color=#ecd39a>Command View</color></b>");
    title->setRect(odai::ui::UiRect::fromXYWH(kPanelX + kCntPadX, kPanelY + kCntPadY, kCntW, 30.0f * s));
    panel->addChild(std::move(title));

    auto foundCityBtn = std::make_unique<odai::ui::Button>(fonts.regular, "Found City", [this]() {
        ++m_uiDemoClicks;
        VOX_LOGI("ui") << "founded city (" << m_uiDemoClicks << ")";
    });
    foundCityBtn->setRect(odai::ui::UiRect::fromXYWH(kPanelX + kCntPadX, kPanelY + kCntPadY + 42.0f * s, 200.0f * s, 46.0f * s));
    foundCityBtn->cornerRadiusPx = 10.0f * s;
    foundCityBtn->glowSizePx = 16.0f * s;
    panel->addChild(std::move(foundCityBtn));

    auto pediaBtn = std::make_unique<odai::ui::Button>(fonts.regular, "Open CivPedia", [this]() {
        if (m_civpediaWindow != nullptr) {
            m_civpediaWindow->visible = true;
        }
    });
    pediaBtn->setRect(odai::ui::UiRect::fromXYWH(kPanelX + kCntPadX, kPanelY + kCntPadY + 96.0f * s, 200.0f * s, 46.0f * s));
    pediaBtn->cornerRadiusPx = 10.0f * s;
    pediaBtn->glowSizePx = 16.0f * s;
    panel->addChild(std::move(pediaBtn));

    auto citiesLabel = std::make_unique<odai::ui::Label>(fonts, "Cities founded: 0");
    citiesLabel->setRect(odai::ui::UiRect::fromXYWH(kPanelX + kCntPadX, kPanelY + kCntPadY + 156.0f * s, kCntW, 30.0f * s));
    m_uiStatusLabel = citiesLabel.get();
    panel->addChild(std::move(citiesLabel));

    auto faithLabel = std::make_unique<odai::ui::Label>(fonts, "[icon=religion 44] <color=#c8963a>Faith  +3</color>");
    faithLabel->setRect(odai::ui::UiRect::fromXYWH(kPanelX + kCntPadX, kPanelY + kCntPadY + 196.0f * s, kCntW, 48.0f * s));
    panel->addChild(std::move(faithLabel));

    root->addChild(std::move(panel));

    // Tile info window (bottom-right, hidden until a tile is hovered).
    const float tileWinW = 250.0f * s;
    const float tileWinH = 130.0f * s;
    const float tileWinX = viewW - tileWinW - 16.0f * s;
    const float tileWinY = viewH - kBarH - tileWinH - 12.0f * s;
    auto tileWin = std::make_unique<odai::ui::Window>(fonts.regular, "Tile Info");
    tileWin->setRect(odai::ui::UiRect::fromXYWH(tileWinX, tileWinY, tileWinW, tileWinH));
    tileWin->titleBarH = odai::ui::Window::kDefaultTitleBarH * s;
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
        const float civX       = 16.0f * s;
        const float civY       = kToolbarH + 12.0f * s;
        const float civH       = 12.0f * s         // top padding
                               + 20.0f * s         // title label
                               + 6.0f  * s         // gap
                               + 3.0f * civIconSz  // 3 rows of icons
                               + 10.0f * s         // gap
                               + civPortSz          // leader portrait
                               + 6.0f  * s         // gap
                               + 18.0f * s         // name label
                               + 12.0f * s;        // bottom padding

        auto civPanel = std::make_unique<odai::ui::Panel>();
        civPanel->setRect(odai::ui::UiRect::fromXYWH(civX, civY, civW, civH));
        civPanel->cornerRadiusPx = 8.0f * s;
        civPanel->background = odai::ui::UiColor{0.08f, 0.10f, 0.13f, 0.88f};

        float cy = civY + 12.0f * s;

        // "Civilizations" title label.
        auto civTitle = std::make_unique<odai::ui::Label>(fonts.bold, "Civilizations");
        civTitle->color = odai::ui::UiColor{0.82f, 0.72f, 0.50f, 1.0f};
        civTitle->setRect(odai::ui::UiRect::fromXYWH(civX, cy, civW, 20.0f * s));
        civTitle->align = odai::ui::UiTextAlign::Center;
        civPanel->addChild(std::move(civTitle));
        cy += 20.0f * s + 6.0f * s;

        // 4×3 grid of empire icons — 12 civilizations.
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
        cy += 3.0f * civIconSz + 10.0f * s;

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
        auto nameLabel = std::make_unique<odai::ui::Label>(fonts.regular, "Caesar \xe2\x80\x94 Rome");
        nameLabel->color = odai::ui::UiColor{0.90f, 0.85f, 0.70f, 1.0f};
        nameLabel->setRect(odai::ui::UiRect::fromXYWH(civX, cy, civW, 18.0f * s));
        nameLabel->align = odai::ui::UiTextAlign::Center;
        m_civLeaderNameLabel = nameLabel.get();
        civPanel->addChild(std::move(nameLabel));

        root->addChild(std::move(civPanel));
    }

    // CivPedia window (top-right): a draggable framed window. When the parchment
    // 9-slice is loaded, it replaces the solid fill and the title bar / body colors
    // are made transparent so the parchment texture shows through. Text colors shift
    // to dark-brown so they read on the light background.
    const float pediaW = parchmentSliceReady ? 430.0f * s : 360.0f * s;
    const float pediaH = parchmentSliceReady ? 310.0f * s : 232.0f * s;
    const float pediaX = viewW - pediaW - 16.0f * s;
    const float pediaY = kToolbarH + 12.0f * s;
    // The parchment frame's top border is ornate gold, so the title bar is made
    // tall enough to seat the "CivPedia" title in the cream just below it.
    const float pediaTitleH = parchmentSliceReady ? 86.0f * s
                                                  : odai::ui::Window::kDefaultTitleBarH * s;
    auto pediaWin = std::make_unique<odai::ui::Window>(
        fonts.bold, "CivPedia",
        [this]() {
            if (m_civpediaWindow != nullptr) {
                m_civpediaWindow->visible = false;
            }
        });
    pediaWin->setRect(odai::ui::UiRect::fromXYWH(pediaX, pediaY, pediaW, pediaH));
    pediaWin->titleBarH = pediaTitleH;
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
        pediaWin->padding      = {14.0f * s, 10.0f * s};
        pediaWin->titleBarColor = odai::ui::UiColor{0.22f, 0.26f, 0.30f, 1.0f};
        pediaWin->bodyColor     = odai::ui::UiColor{0.27f, 0.31f, 0.35f, 1.0f};
        pediaWin->opacity       = 0.80f;
        if (windowFrameReady) {
            pediaWin->frame = windowFrame;
        }
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
        pediaWin->addChild(std::move(portrait));

        const float nameLabelX = pediaX + pBodyPadX + headerIconSz + 8.0f * s;
        const float nameLabelW = pediaW - pBodyPadX - headerIconSz - 8.0f * s - pBodyPadX;
        auto nameLabel = std::make_unique<odai::ui::Label>(
            fonts, "<b><color=#c06820>Spearman</color></b>\n<color=#9fa8a8><i>Ancient Melee Unit</i></color>");
        nameLabel->setRect(odai::ui::UiRect::fromXYWH(nameLabelX, headerY, nameLabelW, headerIconSz));
        pediaWin->addChild(std::move(nameLabel));
    }

    // Rich-text body — inline unit icons prefix each hoverable term.
    const std::string pediaText =
        "The <tip=Spearman \xe2\x80\x94 25 HP, melee. +100% vs mounted units.>"
        "[icon=spearman 18]<b><color=#c06820>Spearman</color></b></tip> is an <i>ancient</i> melee unit. "
        "It stands firm against <tip=Cavalry \xe2\x80\x94 fast mounted units, weak to spears.>"
        "[icon=cavalry 18]<b><color=#c06820>Cavalry</color></b></tip>, anchoring your early army. "
        "Train one in any <tip=Cities grow your empire and produce units and wonders.>"
        "<b><color=#c06820>City</color></b></tip> that has built a <i>Barracks</i>.";
    auto pediaBody = std::make_unique<odai::ui::RichTextView>(fonts, pediaText);
    pediaBody->padding = {6.0f * s, 0.0f};
    pediaBody->setRect(odai::ui::UiRect::fromXYWH(
        pediaX + pBodyPadX,
        headerY + headerH,
        pediaW - pBodyPadX * 2.0f,
        pediaH - pediaTitleH - pBodyPadY - headerH - pBodyBotPad));
    m_civpediaView = pediaBody.get();
    pediaWin->addChild(std::move(pediaBody));
    m_civpediaWindow = root->addChild(std::move(pediaWin));

    // Unit roster panel — all 9 unit icons in a horizontal strip below the CivPedia.
    {
        static const std::array<const char*, 9> kUnitIds = {{
            "spearman", "warrior", "archer",
            "scout",    "settler", "builder",
            "cavalry",  "ship",    "siege",
        }};
        const float rosterW       = pediaW;
        const float rosterX       = pediaX;
        const float rosterY       = pediaY + pediaH + 10.0f * s;
        const float rosterIconSz  = 44.0f * s;
        const float rosterPadX    = 14.0f * s;
        const float rosterTitleH  = 26.0f * s;
        const float rosterH       = rosterTitleH + rosterIconSz + 10.0f * s;

        auto rosterPanel = std::make_unique<odai::ui::Panel>();
        rosterPanel->setRect(odai::ui::UiRect::fromXYWH(rosterX, rosterY, rosterW, rosterH));
        rosterPanel->cornerRadiusPx     = 8.0f * s;
        rosterPanel->background         = odai::ui::UiColor{0.08f, 0.10f, 0.13f, 0.88f};
        rosterPanel->borderColor        = odai::ui::UiColor{0.75f, 0.62f, 0.34f, 0.40f};
        rosterPanel->borderThicknessPx  = 1.0f * s;

        auto rosterTitle = std::make_unique<odai::ui::Label>(fonts.bold, "Military Units");
        rosterTitle->color = odai::ui::UiColor{0.82f, 0.72f, 0.50f, 1.0f};
        rosterTitle->align = odai::ui::UiTextAlign::Center;
        rosterTitle->setRect(odai::ui::UiRect::fromXYWH(rosterX, rosterY + 4.0f * s, rosterW, 18.0f * s));
        rosterPanel->addChild(std::move(rosterTitle));

        const float gridLeft = rosterX + rosterPadX;
        const float gridW    = rosterW - 2.0f * rosterPadX;
        const float iconStep = gridW / static_cast<float>(kUnitIds.size());
        const float iconY    = rosterY + rosterTitleH;
        for (std::size_t ui = 0; ui < kUnitIds.size(); ++ui) {
            odai::ui::UiIconEntry entry{};
            if (odai::ui::UiIconRegistry::global().resolve(kUnitIds[ui], entry)) {
                const float ix = gridLeft + static_cast<float>(ui) * iconStep
                                 + (iconStep - rosterIconSz) * 0.5f;
                auto img = std::make_unique<odai::ui::Image>(entry.textureId);
                img->uvRect = entry.uv;
                img->setRect(odai::ui::UiRect::fromXYWH(ix, iconY, rosterIconSz, rosterIconSz));
                rosterPanel->addChild(std::move(img));
            }
        }
        root->addChild(std::move(rosterPanel));
    }

    // --- Vector UI demo card: a rounded panel containing labels, an icon, an
    // editable textbox, and a progress bar -- all drawn with the DPI-aware SDF
    // rounded-rect primitives (no bitmaps). Anchored bottom-centre above the HUD. ---
    {
        const float cardW = 420.0f * s;
        const float cardH = 250.0f * s;
        const float cardX = viewW * 0.5f - cardW * 0.5f;
        const float cardY = viewH - kBarH - cardH - 20.0f * s;
        const float padX  = 20.0f * s;
        float cy = cardY + 18.0f * s;

        auto card = std::make_unique<odai::ui::Panel>();
        card->setRect(odai::ui::UiRect::fromXYWH(cardX, cardY, cardW, cardH));
        card->background       = odai::ui::UiColor{0.06f, 0.12f, 0.18f, 0.96f};
        card->borderColor      = odai::ui::UiColor{0.40f, 0.55f, 0.62f, 0.55f};
        card->borderThicknessPx = 1.5f * s;
        card->cornerRadiusPx   = 16.0f * s;
        card->showShadow       = true;
        card->shadowColor      = odai::ui::UiColor{0.0f, 0.0f, 0.0f, 0.45f};
        card->shadowBlurPx     = 10.0f * s;

        auto heading = std::make_unique<odai::ui::Label>(
            fonts, "<b><color=#ecd39a>Vector UI Demo</color></b>");
        heading->setRect(odai::ui::UiRect::fromXYWH(cardX + padX, cy, cardW - padX * 2.0f, 28.0f * s));
        card->addChild(std::move(heading));

        // Icon + caption (the icon comes from the global icon atlas via rich text).
        auto iconRow = std::make_unique<odai::ui::Label>(
            fonts, "[icon=religion 26] <color=#9fb4bd>Labels, icons &amp; live text input</color>");
        iconRow->setRect(odai::ui::UiRect::fromXYWH(cardX + padX, cy + 30.0f * s, cardW - padX * 2.0f, 30.0f * s));
        card->addChild(std::move(iconRow));

        cy += 76.0f * s;

        auto box = std::make_unique<odai::ui::TextBox>(
            fonts.regular, "Type here, then Backspace / Enter...");
        box->setRect(odai::ui::UiRect::fromXYWH(cardX + padX, cy, cardW - padX * 2.0f, 44.0f * s));
        box->cornerRadiusPx    = 12.0f * s;
        box->borderThicknessPx = 1.5f * s;
        box->padding           = {16.0f * s, 0.0f};
        m_demoTextBox = box.get();
        card->addChild(std::move(box));

        cy += 56.0f * s;

        auto echo = std::make_unique<odai::ui::Label>(fonts, "<color=#7f9098>You typed: </color>");
        echo->setRect(odai::ui::UiRect::fromXYWH(cardX + padX, cy, cardW - padX * 2.0f, 26.0f * s));
        m_demoEchoLabel = echo.get();
        card->addChild(std::move(echo));

        cy += 34.0f * s;

        auto prog = std::make_unique<odai::ui::ProgressBar>();
        prog->setRect(odai::ui::UiRect::fromXYWH(cardX + padX, cy, cardW - padX * 2.0f, 16.0f * s));
        prog->cornerRadiusPx    = 8.0f * s;   // half-height -> pill
        prog->background        = odai::ui::UiColor{0.12f, 0.18f, 0.22f, 1.0f};
        prog->foreground        = odai::ui::UiColor{0.91f, 0.55f, 0.22f, 1.0f};
        prog->borderColor       = odai::ui::UiColor{0.40f, 0.55f, 0.62f, 0.50f};
        prog->borderThicknessPx = 1.0f * s;
        prog->value             = m_demoProgressValue;
        m_demoProgress = prog.get();
        card->addChild(std::move(prog));

        root->addChild(std::move(card));
    }

    // Tooltip fade: ease-in-out over ~0.16s.
    m_tooltipFade.durationSec = 0.16f;
    m_tooltipFade.easing = odai::ui::Easing::EaseInOut;

    m_uiContext.setRoot(std::move(root));
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

    if (m_uiContext.root() == nullptr) {
        setupDemoUi(static_cast<float>(fbW), static_cast<float>(fbH));
    }

    double mouseX = 0.0, mouseY = 0.0;
    glfwGetCursorPos(m_window, &mouseX, &mouseY);
    const bool leftDown = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

    m_uiInput.beginFrame();
    m_uiInput.mousePx = {static_cast<float>(mouseX), static_cast<float>(mouseY)};
    m_uiInput.setButton(odai::ui::UiMouseButton::Left, leftDown);

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

    // Update turn label.
    if (m_hudTurnLabel != nullptr) {
        m_hudTurnLabel->setText("<b><color=#ecd39a>Turn " + std::to_string(m_currentTurn) + "</color></b>");
    }

    // Update map stats label.
    if (m_hudStatsLabel != nullptr && m_strategyMap.width > 0) {
        const auto cnt = static_cast<int>(m_strategyMap.settlements.size());
        m_hudStatsLabel->setText(
            std::to_string(m_strategyMap.width) + " \xc3\x97 " +
            std::to_string(m_strategyMap.height) + "  \xe2\x80\xa2  " +
            std::to_string(cnt) + (cnt == 1 ? " settlement" : " settlements"));
    }

    // Update cities-founded label.
    if (m_uiStatusLabel != nullptr) {
        m_uiStatusLabel->setText("Cities founded: " + std::to_string(m_uiDemoClicks));
    }

    // Vector UI demo: animate the progress bar and echo the textbox contents.
    m_demoProgressValue += dt * 0.15f;
    if (m_demoProgressValue > 1.0f) {
        m_demoProgressValue -= 1.0f;
    }
    if (m_demoProgress != nullptr) {
        m_demoProgress->value = m_demoProgressValue;
    }
    if (m_demoEchoLabel != nullptr && m_demoTextBox != nullptr) {
        const std::string& typed = m_demoTextBox->value();
        m_demoEchoLabel->setText(
            "<color=#7f9098>You typed:</color> <color=#e8d9b0>" +
            (typed.empty() ? std::string("<i>(nothing yet)</i>") : typed) + "</color>");
    }

    // Top toolbar: accumulate gold and reflect the current turn.
    if (m_toolbar != nullptr) {
        m_tbGold += dt * 39.0f / 60.0f;  // ~+39/turn at ~60 fps
        m_toolbar->setValue(m_tbGoldItem, std::to_string(static_cast<int>(m_tbGold)) + "  +39");
    }
    if (m_toolbarTurnLabel != nullptr) {
        m_toolbarTurnLabel->setText(
            "<color=#c9b896>Turn</color> <b><color=#ecd39a>" +
            std::to_string(m_currentTurn) + "</color></b>");
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

            m_hudTileInfoLabel->setText(info);
        } else {
            m_hoveredHexCol = -1;
            m_hoveredHexRow = -1;
            m_hudTileInfoWindow->visible = false;
        }
    }

    m_uiContext.build(m_uiDrawList);

    // Tooltip overlay: drawn after the widget tree (so it sits on top, unclipped),
    // with an ease-in-out fade. Latch the text/anchor while hovering so the tooltip
    // can keep fading out for a moment after the cursor leaves the term.
    const bool tipActive = m_uiFontReady && m_civpediaView != nullptr && m_civpediaView->hasTooltip();
    if (tipActive) {
        m_lastTooltipText = m_civpediaView->tooltipText();
        m_lastTooltipAnchor = m_civpediaView->tooltipAnchor();
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

    const float tanHalfFov = std::tan(odai::math::radians(m_camera.fovDegrees) * 0.5f);
    const float aspect = static_cast<float>(fbW) / static_cast<float>(fbH);
    const float scaleX = ndcX * tanHalfFov * aspect;
    const float scaleY = ndcY * tanHalfFov;

    float rdX = fwdX + rgtX * scaleX + upX * scaleY;
    float rdY = fwdY + rgtY * scaleX + upY * scaleY;
    float rdZ = fwdZ + rgtZ * scaleX + upZ * scaleY;
    const float rdLen = std::sqrt(rdX*rdX + rdY*rdY + rdZ*rdZ);
    if (rdLen < 1e-6f) return false;
    rdX /= rdLen; rdY /= rdLen; rdZ /= rdLen;

    // Intersect with y = 0 (base hex plane).
    if (std::abs(rdY) < 1e-4f) return false;
    const float t = -m_camera.y / rdY;
    if (t < 0.0f) return false;

    const float wx = m_camera.x + t * rdX;
    const float wz = m_camera.z + t * rdZ;

    // World → approximate hex (col, row). Search a 3×3 neighborhood for the
    // nearest center to handle the non-rectangular hex grid correctly.
    const float hexSize = m_strategyMap.hexSize;
    constexpr float kSqrt3 = 1.7320508075688772f;

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
            if (d2 < bestDist2) {
                bestDist2 = d2;
                bestCol = c;
                bestRow = r;
            }
        }
    }

    if (bestCol < 0) return false;
    outCol = bestCol;
    outRow = bestRow;
    return true;
}

} // namespace odai::app
