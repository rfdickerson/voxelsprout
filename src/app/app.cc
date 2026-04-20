#include "app/app.h"

#include <GLFW/glfw3.h>

#include "core/grid3.h"
#include "core/log.h"
#include "math/math.h"
#include "sim/network_procedural.h"

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
#include <span>
#include <string>
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
constexpr float kImportedSceneSunPitchDegrees = -8.0f;
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
constexpr const char* kImportedSceneEnvVar = "ODAI_IMPORTED_SCENE";

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

std::optional<std::filesystem::path> findImportedSceneDemoPath() {
    std::string envPathValue;
#ifdef _WIN32
    char* envPath = nullptr;
    std::size_t envPathLength = 0;
    if (_dupenv_s(&envPath, &envPathLength, kImportedSceneEnvVar) == 0 && envPath != nullptr) {
        envPathValue = envPath;
    }
    std::free(envPath);
#else
    if (const char* envPath = std::getenv(kImportedSceneEnvVar); envPath != nullptr) {
        envPathValue = envPath;
    }
#endif
    if (!envPathValue.empty()) {
        return std::filesystem::path(envPathValue);
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
    const float offset = std::max(spanX, spanZ) * 0.38f;
    pose.x = centerX - offset;
    pose.y = maxY + std::max(spanX, spanZ) * 0.18f + 1200.0f;
    pose.z = centerZ - offset * 0.75f;
    pose.yawDegrees = 48.0f;
    pose.pitchDegrees = -24.0f;
    return pose;
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

    if (const std::optional<std::filesystem::path> importedScenePath = findImportedSceneDemoPath();
        importedScenePath.has_value()) {
        m_importedScenePath = *importedScenePath;
        const auto importedSceneLoadStart = Clock::now();
        if (!odai::importer::loadImportedSceneRuntime(m_importedScenePath, m_importedScene)) {
            VOX_LOGE("app") << "failed to load imported scene from "
                            << std::filesystem::absolute(m_importedScenePath).string()
                            << ": " << odai::importer::getImportedSceneLastError();
            return false;
        }
        const ImportedSceneCameraPose importedCameraPose = configureImportedSceneCamera(m_importedScene);
        m_camera.x = importedCameraPose.x;
        m_camera.y = importedCameraPose.y;
        m_camera.z = importedCameraPose.z;
        m_camera.yawDegrees = importedCameraPose.yawDegrees;
        m_camera.pitchDegrees = importedCameraPose.pitchDegrees;
        m_cameraPrevious = m_camera;
        m_importedSceneDemoEnabled = true;
        VOX_LOGI("app") << "imported scene demo enabled from "
                        << std::filesystem::absolute(m_importedScenePath).string()
                        << " in " << elapsedMs(importedSceneLoadStart) << " ms"
                        << " (version=" << m_importedScene.sourceFileVersion
                        << ", meshes=" << m_importedScene.sourceMeshCount
                        << ", instances=" << m_importedScene.sourceInstanceCount
                        << ", terrainCells=" << m_importedScene.sourceLandscapeCellCount
                        << ", packedVertices=" << m_importedScene.packedVertices.size()
                        << ", packedIndices=" << m_importedScene.packedIndices.size() << ")";
    }

    if (m_importedSceneDemoEnabled) {
        const auto rendererInitStart = Clock::now();
        const bool rendererOk = m_renderer.init(m_window, m_world.chunkGrid());
        const auto rendererInitMs = elapsedMs(rendererInitStart);
        VOX_LOGI("app") << "init step renderer init took " << rendererInitMs << " ms";
        if (!rendererOk) {
            VOX_LOGE("app") << "renderer init failed";
            return false;
        }
        if (!m_renderer.uploadImportedScene(m_importedScene)) {
            VOX_LOGE("app") << "failed to upload imported scene demo geometry";
            return false;
        }
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

    const auto worldLoadStart = Clock::now();
    const std::filesystem::path worldPath{kWorldFilePath};
    odai::world::World::LoadResult worldLoadResult{};
    if (m_world.loadOrInitialize(worldPath, &worldLoadResult)) {
        const auto worldLoadMs = elapsedMs(worldLoadStart);
        VOX_LOGI("app") << "loaded world from " << std::filesystem::absolute(worldPath).string()
                        << " in " << worldLoadMs << " ms";
    } else if (worldLoadResult.initializedFallback) {
        const auto worldLoadMs = elapsedMs(worldLoadStart);
        VOX_LOGW("app") << "world file missing/invalid at " << std::filesystem::absolute(worldPath).string()
                        << "; starting procedural chunk streaming window in " << worldLoadMs << " ms";
    }

    const auto magicaStampStart = Clock::now();
    constexpr std::array<odai::world::World::MagicaStampSpec, 3> kMagicaLoadSpecs = {
        odai::world::World::MagicaStampSpec{kMagicaCastlePath, 0.0f, 0.0f, 0.0f, 1.0f},
        odai::world::World::MagicaStampSpec{kMagicaTeapotPath, 64.0f, 0.0f, 0.0f, 0.36f},
        odai::world::World::MagicaStampSpec{kMagicaMonu2Path, -72.0f, 0.0f, 16.0f, 0.25f},
    };
    const odai::world::World::MagicaStampResult stampResult = m_world.stampMagicaResources(kMagicaLoadSpecs);
    VOX_LOGI("app") << "stamped " << stampResult.stampedResourceCount << "/" << kMagicaLoadSpecs.size()
                    << " magica resources into world (voxels=" << stampResult.stampedVoxelCount
                    << ", clipped=" << stampResult.clippedVoxelCount
                    << ", paletteColors=" << static_cast<std::uint32_t>(stampResult.baseColorPaletteCount)
                    << ") in " << elapsedMs(magicaStampStart) << " ms";
    if (stampResult.stampedResourceCount != kMagicaLoadSpecs.size()) {
        VOX_LOGW("app") << "release runtime missing one or more Magica assets; world stamp count is incomplete";
    }
    m_renderer.setVoxelBaseColorPalette(stampResult.baseColorPalette);

    m_world.setStreamingConfig(odai::world::World::ChunkStreamingConfig{2, 2});
    const odai::world::World::ChunkStreamingUpdate initialStreamingUpdate =
        m_world.updateStreamingWindowForWorldPosition(m_camera.x, m_camera.z);
    const odai::world::World::ChunkStreamingStats& initialStreamingStats = initialStreamingUpdate.stats;
    VOX_LOGI("app") << "chunk streaming ready (center="
                    << initialStreamingStats.centerChunkX << "," << initialStreamingStats.centerChunkZ
                    << ", resident=" << initialStreamingStats.residentChunkCount
                    << ", stored=" << initialStreamingStats.storedChunkCount
                    << ", generated=" << initialStreamingUpdate.generatedChunkKeys.size()
                    << ", radius=" << m_world.streamingConfig().radiusChunksX
                    << "x" << m_world.streamingConfig().radiusChunksZ
                    << ")";

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
    VOX_LOGI("app") << "morrowind viewer ready (WASD move, Space up, Shift down, F/C renderer panel, F5 terrain, F6 statics, K textures, F7 flat shading, F8 water debug)";
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
            if (!m_importedSceneDemoEnabled) {
                m_simulation.update(static_cast<float>(kSimulationFixedStepSeconds));
            }
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

void App::update(float dt, float simulationAlpha) {
    if (m_importedSceneDemoEnabled) {
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
        m_renderer.renderFrame(
            m_world.chunkGrid(),
            m_simulation,
            cameraPose,
            odai::render::VoxelPreview{},
            simulationAlpha,
            std::span<const std::size_t>(m_visibleChunkIndices)
        );
        m_camera.fovDegrees = m_renderer.cameraFovDegrees();
        return;
    }

    refreshStreamingWindow(false);

    const bool regeneratePressedThisFrame =
        !isAnyUiVisible() && m_input.regenerateWorldDown && !m_wasRegenerateWorldDown;
    m_wasRegenerateWorldDown = m_input.regenerateWorldDown;
    if (regeneratePressedThisFrame) {
        regenerateWorld();
    }

    const CameraRaycastResult raycast = raycastFromCamera();
    if (m_voxelBreakTargetValid) {
        const bool blockTargetStillValid =
            !isAnyUiVisible() &&
            raycast.hitSolid &&
            raycast.hitDistance <= kBlockInteractMaxDistance &&
            raycast.solidX == m_voxelBreakTargetX &&
            raycast.solidY == m_voxelBreakTargetY &&
            raycast.solidZ == m_voxelBreakTargetZ;
        if (!blockTargetStillValid) {
            resetVoxelBreakProgress();
        }
    }

    int windowWidth = 0;
    int windowHeight = 0;
    glfwGetWindowSize(m_window, &windowWidth, &windowHeight);
    double mouseX = 0.0;
    double mouseY = 0.0;
    glfwGetCursorPos(m_window, &mouseX, &mouseY);

    const bool inventoryClickPressedThisFrame = m_inventoryVisible && m_input.removeBlockDown && !m_wasRemoveBlockDown;
    if (inventoryClickPressedThisFrame) {
        handleInventoryClick(
            static_cast<float>(mouseX),
            static_cast<float>(mouseY),
            static_cast<float>(std::max(windowWidth, 1)),
            static_cast<float>(std::max(windowHeight, 1))
        );
    }

    const bool blockInteractionEnabled = !isAnyUiVisible();
    const bool placePressedThisFrame = blockInteractionEnabled && m_input.placeBlockDown && !m_wasPlaceBlockDown;
    const bool removePressedThisFrame = blockInteractionEnabled && m_input.removeBlockDown && !m_wasRemoveBlockDown;
    m_wasPlaceBlockDown = m_input.placeBlockDown;
    m_wasRemoveBlockDown = m_input.removeBlockDown;

    bool voxelChunkEdited = false;
    std::vector<std::size_t> editedChunkIndices;
    if (placePressedThisFrame) {
        resetVoxelBreakProgress();
        if (tryPlaceVoxelFromCameraRay(editedChunkIndices)) {
            voxelChunkEdited = true;
        }
    }
    if (removePressedThisFrame && tryRemoveVoxelFromCameraRay(editedChunkIndices)) {
        voxelChunkEdited = true;
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
        const float latitudeRadians = odai::math::radians(kDayCycleLatitudeDegrees);
        const float declinationRadians = odai::math::radians(kDayCycleWinterDeclinationDegrees);
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

        const float sunPitchDegrees = odai::math::degrees(std::asin(std::clamp(sunUp, -1.0f, 1.0f)));
        float sunAzimuthDegrees = odai::math::degrees(std::atan2(sunEast, sunNorth));
        if (sunAzimuthDegrees < 0.0f) {
            sunAzimuthDegrees += 360.0f;
        }

        // Convert azimuth (north=0, east=90, south=180) to engine yaw
        // where yaw 0 = +X (east), yaw 90 = +Z (south), yaw -90 = -Z (north).
        const float sunYawDegrees = wrapDegreesSigned((sunAzimuthDegrees - 90.0f) + kDayCycleAzimuthOffsetDegrees);
        m_renderer.setSunAngles(sunYawDegrees, sunPitchDegrees);
    }

    const float renderAlpha = std::clamp(simulationAlpha, 0.0f, 1.0f);
    const float renderCameraX = m_cameraPrevious.x + ((m_camera.x - m_cameraPrevious.x) * renderAlpha);
    const float renderCameraY = m_cameraPrevious.y + ((m_camera.y - m_cameraPrevious.y) * renderAlpha);
    const float renderCameraZ = m_cameraPrevious.z + ((m_camera.z - m_cameraPrevious.z) * renderAlpha);
    const float renderCameraYawDegrees =
        lerpWrappedDegrees(m_cameraPrevious.yawDegrees, m_camera.yawDegrees, renderAlpha);
    const float renderCameraPitchDegrees =
        m_cameraPrevious.pitchDegrees + ((m_camera.pitchDegrees - m_cameraPrevious.pitchDegrees) * renderAlpha);

    odai::render::VoxelPreview preview{};
    if (!isAnyUiVisible()) {
        if (raycast.hitSolid && raycast.hitDistance <= kBlockInteractMaxDistance) {
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
                preview.mode = m_input.removeBlockDown
                    ? odai::render::VoxelPreview::Mode::Remove
                    : odai::render::VoxelPreview::Mode::Add;
            }

            if (!m_input.removeBlockDown) {
                int targetX = 0;
                int targetY = 0;
                int targetZ = 0;
                if (computePlacementVoxelFromRaycast(raycast, targetX, targetY, targetZ)) {
                    if (isWorldVoxelInBounds(targetX, targetY, targetZ) &&
                        selectedPlaceVoxel().type != odai::world::VoxelType::Empty) {
                        // Face highlight is enough for placement; do not show a ghost voxel cube.
                    } else {
                        preview.faceVisible = false;
                    }
                } else {
                    preview.faceVisible = false;
                }
            }
        }
    }

    const odai::render::CameraPose cameraPose{
        renderCameraX,
        renderCameraY,
        renderCameraZ,
        renderCameraYawDegrees,
        renderCameraPitchDegrees,
        m_camera.fovDegrees
    };
    syncGameplayUiState();

    const odai::world::ClipmapConfig requestedClipmapConfig = m_renderer.clipmapQueryConfig();
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
    odai::world::SpatialQueryStats spatialQueryStats{};
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
            odai::math::Vector3{renderCameraX, renderCameraY, renderCameraZ},
            renderCameraYawDegrees,
            renderCameraPitchDegrees,
            m_camera.fovDegrees,
            aspectRatio
        );
        const CameraFrustum keepAliveFrustum = buildCameraFrustum(
            odai::math::Vector3{renderCameraX, renderCameraY, renderCameraZ},
            renderCameraYawDegrees,
            renderCameraPitchDegrees,
            m_camera.fovDegrees,
            aspectRatio
        );
        if (cameraFrustum.valid && m_chunkClipmapIndex.valid()) {
            m_chunkClipmapIndex.updateCamera(renderCameraX, renderCameraY, renderCameraZ, &spatialQueryStats);
            std::vector<std::size_t> candidateChunkIndices =
                m_chunkClipmapIndex.queryChunksIntersecting(cameraFrustum.broadPhaseBounds, &spatialQueryStats);
            spatialQueriesUsed = true;
            const std::vector<odai::world::Chunk>& chunks = m_world.chunkGrid().chunks();
            if (m_visibleChunkGraceFrames.size() != chunks.size() ||
                m_previousVisibleChunkMask.size() != chunks.size() ||
                m_currentVisibleChunkMask.size() != chunks.size() ||
                m_directlyVisibleChunkMask.size() != chunks.size()) {
                m_visibleChunkGraceFrames.assign(chunks.size(), 0u);
                m_previousVisibleChunkMask.assign(chunks.size(), 0u);
                m_currentVisibleChunkMask.assign(chunks.size(), 0u);
                m_directlyVisibleChunkMask.assign(chunks.size(), 0u);
                m_visibleChunkIndices.clear();
            }
            std::fill(m_currentVisibleChunkMask.begin(), m_currentVisibleChunkMask.end(), 0u);
            std::fill(m_directlyVisibleChunkMask.begin(), m_directlyVisibleChunkMask.end(), 0u);
            for (std::size_t chunkIndex : candidateChunkIndices) {
                if (chunkIndex >= chunks.size()) {
                    continue;
                }
                if (chunkIntersectsFrustum(chunks[chunkIndex], cameraFrustum.planes, kRenderFrustumPlaneSlackVoxels)) {
                    m_directlyVisibleChunkMask[chunkIndex] = 1u;
                }
            }
            std::uint32_t retainedChunkCount = 0;
            std::uint32_t newlyVisibleChunkCount = 0;
            std::uint32_t evictedChunkCount = 0;
            for (std::size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
                const bool directlyVisible = m_directlyVisibleChunkMask[chunkIndex] != 0u;
                if (directlyVisible) {
                    m_visibleChunkGraceFrames[chunkIndex] = kSpatialQueryVisibilityGraceFrames;
                    m_currentVisibleChunkMask[chunkIndex] = 1u;
                } else {
                    const bool previouslyVisible = m_previousVisibleChunkMask[chunkIndex] != 0u;
                    const bool keepAliveVisible =
                        previouslyVisible &&
                        keepAliveFrustum.valid &&
                        chunkIntersectsFrustum(
                            chunks[chunkIndex],
                            keepAliveFrustum.planes,
                            kRenderFrustumKeepAlivePlaneSlackVoxels
                        );
                    std::uint8_t& graceFrames = m_visibleChunkGraceFrames[chunkIndex];
                    if (keepAliveVisible) {
                        graceFrames = kSpatialQueryVisibilityGraceFrames;
                        m_currentVisibleChunkMask[chunkIndex] = 1u;
                    } else if (graceFrames > 0) {
                        --graceFrames;
                        m_currentVisibleChunkMask[chunkIndex] = 1u;
                    }
                }

                if (m_currentVisibleChunkMask[chunkIndex] != 0u && !directlyVisible) {
                    ++retainedChunkCount;
                }
                if (m_currentVisibleChunkMask[chunkIndex] != 0u && m_previousVisibleChunkMask[chunkIndex] == 0u) {
                    ++newlyVisibleChunkCount;
                }
                if (m_currentVisibleChunkMask[chunkIndex] == 0u && m_previousVisibleChunkMask[chunkIndex] != 0u) {
                    ++evictedChunkCount;
                }
            }
            if (m_currentVisibleChunkMask != m_previousVisibleChunkMask) {
                m_visibleChunkIndices.clear();
                m_visibleChunkIndices.reserve(candidateChunkIndices.size() + retainedChunkCount);
                for (std::size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
                    if (m_currentVisibleChunkMask[chunkIndex] != 0u) {
                        m_visibleChunkIndices.push_back(chunkIndex);
                    }
                }
            }
            spatialQueryStats.visibleChunkCount = static_cast<std::uint32_t>(m_visibleChunkIndices.size());
            spatialQueryStats.retainedChunkCount = retainedChunkCount;
            spatialQueryStats.newlyVisibleChunkCount = newlyVisibleChunkCount;
            spatialQueryStats.evictedChunkCount = evictedChunkCount;
            m_previousVisibleChunkMask.swap(m_currentVisibleChunkMask);
        }
    }

    if (m_visibleChunkIndices.empty() && (!spatialQueriesUsed || !m_chunkClipmapIndex.valid())) {
        const std::size_t chunkCount = m_world.chunkGrid().chunks().size();
        m_visibleChunkIndices.resize(chunkCount);
        for (std::size_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
            m_visibleChunkIndices[chunkIndex] = chunkIndex;
        }
        m_visibleChunkGraceFrames.assign(chunkCount, 0u);
        m_previousVisibleChunkMask.assign(chunkCount, 1u);
        m_currentVisibleChunkMask.assign(chunkCount, 0u);
        m_directlyVisibleChunkMask.assign(chunkCount, 0u);
    } else if (!spatialQueriesUsed) {
        m_visibleChunkGraceFrames.clear();
        m_previousVisibleChunkMask.clear();
        m_currentVisibleChunkMask.clear();
        m_directlyVisibleChunkMask.clear();
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
        if (m_debugUiVisible) {
            toggleDebugUi();
            uiVisibilityChanged = true;
        }
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

    m_renderer.setDebugUiVisible(m_debugUiVisible);
    const bool rendererUiVisible = m_renderer.isDebugUiVisible();
    if (rendererUiVisible != m_debugUiVisible) {
        m_debugUiVisible = rendererUiVisible;
        uiVisibilityChanged = true;
    }
    if (uiVisibilityChanged) {
        glfwSetInputMode(m_window, GLFW_CURSOR, isAnyUiVisible() ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
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

    const float mouseDeltaX = m_pendingMouseDeltaX;
    const float mouseDeltaY = m_pendingMouseDeltaY;
    m_pendingMouseDeltaX = 0.0f;
    m_pendingMouseDeltaY = 0.0f;

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

    if (m_importedSceneDemoEnabled) {
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

} // namespace odai::app
