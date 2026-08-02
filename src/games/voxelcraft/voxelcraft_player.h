#pragma once

#include "math/geometry.h"
#include "render/renderer_types.h"
#include "world/voxel.h"
#include "world/world.h"

#include <cstddef>
#include <vector>

// VoxelCraft player controller
// Responsible for: first-person movement, voxel collision, block raycast/edit, hotbar state.
// Should NOT do: GameApp/GLFW/UI plumbing (kept engine-loop-free so this is reusable from a
// future headless multiplayer server unchanged -- see docs/minecraft_clone_modernization.md
// and the VoxelCraft plan's "Future: Multiplayer" section).
namespace odai::games::voxelcraft {

// Roughly Minecraft-like player proportions, in voxel units. Mirrors src/app/app.cc's
// (unused-in-this-game) strategy-map/voxel-FPS hybrid constants.
constexpr float kPlayerHeightVoxels = 1.8f;
constexpr float kPlayerDiameterVoxels = 0.8f;
constexpr float kPlayerEyeHeightVoxels = 1.62f;
constexpr float kPlayerRadius = kPlayerDiameterVoxels * 0.5f;
constexpr float kPlayerEyeHeight = kPlayerEyeHeightVoxels;
constexpr float kPlayerHeight = kPlayerHeightVoxels;
constexpr float kPlayerTopOffset = kPlayerHeight - kPlayerEyeHeight;
constexpr float kCollisionEpsilon = 0.001f;

constexpr float kGravity = -24.0f;
constexpr float kJumpSpeed = 7.5f;
constexpr float kMaxFallSpeed = -35.0f;
constexpr float kPitchMinDegrees = -89.0f;
constexpr float kPitchMaxDegrees = 89.0f;
constexpr float kMouseSensitivity = 0.12f;
constexpr float kMouseSmoothingSeconds = 0.035f;
constexpr float kMoveMaxSpeed = 5.0f;
constexpr float kSprintSpeedMultiplier = 1.6f;
constexpr float kSneakSpeedMultiplier = 0.4f;
constexpr float kMoveAcceleration = 40.0f;
constexpr float kMoveDeceleration = 30.0f;
constexpr float kGamepadTriggerPressedThreshold = 0.30f;
constexpr float kGamepadMoveDeadzone = 0.18f;
constexpr float kGamepadLookDeadzone = 0.14f;
constexpr float kGamepadLookDegreesPerSecond = 160.0f;
constexpr float kBlockInteractMaxDistance = 6.0f;
constexpr int kVoxelBreakClicksRequired = 2;

using odai::math::Aabb3f;

[[nodiscard]] Aabb3f makePlayerCollisionAabb(float eyeX, float eyeY, float eyeZ);

// Collision-flavoured overlap: math::aabbOverlaps with this game's
// kCollisionEpsilon bound in, so boxes that merely touch do not collide.
// Named distinctly because Aabb3f now lives in odai::math, which would make
// an identically named two-argument overload ambiguous through ADL.
[[nodiscard]] bool aabbCollides(const Aabb3f& lhs, const Aabb3f& rhs);

// The player's camera/physics state. Named PlayerState (not CameraState) to avoid confusion
// with render::CameraPose, the renderer-facing struct it's converted into for drawing.
struct PlayerState {
    float x = 0.0f;
    float y = 10.0f;
    float z = 0.0f;
    float yawDegrees = -90.0f;
    float pitchDegrees = 0.0f;
    float velocityX = 0.0f;
    float velocityY = 0.0f;
    float velocityZ = 0.0f;
    float smoothedMouseDeltaX = 0.0f;
    float smoothedMouseDeltaY = 0.0f;
    float fovDegrees = 90.0f;
    bool onGround = false;
};

// Per-tick input, sampled by the caller (GLFW keys + gamepad + mouse delta) and handed to
// updatePlayer() as plain data -- keeps this file free of any input-plumbing dependency.
struct PlayerInputSnapshot {
    bool moveForward = false;
    bool moveBackward = false;
    bool moveLeft = false;
    bool moveRight = false;
    bool jumpDown = false;
    bool sprintDown = false;
    bool sneakDown = false;
    float mouseDeltaX = 0.0f;
    float mouseDeltaY = 0.0f;
    float gamepadMoveForward = 0.0f;
    float gamepadMoveRight = 0.0f;
    float gamepadLookX = 0.0f;
    float gamepadLookY = 0.0f;
};

[[nodiscard]] bool isSolidWorldVoxel(const odai::world::World& world, int worldX, int worldY, int worldZ);
[[nodiscard]] bool isWorldVoxelInBounds(const odai::world::World& world, int worldX, int worldY, int worldZ);
[[nodiscard]] bool doesPlayerOverlapSolid(const odai::world::World& world, float eyeX, float eyeY, float eyeZ);

// Mouselook + accelerated WASD/gamepad movement + gravity/jump. Calls resolvePlayerCollisions
// internally. Intended to run on a fixed tick step (see VoxelCraftApp) so movement stays
// frame-rate-independent and replayable -- a prerequisite for future client-side prediction.
void updatePlayer(PlayerState& player, const PlayerInputSnapshot& input, const odai::world::World& world, float dt);

// Per-axis swept-AABB collision resolution against voxel solids. Sets player.onGround.
void resolvePlayerCollisions(PlayerState& player, const odai::world::World& world, float dt);

struct CameraRaycastResult {
    bool hitSolid = false;
    int solidX = 0;
    int solidY = 0;
    int solidZ = 0;
    float hitDistance = 0.0f;
    bool hasHitFaceNormal = false;
    int hitFaceNormalX = 0;
    int hitFaceNormalY = 0;
    int hitFaceNormalZ = 0;
    bool hasAdjacentEmpty = false;
    int adjacentEmptyX = 0;
    int adjacentEmptyY = 0;
    int adjacentEmptyZ = 0;
};

// Amanatides-Woo voxel DDA ray march from the player's eye along their look direction.
[[nodiscard]] CameraRaycastResult raycastFromCamera(const PlayerState& player, const odai::world::World& world);

[[nodiscard]] bool computePlacementVoxelFromRaycast(
    const odai::world::World& world,
    const CameraRaycastResult& raycast,
    int& outX,
    int& outY,
    int& outZ
);

// Writes `voxel` at the target world cell and returns the set of resident chunk array
// indices that need their GPU mesh re-uploaded (the edited chunk, plus any neighbor whose
// mesh depends on this cell because it sits on a chunk boundary).
[[nodiscard]] bool applyVoxelEdit(
    odai::world::World& world,
    int targetX,
    int targetY,
    int targetZ,
    odai::world::Voxel voxel,
    std::vector<std::size_t>& outDirtyChunkIndices
);

// Tracks progress toward the multi-click break requirement for whichever solid voxel the
// player is currently aiming at.
struct VoxelBreakProgress {
    bool targetValid = false;
    int targetX = 0;
    int targetY = 0;
    int targetZ = 0;
    int clicks = 0;
};

void resetVoxelBreakProgress(VoxelBreakProgress& progress);

[[nodiscard]] bool tryPlaceVoxelFromCameraRay(
    odai::world::World& world,
    const PlayerState& player,
    odai::world::Voxel placeVoxel,
    std::vector<std::size_t>& outDirtyChunkIndices
);

// Advances break progress by one click on the currently-aimed-at solid voxel; returns true
// (and applies the edit) only once kVoxelBreakClicksRequired has been reached.
[[nodiscard]] bool tryRemoveVoxelFromCameraRay(
    odai::world::World& world,
    const PlayerState& player,
    VoxelBreakProgress& breakProgress,
    std::vector<std::size_t>& outDirtyChunkIndices
);

[[nodiscard]] const char* inventoryItemLabel(odai::render::InventoryItemId itemId);
[[nodiscard]] odai::world::Voxel itemToVoxel(odai::render::InventoryItemId itemId);

void cycleSelectedHotbar(odai::render::GameplayUiState& ui, int direction);
void selectHotbarSlot(odai::render::GameplayUiState& ui, int hotbarIndex);
[[nodiscard]] odai::world::Voxel selectedPlaceVoxel(const odai::render::GameplayUiState& ui);
void assignInventoryItemToSelectedHotbar(odai::render::GameplayUiState& ui, odai::render::InventoryItemId itemId);

} // namespace odai::games::voxelcraft
