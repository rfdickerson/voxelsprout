#pragma once

#include "bethesda/runtime_ids.h"
#include "bethesda/character_movement.h"
#include "math/math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace odai::importer::bethesda { struct MorrowindTerrainSurface; }

namespace odai::bethesda {

struct RuntimeObject;

inline constexpr float kBethesdaUnitsToJoltMetres = 0.0142875f;
inline constexpr float kBethesdaPhysicsGravityMetresPerSecondSq = 9.81f;
inline constexpr float kMorrowindBaselineJumpHeightUnits = 70.0f;

[[nodiscard]] inline float jumpSpeedForHeightBethesdaUnits(float heightUnits) {
    return std::sqrt(2.0f * kBethesdaPhysicsGravityMetresPerSecondSq *
        std::max(0.0f, heightUnits) / kBethesdaUnitsToJoltMetres);
}

struct PhysicsCharacterConfig {
    CharacterMovementSettings movement;
    // Engine/world space: Y-up, Bethesda units. Position is the authored feet
    // origin; the Jolt capsule-centre offset is private to the adapter.
    odai::math::Vector3 position{};
    odai::math::Quaternion rotation{};
    odai::math::Vector3 boundsHalfExtents{22.0f, 64.0f, 22.0f};
    float maxSlopeDegrees = 50.0f;
    float stepHeight = 18.0f;
};

struct PhysicsCharacterInput {
    bool jumpRequested = false;
    odai::math::Vector3 desiredVelocity{};      // Bethesda units / second
    odai::math::Vector3 rootMotion{};           // Bethesda units this tick
    bool animationDriven = false;
};

struct PhysicsCharacterStep {
    JumpPhase jumpPhase = JumpPhase::Grounded;
    LandingSeverity landingSeverity = LandingSeverity::Light;
    float landingImpactMetres = 0;
    bool leftLedge = false;
    odai::math::Vector3 position{};
    odai::math::Quaternion rotation{};
    odai::math::Vector3 velocity{};
    odai::math::Vector3 groundVelocity{};
    odai::math::Vector3 groundNormal{0.0f, 1.0f, 0.0f};
    bool grounded = false;
    bool falling = false;
    bool landed = false;
    bool blocked = false;
    std::optional<ObjectId> supportingObject;
};

struct PhysicsCharacterSnapshot {
    ObjectId object;
    odai::math::Vector3 position{};
    odai::math::Quaternion rotation{};
    odai::math::Vector3 velocity{};
    odai::math::Vector3 groundNormal{0.0f, 1.0f, 0.0f};
    bool grounded = false;
    std::optional<ObjectId> supportingObject;
    CharacterMovementState movement;
    friend bool operator==(const PhysicsCharacterSnapshot& left,
                           const PhysicsCharacterSnapshot& right) {
        return left.object == right.object &&
            left.position.x == right.position.x &&
            left.position.y == right.position.y &&
            left.position.z == right.position.z &&
            left.rotation.x == right.rotation.x &&
            left.rotation.y == right.rotation.y &&
            left.rotation.z == right.rotation.z &&
            left.rotation.w == right.rotation.w &&
            left.velocity.x == right.velocity.x &&
            left.velocity.y == right.velocity.y &&
            left.velocity.z == right.velocity.z &&
            left.groundNormal.x == right.groundNormal.x &&
            left.groundNormal.y == right.groundNormal.y &&
            left.groundNormal.z == right.groundNormal.z &&
            left.grounded == right.grounded &&
            left.supportingObject == right.supportingObject && left.movement == right.movement;
    }
};

struct TerrainCellCoord {
    int x = 0;
    int z = 0;
    friend bool operator==(const TerrainCellCoord&, const TerrainCellCoord&) = default;
    friend bool operator<(const TerrainCellCoord& a, const TerrainCellCoord& b) {
        return a.x < b.x || (a.x == b.x && a.z < b.z);
    }
};

struct PhysicsCastHit {
    odai::math::Vector3 position{};
    odai::math::Vector3 normal{0.0f, 0.0f, 1.0f};
    float distance = 0.0f;
    std::optional<ObjectId> object;
    std::optional<TerrainCellCoord> terrainCell;
};

struct PhysicsMeleeCandidate {
    ObjectId object;
    float distance = 0.0f;
    friend bool operator==(const PhysicsMeleeCandidate&,
                           const PhysicsMeleeCandidate&) = default;
};

struct PhysicsDynamicBodyConfig {
    odai::math::Vector3 position{};
    odai::math::Quaternion rotation{};
    odai::math::Vector3 boundsHalfExtents{16.0f, 16.0f, 16.0f};
    float massKilograms = 1.0f;
    float friction = 0.5f;
    float restitution = 0.1f;
    bool buoyant = false;
};

struct PhysicsDynamicBodySnapshot {
    ObjectId object;
    odai::math::Vector3 position{};
    odai::math::Quaternion rotation{};
    odai::math::Vector3 linearVelocity{};
    odai::math::Vector3 angularVelocity{};
    bool active = false;
    friend bool operator==(const PhysicsDynamicBodySnapshot&,
                           const PhysicsDynamicBodySnapshot&) = default;
};

struct PhysicsRagdollJointConfig {
    std::string role;
    int parent = -1;
    odai::math::Vector3 position{};
    odai::math::Quaternion rotation{};
    float radius = 8.0f;
    float halfHeight = 12.0f;
    float massKilograms = 4.0f;
};

struct PhysicsRagdollJointPose {
    std::string role;
    odai::math::Vector3 position{};
    odai::math::Quaternion rotation{};
    odai::math::Vector3 linearVelocity{};
};

struct PhysicsRagdollSnapshot {
    ObjectId object;
    bool active = false;
    std::vector<PhysicsRagdollJointPose> joints;
    friend bool operator==(const PhysicsRagdollSnapshot&,
                           const PhysicsRagdollSnapshot&) = default;
};

struct PhysicsHingeConfig {
    odai::math::Vector3 worldAnchor{};
    odai::math::Vector3 hingeAxis{0.0f, 1.0f, 0.0f};
    odai::math::Vector3 normalAxis{1.0f, 0.0f, 0.0f};
    float minimumAngleRadians = -3.14159265f;
    float maximumAngleRadians = 3.14159265f;
    float frictionTorqueNewtonMetres = 0.0f;
};

// Immutable collision acceleration structure. Preparation is independent of a
// physics world and may run concurrently; publication remains main-thread only.
class PreparedStaticCollision {
public:
    struct Impl;
    std::shared_ptr<const Impl> impl;
    [[nodiscard]] bool valid() const { return impl != nullptr; }
};

class BethesdaPhysicsWorld {
public:
    BethesdaPhysicsWorld();
    ~BethesdaPhysicsWorld();
    BethesdaPhysicsWorld(BethesdaPhysicsWorld&&) noexcept;
    BethesdaPhysicsWorld& operator=(BethesdaPhysicsWorld&&) noexcept;
    BethesdaPhysicsWorld(const BethesdaPhysicsWorld&) = delete;
    BethesdaPhysicsWorld& operator=(const BethesdaPhysicsWorld&) = delete;

    bool initialize(std::string& outError);
    void clear();
    bool addStaticCollision(
        ObjectId object, std::span<const odai::math::Vector3> vertices,
        std::span<const std::uint32_t> triangleIndices, std::string& outError);
    // Vertices are in Bethesda model space (Z-up). The runtime object's
    // authored transform is applied before the mesh enters the Y-up world.
    bool addStaticCollision(
        const RuntimeObject& object, std::span<const odai::math::Vector3> localVertices,
        std::span<const std::uint32_t> triangleIndices, std::string& outError);
    // Stream residency owns these aggregate bodies. The caller-provided token
    // is stable only within the active worldspace and is not save state.
    bool addStreamedStaticCollision(
        std::uint64_t residencyToken, std::span<const odai::math::Vector3> vertices,
        std::span<const std::uint32_t> triangleIndices, std::string& outError);
    static PreparedStaticCollision prepareStaticCollision(
        std::span<const odai::math::Vector3> vertices,
        std::span<const std::uint32_t> triangleIndices, std::string& outError);
    bool addPreparedStreamedStaticCollision(
        std::uint64_t residencyToken, const PreparedStaticCollision& prepared,
        std::string& outError);
    bool removeStreamedStaticCollision(std::uint64_t residencyToken);
    void clearStreamedStaticCollision();
    static PreparedStaticCollision prepareTerrainCell(
        const odai::importer::bethesda::MorrowindTerrainSurface& terrain,
        std::string& outError);
    bool addTerrainCell(
        const odai::importer::bethesda::MorrowindTerrainSurface& terrain,
        std::string& outError);
    bool addPreparedTerrainCell(
        TerrainCellCoord cell, const PreparedStaticCollision& prepared,
        std::string& outError);
    bool removeTerrainCell(TerrainCellCoord cell);
    // Streamed cells are added one at a time, but rebuilding Jolt's broad
    // phase after every body turns an 81-cell preload into 81 global rebuilds.
    // The residency owner calls this once after a batch/ring settles.
    void optimizeBroadPhase();
    bool addCharacter(ObjectId object, const PhysicsCharacterConfig& config, std::string& outError);
    bool removeCharacter(ObjectId object);
    bool setCharacterInput(ObjectId object, const PhysicsCharacterInput& input);
    // Adds an instantaneous velocity change without replacing locomotion
    // intent. This is the fixed-tick entry point for knockback, explosions and
    // shoves; gravity continues the resulting fall after support is lost.
    bool addCharacterImpulse(
        ObjectId object, const odai::math::Vector3& velocityChange);
    [[nodiscard]] bool hasCharacter(ObjectId object) const;
    bool addDynamicBody(
        ObjectId object, const PhysicsDynamicBodyConfig& config,
        std::string& outError);
    bool removeDynamicBody(ObjectId object);
    [[nodiscard]] bool hasDynamicBody(ObjectId object) const;
    bool addWorldHingeConstraint(
        ObjectId object, const PhysicsHingeConfig& config, std::string& outError);
    bool removeConstraint(ObjectId object);
    [[nodiscard]] bool hasConstraint(ObjectId object) const;
    bool addDynamicBodyImpulse(
        ObjectId object, const odai::math::Vector3& impulseKilogramUnitsPerSecond);
    bool setDynamicBodyTransform(
        ObjectId object, const odai::math::Vector3& position,
        const odai::math::Quaternion& rotation, bool activate = true);
    // Applies a deterministic centre-of-buoyancy force to marked bodies. This
    // is intentionally a gameplay primitive; water rendering remains in the
    // renderer and never advances physics.
    bool applyBuoyancy(
        ObjectId object, float waterHeightBethesdaUnits,
        float fluidDensityKilogramsPerCubicMetre, float fixedDeltaSeconds);
    [[nodiscard]] std::vector<PhysicsDynamicBodySnapshot> dynamicBodySnapshots() const;
    bool restoreDynamicBody(
        const PhysicsDynamicBodySnapshot& snapshot, std::string& outError);
    // Builds an articulated Jolt body from canonical humanoid roles, transfers
    // actor velocity, and suspends CharacterVirtual capsule authority.
    bool activateRagdoll(ObjectId object,
        std::span<const PhysicsRagdollJointConfig> joints,
        const odai::math::Vector3& linearVelocity, std::string& outError);
    bool removeRagdoll(ObjectId object);
    [[nodiscard]] bool hasActiveRagdoll(ObjectId object) const;
    [[nodiscard]] std::optional<PhysicsRagdollSnapshot> ragdollSnapshot(
        ObjectId object) const;
    [[nodiscard]] std::vector<PhysicsRagdollSnapshot> ragdollSnapshots() const;
    bool restoreRagdoll(const PhysicsRagdollSnapshot& snapshot,
        std::string& outError);
    // Restores capsule authority only when a static support surface is found
    // within the requested distance. outPlacement is the validated feet point.
    bool recoverRagdoll(ObjectId object, float maximumDropBethesdaUnits,
        odai::math::Vector3& outPlacement, std::string& outError);
    // Steps every registered character in stable ObjectId order and then the
    // Jolt world. Results are the only transforms animation/gameplay may apply.
    std::vector<std::pair<ObjectId, PhysicsCharacterStep>> step(float fixedDeltaSeconds);
    [[nodiscard]] std::optional<PhysicsCharacterStep> characterState(ObjectId object) const;
    [[nodiscard]] std::vector<PhysicsCharacterSnapshot> snapshot() const;
    bool restoreCharacter(const PhysicsCharacterSnapshot& snapshot, std::string& outError);
    bool restore(std::span<const PhysicsCharacterSnapshot> snapshots, std::string& outError);
    [[nodiscard]] std::optional<PhysicsCastHit> castDown(
        const odai::math::Vector3& origin, float distanceBethesdaUnits) const;
    [[nodiscard]] std::optional<PhysicsCastHit> castRay(
        const odai::math::Vector3& from, const odai::math::Vector3& to) const;
    [[nodiscard]] bool isCharacterPlacementClear(const PhysicsCharacterConfig& config,
        float penetrationToleranceBethesdaUnits = 1.0f) const;
    // Sweeps a sphere through the authored/dynamic rigid-body world. This is
    // the camera-boom primitive: CharacterVirtual is not a rigid body, but an
    // optional stable object id is still accepted so future player proxy
    // bodies and owned dynamic proxies can be excluded without changing the
    // camera interface.
    [[nodiscard]] std::optional<PhysicsCastHit> castSphere(
        const odai::math::Vector3& from,
        const odai::math::Vector3& to,
        float radiusBethesdaUnits,
        std::optional<ObjectId> ignoredObject = std::nullopt) const;
    [[nodiscard]] bool hasLineOfSight(
        const odai::math::Vector3& from,
        const odai::math::Vector3& to) const;
    // CharacterVirtual instances do not appear as rigid bodies in an ordinary
    // Jolt ray cast. Enumerate them in stable ObjectId order, apply a facing
    // cone, reject targets occluded by authored static collision, then return
    // nearest-first candidates for deterministic melee resolution.
    [[nodiscard]] std::vector<PhysicsMeleeCandidate> meleeCandidates(
        ObjectId attacker, const odai::math::Vector3& forward,
        float rangeBethesdaUnits, float minimumFacingDot = 0.35f) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace odai::bethesda
