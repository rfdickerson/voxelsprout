#pragma once

#include "bethesda/runtime_ids.h"
#include "math/math.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace odai::bethesda {

inline constexpr float kBethesdaUnitsToJoltMetres = 0.0142875f;

struct PhysicsCharacterConfig {
    // Engine/world space: Y-up, Bethesda units. Position is the authored feet
    // origin; the Jolt capsule-centre offset is private to the adapter.
    odai::math::Vector3 position{};
    odai::math::Quaternion rotation{};
    odai::math::Vector3 boundsHalfExtents{22.0f, 64.0f, 22.0f};
    float maxSlopeDegrees = 50.0f;
    float stepHeight = 18.0f;
};

struct PhysicsCharacterInput {
    odai::math::Vector3 desiredVelocity{};      // Bethesda units / second
    odai::math::Vector3 rootMotion{};           // Bethesda units this tick
    bool animationDriven = false;
};

struct PhysicsCharacterStep {
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
            left.supportingObject == right.supportingObject;
    }
};

struct PhysicsCastHit {
    odai::math::Vector3 position{};
    odai::math::Vector3 normal{0.0f, 0.0f, 1.0f};
    float distance = 0.0f;
    std::optional<ObjectId> object;
};

struct PhysicsMeleeCandidate {
    ObjectId object;
    float distance = 0.0f;
    friend bool operator==(const PhysicsMeleeCandidate&,
                           const PhysicsMeleeCandidate&) = default;
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
    // Stream residency owns these aggregate bodies. The caller-provided token
    // is stable only within the active worldspace and is not save state.
    bool addStreamedStaticCollision(
        std::uint64_t residencyToken, std::span<const odai::math::Vector3> vertices,
        std::span<const std::uint32_t> triangleIndices, std::string& outError);
    bool removeStreamedStaticCollision(std::uint64_t residencyToken);
    void clearStreamedStaticCollision();
    bool addCharacter(ObjectId object, const PhysicsCharacterConfig& config, std::string& outError);
    bool removeCharacter(ObjectId object);
    bool setCharacterInput(ObjectId object, const PhysicsCharacterInput& input);
    [[nodiscard]] bool hasCharacter(ObjectId object) const;
    // Steps every registered character in stable ObjectId order and then the
    // Jolt world. Results are the only transforms animation/gameplay may apply.
    std::vector<std::pair<ObjectId, PhysicsCharacterStep>> step(float fixedDeltaSeconds);
    [[nodiscard]] std::optional<PhysicsCharacterStep> characterState(ObjectId object) const;
    [[nodiscard]] std::vector<PhysicsCharacterSnapshot> snapshot() const;
    bool restoreCharacter(const PhysicsCharacterSnapshot& snapshot, std::string& outError);
    bool restore(std::span<const PhysicsCharacterSnapshot> snapshots, std::string& outError);
    [[nodiscard]] std::optional<PhysicsCastHit> castDown(
        const odai::math::Vector3& origin, float distanceBethesdaUnits) const;
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
