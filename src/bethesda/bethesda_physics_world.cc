#include "bethesda/bethesda_physics_world.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cmath>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/NarrowPhaseQuery.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceTable.h>
#include <Jolt/Physics/Collision/ObjectLayerPairFilterTable.h>
#include <Jolt/Physics/Collision/BroadPhase/ObjectVsBroadPhaseLayerFilterTable.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/GroupFilterTable.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/PointConstraint.h>
#include <Jolt/Physics/PhysicsSystem.h>

#include "core/log.h"

namespace odai::bethesda {
namespace {

constexpr JPH::ObjectLayer kStaticLayer = 0u;
constexpr JPH::ObjectLayer kCharacterLayer = 1u;
constexpr JPH::ObjectLayer kDynamicLayer = 2u;

// Jolt deliberately installs a breakpoint-producing dummy trace callback in
// debug builds. MeshShape uses Trace for recoverable author-data warnings (for
// example, when overlapping triangles make its SAH splitter fall back to a
// deterministic half split). Leaving the dummy installed turns valid retail
// collision into a SIGTRAP even though Jolt can finish building the mesh.
void joltTrace(const char* format, ...) {
    char message[1024] = {};
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);
    VOX_LOGW("physics") << "Jolt: " << message;
}

void ensureJoltRegistered() {
    static std::once_flag once;
    std::call_once(once, [] {
        JPH::RegisterDefaultAllocator();
        JPH::Trace = joltTrace;
        if (JPH::Factory::sInstance == nullptr) {
            JPH::Factory::sInstance = new JPH::Factory();
            JPH::RegisterTypes();
        }
    });
}

JPH::Vec3 toJoltVector(const odai::math::Vector3& value) {
    return JPH::Vec3(value.x, value.y, value.z) * kBethesdaUnitsToJoltMetres;
}

JPH::RVec3 toJoltPosition(const odai::math::Vector3& value) {
    return JPH::RVec3(value.x * kBethesdaUnitsToJoltMetres,
        value.y * kBethesdaUnitsToJoltMetres, value.z * kBethesdaUnitsToJoltMetres);
}

odai::math::Vector3 fromJoltVector(JPH::Vec3Arg value) {
    const float scale = 1.0f / kBethesdaUnitsToJoltMetres;
    return {value.GetX() * scale, value.GetY() * scale, value.GetZ() * scale};
}

odai::math::Vector3 fromJoltPosition(JPH::RVec3Arg value) {
    const double scale = 1.0 / static_cast<double>(kBethesdaUnitsToJoltMetres);
    return {static_cast<float>(value.GetX() * scale), static_cast<float>(value.GetY() * scale),
        static_cast<float>(value.GetZ() * scale)};
}

JPH::Quat toJoltRotation(const odai::math::Quaternion& value) {
    return JPH::Quat(value.x, value.y, value.z, value.w).Normalized();
}

odai::math::Quaternion fromJoltRotation(JPH::QuatArg value) {
    return odai::math::normalize({value.GetX(), value.GetY(), value.GetZ(), value.GetW()});
}

std::uint64_t userDataFor(const ObjectId& id) {
    return static_cast<std::uint64_t>(ObjectIdHash{}(id));
}

}  // namespace

class BethesdaPhysicsWorld::Impl {
public:
    struct CharacterEntry {
        JPH::Ref<JPH::CharacterVirtual> character;
        PhysicsCharacterInput input;
        CharacterMovementSettings movementSettings;
        CharacterMovementState movement;
        PhysicsCharacterStep last;
        JPH::Vec3 externalVelocity = JPH::Vec3::sZero();
        float stepHeightMetres = 0.25f;
        float centreOffsetMetres = 0.0f;
        bool suspendedByRagdoll = false;
    };
    struct DynamicEntry {
        JPH::BodyID body;
        PhysicsDynamicBodyConfig config;
    };
    struct RagdollEntry {
        JPH::Ref<JPH::GroupFilterTable> groupFilter;
        std::vector<std::string> roles;
        std::vector<JPH::BodyID> bodies;
        std::vector<JPH::Ref<JPH::Constraint>> constraints;
    };

    Impl()
        : broadPhaseLayers(3u, 2u), objectPairFilter(3u), jobs(JPH::cMaxPhysicsJobs) {
        broadPhaseLayers.MapObjectToBroadPhaseLayer(kStaticLayer, JPH::BroadPhaseLayer(0u));
        broadPhaseLayers.MapObjectToBroadPhaseLayer(kCharacterLayer, JPH::BroadPhaseLayer(1u));
        broadPhaseLayers.MapObjectToBroadPhaseLayer(kDynamicLayer, JPH::BroadPhaseLayer(1u));
        objectPairFilter.EnableCollision(kStaticLayer, kCharacterLayer);
        objectPairFilter.EnableCollision(kCharacterLayer, kCharacterLayer);
        objectPairFilter.EnableCollision(kStaticLayer, kDynamicLayer);
        objectPairFilter.EnableCollision(kCharacterLayer, kDynamicLayer);
        objectPairFilter.EnableCollision(kDynamicLayer, kDynamicLayer);
        broadPhaseFilter = std::make_unique<JPH::ObjectVsBroadPhaseLayerFilterTable>(
            broadPhaseLayers, 2u, objectPairFilter, 3u);
    }

    JPH::BroadPhaseLayerInterfaceTable broadPhaseLayers;
    JPH::ObjectLayerPairFilterTable objectPairFilter;
    std::unique_ptr<JPH::ObjectVsBroadPhaseLayerFilterTable> broadPhaseFilter;
    JPH::PhysicsSystem physics;
    JPH::TempAllocatorMalloc allocator;
    JPH::JobSystemSingleThreaded jobs;
    JPH::CharacterVsCharacterCollisionSimple characterCollision;
    std::map<ObjectId, CharacterEntry> characters;
    std::map<ObjectId, DynamicEntry> dynamicBodies;
    std::map<ObjectId, RagdollEntry> ragdolls;
    JPH::CollisionGroup::GroupID nextRagdollGroup = 1u;
    std::map<ObjectId, JPH::Ref<JPH::Constraint>> constraints;
    std::unordered_map<std::uint64_t, ObjectId> objectsByUserData;
    std::vector<JPH::BodyID> staticBodies;
    std::map<std::uint64_t, JPH::BodyID> streamedStaticBodies;
    bool initialized = false;
};

BethesdaPhysicsWorld::BethesdaPhysicsWorld() {
    // Jolt containers allocate while Impl's filter tables are constructed.
    ensureJoltRegistered();
    m_impl = std::make_unique<Impl>();
}
BethesdaPhysicsWorld::~BethesdaPhysicsWorld() { clear(); }
BethesdaPhysicsWorld::BethesdaPhysicsWorld(BethesdaPhysicsWorld&&) noexcept = default;
BethesdaPhysicsWorld& BethesdaPhysicsWorld::operator=(BethesdaPhysicsWorld&&) noexcept = default;

bool BethesdaPhysicsWorld::initialize(std::string& outError) {
    outError.clear();
    if (m_impl->initialized) return true;
    ensureJoltRegistered();
    m_impl->physics.Init(65536u, 0u, 65536u, 65536u, m_impl->broadPhaseLayers,
        *m_impl->broadPhaseFilter, m_impl->objectPairFilter);
    m_impl->physics.SetGravity(JPH::Vec3(0.0f, -9.81f, 0.0f));
    m_impl->initialized = true;
    return true;
}

void BethesdaPhysicsWorld::clear() {
    if (!m_impl || !m_impl->initialized) return;
    for (auto& [id, entry] : m_impl->characters) {
        (void)id;
        if (!entry.suspendedByRagdoll)
            m_impl->characterCollision.Remove(entry.character);
    }
    m_impl->characters.clear();
    JPH::BodyInterface& bodies = m_impl->physics.GetBodyInterface();
    for (auto& [object, ragdoll] : m_impl->ragdolls) {
        (void)object;
        for (const auto& constraint : ragdoll.constraints)
            m_impl->physics.RemoveConstraint(constraint);
        for (JPH::BodyID body : ragdoll.bodies) {
            bodies.RemoveBody(body);
            bodies.DestroyBody(body);
        }
    }
    m_impl->ragdolls.clear();
    for (const auto& [object, constraint] : m_impl->constraints) {
        (void)object;
        m_impl->physics.RemoveConstraint(constraint);
    }
    m_impl->constraints.clear();
    for (const auto& [object, entry] : m_impl->dynamicBodies) {
        (void)object;
        bodies.RemoveBody(entry.body);
        bodies.DestroyBody(entry.body);
    }
    m_impl->dynamicBodies.clear();
    for (JPH::BodyID id : m_impl->staticBodies) {
        bodies.RemoveBody(id);
        bodies.DestroyBody(id);
    }
    m_impl->staticBodies.clear();
    for (const auto& [token, id] : m_impl->streamedStaticBodies) {
        (void)token;
        bodies.RemoveBody(id);
        bodies.DestroyBody(id);
    }
    m_impl->streamedStaticBodies.clear();
    m_impl->objectsByUserData.clear();
}

bool BethesdaPhysicsWorld::addStaticCollision(
    ObjectId object, std::span<const odai::math::Vector3> vertices,
    std::span<const std::uint32_t> triangleIndices, std::string& outError) {
    if (!initialize(outError)) return false;
    if (!object.valid() || triangleIndices.size() % 3u != 0u) {
        outError = "invalid static collision object or triangle index count";
        return false;
    }
    JPH::TriangleList triangles;
    triangles.reserve(triangleIndices.size() / 3u);
    for (std::size_t offset = 0; offset < triangleIndices.size(); offset += 3u) {
        const std::uint32_t a = triangleIndices[offset];
        const std::uint32_t b = triangleIndices[offset + 1u];
        const std::uint32_t c = triangleIndices[offset + 2u];
        if (a >= vertices.size() || b >= vertices.size() || c >= vertices.size()) {
            outError = "static collision triangle index is out of range";
            return false;
        }
        triangles.emplace_back(toJoltVector(vertices[a]), toJoltVector(vertices[c]),
            toJoltVector(vertices[b]));
    }
    JPH::MeshShapeSettings settings(triangles);
    const auto created = settings.Create();
    if (created.HasError()) {
        outError = "Jolt mesh construction failed: " + created.GetError();
        return false;
    }
    const std::uint64_t userData = userDataFor(object);
    JPH::BodyCreationSettings bodySettings(created.Get(), JPH::RVec3::sZero(),
        JPH::Quat::sIdentity(), JPH::EMotionType::Static, kStaticLayer);
    bodySettings.mUserData = userData;
    const JPH::BodyID body = m_impl->physics.GetBodyInterface().CreateAndAddBody(
        bodySettings, JPH::EActivation::DontActivate);
    if (body.IsInvalid()) {
        outError = "Jolt rejected static collision body";
        return false;
    }
    m_impl->staticBodies.push_back(body);
    m_impl->objectsByUserData[userData] = std::move(object);
    m_impl->physics.OptimizeBroadPhase();
    return true;
}

struct PreparedStaticCollision::Impl {
    JPH::RefConst<JPH::Shape> shape;
};

PreparedStaticCollision BethesdaPhysicsWorld::prepareStaticCollision(
    std::span<const odai::math::Vector3> vertices,
    std::span<const std::uint32_t> triangleIndices, std::string& outError) {
    ensureJoltRegistered();
    if (triangleIndices.empty() || triangleIndices.size() % 3u != 0u) {
        outError = "invalid streamed collision triangle index count";
        return {};
    }
    JPH::TriangleList triangles;
    triangles.reserve(triangleIndices.size() / 3u);
    for (std::size_t offset = 0u; offset < triangleIndices.size(); offset += 3u) {
        const std::uint32_t a = triangleIndices[offset];
        const std::uint32_t b = triangleIndices[offset + 1u];
        const std::uint32_t c = triangleIndices[offset + 2u];
        if (a >= vertices.size() || b >= vertices.size() || c >= vertices.size()) {
            outError = "streamed collision triangle index is out of range";
            return {};
        }
        triangles.emplace_back(toJoltVector(vertices[a]), toJoltVector(vertices[c]),
            toJoltVector(vertices[b]));
    }
    JPH::MeshShapeSettings settings(triangles);
    const auto created = settings.Create();
    if (created.HasError()) {
        outError = "Jolt streamed mesh construction failed: " + created.GetError();
        return {};
    }
    auto impl = std::make_shared<PreparedStaticCollision::Impl>();
    impl->shape = created.Get();
    outError.clear();
    return {std::move(impl)};
}

bool BethesdaPhysicsWorld::addStreamedStaticCollision(
    std::uint64_t residencyToken, std::span<const odai::math::Vector3> vertices,
    std::span<const std::uint32_t> triangleIndices, std::string& outError) {
    const auto prepared = prepareStaticCollision(vertices, triangleIndices, outError);
    return prepared.valid() && addPreparedStreamedStaticCollision(residencyToken, prepared, outError);
}

bool BethesdaPhysicsWorld::addPreparedStreamedStaticCollision(
    std::uint64_t residencyToken, const PreparedStaticCollision& prepared,
    std::string& outError) {
    if (!prepared.valid()) {
        outError = "missing prepared collision shape";
        return false;
    }
    if (!initialize(outError)) return false;
    JPH::BodyCreationSettings bodySettings(prepared.impl->shape, JPH::RVec3::sZero(),
        JPH::Quat::sIdentity(), JPH::EMotionType::Static, kStaticLayer);
    const JPH::BodyID body = m_impl->physics.GetBodyInterface().CreateAndAddBody(
        bodySettings, JPH::EActivation::DontActivate);
    if (body.IsInvalid()) {
        outError = "Jolt rejected streamed collision body";
        return false;
    }
    removeStreamedStaticCollision(residencyToken);
    m_impl->streamedStaticBodies.emplace(residencyToken, body);
    outError.clear();
    return true;
}

bool BethesdaPhysicsWorld::removeStreamedStaticCollision(std::uint64_t residencyToken) {
    const auto found = m_impl->streamedStaticBodies.find(residencyToken);
    if (found == m_impl->streamedStaticBodies.end()) return false;
    JPH::BodyInterface& bodies = m_impl->physics.GetBodyInterface();
    bodies.RemoveBody(found->second);
    bodies.DestroyBody(found->second);
    m_impl->streamedStaticBodies.erase(found);
    return true;
}

void BethesdaPhysicsWorld::clearStreamedStaticCollision() {
    if (!m_impl->initialized) return;
    JPH::BodyInterface& bodies = m_impl->physics.GetBodyInterface();
    for (const auto& [token, id] : m_impl->streamedStaticBodies) {
        (void)token;
        bodies.RemoveBody(id);
        bodies.DestroyBody(id);
    }
    m_impl->streamedStaticBodies.clear();
}

void BethesdaPhysicsWorld::optimizeBroadPhase() {
    if (!m_impl->initialized) return;
    m_impl->physics.OptimizeBroadPhase();
}

bool BethesdaPhysicsWorld::addCharacter(
    ObjectId object, const PhysicsCharacterConfig& config, std::string& outError) {
    if (!initialize(outError)) return false;
    if (!object.valid()) { outError = "invalid character ObjectId"; return false; }
    if (m_impl->characters.contains(object)) { outError = "character already exists"; return false; }
    const float horizontal = std::max(std::fabs(config.boundsHalfExtents.x),
        std::fabs(config.boundsHalfExtents.z));
    const float radius = std::clamp(horizontal * kBethesdaUnitsToJoltMetres, 0.18f, 0.65f);
    const float halfHeight = std::clamp(std::fabs(config.boundsHalfExtents.y) *
        kBethesdaUnitsToJoltMetres, radius + 0.1f, 1.6f);
    JPH::CapsuleShapeSettings capsule(std::max(0.1f, halfHeight - radius), radius);
    const auto shape = capsule.Create();
    if (shape.HasError()) { outError = "Jolt capsule construction failed: " + shape.GetError(); return false; }
    if (!validCharacterMovementSettings(config.movement)) { outError = "invalid native movement settings"; return false; }
    JPH::CharacterVirtualSettings settings;
    settings.mShape = shape.Get();
    settings.mUp = JPH::Vec3::sAxisY();
    settings.mMaxSlopeAngle = JPH::DegreesToRadians(std::clamp(config.maxSlopeDegrees, 0.0f, 89.0f));
    settings.mEnhancedInternalEdgeRemoval = true;
    JPH::RVec3 centre = toJoltPosition(config.position);
    centre += JPH::RVec3(0.0, static_cast<double>(halfHeight), 0.0);
    JPH::Ref<JPH::CharacterVirtual> character = new JPH::CharacterVirtual(&settings,
        centre, toJoltRotation(config.rotation), userDataFor(object),
        &m_impl->physics);
    character->SetCharacterVsCharacterCollision(&m_impl->characterCollision);
    m_impl->characterCollision.Add(character);
    Impl::CharacterEntry entry;
    entry.character = character;
    entry.movementSettings = config.movement;
    entry.stepHeightMetres = std::clamp(config.stepHeight * kBethesdaUnitsToJoltMetres, 0.05f, 0.6f);
    entry.centreOffsetMetres = halfHeight;
    entry.last.position = config.position;
    entry.last.rotation = config.rotation;
    m_impl->characters.emplace(object, std::move(entry));
    m_impl->objectsByUserData[userDataFor(object)] = object;
    return true;
}

bool BethesdaPhysicsWorld::removeCharacter(ObjectId object) {
    const auto found = m_impl->characters.find(object);
    if (found == m_impl->characters.end()) return false;
    (void)removeRagdoll(object);
    if (!found->second.suspendedByRagdoll)
        m_impl->characterCollision.Remove(found->second.character);
    m_impl->objectsByUserData.erase(userDataFor(object));
    m_impl->characters.erase(found);
    return true;
}

bool BethesdaPhysicsWorld::setCharacterInput(ObjectId object, const PhysicsCharacterInput& input) {
    const auto found = m_impl->characters.find(object);
    if (found == m_impl->characters.end()) return false;
    found->second.input = input;
    return true;
}

bool BethesdaPhysicsWorld::addCharacterImpulse(
    ObjectId object, const odai::math::Vector3& velocityChange) {
    const auto found = m_impl->characters.find(object);
    if (found == m_impl->characters.end() ||
        !std::isfinite(velocityChange.x) || !std::isfinite(velocityChange.y) ||
        !std::isfinite(velocityChange.z)) {
        return false;
    }
    found->second.externalVelocity += toJoltVector(velocityChange);
    return true;
}

bool BethesdaPhysicsWorld::hasCharacter(ObjectId object) const {
    return m_impl->characters.contains(object);
}

bool BethesdaPhysicsWorld::addDynamicBody(
    ObjectId object, const PhysicsDynamicBodyConfig& config,
    std::string& outError) {
    if (!initialize(outError)) return false;
    const auto finite = [](const odai::math::Vector3& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
    };
    if (!object.valid() || m_impl->dynamicBodies.contains(object) ||
        !finite(config.position) || !finite(config.boundsHalfExtents) ||
        config.boundsHalfExtents.x <= 0.0f || config.boundsHalfExtents.y <= 0.0f ||
        config.boundsHalfExtents.z <= 0.0f || !std::isfinite(config.massKilograms) ||
        config.massKilograms <= 0.0f) {
        outError = "invalid or duplicate dynamic body";
        return false;
    }
    const JPH::Vec3 half = toJoltVector(config.boundsHalfExtents);
    JPH::BoxShapeSettings shapeSettings(JPH::Vec3(
        std::max(0.01f, half.GetX()), std::max(0.01f, half.GetY()),
        std::max(0.01f, half.GetZ())));
    const auto shape = shapeSettings.Create();
    if (shape.HasError()) {
        outError = "Jolt dynamic box construction failed: " + shape.GetError();
        return false;
    }
    JPH::BodyCreationSettings settings(shape.Get(), toJoltPosition(config.position),
        toJoltRotation(config.rotation), JPH::EMotionType::Dynamic, kDynamicLayer);
    settings.mUserData = userDataFor(object);
    settings.mFriction = std::clamp(config.friction, 0.0f, 1.0f);
    settings.mRestitution = std::clamp(config.restitution, 0.0f, 1.0f);
    settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    settings.mMassPropertiesOverride.mMass = config.massKilograms;
    const JPH::BodyID body = m_impl->physics.GetBodyInterface().CreateAndAddBody(
        settings, JPH::EActivation::Activate);
    if (body.IsInvalid()) {
        outError = "Jolt rejected dynamic body";
        return false;
    }
    m_impl->dynamicBodies.emplace(object, Impl::DynamicEntry{body, config});
    m_impl->objectsByUserData[userDataFor(object)] = object;
    outError.clear();
    return true;
}

bool BethesdaPhysicsWorld::removeDynamicBody(ObjectId object) {
    const auto found = m_impl->dynamicBodies.find(object);
    if (found == m_impl->dynamicBodies.end()) return false;
    (void)removeConstraint(object);
    JPH::BodyInterface& bodies = m_impl->physics.GetBodyInterface();
    bodies.RemoveBody(found->second.body);
    bodies.DestroyBody(found->second.body);
    m_impl->objectsByUserData.erase(userDataFor(object));
    m_impl->dynamicBodies.erase(found);
    return true;
}

bool BethesdaPhysicsWorld::hasDynamicBody(ObjectId object) const {
    return m_impl->dynamicBodies.contains(object);
}

bool BethesdaPhysicsWorld::addWorldHingeConstraint(
    ObjectId object, const PhysicsHingeConfig& config, std::string& outError) {
    const auto found = m_impl->dynamicBodies.find(object);
    const auto finite = [](const odai::math::Vector3& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
    };
    const odai::math::Vector3 cross = odai::math::cross(
        config.hingeAxis, config.normalAxis);
    if (found == m_impl->dynamicBodies.end() || m_impl->constraints.contains(object) ||
        !finite(config.worldAnchor) || !finite(config.hingeAxis) ||
        !finite(config.normalAxis) || odai::math::length(config.hingeAxis) < 1.0e-5f ||
        odai::math::length(config.normalAxis) < 1.0e-5f ||
        odai::math::length(cross) < 1.0e-5f ||
        !std::isfinite(config.minimumAngleRadians) ||
        !std::isfinite(config.maximumAngleRadians) ||
        config.minimumAngleRadians > 0.0f || config.maximumAngleRadians < 0.0f ||
        config.minimumAngleRadians > config.maximumAngleRadians ||
        !std::isfinite(config.frictionTorqueNewtonMetres) ||
        config.frictionTorqueNewtonMetres < 0.0f) {
        outError = "invalid or duplicate world hinge constraint";
        return false;
    }
    JPH::BodyLockWrite lock(
        m_impl->physics.GetBodyLockInterface(), found->second.body);
    if (!lock.Succeeded()) {
        outError = "could not lock dynamic body for hinge constraint";
        return false;
    }
    const odai::math::Vector3 hinge = odai::math::normalize(config.hingeAxis);
    const odai::math::Vector3 normal = odai::math::normalize(config.normalAxis);
    JPH::HingeConstraintSettings settings;
    settings.mSpace = JPH::EConstraintSpace::WorldSpace;
    settings.mPoint1 = settings.mPoint2 = toJoltPosition(config.worldAnchor);
    settings.mHingeAxis1 = settings.mHingeAxis2 =
        JPH::Vec3(hinge.x, hinge.y, hinge.z);
    settings.mNormalAxis1 = settings.mNormalAxis2 =
        JPH::Vec3(normal.x, normal.y, normal.z);
    settings.mLimitsMin = std::clamp(
        config.minimumAngleRadians, -JPH::JPH_PI, 0.0f);
    settings.mLimitsMax = std::clamp(
        config.maximumAngleRadians, 0.0f, JPH::JPH_PI);
    settings.mMaxFrictionTorque = config.frictionTorqueNewtonMetres;
    JPH::Ref<JPH::Constraint> constraint =
        settings.Create(JPH::Body::sFixedToWorld, lock.GetBody());
    if (constraint == nullptr) {
        outError = "Jolt rejected world hinge constraint";
        return false;
    }
    m_impl->physics.AddConstraint(constraint);
    m_impl->constraints.emplace(object, std::move(constraint));
    outError.clear();
    return true;
}

bool BethesdaPhysicsWorld::removeConstraint(ObjectId object) {
    const auto found = m_impl->constraints.find(object);
    if (found == m_impl->constraints.end()) return false;
    m_impl->physics.RemoveConstraint(found->second);
    m_impl->constraints.erase(found);
    return true;
}

bool BethesdaPhysicsWorld::hasConstraint(ObjectId object) const {
    return m_impl->constraints.contains(object);
}

bool BethesdaPhysicsWorld::addDynamicBodyImpulse(
    ObjectId object, const odai::math::Vector3& impulseKilogramUnitsPerSecond) {
    const auto found = m_impl->dynamicBodies.find(object);
    if (found == m_impl->dynamicBodies.end() ||
        !std::isfinite(impulseKilogramUnitsPerSecond.x) ||
        !std::isfinite(impulseKilogramUnitsPerSecond.y) ||
        !std::isfinite(impulseKilogramUnitsPerSecond.z)) return false;
    m_impl->physics.GetBodyInterface().AddImpulse(
        found->second.body, toJoltVector(impulseKilogramUnitsPerSecond));
    return true;
}

bool BethesdaPhysicsWorld::setDynamicBodyTransform(
    ObjectId object, const odai::math::Vector3& position,
    const odai::math::Quaternion& rotation, bool activate) {
    const auto found = m_impl->dynamicBodies.find(object);
    if (found == m_impl->dynamicBodies.end() || !std::isfinite(position.x) ||
        !std::isfinite(position.y) || !std::isfinite(position.z)) return false;
    m_impl->physics.GetBodyInterface().SetPositionAndRotation(found->second.body,
        toJoltPosition(position), toJoltRotation(rotation),
        activate ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
    return true;
}

bool BethesdaPhysicsWorld::applyBuoyancy(
    ObjectId object, float waterHeightBethesdaUnits,
    float fluidDensityKilogramsPerCubicMetre, float fixedDeltaSeconds) {
    const auto found = m_impl->dynamicBodies.find(object);
    if (found == m_impl->dynamicBodies.end() || !found->second.config.buoyant ||
        !std::isfinite(waterHeightBethesdaUnits) ||
        !std::isfinite(fluidDensityKilogramsPerCubicMetre) ||
        fluidDensityKilogramsPerCubicMetre <= 0.0f ||
        !std::isfinite(fixedDeltaSeconds) || fixedDeltaSeconds <= 0.0f) return false;
    JPH::BodyInterface& bodies = m_impl->physics.GetBodyInterface();
    const odai::math::Vector3 centre = fromJoltPosition(bodies.GetPosition(found->second.body));
    const float bottom = centre.y - found->second.config.boundsHalfExtents.y;
    const float height = found->second.config.boundsHalfExtents.y * 2.0f;
    const float submerged = std::clamp(
        (waterHeightBethesdaUnits - bottom) / std::max(1.0f, height), 0.0f, 1.0f);
    if (submerged <= 0.0f) return true;
    const odai::math::Vector3 half = found->second.config.boundsHalfExtents *
        kBethesdaUnitsToJoltMetres;
    const float volume = 8.0f * half.x * half.y * half.z;
    const float force = fluidDensityKilogramsPerCubicMetre * volume * 9.81f * submerged;
    bodies.AddForce(found->second.body, JPH::Vec3(0.0f, force, 0.0f));
    return true;
}

std::vector<PhysicsDynamicBodySnapshot> BethesdaPhysicsWorld::dynamicBodySnapshots() const {
    std::vector<PhysicsDynamicBodySnapshot> snapshots;
    snapshots.reserve(m_impl->dynamicBodies.size());
    const JPH::BodyInterface& bodies = m_impl->physics.GetBodyInterface();
    for (const auto& [object, entry] : m_impl->dynamicBodies) {
        snapshots.push_back(PhysicsDynamicBodySnapshot{object,
            fromJoltPosition(bodies.GetPosition(entry.body)),
            fromJoltRotation(bodies.GetRotation(entry.body)),
            fromJoltVector(bodies.GetLinearVelocity(entry.body)),
            fromJoltVector(bodies.GetAngularVelocity(entry.body)),
            bodies.IsActive(entry.body)});
    }
    return snapshots;
}

bool BethesdaPhysicsWorld::restoreDynamicBody(
    const PhysicsDynamicBodySnapshot& snapshot, std::string& outError) {
    const auto found = m_impl->dynamicBodies.find(snapshot.object);
    if (found == m_impl->dynamicBodies.end()) {
        outError = "saved Jolt dynamic body is not registered: " +
            snapshot.object.toString();
        return false;
    }
    const auto finite = [](const odai::math::Vector3& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
    };
    if (!finite(snapshot.position) || !finite(snapshot.linearVelocity) ||
        !finite(snapshot.angularVelocity)) {
        outError = "invalid saved Jolt dynamic body transform";
        return false;
    }
    JPH::BodyInterface& bodies = m_impl->physics.GetBodyInterface();
    bodies.SetPositionAndRotation(found->second.body, toJoltPosition(snapshot.position),
        toJoltRotation(snapshot.rotation), snapshot.active
            ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
    bodies.SetLinearAndAngularVelocity(found->second.body,
        toJoltVector(snapshot.linearVelocity), toJoltVector(snapshot.angularVelocity));
    outError.clear();
    return true;
}

bool BethesdaPhysicsWorld::activateRagdoll(ObjectId object,
    std::span<const PhysicsRagdollJointConfig> joints,
    const odai::math::Vector3& linearVelocity, std::string& outError) {
    outError.clear();
    const auto character = m_impl->characters.find(object);
    const auto finite = [](const odai::math::Vector3& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
    };
    if (character == m_impl->characters.end() || character->second.suspendedByRagdoll ||
        m_impl->ragdolls.contains(object) || joints.empty() || joints.size() > 32 ||
        !finite(linearVelocity)) {
        outError = "invalid or duplicate ragdoll activation";
        return false;
    }
    std::set<std::string> roles;
    for (std::size_t i = 0; i < joints.size(); ++i) {
        const PhysicsRagdollJointConfig& joint = joints[i];
        if (joint.role.empty() || !roles.insert(joint.role).second ||
            joint.parent < -1 || joint.parent >= static_cast<int>(i) ||
            !finite(joint.position) || !std::isfinite(joint.radius) ||
            !std::isfinite(joint.halfHeight) || !std::isfinite(joint.massKilograms) ||
            joint.radius <= 0.0f || joint.halfHeight < joint.radius ||
            joint.massKilograms <= 0.0f) {
            outError = "invalid canonical ragdoll joint: " + joint.role;
            return false;
        }
    }

    JPH::BodyInterface& bodies = m_impl->physics.GetBodyInterface();
    Impl::RagdollEntry entry;
    entry.groupFilter = new JPH::GroupFilterTable(
        static_cast<JPH::uint>(joints.size()));
    for (std::size_t i = 0; i < joints.size(); ++i)
        if (joints[i].parent >= 0)
            entry.groupFilter->DisableCollision(
                static_cast<JPH::CollisionGroup::SubGroupID>(joints[i].parent),
                static_cast<JPH::CollisionGroup::SubGroupID>(i));
    const JPH::CollisionGroup::GroupID group = m_impl->nextRagdollGroup++;
    if (m_impl->nextRagdollGroup == JPH::CollisionGroup::cInvalidGroup)
        m_impl->nextRagdollGroup = 1u;
    std::vector<JPH::Body*> created;
    created.reserve(joints.size());
    entry.roles.reserve(joints.size());
    entry.bodies.reserve(joints.size());
    for (const PhysicsRagdollJointConfig& joint : joints) {
        const float radius = std::clamp(
            joint.radius * kBethesdaUnitsToJoltMetres, 0.025f, 0.45f);
        const float halfHeight = std::clamp(
            joint.halfHeight * kBethesdaUnitsToJoltMetres, radius, 0.8f);
        JPH::CapsuleShapeSettings shapeSettings(
            std::max(0.01f, halfHeight - radius), radius);
        const auto shape = shapeSettings.Create();
        if (shape.HasError()) {
            outError = "Jolt ragdoll capsule construction failed: " + shape.GetError();
            break;
        }
        JPH::BodyCreationSettings settings(shape.Get(), toJoltPosition(joint.position),
            toJoltRotation(joint.rotation), JPH::EMotionType::Dynamic, kDynamicLayer);
        settings.mUserData = userDataFor(object);
        settings.mCollisionGroup = JPH::CollisionGroup(entry.groupFilter, group,
            static_cast<JPH::CollisionGroup::SubGroupID>(created.size()));
        settings.mFriction = 0.7f;
        settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
        settings.mMassPropertiesOverride.mMass = joint.massKilograms;
        JPH::Body* body = bodies.CreateBody(settings);
        if (body == nullptr) {
            outError = "Jolt ran out of bodies while building ragdoll";
            break;
        }
        created.push_back(body);
        entry.roles.push_back(joint.role);
        entry.bodies.push_back(body->GetID());
    }
    if (!outError.empty()) {
        for (JPH::Body* body : created) bodies.DestroyBody(body->GetID());
        return false;
    }
    for (std::size_t i = 0; i < joints.size(); ++i) {
        if (joints[i].parent < 0) continue;
        JPH::PointConstraintSettings settings;
        settings.mSpace = JPH::EConstraintSpace::WorldSpace;
        settings.mPoint1 = settings.mPoint2 = toJoltPosition(joints[i].position);
        JPH::Ref<JPH::Constraint> constraint = settings.Create(
            *created[static_cast<std::size_t>(joints[i].parent)], *created[i]);
        if (constraint == nullptr) {
            outError = "Jolt rejected ragdoll parent constraint";
            for (JPH::Body* body : created) bodies.DestroyBody(body->GetID());
            return false;
        }
        entry.constraints.push_back(std::move(constraint));
    }
    for (JPH::BodyID body : entry.bodies) {
        bodies.AddBody(body, JPH::EActivation::Activate);
        bodies.SetLinearVelocity(body, toJoltVector(linearVelocity));
    }
    for (const auto& constraint : entry.constraints)
        m_impl->physics.AddConstraint(constraint);
    m_impl->characterCollision.Remove(character->second.character);
    character->second.suspendedByRagdoll = true;
    m_impl->ragdolls.emplace(object, std::move(entry));
    return true;
}

bool BethesdaPhysicsWorld::removeRagdoll(ObjectId object) {
    const auto found = m_impl->ragdolls.find(object);
    if (found == m_impl->ragdolls.end()) return false;
    for (const auto& constraint : found->second.constraints)
        m_impl->physics.RemoveConstraint(constraint);
    JPH::BodyInterface& bodies = m_impl->physics.GetBodyInterface();
    for (JPH::BodyID body : found->second.bodies) {
        bodies.RemoveBody(body);
        bodies.DestroyBody(body);
    }
    m_impl->ragdolls.erase(found);
    if (const auto character = m_impl->characters.find(object);
        character != m_impl->characters.end() && character->second.suspendedByRagdoll) {
        character->second.suspendedByRagdoll = false;
        m_impl->characterCollision.Add(character->second.character);
    }
    return true;
}

bool BethesdaPhysicsWorld::hasActiveRagdoll(ObjectId object) const {
    return m_impl->ragdolls.contains(object);
}

std::optional<PhysicsRagdollSnapshot> BethesdaPhysicsWorld::ragdollSnapshot(
    ObjectId object) const {
    const auto found = m_impl->ragdolls.find(object);
    if (found == m_impl->ragdolls.end()) return std::nullopt;
    PhysicsRagdollSnapshot snapshot;
    snapshot.object = object;
    snapshot.active = true;
    const JPH::BodyInterface& bodies = m_impl->physics.GetBodyInterface();
    for (std::size_t i = 0; i < found->second.bodies.size(); ++i) {
        const JPH::BodyID body = found->second.bodies[i];
        snapshot.joints.push_back({found->second.roles[i],
            fromJoltPosition(bodies.GetPosition(body)),
            fromJoltRotation(bodies.GetRotation(body)),
            fromJoltVector(bodies.GetLinearVelocity(body))});
    }
    return snapshot;
}

std::vector<PhysicsRagdollSnapshot> BethesdaPhysicsWorld::ragdollSnapshots() const {
    std::vector<PhysicsRagdollSnapshot> result;
    result.reserve(m_impl->ragdolls.size());
    for (const auto& [object, entry] : m_impl->ragdolls) {
        (void)entry;
        if (auto snapshot = ragdollSnapshot(object)) result.push_back(std::move(*snapshot));
    }
    return result;
}

bool BethesdaPhysicsWorld::restoreRagdoll(
    const PhysicsRagdollSnapshot& snapshot, std::string& outError) {
    if (!snapshot.active || snapshot.joints.empty()) {
        outError = "saved ragdoll is inactive or empty";
        return false;
    }
    const std::map<std::string, std::string> parents{
        {"spine", "pelvis"}, {"head", "spine"},
        {"left_upper_arm", "spine"}, {"left_forearm", "left_upper_arm"},
        {"left_hand", "left_forearm"}, {"right_upper_arm", "spine"},
        {"right_forearm", "right_upper_arm"}, {"right_hand", "right_forearm"},
        {"left_thigh", "pelvis"}, {"left_calf", "left_thigh"},
        {"left_foot", "left_calf"}, {"right_thigh", "pelvis"},
        {"right_calf", "right_thigh"}, {"right_foot", "right_calf"}};
    std::map<std::string, int> indices;
    std::vector<PhysicsRagdollJointConfig> config;
    config.reserve(snapshot.joints.size());
    for (const PhysicsRagdollJointPose& saved : snapshot.joints) {
        PhysicsRagdollJointConfig joint;
        joint.role = saved.role;
        joint.position = saved.position;
        joint.rotation = saved.rotation;
        if (const auto parent = parents.find(saved.role); parent != parents.end()) {
            const auto index = indices.find(parent->second);
            if (index == indices.end()) {
                outError = "saved ragdoll joints are not parent-before-child";
                return false;
            }
            joint.parent = index->second;
        }
        const bool torso = saved.role == "pelvis" || saved.role == "spine";
        const bool head = saved.role == "head";
        joint.radius = torso ? 10.0f : (head ? 9.0f : 5.0f);
        joint.halfHeight = torso ? 17.0f : (head ? 9.0f : 13.0f);
        joint.massKilograms = torso ? 11.0f : (head ? 5.0f : 3.0f);
        indices.emplace(saved.role, static_cast<int>(config.size()));
        config.push_back(std::move(joint));
    }
    if (!activateRagdoll(snapshot.object, config, {}, outError)) return false;
    const auto found = m_impl->ragdolls.find(snapshot.object);
    JPH::BodyInterface& bodies = m_impl->physics.GetBodyInterface();
    for (std::size_t i = 0; i < snapshot.joints.size(); ++i)
        bodies.SetLinearVelocity(found->second.bodies[i],
            toJoltVector(snapshot.joints[i].linearVelocity));
    return true;
}

bool BethesdaPhysicsWorld::recoverRagdoll(ObjectId object,
    float maximumDropBethesdaUnits, odai::math::Vector3& outPlacement,
    std::string& outError) {
    outError.clear();
    const auto ragdoll = ragdollSnapshot(object);
    const auto character = m_impl->characters.find(object);
    if (!ragdoll || ragdoll->joints.empty() || character == m_impl->characters.end() ||
        !std::isfinite(maximumDropBethesdaUnits) || maximumDropBethesdaUnits <= 0.0f) {
        outError = "ragdoll is unavailable for recovery";
        return false;
    }
    const auto support = castDown(ragdoll->joints.front().position, maximumDropBethesdaUnits);
    if (!support || support->normal.y < std::cos(JPH::DegreesToRadians(50.0f))) {
        outError = "no valid get-up support surface";
        return false;
    }
    outPlacement = support->position;
    if (!removeRagdoll(object)) {
        outError = "ragdoll disappeared during recovery";
        return false;
    }
    Impl::CharacterEntry& entry = character->second;
    JPH::RVec3 centre = toJoltPosition(outPlacement) +
        JPH::RVec3(0.0, static_cast<double>(entry.centreOffsetMetres), 0.0);
    entry.character->SetPosition(centre);
    entry.character->SetLinearVelocity(JPH::Vec3::sZero());
    entry.last.position = outPlacement;
    entry.last.velocity = {};
    entry.last.grounded = true;
    entry.last.landed = false;
    entry.externalVelocity = JPH::Vec3::sZero();
    entry.suspendedByRagdoll = false;
    return true;
}

std::vector<std::pair<ObjectId, PhysicsCharacterStep>> BethesdaPhysicsWorld::step(float fixedDeltaSeconds) {
    std::vector<std::pair<ObjectId, PhysicsCharacterStep>> results;
    if (!m_impl->initialized) return results;
    const float delta = std::clamp(fixedDeltaSeconds, 1.0e-5f, 0.25f);
    const JPH::Vec3 gravity(0.0f, -9.81f, 0.0f);
    for (auto& [id, entry] : m_impl->characters) {
        if (entry.suspendedByRagdoll) continue;
        const bool wasGrounded = entry.last.grounded;
        JPH::Vec3 desired = toJoltVector(entry.input.desiredVelocity);
        if (entry.input.animationDriven) desired = toJoltVector(entry.input.rootMotion) / delta;
        const JPH::Vec3 oldVelocity = entry.character->GetLinearVelocity();
        const bool wasSupported = entry.character->IsSupported();
        bool launched = false;
        if (entry.movementSettings.enabled && !entry.input.animationDriven) {
            float x = desired.GetX(), z = desired.GetZ();
            launched = advanceCharacterMovement(entry.movement, entry.movementSettings,
                wasSupported, entry.last.landed, entry.input.jumpRequested, delta, x, z);
            desired.SetX(x); desired.SetZ(z); desired.SetY(0);
        }
        if (wasSupported) {
            // A positive controller Y is a jump request. Move it into the
            // external channel once so holding jump cannot reapply it in air.
            if (!entry.movementSettings.enabled && desired.GetY() > 0.0f) {
                entry.externalVelocity.SetY(
                    std::max(entry.externalVelocity.GetY(), desired.GetY()));
            } else if (entry.externalVelocity.GetY() < 0.0f) {
                entry.externalVelocity.SetY(0.0f);
            }
            desired.SetY(0.0f);
            desired += entry.character->GetGroundVelocity();
        } else {
            desired.SetY(0.0f);
            entry.externalVelocity.SetY(oldVelocity.GetY() + gravity.GetY() * delta);
        }
        if (launched) entry.externalVelocity.SetY(std::max(entry.externalVelocity.GetY(), entry.movementSettings.jumpSpeedMetres));
        desired += entry.externalVelocity;
        const JPH::RVec3 before = entry.character->GetPosition();
        entry.character->SetLinearVelocity(desired);
        JPH::CharacterVirtual::ExtendedUpdateSettings settings;
        settings.mStickToFloorStepDown = JPH::Vec3(0.0f, -entry.stepHeightMetres, 0.0f);
        settings.mWalkStairsStepUp = JPH::Vec3(0.0f, entry.stepHeightMetres, 0.0f);
        entry.character->ExtendedUpdate(delta, gravity, settings,
            m_impl->physics.GetDefaultBroadPhaseLayerFilter(kCharacterLayer),
            m_impl->physics.GetDefaultLayerFilter(kCharacterLayer), JPH::BodyFilter{},
            JPH::ShapeFilter{}, m_impl->allocator);
        const JPH::RVec3 after = entry.character->GetPosition();
        entry.last.position = fromJoltPosition(
            after - JPH::RVec3(0.0, static_cast<double>(entry.centreOffsetMetres), 0.0));
        entry.last.rotation = fromJoltRotation(entry.character->GetRotation());
        entry.last.velocity = fromJoltVector(entry.character->GetLinearVelocity());
        entry.last.groundVelocity = fromJoltVector(entry.character->GetGroundVelocity());
        entry.last.groundNormal = fromJoltVector(entry.character->GetGroundNormal()) *
            kBethesdaUnitsToJoltMetres;
        entry.last.grounded = entry.character->IsSupported();
        entry.last.falling = !entry.last.grounded && entry.last.velocity.y < 0.0f;
        entry.last.landed = !wasGrounded && entry.last.grounded;
        entry.last.leftLedge = wasGrounded && !entry.last.grounded && !launched;
        entry.last.landingImpactMetres = entry.last.landed
            ? std::max(0.f, -(desired - entry.character->GetGroundVelocity()).Dot(entry.character->GetGroundNormal())) : 0.f;
        entry.last.landingSeverity = classifyLanding(entry.last.landingImpactMetres, entry.movementSettings);
        entry.last.jumpPhase = launched ? JumpPhase::Takeoff : entry.last.landed ? JumpPhase::Landing :
            entry.last.grounded ? JumpPhase::Grounded : std::abs(entry.last.velocity.y * kBethesdaUnitsToJoltMetres) < .2f
                ? JumpPhase::Apex : entry.last.falling ? JumpPhase::Falling : JumpPhase::Ascending;
        const JPH::Vec3 actual = JPH::Vec3(after - before) / delta;
        const float desiredHorizontal = std::sqrt(desired.GetX() * desired.GetX() + desired.GetZ() * desired.GetZ());
        const float actualHorizontal = std::sqrt(actual.GetX() * actual.GetX() + actual.GetZ() * actual.GetZ());
        entry.last.blocked = desiredHorizontal > 0.1f && actualHorizontal < desiredHorizontal * 0.5f;
        // Knockback retains momentum in the air but settles quickly on the
        // ground. Locomotion remains a separate input and is never damped.
        const float horizontalDamping =
            std::exp(-(entry.last.grounded ? 3.5f : 0.25f) * delta);
        entry.externalVelocity.SetX(entry.externalVelocity.GetX() * horizontalDamping);
        entry.externalVelocity.SetZ(entry.externalVelocity.GetZ() * horizontalDamping);
        entry.externalVelocity.SetY(entry.last.grounded
                ? std::max(0.0f, entry.externalVelocity.GetY())
                : entry.character->GetLinearVelocity().GetY());
        const auto support = m_impl->objectsByUserData.find(entry.character->GetGroundUserData());
        entry.last.supportingObject = support == m_impl->objectsByUserData.end() ?
            std::optional<ObjectId>{} : std::optional<ObjectId>{support->second};
        results.emplace_back(id, entry.last);
    }
    m_impl->physics.Update(delta, 1, &m_impl->allocator, &m_impl->jobs);
    return results;
}

std::optional<PhysicsCharacterStep> BethesdaPhysicsWorld::characterState(ObjectId object) const {
    const auto found = m_impl->characters.find(object);
    return found == m_impl->characters.end() ? std::nullopt : std::optional(found->second.last);
}

std::vector<PhysicsCharacterSnapshot> BethesdaPhysicsWorld::snapshot() const {
    std::vector<PhysicsCharacterSnapshot> result;
    result.reserve(m_impl->characters.size());
    for (const auto& [id, entry] : m_impl->characters) {
        result.push_back({id, entry.last.position, entry.last.rotation, entry.last.velocity,
            entry.last.groundNormal, entry.last.grounded, entry.last.supportingObject, entry.movement});
    }
    return result;
}

bool BethesdaPhysicsWorld::restoreCharacter(
    const PhysicsCharacterSnapshot& saved, std::string& outError) {
    const auto found = m_impl->characters.find(saved.object);
    if (found == m_impl->characters.end()) {
        outError = "saved Jolt character is not registered: " + saved.object.toString();
        return false;
    }
    if (!std::isfinite(saved.position.x) || !std::isfinite(saved.position.y) ||
        !std::isfinite(saved.position.z) || !std::isfinite(saved.rotation.x) ||
        !std::isfinite(saved.rotation.y) || !std::isfinite(saved.rotation.z) ||
        !std::isfinite(saved.rotation.w) || !std::isfinite(saved.velocity.x) ||
        !std::isfinite(saved.velocity.y) || !std::isfinite(saved.velocity.z) ||
        !std::isfinite(saved.groundNormal.x) || !std::isfinite(saved.groundNormal.y) ||
        !std::isfinite(saved.groundNormal.z) || !validCharacterMovementState(saved.movement)) {
        outError = "invalid saved Jolt character transform";
        return false;
    }
    if (!validCharacterMovementState(saved.movement)) { outError = "invalid saved movement state"; return false; }
    found->second.movement = saved.movement;
    found->second.externalVelocity = JPH::Vec3(0, saved.velocity.y * kBethesdaUnitsToJoltMetres, 0);
    JPH::RVec3 centre = toJoltPosition(saved.position);
    centre += JPH::RVec3(
        0.0, static_cast<double>(found->second.centreOffsetMetres), 0.0);
    found->second.character->SetPosition(centre);
    found->second.character->SetRotation(toJoltRotation(saved.rotation));
    found->second.character->SetLinearVelocity(toJoltVector(saved.velocity));
    found->second.last.position = saved.position;
    found->second.last.rotation = saved.rotation;
    found->second.last.velocity = saved.velocity;
    found->second.last.groundNormal = saved.groundNormal;
    found->second.last.grounded = saved.grounded;
    found->second.last.supportingObject = saved.supportingObject;
    found->second.last.landed = false;
    found->second.last.leftLedge = false;
    found->second.last.landingImpactMetres = 0;
    found->second.last.falling = !saved.grounded && saved.velocity.y < 0;
    outError.clear();
    return true;
}

bool BethesdaPhysicsWorld::restore(
    std::span<const PhysicsCharacterSnapshot> snapshots, std::string& outError) {
    outError.clear();
    if (snapshots.size() != m_impl->characters.size()) {
        outError = "saved Jolt character set does not match the registered runtime actors";
        return false;
    }
    std::set<ObjectId> seen;
    for (const PhysicsCharacterSnapshot& saved : snapshots) {
        const auto found = m_impl->characters.find(saved.object);
        if (found == m_impl->characters.end() || !seen.insert(saved.object).second) {
            outError = "saved Jolt character is missing or duplicated: " + saved.object.toString();
            return false;
        }
        if (!std::isfinite(saved.position.x) || !std::isfinite(saved.position.y) ||
            !std::isfinite(saved.position.z) || !std::isfinite(saved.rotation.x) ||
            !std::isfinite(saved.rotation.y) || !std::isfinite(saved.rotation.z) ||
            !std::isfinite(saved.rotation.w) || !std::isfinite(saved.velocity.x) ||
            !std::isfinite(saved.velocity.y) || !std::isfinite(saved.velocity.z) ||
            !std::isfinite(saved.groundNormal.x) || !std::isfinite(saved.groundNormal.y) ||
            !std::isfinite(saved.groundNormal.z) || !validCharacterMovementState(saved.movement)) {
            outError = "invalid saved Jolt character transform";
            return false;
        }
    }
    for (const PhysicsCharacterSnapshot& saved : snapshots) {
        if (!restoreCharacter(saved, outError)) return false;
    }
    return true;
}

bool BethesdaPhysicsWorld::isCharacterPlacementClear(const PhysicsCharacterConfig& config,
    float penetrationToleranceBethesdaUnits) const {
    if (!m_impl->initialized || !std::isfinite(config.position.x) || !std::isfinite(config.position.y) ||
        !std::isfinite(config.position.z) || !std::isfinite(penetrationToleranceBethesdaUnits) || penetrationToleranceBethesdaUnits < 0)
        return false;
    for (float v : {config.boundsHalfExtents.x, config.boundsHalfExtents.y, config.boundsHalfExtents.z,
                    config.rotation.x, config.rotation.y, config.rotation.z, config.rotation.w})
        if (!std::isfinite(v)) return false;
    if (config.rotation.x*config.rotation.x + config.rotation.y*config.rotation.y +
        config.rotation.z*config.rotation.z + config.rotation.w*config.rotation.w < 1.e-8f) return false;
    const float horizontal = std::max(std::fabs(config.boundsHalfExtents.x),std::fabs(config.boundsHalfExtents.z));
    const float radius = std::clamp(horizontal*kBethesdaUnitsToJoltMetres,.18f,.65f);
    const float halfHeight = std::clamp(std::fabs(config.boundsHalfExtents.y)*kBethesdaUnitsToJoltMetres,radius+.1f,1.6f);
    JPH::CapsuleShape shape(std::max(.1f,halfHeight-radius),radius);
    auto center=toJoltPosition(config.position)+JPH::RVec3(0,halfHeight,0);
    class SolidFilter final : public JPH::ObjectLayerFilter {
        bool ShouldCollide(JPH::ObjectLayer layer) const override { return layer==kStaticLayer || layer==kDynamicLayer; }
    } filter;
    JPH::CollideShapeSettings settings;
    JPH::AllHitCollisionCollector<JPH::CollideShapeCollector> collector;
    m_impl->physics.GetNarrowPhaseQuery().CollideShape(&shape,JPH::Vec3::sReplicate(1),
        JPH::RMat44::sRotationTranslation(toJoltRotation(config.rotation),center),settings,center,collector,
        m_impl->physics.GetDefaultBroadPhaseLayerFilter(kCharacterLayer),filter);
    const float tolerance=penetrationToleranceBethesdaUnits*kBethesdaUnitsToJoltMetres;
    return std::none_of(collector.mHits.begin(),collector.mHits.end(),
        [&](const auto& hit){return hit.mPenetrationDepth>tolerance;});
}

std::optional<PhysicsCastHit> BethesdaPhysicsWorld::castDown(
    const odai::math::Vector3& origin, float distanceBethesdaUnits) const {
    if (!m_impl->initialized || !std::isfinite(distanceBethesdaUnits) ||
        distanceBethesdaUnits <= 0.0f) return std::nullopt;
    class StaticLayerFilter final : public JPH::ObjectLayerFilter {
    public:
        bool ShouldCollide(JPH::ObjectLayer layer) const override {
            return layer == kStaticLayer;
        }
    } staticLayerFilter;
    const JPH::RRayCast ray(toJoltPosition(origin),
        JPH::Vec3(0.0f, -distanceBethesdaUnits * kBethesdaUnitsToJoltMetres, 0.0f));
    JPH::RayCastResult hit;
    if (!m_impl->physics.GetNarrowPhaseQuery().CastRay(
            ray, hit, m_impl->physics.GetDefaultBroadPhaseLayerFilter(kCharacterLayer),
            staticLayerFilter)) return std::nullopt;
    PhysicsCastHit result;
    const JPH::RVec3 hitPosition = ray.GetPointOnRay(hit.mFraction);
    result.position = fromJoltPosition(hitPosition);
    result.distance = distanceBethesdaUnits * hit.mFraction;
    JPH::BodyLockRead lock(m_impl->physics.GetBodyLockInterface(), hit.mBodyID);
    if (lock.Succeeded()) {
        const JPH::Body& body = lock.GetBody();
        result.normal = fromJoltVector(
            body.GetWorldSpaceSurfaceNormal(hit.mSubShapeID2, hitPosition)) *
            kBethesdaUnitsToJoltMetres;
        const auto object = m_impl->objectsByUserData.find(body.GetUserData());
        if (object != m_impl->objectsByUserData.end()) result.object = object->second;
    }
    return result;
}

std::optional<PhysicsCastHit> BethesdaPhysicsWorld::castSphere(
    const odai::math::Vector3& from,
    const odai::math::Vector3& to,
    float radiusBethesdaUnits,
    std::optional<ObjectId> ignoredObject) const {
    const odai::math::Vector3 delta = to - from;
    const float length = odai::math::length(delta);
    if (!m_impl->initialized || !std::isfinite(from.x) || !std::isfinite(from.y) ||
        !std::isfinite(from.z) || !std::isfinite(to.x) || !std::isfinite(to.y) ||
        !std::isfinite(to.z) || !std::isfinite(radiusBethesdaUnits) ||
        radiusBethesdaUnits <= 0.0f || length <= 1.0e-5f) {
        return std::nullopt;
    }

    class CameraLayerFilter final : public JPH::ObjectLayerFilter {
    public:
        bool ShouldCollide(JPH::ObjectLayer layer) const override {
            return layer == kStaticLayer || layer == kDynamicLayer;
        }
    } cameraLayerFilter;
    class IgnoredObjectFilter final : public JPH::BodyFilter {
    public:
        explicit IgnoredObjectFilter(std::optional<ObjectId> object)
            : ignoredUserData(object.has_value()
                    ? std::optional<std::uint64_t>{userDataFor(*object)}
                    : std::nullopt) {}
        bool ShouldCollideLocked(const JPH::Body& body) const override {
            return !ignoredUserData.has_value() ||
                body.GetUserData() != *ignoredUserData;
        }
    private:
        std::optional<std::uint64_t> ignoredUserData;
    } bodyFilter(ignoredObject);

    JPH::SphereShape sphere(radiusBethesdaUnits * kBethesdaUnitsToJoltMetres);
    const JPH::RVec3 start = toJoltPosition(from);
    const JPH::Vec3 direction = toJoltPosition(to) - start;
    const JPH::RShapeCast cast(
        &sphere, JPH::Vec3::sReplicate(1.0f),
        JPH::RMat44::sTranslation(start), direction);
    JPH::ShapeCastSettings settings;
    settings.mReturnDeepestPoint = true;
    JPH::ClosestHitCollisionCollector<JPH::CastShapeCollector> collector;
    m_impl->physics.GetNarrowPhaseQuery().CastShape(
        cast, settings, start, collector,
        m_impl->physics.GetDefaultBroadPhaseLayerFilter(kCharacterLayer),
        cameraLayerFilter, bodyFilter);
    if (!collector.HadHit()) return std::nullopt;

    const JPH::ShapeCastResult& hit = collector.mHit;
    PhysicsCastHit result;
    result.position = fromJoltPosition(cast.GetPointOnRay(hit.mFraction));
    result.distance = length * std::clamp(hit.mFraction, 0.0f, 1.0f);
    const JPH::Vec3 normal = -hit.mPenetrationAxis.NormalizedOr(JPH::Vec3::sAxisZ());
    result.normal = {normal.GetX(), normal.GetY(), normal.GetZ()};
    JPH::BodyLockRead lock(m_impl->physics.GetBodyLockInterface(), hit.mBodyID2);
    if (lock.Succeeded()) {
        const auto object = m_impl->objectsByUserData.find(lock.GetBody().GetUserData());
        if (object != m_impl->objectsByUserData.end()) result.object = object->second;
    }
    return result;
}

bool BethesdaPhysicsWorld::hasLineOfSight(
    const odai::math::Vector3& from, const odai::math::Vector3& to) const {
    if (!std::isfinite(from.x) || !std::isfinite(from.y) ||
        !std::isfinite(from.z) || !std::isfinite(to.x) || !std::isfinite(to.y) ||
        !std::isfinite(to.z)) return false;
    if (!m_impl->initialized) return true;
    class OccluderLayerFilter final : public JPH::ObjectLayerFilter {
    public:
        bool ShouldCollide(JPH::ObjectLayer layer) const override {
            return layer == kStaticLayer || layer == kDynamicLayer;
        }
    } occluderFilter;
    const JPH::RRayCast ray(toJoltPosition(from),
        toJoltPosition(to) - toJoltPosition(from));
    JPH::RayCastResult hit;
    return !m_impl->physics.GetNarrowPhaseQuery().CastRay(
        ray, hit, m_impl->physics.GetDefaultBroadPhaseLayerFilter(kCharacterLayer),
        occluderFilter);
}

std::vector<PhysicsMeleeCandidate> BethesdaPhysicsWorld::meleeCandidates(
    ObjectId attacker,
    const odai::math::Vector3& forward,
    float rangeBethesdaUnits,
    float minimumFacingDot) const {
    std::vector<PhysicsMeleeCandidate> result;
    if (!m_impl->initialized || !attacker.valid() ||
        !std::isfinite(rangeBethesdaUnits) || rangeBethesdaUnits <= 0.0f ||
        !std::isfinite(forward.x) || !std::isfinite(forward.y) ||
        !std::isfinite(forward.z)) {
        return result;
    }
    const auto source = m_impl->characters.find(attacker);
    const float forwardLength = odai::math::length(forward);
    if (source == m_impl->characters.end() || forwardLength <= 1.0e-5f) return result;
    const odai::math::Vector3 facing = forward * (1.0f / forwardLength);
    const float sourceHalfHeight =
        source->second.centreOffsetMetres / kBethesdaUnitsToJoltMetres;
    const odai::math::Vector3 origin = source->second.last.position +
        odai::math::Vector3{0.0f, sourceHalfHeight, 0.0f};
    const float rangeSquared = rangeBethesdaUnits * rangeBethesdaUnits;
    class StaticLayerFilter final : public JPH::ObjectLayerFilter {
    public:
        bool ShouldCollide(JPH::ObjectLayer layer) const override {
            return layer == kStaticLayer;
        }
    } staticLayerFilter;

    for (const auto& [object, entry] : m_impl->characters) {
        if (object == attacker) continue;
        const float targetHalfHeight =
            entry.centreOffsetMetres / kBethesdaUnitsToJoltMetres;
        const odai::math::Vector3 target = entry.last.position +
            odai::math::Vector3{0.0f, targetHalfHeight, 0.0f};
        const odai::math::Vector3 offset = target - origin;
        const float distanceSquared =
            (offset.x * offset.x) + (offset.y * offset.y) + (offset.z * offset.z);
        if (distanceSquared <= 1.0e-6f || distanceSquared > rangeSquared) continue;
        const float distance = std::sqrt(distanceSquared);
        const odai::math::Vector3 direction = offset * (1.0f / distance);
        if (odai::math::dot(facing, direction) <
            std::clamp(minimumFacingDot, -1.0f, 1.0f)) continue;

        const JPH::RRayCast ray(toJoltPosition(origin), toJoltVector(offset));
        JPH::RayCastResult obstruction;
        if (m_impl->physics.GetNarrowPhaseQuery().CastRay(
                ray, obstruction,
                m_impl->physics.GetDefaultBroadPhaseLayerFilter(kCharacterLayer),
                staticLayerFilter) && obstruction.mFraction < 0.98f) {
            continue;
        }
        result.push_back({object, distance});
    }
    std::sort(result.begin(), result.end(),
        [](const PhysicsMeleeCandidate& left, const PhysicsMeleeCandidate& right) {
            if (left.distance != right.distance) return left.distance < right.distance;
            return left.object < right.object;
        });
    return result;
}

}  // namespace odai::bethesda
