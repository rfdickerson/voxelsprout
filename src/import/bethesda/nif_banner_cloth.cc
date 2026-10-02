#include "import/bethesda/nif_scene.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <mutex>
#include <unordered_map>

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceTable.h>
#include <Jolt/Physics/Collision/BroadPhase/ObjectVsBroadPhaseLayerFilterTable.h>
#include <Jolt/Physics/Collision/ObjectLayerPairFilterTable.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/SoftBody/SoftBodyCreationSettings.h>
#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>

namespace odai::importer::bethesda {

namespace {

struct QuantizedClothVertex {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const QuantizedClothVertex&) const = default;
};

struct QuantizedClothVertexHash {
    std::size_t operator()(const QuantizedClothVertex& key) const noexcept {
        std::size_t hash = static_cast<std::size_t>(key.x) * 73856093u;
        hash ^= static_cast<std::size_t>(key.y) * 19349663u;
        hash ^= static_cast<std::size_t>(key.z) * 83492791u;
        return hash;
    }
};

void ensureJoltClothTypesRegistered() {
    static std::once_flag once;
    std::call_once(once, [] {
        JPH::RegisterDefaultAllocator();
        if (JPH::Factory::sInstance == nullptr) {
            JPH::Factory::sInstance = new JPH::Factory();
        }
        JPH::RegisterTypes();
    });
}

bool settleBannerClothWithJolt(NifModel& model, float topZ, float topBand) {
    ensureJoltClothTypesRegistered();

    JPH::Ref<JPH::SoftBodySharedSettings> settings = new JPH::SoftBodySharedSettings();
    std::unordered_map<QuantizedClothVertex, std::uint32_t, QuantizedClothVertexHash> welded;
    std::vector<std::vector<std::uint32_t>> shapeToSoftVertex(model.shapes.size());
    for (std::size_t shapeIndex = 0; shapeIndex < model.shapes.size(); ++shapeIndex) {
        const NifShape& shape = model.shapes[shapeIndex];
        const std::size_t count = shape.positions.size() / 3u;
        shapeToSoftVertex[shapeIndex].resize(count);
        for (std::size_t v = 0; v < count; ++v) {
            const float* p = shape.positions.data() + (v * 3u);
            const QuantizedClothVertex key{
                static_cast<int>(std::lround(p[0] * 10.0f)),
                static_cast<int>(std::lround(p[1] * 10.0f)),
                static_cast<int>(std::lround(p[2] * 10.0f))};
            auto [it, inserted] = welded.emplace(
                key, static_cast<std::uint32_t>(settings->mVertices.size()));
            if (inserted) {
                // NIF is Z-up; Jolt is Y-up. Vertex mass zero pins the authored
                // top attachment row, which is the physical role of the root
                // banner bone while the remaining skin follows the soft body.
                const float invMass = p[2] >= topZ - topBand ? 0.0f : 1.0f;
                settings->mVertices.emplace_back(
                    JPH::Float3(p[0], p[2], p[1]), JPH::Float3(0, 0, 0), invMass);
            }
            shapeToSoftVertex[shapeIndex][v] = it->second;
        }
    }
    if (settings->mVertices.size() < 4u) {
        return false;
    }

    for (std::size_t shapeIndex = 0; shapeIndex < model.shapes.size(); ++shapeIndex) {
        const NifShape& shape = model.shapes[shapeIndex];
        const auto& mapping = shapeToSoftVertex[shapeIndex];
        for (std::size_t tri = 0; (tri * 3u) + 2u < shape.triangleIndices.size(); ++tri) {
            const std::uint32_t ia = shape.triangleIndices[tri * 3u];
            const std::uint32_t ib = shape.triangleIndices[(tri * 3u) + 1u];
            const std::uint32_t ic = shape.triangleIndices[(tri * 3u) + 2u];
            if (ia >= mapping.size() || ib >= mapping.size() || ic >= mapping.size()) {
                continue;
            }
            const JPH::SoftBodySharedSettings::Face face(
                mapping[ia], mapping[ib], mapping[ic]);
            if (!face.IsDegenerate()) {
                settings->mFaces.push_back(face);
            }
        }
    }
    if (settings->mFaces.empty()) {
        return false;
    }

    // Edge/shear constraints preserve the authored weave; dihedral bending
    // keeps the result cloth-like rather than a collection of independent
    // strings. Long-range attachment prevents a low-iteration solve from
    // stretching the banner away from its pinned top bone.
    const JPH::SoftBodySharedSettings::VertexAttributes attributes(
        0.0f, 0.0f, 5.0e-4f,
        JPH::SoftBodySharedSettings::ELRAType::GeodesicDistance, 1.02f);
    settings->CreateConstraints(
        &attributes, 1u, JPH::SoftBodySharedSettings::EBendType::Dihedral);
    settings->Optimize();

    constexpr JPH::ObjectLayer kClothLayer = 0u;
    JPH::BroadPhaseLayerInterfaceTable broadPhaseLayers(1u, 1u);
    broadPhaseLayers.MapObjectToBroadPhaseLayer(kClothLayer, JPH::BroadPhaseLayer(0u));
    JPH::ObjectLayerPairFilterTable objectPairFilter(1u);
    JPH::ObjectVsBroadPhaseLayerFilterTable broadPhaseFilter(
        broadPhaseLayers, 1u, objectPairFilter, 1u);
    JPH::PhysicsSystem physics;
    physics.Init(
        16u, 0u, 16u, 16u, broadPhaseLayers, broadPhaseFilter,
        objectPairFilter);
    // Bethesda's model units are roughly centimetres; use the same scale for
    // gravity so a several-metre hall banner settles in seconds, not minutes.
    physics.SetGravity(JPH::Vec3(0.0f, -980.0f, 0.0f));

    JPH::SoftBodyCreationSettings bodySettings(
        settings, JPH::RVec3::sZero(), JPH::Quat::sIdentity(), kClothLayer);
    bodySettings.mNumIterations = 10u;
    bodySettings.mLinearDamping = 3.0f;
    bodySettings.mMaxLinearVelocity = 1000.0f;
    bodySettings.mUpdatePosition = false;
    bodySettings.mAllowSleeping = false;
    JPH::BodyInterface& bodyInterface = physics.GetBodyInterface();
    const JPH::BodyID bodyId =
        bodyInterface.CreateAndAddSoftBody(bodySettings, JPH::EActivation::Activate);
    if (bodyId.IsInvalid()) {
        return false;
    }

    JPH::TempAllocatorMalloc allocator;
    JPH::JobSystemSingleThreaded jobs(JPH::cMaxPhysicsJobs);
    constexpr float kStepSeconds = 1.0f / 60.0f;
    for (int step = 0; step < 240; ++step) {
        physics.Update(kStepSeconds, 1, &allocator, &jobs);
    }

    bool copied = false;
    {
        JPH::BodyLockRead lock(physics.GetBodyLockInterface(), bodyId);
        if (lock.Succeeded()) {
            const JPH::Body& body = lock.GetBody();
            const auto* motion = static_cast<const JPH::SoftBodyMotionProperties*>(
                body.GetMotionProperties());
            const auto& vertices = motion->GetVertices();
            const JPH::RMat44 centerOfMass = body.GetCenterOfMassTransform();
            if (vertices.size() == settings->mVertices.size()) {
                for (std::size_t shapeIndex = 0; shapeIndex < model.shapes.size(); ++shapeIndex) {
                    NifShape& shape = model.shapes[shapeIndex];
                    for (std::size_t v = 0; v < shapeToSoftVertex[shapeIndex].size(); ++v) {
                        const std::uint32_t softIndex = shapeToSoftVertex[shapeIndex][v];
                        const JPH::RVec3 world = centerOfMass * vertices[softIndex].mPosition;
                        float* p = shape.positions.data() + (v * 3u);
                        p[0] = static_cast<float>(world.GetX());
                        p[1] = static_cast<float>(world.GetZ());
                        p[2] = static_cast<float>(world.GetY());
                    }
                }
                copied = true;
            }
        }
    }
    bodyInterface.RemoveBody(bodyId);
    bodyInterface.DestroyBody(bodyId);
    return copied;
}

}  // namespace

bool applyNifBannerGravityRestPose(std::string_view modelPath, NifModel& model) {
    if (!model.hasEmbeddedTransformAnimation || model.shapes.empty()) {
        return false;
    }

    std::string lowered(modelPath);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return c == '/' ? '\\' : static_cast<char>(std::tolower(c));
    });
    if (lowered.find("banner") == std::string::npos) {
        return false;
    }

    // The controller-less static importer sees the skin partition's bind pose.
    // Skyrim's animated banners author that pose under strong lateral wind:
    // their Z drop and horizontal displacement are comparable, so the cloth
    // points into the room. Find the horizontal wind axis from the top edge --
    // it has almost no spread there, while the cross axis spans the banner's
    // full width -- then settle that one axis toward the attachment under
    // gravity. The hypotenuse is preserved so settling does not shorten cloth.
    float minPosition[3] = {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    float maxPosition[3] = {
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
    std::size_t vertexCount = 0u;
    for (const NifShape& shape : model.shapes) {
        for (std::size_t v = 0; (v * 3u) + 2u < shape.positions.size(); ++v) {
            for (int axis = 0; axis < 3; ++axis) {
                const float value = shape.positions[(v * 3u) + static_cast<std::size_t>(axis)];
                minPosition[axis] = std::min(minPosition[axis], value);
                maxPosition[axis] = std::max(maxPosition[axis], value);
            }
            ++vertexCount;
        }
    }
    const float height = maxPosition[2] - minPosition[2];
    if (vertexCount < 4u || height <= 1.0e-3f) {
        return false;
    }

    const float topBand = std::max(height * 0.08f, 1.0f);
    const bool settledByJolt = settleBannerClothWithJolt(model, maxPosition[2], topBand);
    float topMin[2] = {
        std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    float topMax[2] = {
        std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
    double topSum[2] = {0.0, 0.0};
    std::size_t topCount = 0u;
    for (const NifShape& shape : model.shapes) {
        for (std::size_t v = 0; (v * 3u) + 2u < shape.positions.size(); ++v) {
            const float* position = shape.positions.data() + (v * 3u);
            if (position[2] < maxPosition[2] - topBand) {
                continue;
            }
            for (int axis = 0; axis < 2; ++axis) {
                topMin[axis] = std::min(topMin[axis], position[axis]);
                topMax[axis] = std::max(topMax[axis], position[axis]);
                topSum[axis] += position[axis];
            }
            ++topCount;
        }
    }
    if (topCount == 0u) {
        return false;
    }

    const auto normalizedTopSpread = [&](int axis) {
        const float fullSpread = std::max(maxPosition[axis] - minPosition[axis], 1.0e-3f);
        return (topMax[axis] - topMin[axis]) / fullSpread;
    };
    const int windAxis = normalizedTopSpread(0) <= normalizedTopSpread(1) ? 0 : 1;
    const float anchor = static_cast<float>(topSum[windAxis] / static_cast<double>(topCount));

    for (NifShape& shape : model.shapes) {
        if (!settledByJolt) {
            for (std::size_t v = 0; (v * 3u) + 2u < shape.positions.size(); ++v) {
                float* position = shape.positions.data() + (v * 3u);
                const float verticalDrop = std::max(maxPosition[2] - position[2], 0.0f);
                const float drop01 = std::clamp(verticalDrop / height, 0.0f, 1.0f);
                const float smoothDrop = drop01 * drop01 * (3.0f - (2.0f * drop01));
                const float windOffset = position[windAxis] - anchor;
                const float retention = 1.0f - (0.88f * smoothDrop);
                const float restedOffset = windOffset * retention;
                const float restLengthSquared =
                    (verticalDrop * verticalDrop) + (windOffset * windOffset);
                const float restedDrop = std::sqrt(std::max(
                    restLengthSquared - (restedOffset * restedOffset), 0.0f));
                position[windAxis] = anchor + restedOffset;
                position[2] = maxPosition[2] - restedDrop;
            }
        }

        // The deformation changes the surface basis. Rebuild smooth normals
        // from the authored winding rather than lighting the newly vertical
        // cloth with normals from its old wind-blown pose.
        const std::size_t shapeVertexCount = shape.positions.size() / 3u;
        shape.normals.assign(shapeVertexCount * 3u, 0.0f);
        for (std::size_t tri = 0; (tri * 3u) + 2u < shape.triangleIndices.size(); ++tri) {
            const std::uint32_t ia = shape.triangleIndices[tri * 3u];
            const std::uint32_t ib = shape.triangleIndices[(tri * 3u) + 1u];
            const std::uint32_t ic = shape.triangleIndices[(tri * 3u) + 2u];
            if (ia >= shapeVertexCount || ib >= shapeVertexCount || ic >= shapeVertexCount) {
                continue;
            }
            const float* a = shape.positions.data() + (static_cast<std::size_t>(ia) * 3u);
            const float* b = shape.positions.data() + (static_cast<std::size_t>(ib) * 3u);
            const float* c = shape.positions.data() + (static_cast<std::size_t>(ic) * 3u);
            const float e0[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
            const float e1[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
            const float faceNormal[3] = {
                (e0[1] * e1[2]) - (e0[2] * e1[1]),
                (e0[2] * e1[0]) - (e0[0] * e1[2]),
                (e0[0] * e1[1]) - (e0[1] * e1[0])};
            for (const std::uint32_t index : {ia, ib, ic}) {
                float* normal = shape.normals.data() + (static_cast<std::size_t>(index) * 3u);
                normal[0] += faceNormal[0];
                normal[1] += faceNormal[1];
                normal[2] += faceNormal[2];
            }
        }
        for (std::size_t v = 0; v < shapeVertexCount; ++v) {
            float* normal = shape.normals.data() + (v * 3u);
            const float length = std::sqrt(
                (normal[0] * normal[0]) + (normal[1] * normal[1]) +
                (normal[2] * normal[2]));
            if (length > 1.0e-6f) {
                normal[0] /= length;
                normal[1] /= length;
                normal[2] /= length;
            }
        }
    }
    return true;
}

}  // namespace odai::importer::bethesda
