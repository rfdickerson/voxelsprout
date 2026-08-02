#include "procgen/humanoid_generator.h"

#include "anim/biped_rig.h"

#include <array>
#include <cmath>
#include <cstddef>

namespace odai::procgen {

namespace {

using odai::math::Vector3;
using odai::render::ImportedSkinnedMeshVertex;

constexpr int kCylinderSegmentsLimb = 6;
constexpr int kCylinderSegmentsTorso = 8;
constexpr float kPi = 3.14159265358979323846f;

// Accumulates skinned-mesh geometry. Every vertex is rigidly bound to a
// single bone (weight 1.0, matching HumanoidMeshColors' doc comment on why
// this generator skips joint blending).
struct SkinnedMeshBuilder {
    std::vector<ImportedSkinnedMeshVertex>& vertices;
    std::vector<std::uint32_t>& indices;

    std::uint32_t addVertex(const Vector3& p, const Vector3& n, const Vector3& color, std::uint16_t boneIndex) {
        ImportedSkinnedMeshVertex v{};
        v.position[0] = p.x;
        v.position[1] = p.y;
        v.position[2] = p.z;
        v.normal[0] = n.x;
        v.normal[1] = n.y;
        v.normal[2] = n.z;
        v.color[0] = color.x;
        v.color[1] = color.y;
        v.color[2] = color.z;
        v.textureIndex = 0xffffffffu;  // packed vertex-color path, no texture.
        v.flags = 0u;
        v.boneIndices[0] = boneIndex;
        v.boneIndices[1] = 0;
        v.boneIndices[2] = 0;
        v.boneIndices[3] = 0;
        v.boneWeights[0] = 1.0f;
        v.boneWeights[1] = 0.0f;
        v.boneWeights[2] = 0.0f;
        v.boneWeights[3] = 0.0f;
        const auto index = static_cast<std::uint32_t>(vertices.size());
        vertices.push_back(v);
        return index;
    }

    void addTriangle(std::uint32_t a, std::uint32_t b, std::uint32_t c) {
        indices.push_back(a);
        indices.push_back(b);
        indices.push_back(c);
    }

    void addFlatTriangle(const Vector3& a, const Vector3& b, const Vector3& c,
                          const Vector3& color, std::uint16_t boneIndex) {
        const Vector3 n = odai::math::normalize(odai::math::cross(b - a, c - a));
        addTriangle(addVertex(a, n, color, boneIndex), addVertex(b, n, color, boneIndex),
                    addVertex(c, n, color, boneIndex));
    }

    // a-b-c-d wound counter-clockwise when viewed from the face's outward
    // side -- same convention as CityMeshBuilder::addQuad (citybuilder_app.cc).
    void addQuad(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d,
                 const Vector3& color, std::uint16_t boneIndex) {
        addFlatTriangle(a, d, c, color, boneIndex);
        addFlatTriangle(a, c, b, color, boneIndex);
    }
};

// Perpendicular basis (right, up) for a ring of points around `axis`
// (unit length). Picks a reference "up" that isn't near-parallel to axis to
// avoid a degenerate cross product.
void perpendicularBasis(const Vector3& axis, Vector3& outRight, Vector3& outUp) {
    Vector3 reference{0.0f, 1.0f, 0.0f};
    if (std::fabs(axis.y) > 0.98f) {
        reference = Vector3{1.0f, 0.0f, 0.0f};
    }
    outRight = odai::math::normalize(odai::math::cross(reference, axis));
    outUp = odai::math::cross(axis, outRight);
}

// One tapered cylinder segment from `pointA` (radiusA) to `pointB` (radiusB),
// capped at both ends, every vertex bound to `boneIndex`.
void addCylinderSegment(
    SkinnedMeshBuilder& builder, const Vector3& pointA, const Vector3& pointB,
    float radiusA, float radiusB, int segments, const Vector3& color, std::uint16_t boneIndex
) {
    const Vector3 delta = pointB - pointA;
    const float len = odai::math::length(delta);
    if (len <= 1e-5f) {
        return;
    }
    const Vector3 axis = delta * (1.0f / len);
    Vector3 right, up;
    perpendicularBasis(axis, right, up);

    std::vector<Vector3> ringA(static_cast<std::size_t>(segments));
    std::vector<Vector3> ringB(static_cast<std::size_t>(segments));
    for (int i = 0; i < segments; ++i) {
        const float theta = (2.0f * kPi * static_cast<float>(i)) / static_cast<float>(segments);
        const float c = std::cos(theta);
        const float s = std::sin(theta);
        const Vector3 offsetDirection = (right * c) + (up * s);
        ringA[static_cast<std::size_t>(i)] = pointA + (offsetDirection * radiusA);
        ringB[static_cast<std::size_t>(i)] = pointB + (offsetDirection * radiusB);
    }

    for (int i = 0; i < segments; ++i) {
        const int next = (i + 1) % segments;
        builder.addQuad(
            ringA[static_cast<std::size_t>(i)], ringA[static_cast<std::size_t>(next)],
            ringB[static_cast<std::size_t>(next)], ringB[static_cast<std::size_t>(i)],
            color, boneIndex);
    }

    // End caps: fans wound so addFlatTriangle's derived normal points
    // outward (-axis at pointA, +axis at pointB).
    for (int i = 0; i < segments; ++i) {
        const int next = (i + 1) % segments;
        builder.addFlatTriangle(pointA, ringA[static_cast<std::size_t>(next)], ringA[static_cast<std::size_t>(i)],
                                 color, boneIndex);
        builder.addFlatTriangle(pointB, ringB[static_cast<std::size_t>(i)], ringB[static_cast<std::size_t>(next)],
                                 color, boneIndex);
    }
}

// Axis-aligned box centered at `center` with `halfExtents`, every vertex
// bound to `boneIndex`. Mirrors CityMeshBuilder's addBox winding.
void addBox(SkinnedMeshBuilder& builder, const Vector3& center, const Vector3& halfExtents,
            const Vector3& color, std::uint16_t boneIndex) {
    const float minX = center.x - halfExtents.x, maxX = center.x + halfExtents.x;
    const float minY = center.y - halfExtents.y, maxY = center.y + halfExtents.y;
    const float minZ = center.z - halfExtents.z, maxZ = center.z + halfExtents.z;
    const Vector3 corners[8] = {
        {minX, minY, minZ}, {maxX, minY, minZ}, {maxX, minY, maxZ}, {minX, minY, maxZ},
        {minX, maxY, minZ}, {maxX, maxY, minZ}, {maxX, maxY, maxZ}, {minX, maxY, maxZ},
    };
    builder.addQuad(corners[4], corners[5], corners[6], corners[7], color, boneIndex);  // top
    builder.addQuad(corners[3], corners[2], corners[1], corners[0], color, boneIndex);  // bottom
    builder.addQuad(corners[0], corners[1], corners[5], corners[4], color, boneIndex);  // -Z
    builder.addQuad(corners[2], corners[3], corners[7], corners[6], color, boneIndex);  // +Z
    builder.addQuad(corners[3], corners[0], corners[4], corners[7], color, boneIndex);  // -X
    builder.addQuad(corners[1], corners[2], corners[6], corners[5], color, boneIndex);  // +X
}

// Bind-pose bone hierarchy carries no rotation anywhere in makeBipedSkeleton
// (every joint is a pure translation offset from its parent), so each bone's
// bind-pose world position is just the running sum of localTranslation up
// the parent chain -- no need to compose full Matrix4 transforms.
std::vector<Vector3> computeBindWorldPositions(const odai::anim::Skeleton& skeleton) {
    std::vector<Vector3> world(skeleton.bones.size());
    for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
        const odai::anim::Bone& bone = skeleton.bones[i];
        const Vector3 parentWorld =
            (bone.parentIndex >= 0) ? world[static_cast<std::size_t>(bone.parentIndex)] : Vector3{};
        world[i] = parentWorld + bone.localTranslation;
    }
    return world;
}

struct Segment {
    const char* parentBone;
    const char* childBone;
    float radiusAtParent;
    float radiusAtChild;
    int colorGroup;  // 0 = torso, 1 = limbs, 2 = accent
};

constexpr std::array<Segment, 16> kSegments = {{
    {odai::anim::kBipedBonePelvis, odai::anim::kBipedBoneSpine, 0.16f, 0.15f, 0},
    {odai::anim::kBipedBoneSpine, odai::anim::kBipedBoneChest, 0.15f, 0.17f, 0},
    {odai::anim::kBipedBoneChest, odai::anim::kBipedBoneNeck, 0.06f, 0.05f, 1},

    {odai::anim::kBipedBoneChest, odai::anim::kBipedBoneShoulderL, 0.07f, 0.06f, 0},
    {odai::anim::kBipedBoneShoulderL, odai::anim::kBipedBoneElbowL, 0.055f, 0.045f, 1},
    {odai::anim::kBipedBoneElbowL, odai::anim::kBipedBoneWristL, 0.045f, 0.04f, 1},
    {odai::anim::kBipedBoneWristL, odai::anim::kBipedBoneHandL, 0.04f, 0.035f, 1},

    {odai::anim::kBipedBoneChest, odai::anim::kBipedBoneShoulderR, 0.07f, 0.06f, 0},
    {odai::anim::kBipedBoneShoulderR, odai::anim::kBipedBoneElbowR, 0.055f, 0.045f, 1},
    {odai::anim::kBipedBoneElbowR, odai::anim::kBipedBoneWristR, 0.045f, 0.04f, 1},
    {odai::anim::kBipedBoneWristR, odai::anim::kBipedBoneHandR, 0.04f, 0.035f, 1},

    {odai::anim::kBipedBonePelvis, odai::anim::kBipedBoneHipL, 0.09f, 0.08f, 0},
    {odai::anim::kBipedBoneHipL, odai::anim::kBipedBoneKneeL, 0.08f, 0.065f, 1},
    {odai::anim::kBipedBoneKneeL, odai::anim::kBipedBoneAnkleL, 0.065f, 0.05f, 1},
    {odai::anim::kBipedBonePelvis, odai::anim::kBipedBoneHipR, 0.09f, 0.08f, 0},
    {odai::anim::kBipedBoneHipR, odai::anim::kBipedBoneKneeR, 0.08f, 0.065f, 1},
}};

// Boot/foot segments kept separate from kSegments since they use the accent
// color group.
constexpr std::array<Segment, 2> kBootSegments = {{
    {odai::anim::kBipedBoneAnkleL, odai::anim::kBipedBoneFootL, 0.05f, 0.045f, 2},
    {odai::anim::kBipedBoneAnkleR, odai::anim::kBipedBoneFootR, 0.05f, 0.045f, 2},
}};

}  // namespace

HumanoidSkinnedMesh buildHumanoidSkinnedMesh(
    const odai::anim::Skeleton& skeleton, const HumanoidMeshColors& colors
) {
    HumanoidSkinnedMesh mesh;
    mesh.boneCount = static_cast<std::uint32_t>(skeleton.bones.size());
    SkinnedMeshBuilder builder{mesh.vertices, mesh.indices};

    const std::vector<Vector3> bindWorldPositions = computeBindWorldPositions(skeleton);
    const std::array<Vector3, 3> colorByGroup = {colors.torso, colors.limbs, colors.accent};

    const auto emitSegment = [&](const Segment& segment) {
        const int parentIndex = skeleton.findBone(segment.parentBone);
        const int childIndex = skeleton.findBone(segment.childBone);
        if (parentIndex < 0 || childIndex < 0) {
            return;
        }
        addCylinderSegment(
            builder,
            bindWorldPositions[static_cast<std::size_t>(parentIndex)],
            bindWorldPositions[static_cast<std::size_t>(childIndex)],
            segment.radiusAtParent, segment.radiusAtChild,
            (segment.colorGroup == 0) ? kCylinderSegmentsTorso : kCylinderSegmentsLimb,
            colorByGroup[static_cast<std::size_t>(segment.colorGroup)],
            static_cast<std::uint16_t>(parentIndex));
    };
    for (const Segment& segment : kSegments) {
        emitSegment(segment);
    }
    for (const Segment& segment : kBootSegments) {
        emitSegment(segment);
    }

    // Head: a small box sitting on top of the neck-to-head segment, bound to
    // the head bone (a leaf, so it has no outgoing segment of its own).
    const int headIndex = skeleton.findBone(odai::anim::kBipedBoneHead);
    if (headIndex >= 0) {
        const Vector3 headJoint = bindWorldPositions[static_cast<std::size_t>(headIndex)];
        const Vector3 headCenter = headJoint + Vector3{0.0f, 0.08f, 0.0f};
        addBox(builder, headCenter, Vector3{0.08f, 0.09f, 0.09f}, colors.limbs,
               static_cast<std::uint16_t>(headIndex));
    }

    return mesh;
}

}  // namespace odai::procgen
