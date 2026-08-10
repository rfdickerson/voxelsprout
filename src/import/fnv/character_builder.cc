#include "import/fnv/character_builder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace odai::importer::fnv {

namespace {

using odai::math::Matrix4;
using odai::math::Quaternion;
using odai::math::Vector3;

// Bethesda Z-up -> engine Y-up, as a 3x3 basis change: (x, y, z) -> (x, z, -y).
// The same mapping cell_builder's bethesdaToEngine applies to points, written
// as a matrix because the bone path needs it on both sides of a product.
//
// det(M) = +1, so it is a proper rotation: normals convert with the same matrix
// as positions and no handedness flip sneaks in.
constexpr float kBasisChange[9] = {
    1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f,
    0.0f, -1.0f, 0.0f,
};

void multiply3x3(const float a[9], const float b[9], float out[9]) {
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k) {
                sum += a[(row * 3) + k] * b[(k * 3) + col];
            }
            out[(row * 3) + col] = sum;
        }
    }
}

void transpose3x3(const float a[9], float out[9]) {
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            out[(row * 3) + col] = a[(col * 3) + row];
        }
    }
}

// R -> M * R * transpose(M). See buildFalloutSkeleton's header comment for why
// this and not the single-sided form the static path uses.
void changeRotationBasis(const float rotation[9], float out[9]) {
    float basisTranspose[9] = {};
    transpose3x3(kBasisChange, basisTranspose);
    float temp[9] = {};
    multiply3x3(kBasisChange, rotation, temp);
    multiply3x3(temp, basisTranspose, out);
}

Vector3 changePointBasis(float x, float y, float z) {
    return Vector3{x, z, -y};
}

// Shepperd's method: pick the largest-magnitude component to divide by, so the
// division is never by something near zero. The naive w-first form loses all
// precision on a 180-degree rotation, which a bone that faces backward down its
// own chain genuinely is.
Quaternion quaternionFromRotation(const float r[9]) {
    const float trace = r[0] + r[4] + r[8];
    Quaternion q;
    if (trace > 0.0f) {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        q.w = 0.25f * s;
        q.x = (r[7] - r[5]) / s;
        q.y = (r[2] - r[6]) / s;
        q.z = (r[3] - r[1]) / s;
    } else if (r[0] > r[4] && r[0] > r[8]) {
        const float s = std::sqrt(1.0f + r[0] - r[4] - r[8]) * 2.0f;
        q.w = (r[7] - r[5]) / s;
        q.x = 0.25f * s;
        q.y = (r[1] + r[3]) / s;
        q.z = (r[2] + r[6]) / s;
    } else if (r[4] > r[8]) {
        const float s = std::sqrt(1.0f + r[4] - r[0] - r[8]) * 2.0f;
        q.w = (r[2] - r[6]) / s;
        q.x = (r[1] + r[3]) / s;
        q.y = 0.25f * s;
        q.z = (r[5] + r[7]) / s;
    } else {
        const float s = std::sqrt(1.0f + r[8] - r[0] - r[4]) * 2.0f;
        q.w = (r[3] - r[1]) / s;
        q.x = (r[2] + r[6]) / s;
        q.y = (r[5] + r[7]) / s;
        q.z = 0.25f * s;
    }
    return normalize(q);
}

// A NIF 4x4 (row-major, translation in the last column) rebased into engine
// space: M4 * matrix * transpose(M4). Same change of basis as the rotations,
// extended to the translation column, which is what keeps the inverse bind
// consistent with the bone world transforms it multiplies against.
Matrix4 changeMatrixBasis(const float source[16]) {
    float rotation[9] = {};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            rotation[(row * 3) + col] = source[(row * 4) + col];
        }
    }
    float rebased[9] = {};
    changeRotationBasis(rotation, rebased);
    const Vector3 translation =
        changePointBasis(source[3], source[7], source[11]);

    Matrix4 out = Matrix4::identity();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            out(row, col) = rebased[(row * 3) + col];
        }
    }
    out(0, 3) = translation.x;
    out(1, 3) = translation.y;
    out(2, 3) = translation.z;
    return out;
}

// A bone's local bind transform as a matrix, for accumulating world transforms.
Matrix4 boneLocalMatrix(const anim::Bone& bone) {
    Matrix4 out = toMatrix(bone.localRotation);
    for (int row = 0; row < 3; ++row) {
        out(row, 0) *= bone.localScale.x;
        out(row, 1) *= bone.localScale.y;
        out(row, 2) *= bone.localScale.z;
    }
    out(0, 3) = bone.localTranslation.x;
    out(1, 3) = bone.localTranslation.y;
    out(2, 3) = bone.localTranslation.z;
    return out;
}

bool matricesDiffer(const Matrix4& a, const Matrix4& b) {
    // Loose: these come from independently exported body parts, so bitwise
    // equality is not expected even when they agree. The threshold is well
    // under a millimetre at Fallout's scale (1 unit ~= 1.4 cm) and well over
    // float round-trip noise.
    for (int i = 0; i < 16; ++i) {
        if (std::fabs(a.m[i] - b.m[i]) > 1e-3f) {
            return true;
        }
    }
    return false;
}

}  // namespace

bool buildFalloutSkeleton(const NifSkeleton& source, anim::Skeleton& outSkeleton) {
    outSkeleton = anim::Skeleton{};
    if (source.bones.empty()) {
        return false;
    }
    outSkeleton.bones.reserve(source.bones.size());
    for (const NifSkeletonBone& sourceBone : source.bones) {
        anim::Bone bone;
        bone.name = sourceBone.name;
        // NifSkeleton guarantees parents precede children, which anim::Skeleton
        // requires, so the index carries over unchanged.
        bone.parentIndex = sourceBone.parentIndex;
        bone.localTranslation = changePointBasis(
            sourceBone.translation[0], sourceBone.translation[1], sourceBone.translation[2]);
        float rebased[9] = {};
        changeRotationBasis(sourceBone.rotation, rebased);
        bone.localRotation = quaternionFromRotation(rebased);
        // NIF scale is a single float; a basis change cannot make it
        // non-uniform, so all three axes take it.
        bone.localScale = Vector3{sourceBone.scale, sourceBone.scale, sourceBone.scale};
        outSkeleton.bones.push_back(std::move(bone));
    }
    return true;
}

bool appendFalloutCharacterMesh(
    const NifSkinnedModel& model, FalloutCharacter& character, std::string& outError) {
    if (character.skeleton.bones.empty()) {
        outError = "character has no skeleton; call buildFalloutSkeleton first";
        return false;
    }
    if (model.shapes.empty()) {
        outError = "skinned model has no skinned shapes";
        return false;
    }

    // Bone lookup by name, built once rather than per shape: anim::Skeleton's
    // own findBone is a linear scan, and a 6-part character with ~20 bones per
    // part against a 66-bone skeleton would run it thousands of times.
    std::unordered_map<std::string, int> boneIndexByName;
    boneIndexByName.reserve(character.skeleton.bones.size());
    for (std::size_t i = 0; i < character.skeleton.bones.size(); ++i) {
        boneIndexByName.emplace(character.skeleton.bones[i].name, static_cast<int>(i));
    }

    character.inverseBindMatrices.resize(character.skeleton.bones.size(), Matrix4::identity());
    // Which entries hold a real value rather than the identity placeholder, so
    // a genuine identity inverse bind is not mistaken for "not yet set".
    if (character.parts.empty()) {
        character.parts.reserve(model.shapes.size());
    }
    std::vector<bool> inverseBindWritten(character.skeleton.bones.size(), false);
    for (std::size_t i = 0; i < character.skeleton.bones.size(); ++i) {
        // A bone already carrying a non-identity matrix from an earlier call
        // counts as written; the identity ones are genuinely unset.
        inverseBindWritten[i] = matricesDiffer(character.inverseBindMatrices[i], Matrix4::identity());
    }

    for (const NifSkinnedShape& shape : model.shapes) {
        const std::size_t vertexCount = shape.positions.size() / 3u;
        if (vertexCount == 0u || shape.triangleIndices.empty()) {
            continue;
        }

        // This shape's local bone list -> skeleton bone indices, resolved once
        // per shape and then applied per vertex.
        std::vector<int> skeletonBoneIndex(shape.boneNames.size(), -1);
        for (std::size_t b = 0; b < shape.boneNames.size(); ++b) {
            const auto found = boneIndexByName.find(shape.boneNames[b]);
            if (found == boneIndexByName.end()) {
                ++character.unresolvedBoneCount;
                continue;
            }
            skeletonBoneIndex[b] = found->second;

            const auto boneSlot = static_cast<std::size_t>(found->second);
            if ((b * 16u) + 16u > shape.inverseBindMatrices.size()) {
                continue;
            }
            // The full inverse bind is boneData.skinTransform composed with
            // NiSkinData's overall skinTransform: the per-bone half maps the
            // skeleton root's space into the bone's, and the overall half maps
            // this shape's geometry space into the root's. A vertex needs both,
            // in that order.
            const Matrix4 inverseBind =
                changeMatrixBasis(shape.inverseBindMatrices.data() + (b * 16u)) *
                changeMatrixBasis(shape.skinTransform);
            if (!inverseBindWritten[boneSlot]) {
                character.inverseBindMatrices[boneSlot] = inverseBind;
                inverseBindWritten[boneSlot] = true;
            } else if (matricesDiffer(character.inverseBindMatrices[boneSlot], inverseBind)) {
                ++character.conflictingInverseBindCount;
            }
        }

        const auto baseVertex = static_cast<std::uint32_t>(character.vertices.size());
        FalloutCharacterPart part;
        part.name = shape.name;
        part.diffuseTexturePath = shape.diffuseTexturePath;
        part.alphaTest = shape.alphaTest;
        part.alphaThreshold = shape.alphaThreshold;
        part.alphaBlend = shape.alphaBlend;
        part.twoSided = shape.twoSided;
        part.firstIndex = static_cast<std::uint32_t>(character.indices.size());
        part.indexCount = static_cast<std::uint32_t>(shape.triangleIndices.size());

        character.vertices.reserve(character.vertices.size() + vertexCount);
        for (std::size_t v = 0; v < vertexCount; ++v) {
            odai::render::ImportedSkinnedMeshVertex vertex;
            const Vector3 position = changePointBasis(
                shape.positions[v * 3u], shape.positions[(v * 3u) + 1u],
                shape.positions[(v * 3u) + 2u]);
            vertex.position[0] = position.x;
            vertex.position[1] = position.y;
            vertex.position[2] = position.z;
            if ((v * 3u) + 2u < shape.normals.size()) {
                const Vector3 normal = changePointBasis(
                    shape.normals[v * 3u], shape.normals[(v * 3u) + 1u],
                    shape.normals[(v * 3u) + 2u]);
                vertex.normal[0] = normal.x;
                vertex.normal[1] = normal.y;
                vertex.normal[2] = normal.z;
            } else {
                vertex.normal[1] = 1.0f;
            }
            if ((v * 2u) + 1u < shape.uvs.size()) {
                vertex.uv[0] = shape.uvs[v * 2u];
                vertex.uv[1] = shape.uvs[(v * 2u) + 1u];
            }
            vertex.color[0] = 1.0f;
            vertex.color[1] = 1.0f;
            vertex.color[2] = 1.0f;
            // Texture binding is the caller's job: it owns the bindless slot
            // table and resolves part.diffuseTexturePath against the archives.
            vertex.textureIndex = 0xffffffffu;

            for (int k = 0; k < kNifMaxBoneInfluences; ++k) {
                const std::size_t slot = (v * kNifMaxBoneInfluences) + static_cast<std::size_t>(k);
                const float weight = shape.boneWeights[slot];
                const std::uint16_t localBone = shape.boneIndices[slot];
                const int resolved = (localBone < skeletonBoneIndex.size())
                    ? skeletonBoneIndex[localBone]
                    : -1;
                if (weight <= 0.0f || resolved < 0) {
                    // Weight zero, so the index is never read by the shader.
                    // Bone 0 rather than the unresolved index keeps it in range
                    // regardless.
                    vertex.boneIndices[k] = 0u;
                    vertex.boneWeights[k] = 0.0f;
                    continue;
                }
                vertex.boneIndices[k] = static_cast<std::uint16_t>(resolved);
                vertex.boneWeights[k] = weight;
            }
            character.vertices.push_back(vertex);
        }

        character.indices.reserve(character.indices.size() + shape.triangleIndices.size());
        for (const std::uint32_t index : shape.triangleIndices) {
            character.indices.push_back(baseVertex + index);
        }
        character.parts.push_back(std::move(part));
    }

    return true;
}

void computeFalloutBindPose(
    const FalloutCharacter& character, std::vector<Matrix4>& outMatrices) {
    const std::size_t boneCount = character.skeleton.bones.size();
    outMatrices.assign(boneCount, Matrix4::identity());
    // One forward pass: anim::Skeleton guarantees parents precede children, so
    // a parent's world transform is always already final when its child is
    // reached.
    std::vector<Matrix4> world(boneCount, Matrix4::identity());
    for (std::size_t i = 0; i < boneCount; ++i) {
        const anim::Bone& bone = character.skeleton.bones[i];
        const Matrix4 local = boneLocalMatrix(bone);
        world[i] = (bone.parentIndex >= 0)
            ? world[static_cast<std::size_t>(bone.parentIndex)] * local
            : local;
        outMatrices[i] = (i < character.inverseBindMatrices.size())
            ? world[i] * character.inverseBindMatrices[i]
            : world[i];
    }
}

}  // namespace odai::importer::fnv
