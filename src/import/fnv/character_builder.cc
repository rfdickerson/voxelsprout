#include "import/fnv/character_builder.h"

#include <cctype>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace odai::importer::fnv {

namespace {

using odai::math::Matrix4;
using odai::math::Quaternion;
using odai::math::Vector3;
using odai::math::Vector4;

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

bool buildFalloutAnimationClip(
    const KfAnimation& source,
    const anim::Skeleton& skeleton,
    anim::AnimationClip& outClip,
    FalloutAnimationStats& outStats
) {
    outClip = anim::AnimationClip{};
    outStats = FalloutAnimationStats{};
    if (skeleton.bones.empty()) {
        return false;
    }

    std::unordered_map<std::string, int> boneIndexByName;
    boneIndexByName.reserve(skeleton.bones.size());
    for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
        boneIndexByName.emplace(skeleton.bones[i].name, static_cast<int>(i));
    }

    outClip.name = source.name;
    outClip.duration = source.duration();
    outClip.loop = source.loops();
    outStats.tracks = source.tracks.size();
    outClip.tracks.reserve(source.tracks.size());

    // Quaternion -> row-major 3x3 -> rebased 3x3 -> quaternion. The round trip
    // through matrices is deliberate: it runs the keys through the SAME
    // changeRotationBasis/quaternionFromRotation pair the bind pose went
    // through, so the two cannot drift apart. It costs nothing that matters --
    // this runs once per key at load, never per frame.
    const auto rebaseRotation = [](const Quaternion& q) {
        const Matrix4 asMatrix = odai::math::toMatrix(q);
        float rotation[9] = {};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                rotation[(row * 3) + col] = asMatrix(row, col);
            }
        }
        float rebased[9] = {};
        changeRotationBasis(rotation, rebased);
        return quaternionFromRotation(rebased);
    };

    for (const KfBoneTrack& sourceTrack : source.tracks) {
        const auto found = boneIndexByName.find(sourceTrack.nodeName);
        if (found == boneIndexByName.end()) {
            ++outStats.unresolvedNodes;
            continue;
        }
        anim::BoneTrack track;
        track.boneIndex = found->second;
        track.translationKeys.reserve(sourceTrack.translationKeys.size());
        for (const KfVector3Key& key : sourceTrack.translationKeys) {
            anim::Vector3Key converted;
            converted.time = key.time;
            converted.value = changePointBasis(key.value.x, key.value.y, key.value.z);
            track.translationKeys.push_back(converted);
        }
        track.rotationKeys.reserve(sourceTrack.rotationKeys.size());
        for (const KfQuaternionKey& key : sourceTrack.rotationKeys) {
            anim::QuaternionKey converted;
            converted.time = key.time;
            converted.value = rebaseRotation(normalize(key.value));
            track.rotationKeys.push_back(converted);
        }
        // Scale is a scalar in the file and stays one: a rotation cannot make
        // a uniform scale non-uniform, so no basis change applies.
        track.scaleKeys.reserve(sourceTrack.scaleKeys.size());
        for (const KfVector3Key& key : sourceTrack.scaleKeys) {
            anim::Vector3Key converted;
            converted.time = key.time;
            converted.value = key.value;
            track.scaleKeys.push_back(converted);
        }
        outClip.tracks.push_back(std::move(track));
        ++outStats.boundTracks;
    }

    // THE ACCUMULATION ROOT'S TRANSLATION IS ROOT MOTION, NOT POSE.
    //
    // Gamebryo names one node in a NiControllerSequence as its accumulation
    // root -- "Bip01 NonAccum" on every Bethesda human rig -- and the engine
    // consumes that node's translation to MOVE THE CHARACTER, never to pose it.
    // Applied to the pose it is counted twice, because the skeleton's own
    // "Bip01" parent already carries the character's pelvis height.
    //
    // Measured, and it was not subtle: every human in Goodsprings stood 66.5
    // units -- most of a metre -- above the ground, because NonAccum binds at
    // (0,0,0) and every idle sets it to (0.2, 66.331, 0.27). Bind pose put them
    // on the ground correctly, which is what made it look like an animation
    // problem rather than a placement one. Creatures were unaffected throughout:
    // a bighorner's accumulation root is (0, 0.000001, 0), which is why this
    // survived so long -- Victor, the one actor anybody had studied, is a robot.
    //
    // Made RELATIVE to the first key rather than deleted, so the vertical bob a
    // clip authors around that offset survives; the horizontal channel is
    // flattened outright, because the wander code is what moves an actor here
    // and leaving it in would slide the body out from under its own placement.
    for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
        const std::string& name = skeleton.bones[i].name;
        constexpr const char* kAccumSuffix = "nonaccum";
        constexpr std::size_t kAccumLength = 8u;
        if (name.size() < kAccumLength) {
            continue;
        }
        std::string tail = name.substr(name.size() - kAccumLength);
        std::transform(tail.begin(), tail.end(), tail.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (tail != kAccumSuffix) {
            continue;
        }
        for (anim::BoneTrack& track : outClip.tracks) {
            if (track.boneIndex != static_cast<int>(i) || track.translationKeys.empty()) {
                continue;
            }
            const odai::math::Vector3 origin = track.translationKeys.front().value;
            for (anim::Vector3Key& key : track.translationKeys) {
                key.value.x = 0.0f;
                key.value.y -= origin.y;
                key.value.z = 0.0f;
            }
        }
    }
    return !outClip.tracks.empty() && outClip.duration > 0.0f;
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

    // Skeleton bind-pose world matrices are the canonical GPU binding for
    // FaceGen. Most body parts already author this same binding (within export
    // noise); generated face pieces do not, because eyes, mouth, hair and head
    // each carry their own skin space while the renderer has one palette per
    // actor. Their vertices are baked through the authored bind below, then use
    // these canonical inverses for animation.
    std::vector<Matrix4> skeletonBindWorld(character.skeleton.bones.size(), Matrix4::identity());
    for (std::size_t i = 0; i < character.skeleton.bones.size(); ++i) {
        const Matrix4 local = boneLocalMatrix(character.skeleton.bones[i]);
        const int parent = character.skeleton.bones[i].parentIndex;
        skeletonBindWorld[i] = parent >= 0
            ? skeletonBindWorld[static_cast<std::size_t>(parent)] * local
            : local;
    }

    for (const NifSkinnedShape& shape : model.shapes) {
        const std::size_t vertexCount = shape.positions.size() / 3u;
        if (vertexCount == 0u || shape.triangleIndices.empty()) {
            continue;
        }

        // This shape's local bone list -> skeleton bone indices, resolved once
        // per shape and then applied per vertex.
        std::vector<int> skeletonBoneIndex(shape.boneNames.size(), -1);
        std::vector<Matrix4> authoredBindSkin(shape.boneNames.size(), Matrix4::identity());
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
            authoredBindSkin[b] = skeletonBindWorld[boneSlot] * inverseBind;
            if (shape.requiresCanonicalBindBake) {
                const Matrix4 canonicalInverse = inverse(skeletonBindWorld[boneSlot]);
                if (!inverseBindWritten[boneSlot]) {
                    character.inverseBindMatrices[boneSlot] = canonicalInverse;
                    inverseBindWritten[boneSlot] = true;
                } else if (matricesDiffer(
                               character.inverseBindMatrices[boneSlot], canonicalInverse)) {
                    ++character.conflictingInverseBindCount;
                }
                continue;
            }
            if (shape.usesDynamicPositions) {
                if (!inverseBindWritten[boneSlot]) {
                    character.inverseBindMatrices[boneSlot] = inverse(skeletonBindWorld[boneSlot]);
                    inverseBindWritten[boneSlot] = true;
                }
                // A disagreement is expected across generated face pieces and
                // is resolved by baking each piece through authoredBindSkin.
                continue;
            }
            if (!inverseBindWritten[boneSlot]) {
                character.inverseBindMatrices[boneSlot] = inverseBind;
                inverseBindWritten[boneSlot] = true;
            } else if (matricesDiffer(character.inverseBindMatrices[boneSlot], inverseBind)) {
                ++character.conflictingInverseBindCount;
            }
        }

        // A SHAPE'S VERTICES ARE NOT NECESSARILY IN THE CHARACTER'S SPACE.
        //
        // NiSkinData's overall skinTransform maps the character's space into
        // this shape's own geometry space. That is what makes the composition
        // above agree across shapes authored in different spaces -- and it is
        // why a bone two such shapes share reports no conflict. But composing
        // it normalizes the BINDING only. Nothing has normalized the GEOMETRY,
        // so the vertices are still in the shape's own space and the skinning
        // product faithfully leaves them there.
        //
        // Fallout's human parts are where this becomes visible: a hand NIF is
        // authored around the hand, a head NIF around the head, and a body NIF
        // around the whole character. Left uncorrected, a settler renders as a
        // clothed torso with his head and both hands piled at his feet -- each
        // the right shape, in the wrong place. Every creature part in the game
        // has an identity skinTransform, which is why the actors built before
        // the townsfolk never showed it.
        const Matrix4 geometryToCharacter =
            shape.requiresCanonicalBindBake &&
                !shape.canonicalBindCancelsSkinTransform
            ? Matrix4::identity()
            : inverse(changeMatrixBasis(shape.skinTransform));
        const bool moveGeometry = matricesDiffer(geometryToCharacter, Matrix4::identity());

        const auto baseVertex = static_cast<std::uint32_t>(character.vertices.size());
        FalloutCharacterPart part;
        part.name = shape.name;
        part.diffuseTexturePath = shape.diffuseTexturePath;
        part.alphaTest = shape.alphaTest;
        part.alphaThreshold = shape.alphaThreshold;
        part.alphaBlend = shape.alphaBlend;
        part.twoSided = shape.twoSided;
        part.unlit = shape.unlit;
        part.firstIndex = static_cast<std::uint32_t>(character.indices.size());
        part.indexCount = 0u;

        character.vertices.reserve(character.vertices.size() + vertexCount);
        std::vector<bool> vertexHasUnresolvedInfluence(vertexCount, false);
        for (std::size_t v = 0; v < vertexCount; ++v) {
            odai::render::ImportedSkinnedMeshVertex vertex;
            Vector3 position = changePointBasis(
                shape.positions[v * 3u], shape.positions[(v * 3u) + 1u],
                shape.positions[(v * 3u) + 2u]);
            if (moveGeometry) {
                position = transformPoint(geometryToCharacter, position);
            }
            if (shape.usesDynamicPositions || shape.requiresCanonicalBindBake) {
                Vector3 baked{0.0f, 0.0f, 0.0f};
                float totalWeight = 0.0f;
                for (int k = 0; k < kNifMaxBoneInfluences; ++k) {
                    const std::size_t slot =
                        (v * kNifMaxBoneInfluences) + static_cast<std::size_t>(k);
                    const float weight = shape.boneWeights[slot];
                    const std::uint16_t localBone = shape.boneIndices[slot];
                    if (weight <= 0.0f || localBone >= authoredBindSkin.size()) {
                        continue;
                    }
                    const Vector3 contribution =
                        transformPoint(authoredBindSkin[localBone], position);
                    baked.x += contribution.x * weight;
                    baked.y += contribution.y * weight;
                    baked.z += contribution.z * weight;
                    totalWeight += weight;
                }
                if (totalWeight > 0.0f) {
                    position = baked * (1.0f / totalWeight);
                }
            }
            vertex.position[0] = position.x;
            vertex.position[1] = position.y;
            vertex.position[2] = position.z;
            if ((v * 3u) + 2u < shape.normals.size()) {
                Vector3 normal = changePointBasis(
                    shape.normals[v * 3u], shape.normals[(v * 3u) + 1u],
                    shape.normals[(v * 3u) + 2u]);
                if (moveGeometry) {
                    // A direction, so translation must not apply.
                    normal = normalize(transformDirection(geometryToCharacter, normal));
                }
                if (shape.usesDynamicPositions || shape.requiresCanonicalBindBake) {
                    Vector3 baked{0.0f, 0.0f, 0.0f};
                    for (int k = 0; k < kNifMaxBoneInfluences; ++k) {
                        const std::size_t slot =
                            (v * kNifMaxBoneInfluences) + static_cast<std::size_t>(k);
                        const float weight = shape.boneWeights[slot];
                        const std::uint16_t localBone = shape.boneIndices[slot];
                        if (weight <= 0.0f || localBone >= authoredBindSkin.size()) {
                            continue;
                        }
                        const Vector3 contribution =
                            transformDirection(authoredBindSkin[localBone], normal);
                        baked.x += contribution.x * weight;
                        baked.y += contribution.y * weight;
                        baked.z += contribution.z * weight;
                    }
                    if (lengthSquared(baked) > 1e-12f) {
                        normal = normalize(baked);
                    }
                }
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
                    if (weight > 0.0f) {
                        vertexHasUnresolvedInfluence[v] = true;
                    }
                    // This slot is never read by the shader. Bone 0 rather
                    // than the unresolved index keeps it in range; triangles
                    // touching a positive unresolved slot are culled below.
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
        for (std::size_t triangle = 0u;
             triangle + 2u < shape.triangleIndices.size(); triangle += 3u) {
            const std::uint32_t a = shape.triangleIndices[triangle];
            const std::uint32_t b = shape.triangleIndices[triangle + 1u];
            const std::uint32_t c = shape.triangleIndices[triangle + 2u];
            if (a >= vertexCount || b >= vertexCount || c >= vertexCount ||
                vertexHasUnresolvedInfluence[a] ||
                vertexHasUnresolvedInfluence[b] ||
                vertexHasUnresolvedInfluence[c]) {
                ++character.droppedUnresolvedBoneTriangleCount;
                continue;
            }
            character.indices.push_back(baseVertex + a);
            character.indices.push_back(baseVertex + b);
            character.indices.push_back(baseVertex + c);
        }
        part.indexCount = static_cast<std::uint32_t>(character.indices.size()) -
            part.firstIndex;
        if (part.indexCount != 0u) {
            character.parts.push_back(std::move(part));
        }
    }

    return true;
}

bool appendFalloutCharacterRigidMesh(
    const NifModel& model,
    const std::string& rootNodeName,
    FalloutCharacter& character,
    std::string& outError) {
    const int boneIndex = character.skeleton.findBone(rootNodeName);
    if (boneIndex < 0) {
        outError = "no skeleton bone named " + rootNodeName;
        return false;
    }
    if (model.shapes.empty()) {
        outError = "static parse produced no shapes";
        return false;
    }

    // Where to bake the prop's vertices so that skinning them to this one bone
    // reproduces "rigidly parented to the bone".
    //
    // Skinning computes  actorWorld * boneWorld * inverseBind * v, so for the
    // prop to land at  actorWorld * boneWorld * vLocal  the baked vertex has to
    // satisfy  inverseBind * vBaked == vLocal  -- that is, vBaked is vLocal put
    // through the INVERSE of this bone's stored inverse bind.
    //
    // Deriving the bone's bind-pose world transform from the skeleton instead
    // is the obvious thing and it is wrong whenever NiSkinData disagrees with
    // the skeleton's own bind pose, which is exactly the case
    // FalloutCharacter::inverseBindMatrices exists to record. A bone no skinned
    // shape binds keeps identity there, and identity is also the right answer
    // for it: vBaked == vLocal.
    //
    // This READS the shared inverse-bind entry rather than overwriting it,
    // which is what makes weighting the prop safe -- an earlier version emitted
    // these vertices unweighted to avoid touching that shared state, and
    // unweighted is not merely un-animated: the skinning shader passes an
    // unweighted vertex through at its authored position, so it never receives
    // the actor's world placement either and the prop renders at the world
    // origin, thousands of units from the character.
    const auto boneSlot = static_cast<std::size_t>(boneIndex);
    const Matrix4 attach = (boneSlot < character.inverseBindMatrices.size())
        ? odai::math::inverse(character.inverseBindMatrices[boneSlot])
        : Matrix4::identity();

    bool appended = false;
    for (const NifShape& shape : model.shapes) {
        const std::size_t vertexCount = shape.positions.size() / 3u;
        if (vertexCount == 0u || shape.triangleIndices.empty()) {
            continue;
        }
        const auto baseVertex = static_cast<std::uint32_t>(character.vertices.size());
        FalloutCharacterPart part;
        part.name = shape.name;
        part.diffuseTexturePath = shape.diffuseTexturePath;
        part.alphaTest = shape.alphaTest;
        part.alphaThreshold = shape.alphaThreshold;
        part.alphaBlend = shape.alphaBlend;
        part.twoSided = shape.twoSided;
        part.unlit = shape.unlit;
        part.firstIndex = static_cast<std::uint32_t>(character.indices.size());
        part.indexCount = static_cast<std::uint32_t>(shape.triangleIndices.size());

        character.vertices.reserve(character.vertices.size() + vertexCount);
        for (std::size_t v = 0; v < vertexCount; ++v) {
            const Vector3 local = changePointBasis(
                shape.positions[v * 3u], shape.positions[(v * 3u) + 1u],
                shape.positions[(v * 3u) + 2u]);
            const Vector4 posed = attach * Vector4{local.x, local.y, local.z, 1.0f};

            odai::render::ImportedSkinnedMeshVertex vertex{};
            vertex.position[0] = posed.x;
            vertex.position[1] = posed.y;
            vertex.position[2] = posed.z;
            if ((v * 3u) + 2u < shape.normals.size()) {
                const Vector3 n = changePointBasis(
                    shape.normals[v * 3u], shape.normals[(v * 3u) + 1u],
                    shape.normals[(v * 3u) + 2u]);
                const Vector4 posedNormal = attach * Vector4{n.x, n.y, n.z, 0.0f};
                const float length = std::sqrt((posedNormal.x * posedNormal.x) +
                                               (posedNormal.y * posedNormal.y) +
                                               (posedNormal.z * posedNormal.z));
                if (length > 1e-6f) {
                    vertex.normal[0] = posedNormal.x / length;
                    vertex.normal[1] = posedNormal.y / length;
                    vertex.normal[2] = posedNormal.z / length;
                } else {
                    vertex.normal[1] = 1.0f;
                }
            } else {
                vertex.normal[1] = 1.0f;
            }
            if ((v * 2u) + 1u < shape.uvs.size()) {
                vertex.uv[0] = shape.uvs[v * 2u];
                vertex.uv[1] = shape.uvs[(v * 2u) + 1u];
            }
            // Rigidly bound: all of the weight on the one attachment bone, so
            // the prop rides that bone through every pose the way a parented
            // node would.
            vertex.boneIndices[0] = static_cast<std::uint16_t>(boneIndex);
            vertex.boneWeights[0] = 1.0f;
            character.vertices.push_back(vertex);
        }
        for (const std::uint32_t index : shape.triangleIndices) {
            character.indices.push_back(baseVertex + index);
        }
        character.parts.push_back(std::move(part));
        appended = true;
    }
    if (!appended) {
        outError = "every shape was empty";
        return false;
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
