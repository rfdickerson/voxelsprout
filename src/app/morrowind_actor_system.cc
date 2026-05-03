#include "app/morrowind_actor_system.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <limits>
#include <unordered_set>

namespace {

constexpr std::uint32_t kImportedSceneMaterialFlagNpcGpuTransform = 1u << 24u;
constexpr std::uint32_t kImportedSceneMaterialFlagNpcGpuSkinned = 1u << 26u;

std::string lowerActorKey(std::string value) {
    std::replace(value.begin(), value.end(), '\\', '/');
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

bool assignImportedActorDrawMaterial(
    std::vector<odai::importer::ImportedScenePackedVertex>& vertices,
    const std::vector<std::uint32_t>& indices,
    const odai::importer::ImportedScenePackedDraw& draw,
    std::uint32_t textureIndex,
    std::uint32_t materialFlags
) {
    const std::size_t firstIndex = draw.firstIndex;
    const std::size_t indexEnd = firstIndex + draw.indexCount;
    if (draw.indexCount == 0u || firstIndex >= indices.size() || indexEnd > indices.size()) {
        return false;
    }
    for (std::size_t indexOffset = firstIndex; indexOffset < indexEnd; ++indexOffset) {
        const std::uint32_t vertexIndex = indices[indexOffset];
        if (vertexIndex >= vertices.size()) {
            return false;
        }
        vertices[vertexIndex].textureIndex = textureIndex;
        vertices[vertexIndex].flags = materialFlags;
    }
    return true;
}

std::array<float, 16> identityActorMatrix() {
    return {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

std::array<float, 16> multiplyActorMatrices(
    const std::array<float, 16>& lhs,
    const std::array<float, 16>& rhs
) {
    std::array<float, 16> out{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += lhs[static_cast<std::size_t>(row * 4 + k)] *
                    rhs[static_cast<std::size_t>(k * 4 + col)];
            }
            out[static_cast<std::size_t>(row * 4 + col)] = sum;
        }
    }
    return out;
}

std::array<float, 4> normalizeActorQuat(std::array<float, 4> q) {
    const float len = std::sqrt((q[0] * q[0]) + (q[1] * q[1]) + (q[2] * q[2]) + (q[3] * q[3]));
    if (len <= 0.00001f) {
        return {0.0f, 0.0f, 0.0f, 1.0f};
    }
    return {q[0] / len, q[1] / len, q[2] / len, q[3] / len};
}

std::array<float, 4> multiplyActorQuats(
    const std::array<float, 4>& a,
    const std::array<float, 4>& b
) {
    return {
        (a[3] * b[0]) + (a[0] * b[3]) + (a[1] * b[2]) - (a[2] * b[1]),
        (a[3] * b[1]) - (a[0] * b[2]) + (a[1] * b[3]) + (a[2] * b[0]),
        (a[3] * b[2]) + (a[0] * b[1]) - (a[1] * b[0]) + (a[2] * b[3]),
        (a[3] * b[3]) - (a[0] * b[0]) - (a[1] * b[1]) - (a[2] * b[2])
    };
}

std::array<float, 4> axisAngleActorQuat(float x, float y, float z, float radians) {
    const float half = radians * 0.5f;
    const float s = std::sin(half);
    return normalizeActorQuat({x * s, y * s, z * s, std::cos(half)});
}

std::array<float, 16> actorMatrixFromQuat(
    const std::array<float, 4>& quat,
    const std::array<float, 3>& translation,
    float scale
) {
    const std::array<float, 4> q = normalizeActorQuat(quat);
    const float x = q[0];
    const float y = q[1];
    const float z = q[2];
    const float w = q[3];
    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float wx = w * x;
    const float wy = w * y;
    const float wz = w * z;
    return {
        (1.0f - (2.0f * (yy + zz))) * scale,
        (2.0f * (xy - wz)) * scale,
        (2.0f * (xz + wy)) * scale,
        translation[0],
        (2.0f * (xy + wz)) * scale,
        (1.0f - (2.0f * (xx + zz))) * scale,
        (2.0f * (yz - wx)) * scale,
        translation[1],
        (2.0f * (xz - wy)) * scale,
        (2.0f * (yz + wx)) * scale,
        (1.0f - (2.0f * (xx + yy))) * scale,
        translation[2],
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

float actorMatrixUniformScale(const std::array<float, 16>& matrix) {
    const float sx = std::sqrt((matrix[0] * matrix[0]) + (matrix[4] * matrix[4]) + (matrix[8] * matrix[8]));
    const float sy = std::sqrt((matrix[1] * matrix[1]) + (matrix[5] * matrix[5]) + (matrix[9] * matrix[9]));
    const float sz = std::sqrt((matrix[2] * matrix[2]) + (matrix[6] * matrix[6]) + (matrix[10] * matrix[10]));
    return (sx + sy + sz) / 3.0f;
}

float sampleActorFloatKeys(
    std::span<const odai::importer::ImportedNifFloatKey> keys,
    float time,
    float fallback
) {
    if (keys.empty()) return fallback;
    if (time <= keys.front().time) return keys.front().value;
    if (time >= keys.back().time) return keys.back().value;
    for (std::size_t keyIndex = 1; keyIndex < keys.size(); ++keyIndex) {
        const odai::importer::ImportedNifFloatKey& b = keys[keyIndex];
        if (time > b.time) continue;
        const odai::importer::ImportedNifFloatKey& a = keys[keyIndex - 1u];
        const float t = std::clamp((time - a.time) / std::max(b.time - a.time, 0.00001f), 0.0f, 1.0f);
        return a.value + ((b.value - a.value) * t);
    }
    return keys.back().value;
}

std::array<float, 3> sampleActorVec3Keys(
    std::span<const odai::importer::ImportedNifVec3Key> keys,
    float time,
    const std::array<float, 3>& fallback
) {
    if (keys.empty()) return fallback;
    if (time <= keys.front().time) return {keys.front().value[0], keys.front().value[1], keys.front().value[2]};
    if (time >= keys.back().time) return {keys.back().value[0], keys.back().value[1], keys.back().value[2]};
    for (std::size_t keyIndex = 1; keyIndex < keys.size(); ++keyIndex) {
        const odai::importer::ImportedNifVec3Key& b = keys[keyIndex];
        if (time > b.time) continue;
        const odai::importer::ImportedNifVec3Key& a = keys[keyIndex - 1u];
        const float t = std::clamp((time - a.time) / std::max(b.time - a.time, 0.00001f), 0.0f, 1.0f);
        return {
            a.value[0] + ((b.value[0] - a.value[0]) * t),
            a.value[1] + ((b.value[1] - a.value[1]) * t),
            a.value[2] + ((b.value[2] - a.value[2]) * t)
        };
    }
    return {keys.back().value[0], keys.back().value[1], keys.back().value[2]};
}

std::array<float, 4> sampleActorQuatKeys(
    std::span<const odai::importer::ImportedNifQuatKey> keys,
    float time,
    const std::array<float, 4>& fallback
) {
    if (keys.empty()) return fallback;
    const auto keyQuat = [](const odai::importer::ImportedNifQuatKey& key) {
        return normalizeActorQuat({key.value[1], key.value[2], key.value[3], key.value[0]});
    };
    if (time <= keys.front().time) return keyQuat(keys.front());
    if (time >= keys.back().time) return keyQuat(keys.back());
    for (std::size_t keyIndex = 1; keyIndex < keys.size(); ++keyIndex) {
        const odai::importer::ImportedNifQuatKey& bKey = keys[keyIndex];
        if (time > bKey.time) continue;
        const odai::importer::ImportedNifQuatKey& aKey = keys[keyIndex - 1u];
        std::array<float, 4> a = keyQuat(aKey);
        std::array<float, 4> b = keyQuat(bKey);
        float dot = (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]) + (a[3] * b[3]);
        if (dot < 0.0f) {
            b = {-b[0], -b[1], -b[2], -b[3]};
            dot = -dot;
        }
        const float t = std::clamp((time - aKey.time) / std::max(bKey.time - aKey.time, 0.00001f), 0.0f, 1.0f);
        if (dot > 0.9995f) {
            return normalizeActorQuat({
                a[0] + ((b[0] - a[0]) * t),
                a[1] + ((b[1] - a[1]) * t),
                a[2] + ((b[2] - a[2]) * t),
                a[3] + ((b[3] - a[3]) * t)
            });
        }
        const float theta0 = std::acos(std::clamp(dot, -1.0f, 1.0f));
        const float theta = theta0 * t;
        const float sinTheta = std::sin(theta);
        const float sinTheta0 = std::sin(theta0);
        const float s0 = std::cos(theta) - (dot * sinTheta / std::max(sinTheta0, 0.00001f));
        const float s1 = sinTheta / std::max(sinTheta0, 0.00001f);
        return normalizeActorQuat({
            (a[0] * s0) + (b[0] * s1),
            (a[1] * s0) + (b[1] * s1),
            (a[2] * s0) + (b[2] * s1),
            (a[3] * s0) + (b[3] * s1)
        });
    }
    return keyQuat(keys.back());
}

std::array<float, 16> actorMatrixFromArray(const float matrix[16]) {
    std::array<float, 16> out{};
    std::copy(matrix, matrix + 16, out.begin());
    return out;
}

std::array<float, 16> nifMatrixToEngineMatrix(const std::array<float, 16>& nifMatrix) {
    const std::array<float, 16> nifToEngine = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    return multiplyActorMatrices(multiplyActorMatrices(nifToEngine, nifMatrix), nifToEngine);
}

const odai::importer::ImportedNifNodeAnimation* findActorNodeAnimation(
    std::span<const odai::importer::ImportedNifNodeAnimation> animations,
    std::uint32_t nodeIndex
) {
    for (const odai::importer::ImportedNifNodeAnimation& animation : animations) {
        if (animation.nodeIndex == nodeIndex) return &animation;
    }
    return nullptr;
}

std::array<float, 4> sampleActorXyzRotation(
    const odai::importer::ImportedNifNodeAnimation& animation,
    float time
) {
    const float xrot = sampleActorFloatKeys(animation.xRotationKeys, time, 0.0f);
    const float yrot = sampleActorFloatKeys(animation.yRotationKeys, time, 0.0f);
    const float zrot = sampleActorFloatKeys(animation.zRotationKeys, time, 0.0f);
    const std::array<float, 4> xr = axisAngleActorQuat(1.0f, 0.0f, 0.0f, xrot);
    const std::array<float, 4> yr = axisAngleActorQuat(0.0f, 1.0f, 0.0f, yrot);
    const std::array<float, 4> zr = axisAngleActorQuat(0.0f, 0.0f, 1.0f, zrot);
    return multiplyActorQuats(multiplyActorQuats(xr, yr), zr);
}

std::array<float, 16> animatedActorLocalMatrix(
    const odai::importer::ImportedSkeletonNode& node,
    std::uint32_t nodeIndex,
    std::span<const odai::importer::ImportedNifNodeAnimation> nodeAnimations,
    const odai::importer::ImportedAnimationClip* animationClip,
    float animationTime
) {
    std::array<float, 16> local = actorMatrixFromArray(node.localTransform);
    if (animationClip == nullptr) {
        return local;
    }
    const odai::importer::ImportedNifNodeAnimation* animation =
        findActorNodeAnimation(nodeAnimations, nodeIndex);
    if (animation == nullptr) {
        return local;
    }
    const float duration = animationClip->stopTime - animationClip->startTime;
    float sampleTime = animationClip->startTime;
    if (duration > 0.0001f) {
        sampleTime += std::fmod(std::max(animationTime, 0.0f), duration);
    }
    const std::array<float, 3> baseTranslation{local[3], local[7], local[11]};
    const std::array<float, 3> translation =
        sampleActorVec3Keys(animation->translationKeys, sampleTime, baseTranslation);
    const float scale = sampleActorFloatKeys(animation->scaleKeys, sampleTime, actorMatrixUniformScale(local));
    bool hasRotation = false;
    std::array<float, 4> rotation{0.0f, 0.0f, 0.0f, 1.0f};
    if (!animation->rotationKeys.empty()) {
        rotation = sampleActorQuatKeys(animation->rotationKeys, sampleTime, rotation);
        hasRotation = true;
    } else if (!animation->xRotationKeys.empty() ||
               !animation->yRotationKeys.empty() ||
               !animation->zRotationKeys.empty()) {
        rotation = sampleActorXyzRotation(*animation, sampleTime);
        hasRotation = true;
    }
    if (hasRotation) {
        return actorMatrixFromQuat(rotation, translation, scale);
    }
    local[3] = translation[0];
    local[7] = translation[1];
    local[11] = translation[2];
    return local;
}

odai::math::Vector3 nifPointToEnginePoint(const std::array<float, 16>& nifMatrix) {
    return {nifMatrix[3], nifMatrix[11], nifMatrix[7]};
}

odai::math::Vector3 rotateActorLocalPointToWorld(
    const odai::math::Vector3& localPoint,
    const odai::math::Vector3& actorPosition,
    float yawRadians
) {
    const float c = std::cos(yawRadians);
    const float s = std::sin(yawRadians);
    return {
        actorPosition.x + (localPoint.x * c) + (localPoint.z * s),
        actorPosition.y + localPoint.y,
        actorPosition.z + (-localPoint.x * s) + (localPoint.z * c)
    };
}

std::array<float, 3> actorBoneDebugColor(std::string_view boneName) {
    const std::string name = lowerActorKey(std::string(boneName));
    if (name.find(" l ") != std::string::npos) return {0.25f, 0.55f, 1.0f};
    if (name.find(" r ") != std::string::npos) return {1.0f, 0.32f, 0.26f};
    if (name.find("spine") != std::string::npos ||
        name.find("neck") != std::string::npos ||
        name.find("head") != std::string::npos ||
        name.find("pelvis") != std::string::npos) {
        return {0.20f, 1.0f, 0.42f};
    }
    return {1.0f, 0.90f, 0.25f};
}

const odai::importer::ImportedAnimationClip* findWalkClip(
    std::span<const odai::importer::ImportedAnimationClip> clips
) {
    const auto normalizedClipName = [](const std::string& clipName) {
        const std::string lower = lowerActorKey(clipName);
        std::string out;
        out.reserve(lower.size());
        for (const char ch : lower) {
            if (std::isalnum(static_cast<unsigned char>(ch))) {
                out.push_back(ch);
            }
        }
        return out;
    };
    const odai::importer::ImportedAnimationClip* genericWalk = nullptr;
    const odai::importer::ImportedAnimationClip* directionalWalk = nullptr;
    for (const odai::importer::ImportedAnimationClip& clip : clips) {
        if (clip.stopTime <= clip.startTime) continue;
        const std::string name = normalizedClipName(clip.name);
        if (name.find("swim") != std::string::npos ||
            name.find("run") != std::string::npos ||
            name.find("sneak") != std::string::npos) {
            continue;
        }
        if (name == "walkforward") return &clip;
        if (directionalWalk == nullptr && name.find("walkforward") != std::string::npos) {
            directionalWalk = &clip;
            continue;
        }
        if (genericWalk == nullptr && name.find("walk") != std::string::npos) {
            genericWalk = &clip;
        }
    }
    return directionalWalk != nullptr ? directionalWalk : genericWalk;
}

const odai::importer::ImportedAnimationClip* findIdleClip(
    std::span<const odai::importer::ImportedAnimationClip> clips
) {
    const auto normalizedClipName = [](const std::string& clipName) {
        const std::string lower = lowerActorKey(clipName);
        std::string out;
        out.reserve(lower.size());
        for (const char ch : lower) {
            if (std::isalnum(static_cast<unsigned char>(ch))) {
                out.push_back(ch);
            }
        }
        return out;
    };
    const odai::importer::ImportedAnimationClip* genericIdle = nullptr;
    for (const odai::importer::ImportedAnimationClip& clip : clips) {
        if (clip.stopTime <= clip.startTime) continue;
        const std::string name = normalizedClipName(clip.name);
        if (name.find("swim") != std::string::npos ||
            name.find("walk") != std::string::npos ||
            name.find("run") != std::string::npos ||
            name.find("sneak") != std::string::npos) {
            continue;
        }
        if (name == "idle") return &clip;
        if (genericIdle == nullptr && name.find("idle") != std::string::npos) {
            genericIdle = &clip;
        }
    }
    return genericIdle;
}

} // namespace

namespace odai::app {

void MorrowindActorSystem::clear() {
    m_skeletonTemplate = {};
    m_vertices.clear();
    m_indices.clear();
    m_draws.clear();
    m_boneIndices.clear();
    m_boneWeights.clear();
    m_prototypes.clear();
    m_prototypeBySignature.clear();
    m_frameInstances.clear();
    m_frameBonePalette.clear();
    m_frameDebugBoneLines.clear();
    m_stats = {};
    m_skeletonLoaded = false;
    m_assetDirty = true;
    m_frameDebugBoneLinesEnabled = false;
}

bool MorrowindActorSystem::loadHumanoidSkeleton(
    const std::filesystem::path& dataFilesPath,
    std::string& outError
) {
    if (m_skeletonLoaded) {
        return true;
    }
    const std::filesystem::path baseAnimPath = dataFilesPath / "Meshes" / "base_anim.nif";
    if (!odai::importer::loadMorrowindSkinnedActorSkeleton(baseAnimPath, m_skeletonTemplate, outError)) {
        return false;
    }
    m_skeletonLoaded =
        !m_skeletonTemplate.skeleton.empty() &&
        !m_skeletonTemplate.animationClips.empty();
    if (!m_skeletonLoaded) {
        outError = "base_anim.nif did not expose a usable humanoid skeleton and clips";
        return false;
    }
    refreshStats();
    return true;
}

std::uint32_t MorrowindActorSystem::findOrBuildHumanoidPrototype(
    const std::filesystem::path& dataFilesPath,
    const odai::importer::MorrowindActorRecord& actor,
    const odai::importer::MorrowindEquipmentCatalog& equipmentCatalog,
    const TextureSlotFn& textureSlotFn,
    std::string& outError
) {
    const std::vector<odai::importer::MorrowindEquipmentCatalog::ResolvedActorPart> parts =
        odai::importer::resolveMorrowindNpcParts(actor, equipmentCatalog);
    if (parts.empty()) {
        outError = "actor resolved no catalog-driven humanoid parts: " + actor.id;
        return std::numeric_limits<std::uint32_t>::max();
    }
    const std::string signature = makeAppearanceSignature(actor, parts);
    const auto existing = m_prototypeBySignature.find(signature);
    if (existing != m_prototypeBySignature.end()) {
        return existing->second;
    }
    const std::uint32_t prototypeIndex =
        findOrBuildHumanoidPrototypeFromParts(dataFilesPath, actor.id, parts, textureSlotFn, outError);
    if (prototypeIndex != std::numeric_limits<std::uint32_t>::max()) {
        m_prototypeBySignature.emplace(signature, prototypeIndex);
    }
    return prototypeIndex;
}

std::uint32_t MorrowindActorSystem::findOrBuildHumanoidPrototypeFromParts(
    const std::filesystem::path& dataFilesPath,
    const std::string& actorId,
    const std::vector<odai::importer::MorrowindEquipmentCatalog::ResolvedActorPart>& parts,
    const TextureSlotFn& textureSlotFn,
    std::string& outError
) {
    if (!loadHumanoidSkeleton(dataFilesPath, outError)) {
        outError = "base_anim.nif failed for actor " + actorId + ": " + outError;
        return std::numeric_limits<std::uint32_t>::max();
    }
    odai::importer::ImportedSkinnedActorAsset actorAsset = m_skeletonTemplate;
    std::unordered_set<std::string> seenPartKeys;
    for (const odai::importer::MorrowindEquipmentCatalog::ResolvedActorPart& part : parts) {
        const std::string partKey = lowerActorKey(
            part.modelPath + "|" +
            part.slot + "|" +
            part.side + "|" +
            part.bodyPartId + "|" +
            part.attachBone + "|" +
            part.meshFilter + "|" +
            std::to_string(part.partReferenceType));
        if (!seenPartKeys.insert(partKey).second) {
            continue;
        }
        const std::filesystem::path nifPath =
            dataFilesPath / "Meshes" / std::filesystem::path(part.modelPath);
        if (!std::filesystem::exists(nifPath)) {
            outError = "actor=" + actorId +
                " part=" + part.modelPath +
                " slot=" + part.slot +
                " side=" + part.side +
                " attachBone=" + part.attachBone +
                " meshFilter=" + part.meshFilter +
                " reason=missing file";
            return std::numeric_limits<std::uint32_t>::max();
        }
        std::string partError;
        if (!odai::importer::appendMorrowindSkinnedActorPartNif(
                nifPath,
                odai::importer::toMorrowindActorPartMetadata(part),
                actorAsset,
                partError)) {
            outError = "actor=" + actorId +
                " part=" + part.modelPath +
                " slot=" + part.slot +
                " side=" + part.side +
                " attachBone=" + part.attachBone +
                " meshFilter=" + part.meshFilter +
                " reason=" + partError;
            return std::numeric_limits<std::uint32_t>::max();
        }
    }
    const bool valid =
        !actorAsset.vertices.empty() &&
        !actorAsset.indices.empty() &&
        !actorAsset.draws.empty() &&
        actorAsset.boneIndices.size() == actorAsset.vertices.size() &&
        actorAsset.boneWeights.size() == actorAsset.vertices.size() &&
        actorAsset.unweightedVertexCount == 0u;
    if (!valid) {
        outError = "actor=" + actorId +
            " reason=invalid humanoid actor asset" +
            " vertices=" + std::to_string(actorAsset.vertices.size()) +
            " indices=" + std::to_string(actorAsset.indices.size()) +
            " draws=" + std::to_string(actorAsset.draws.size()) +
            " boneIndices=" + std::to_string(actorAsset.boneIndices.size()) +
            " boneWeights=" + std::to_string(actorAsset.boneWeights.size()) +
            " unweightedVertices=" + std::to_string(actorAsset.unweightedVertexCount);
        return std::numeric_limits<std::uint32_t>::max();
    }
    std::uint32_t prototypeIndex = std::numeric_limits<std::uint32_t>::max();
    if (!appendBuiltActor(actorId, actorAsset, textureSlotFn, prototypeIndex, outError)) {
        return std::numeric_limits<std::uint32_t>::max();
    }
    m_assetDirty = true;
    refreshStats();
    return prototypeIndex;
}

bool MorrowindActorSystem::appendBuiltActor(
    const std::string& actorId,
    const odai::importer::ImportedSkinnedActorAsset& actorAsset,
    const TextureSlotFn& textureSlotFn,
    std::uint32_t& outPrototypeIndex,
    std::string& outError
) {
    const std::uint32_t vertexBase = static_cast<std::uint32_t>(m_vertices.size());
    const std::uint32_t indexBase = static_cast<std::uint32_t>(m_indices.size());
    const std::uint32_t drawBase = static_cast<std::uint32_t>(m_draws.size());

    m_vertices.insert(m_vertices.end(), actorAsset.vertices.begin(), actorAsset.vertices.end());
    m_boneIndices.insert(m_boneIndices.end(), actorAsset.boneIndices.begin(), actorAsset.boneIndices.end());
    m_boneWeights.insert(m_boneWeights.end(), actorAsset.boneWeights.begin(), actorAsset.boneWeights.end());
    m_indices.reserve(m_indices.size() + actorAsset.indices.size());
    for (const std::uint32_t index : actorAsset.indices) {
        m_indices.push_back(vertexBase + index);
    }
    m_draws.reserve(m_draws.size() + actorAsset.draws.size());
    for (const odai::importer::ImportedScenePackedDraw& sourceDraw : actorAsset.draws) {
        odai::importer::ImportedScenePackedDraw draw = sourceDraw;
        draw.firstIndex += indexBase;
        m_draws.push_back(draw);
    }

    for (std::size_t localDrawIndex = 0; localDrawIndex < actorAsset.draws.size(); ++localDrawIndex) {
        const std::size_t globalDrawIndex = static_cast<std::size_t>(drawBase) + localDrawIndex;
        const std::string texturePath = localDrawIndex < actorAsset.partDiffuseTexturePaths.size()
            ? actorAsset.partDiffuseTexturePaths[localDrawIndex]
            : std::string{};
        const std::uint32_t textureIndex = textureSlotFn(texturePath);
        if (textureIndex == std::numeric_limits<std::uint32_t>::max()) {
            outError = "actor=" + actorId +
                " draw=" + std::to_string(localDrawIndex) +
                " texture=" + texturePath +
                " reason=texture load failed";
            return false;
        }
        const odai::importer::ImportedScenePackedDraw& draw = m_draws[globalDrawIndex];
        if (!assignImportedActorDrawMaterial(
                m_vertices,
                m_indices,
                draw,
                textureIndex,
                kImportedSceneMaterialFlagNpcGpuTransform |
                    kImportedSceneMaterialFlagNpcGpuSkinned)) {
            outError = "actor=" + actorId +
                " draw=" + std::to_string(localDrawIndex) +
                " firstIndex=" + std::to_string(draw.firstIndex) +
                " indexCount=" + std::to_string(draw.indexCount) +
                " reason=invalid draw range";
            return false;
        }
    }

    PrototypeRange range{};
    range.firstDraw = drawBase;
    range.drawCount = static_cast<std::uint32_t>(m_draws.size()) - drawBase;
    if (range.drawCount == 0u) {
        outError = "actor=" + actorId + " reason=prototype produced no draw ranges";
        return false;
    }
    outPrototypeIndex = static_cast<std::uint32_t>(m_prototypes.size());
    m_prototypes.push_back(range);
    m_stats.weightedVertexCount += actorAsset.weightedVertexCount;
    m_stats.unweightedVertexCount += actorAsset.unweightedVertexCount;
    return true;
}

std::string MorrowindActorSystem::makeAppearanceSignature(
    const odai::importer::MorrowindActorRecord& actor,
    std::span<const odai::importer::MorrowindEquipmentCatalog::ResolvedActorPart> parts
) const {
    std::string signature = "humanoid:" + actor.id + ":" +
        actor.raceId + ":" +
        actor.headBodyPartId + ":" +
        actor.hairBodyPartId;
    for (const std::string& itemId : actor.inventoryItemIds) {
        signature += "|item=" + itemId;
    }
    for (const odai::importer::MorrowindEquipmentCatalog::ResolvedActorPart& part : parts) {
        signature += "|" + part.modelPath +
            "|" + part.slot +
            "|" + part.side +
            "|" + part.bodyPartId +
            "|" + part.attachBone +
            "|" + part.meshFilter +
            "|" + std::to_string(part.meshPart) +
            "|" + std::to_string(part.partReferenceType);
    }
    return lowerActorKey(signature);
}

void MorrowindActorSystem::beginFrame(std::size_t visibleActorCapacity, bool debugBoneLines) {
    m_frameInstances.clear();
    m_frameBonePalette.clear();
    m_frameDebugBoneLines.clear();
    m_frameDebugBoneLinesEnabled = debugBoneLines;
    m_frameInstances.reserve(visibleActorCapacity);
    m_frameBonePalette.reserve(visibleActorCapacity * m_skeletonTemplate.skeleton.size());
    if (debugBoneLines) {
        m_frameDebugBoneLines.reserve(visibleActorCapacity * m_skeletonTemplate.skeleton.size() * 2u);
    }
}

void MorrowindActorSystem::appendFrameInstance(
    const FrameInput& input,
    PoseMode poseMode,
    float nowSeconds
) {
    if (input.prototypeIndex >= m_prototypes.size() || m_skeletonTemplate.skeleton.empty()) {
        return;
    }
    const PrototypeRange& range = m_prototypes[input.prototypeIndex];
    odai::render::ImportedActorInstanceData instance{};
    instance.position[0] = input.position.x;
    instance.position[1] = input.position.y;
    instance.position[2] = input.position.z;
    instance.yawRadians = input.yawRadians;
    instance.animationTime = input.animationTime;
    instance.flags = kImportedSceneMaterialFlagNpcGpuTransform;
    instance.firstDraw = range.firstDraw;
    instance.drawCount = range.drawCount;
    instance.bonePaletteOffset = static_cast<std::uint32_t>(m_frameBonePalette.size());

    const odai::importer::ImportedAnimationClip* animationClip = nullptr;
    if (poseMode == PoseMode::WalkClip) {
        animationClip = walkClip();
        instance.animationTime = nowSeconds;
    } else if (poseMode == PoseMode::Movement && input.moving) {
        animationClip = walkClip();
    } else if (poseMode == PoseMode::Movement) {
        animationClip = idleClip();
        instance.animationTime = nowSeconds;
    }
    instance.clipIndex = 0u;
    if (animationClip != nullptr) {
        for (std::size_t clipIndex = 0; clipIndex < m_skeletonTemplate.animationClips.size(); ++clipIndex) {
            if (&m_skeletonTemplate.animationClips[clipIndex] == animationClip) {
                instance.clipIndex = static_cast<std::uint32_t>(clipIndex);
                break;
            }
        }
    }

    std::vector<std::array<float, 16>> worldMatrices(m_skeletonTemplate.skeleton.size(), identityActorMatrix());
    for (std::size_t nodeIndex = 0; nodeIndex < m_skeletonTemplate.skeleton.size(); ++nodeIndex) {
        const std::array<float, 16> local =
            animatedActorLocalMatrix(
                m_skeletonTemplate.skeleton[nodeIndex],
                static_cast<std::uint32_t>(nodeIndex),
                m_skeletonTemplate.nodeAnimations,
                animationClip,
                instance.animationTime);
        const std::int32_t parentIndex = m_skeletonTemplate.skeleton[nodeIndex].parentIndex;
        if (parentIndex >= 0 && static_cast<std::size_t>(parentIndex) < worldMatrices.size()) {
            worldMatrices[nodeIndex] =
                multiplyActorMatrices(worldMatrices[static_cast<std::size_t>(parentIndex)], local);
        } else {
            worldMatrices[nodeIndex] = local;
        }

        const std::array<float, 16> inverseBind =
            actorMatrixFromArray(m_skeletonTemplate.skeleton[nodeIndex].inverseBindWorldTransform);
        const std::array<float, 16> paletteNif = multiplyActorMatrices(worldMatrices[nodeIndex], inverseBind);
        const std::array<float, 16> paletteEngine = nifMatrixToEngineMatrix(paletteNif);
        odai::render::ImportedActorBonePaletteMatrix matrix{};
        matrix.rows[0] = paletteEngine[0];
        matrix.rows[1] = paletteEngine[1];
        matrix.rows[2] = paletteEngine[2];
        matrix.rows[3] = paletteEngine[3];
        matrix.rows[4] = paletteEngine[4];
        matrix.rows[5] = paletteEngine[5];
        matrix.rows[6] = paletteEngine[6];
        matrix.rows[7] = paletteEngine[7];
        matrix.rows[8] = paletteEngine[8];
        matrix.rows[9] = paletteEngine[9];
        matrix.rows[10] = paletteEngine[10];
        matrix.rows[11] = paletteEngine[11];
        m_frameBonePalette.push_back(matrix);
    }

    if (!m_frameDebugBoneLinesEnabled) {
        m_frameInstances.push_back(instance);
        return;
    }
    for (std::size_t nodeIndex = 0; nodeIndex < m_skeletonTemplate.skeleton.size(); ++nodeIndex) {
        const std::int32_t parentIndex = m_skeletonTemplate.skeleton[nodeIndex].parentIndex;
        if (parentIndex < 0 || static_cast<std::size_t>(parentIndex) >= worldMatrices.size()) {
            continue;
        }
        const odai::math::Vector3 parentLocal =
            nifPointToEnginePoint(worldMatrices[static_cast<std::size_t>(parentIndex)]);
        const odai::math::Vector3 childLocal = nifPointToEnginePoint(worldMatrices[nodeIndex]);
        const odai::math::Vector3 parentWorld =
            rotateActorLocalPointToWorld(parentLocal, input.position, input.yawRadians);
        const odai::math::Vector3 childWorld =
            rotateActorLocalPointToWorld(childLocal, input.position, input.yawRadians);
        const std::array<float, 3> color = actorBoneDebugColor(m_skeletonTemplate.skeleton[nodeIndex].name);
        odai::render::ImportedActorDebugLineVertex a{};
        a.position[0] = parentWorld.x;
        a.position[1] = parentWorld.y;
        a.position[2] = parentWorld.z;
        a.color[0] = color[0];
        a.color[1] = color[1];
        a.color[2] = color[2];
        odai::render::ImportedActorDebugLineVertex b = a;
        b.position[0] = childWorld.x;
        b.position[1] = childWorld.y;
        b.position[2] = childWorld.z;
        m_frameDebugBoneLines.push_back(a);
        m_frameDebugBoneLines.push_back(b);
    }
    m_frameInstances.push_back(instance);
}

odai::render::ImportedActorRenderAssetData MorrowindActorSystem::renderAssetData() const {
    odai::render::ImportedActorRenderAssetData data{};
    data.vertices = m_vertices;
    data.indices = m_indices;
    data.draws = m_draws;
    data.boneIndices = m_boneIndices;
    data.boneWeights = m_boneWeights;
    return data;
}

odai::render::ImportedActorFrameData MorrowindActorSystem::frameData() const {
    odai::render::ImportedActorFrameData data{};
    data.instances = m_frameInstances;
    data.bonePalette = m_frameBonePalette;
    data.debugBoneLines = m_frameDebugBoneLines;
    return data;
}

bool MorrowindActorSystem::hasRenderableAsset() const {
    return !m_vertices.empty() && !m_indices.empty() && !m_draws.empty();
}

std::uint32_t MorrowindActorSystem::prototypeCount() const {
    return static_cast<std::uint32_t>(std::min<std::size_t>(m_prototypes.size(), std::numeric_limits<std::uint32_t>::max()));
}

std::uint32_t MorrowindActorSystem::vertexCount() const {
    return static_cast<std::uint32_t>(std::min<std::size_t>(m_vertices.size(), std::numeric_limits<std::uint32_t>::max()));
}

std::uint32_t MorrowindActorSystem::indexCount() const {
    return static_cast<std::uint32_t>(std::min<std::size_t>(m_indices.size(), std::numeric_limits<std::uint32_t>::max()));
}

std::uint32_t MorrowindActorSystem::drawCount() const {
    return static_cast<std::uint32_t>(std::min<std::size_t>(m_draws.size(), std::numeric_limits<std::uint32_t>::max()));
}

std::uint32_t MorrowindActorSystem::skeletonNodeCount() const {
    return static_cast<std::uint32_t>(std::min<std::size_t>(m_skeletonTemplate.skeleton.size(), std::numeric_limits<std::uint32_t>::max()));
}

std::uint32_t MorrowindActorSystem::paletteMatrixCount() const {
    return static_cast<std::uint32_t>(std::min<std::size_t>(m_frameBonePalette.size(), std::numeric_limits<std::uint32_t>::max()));
}

std::uint32_t MorrowindActorSystem::visibleInstanceCount() const {
    return static_cast<std::uint32_t>(std::min<std::size_t>(m_frameInstances.size(), std::numeric_limits<std::uint32_t>::max()));
}

const odai::importer::ImportedAnimationClip* MorrowindActorSystem::walkClip() const {
    return findWalkClip(m_skeletonTemplate.animationClips);
}

const odai::importer::ImportedAnimationClip* MorrowindActorSystem::idleClip() const {
    return findIdleClip(m_skeletonTemplate.animationClips);
}

void MorrowindActorSystem::refreshStats() {
    m_stats.prototypeCount = prototypeCount();
    m_stats.skeletonNodeCount = skeletonNodeCount();
    m_stats.nodeAnimationCount = static_cast<std::uint32_t>(
        std::min<std::size_t>(m_skeletonTemplate.nodeAnimations.size(), std::numeric_limits<std::uint32_t>::max()));
    m_stats.clipCount = static_cast<std::uint32_t>(
        std::min<std::size_t>(m_skeletonTemplate.animationClips.size(), std::numeric_limits<std::uint32_t>::max()));
    m_stats.firstBoneNames.clear();
    for (std::size_t i = 0; i < std::min<std::size_t>(m_skeletonTemplate.skeleton.size(), 8u); ++i) {
        if (!m_stats.firstBoneNames.empty()) m_stats.firstBoneNames += ", ";
        m_stats.firstBoneNames += m_skeletonTemplate.skeleton[i].name;
    }
    m_stats.firstClipNames.clear();
    for (std::size_t i = 0; i < std::min<std::size_t>(m_skeletonTemplate.animationClips.size(), 6u); ++i) {
        if (!m_stats.firstClipNames.empty()) m_stats.firstClipNames += ", ";
        m_stats.firstClipNames += m_skeletonTemplate.animationClips[i].name;
    }
}

} // namespace odai::app
