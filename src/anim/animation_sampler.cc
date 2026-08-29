#include "anim/animation_sampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace odai::anim {

namespace {

using odai::math::Matrix4;
using odai::math::Quaternion;
using odai::math::Vector3;

Matrix4 composeLocal(const Vector3& translation, const Quaternion& rotation, const Vector3& scale) {
    return Matrix4::translation(translation) * odai::math::toMatrix(rotation) * Matrix4::scale(scale);
}

// Composes each bone's local transform into world space. Requires
// Skeleton::bones to be stored parent-before-child (see skeleton.h).
std::vector<Matrix4> composeWorldMatrices(const Skeleton& skeleton,
                                           const std::vector<Matrix4>& localMatrices) {
    std::vector<Matrix4> world(skeleton.bones.size());
    for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
        const int parent = skeleton.bones[i].parentIndex;
        world[i] = (parent >= 0) ? (world[static_cast<std::size_t>(parent)] * localMatrices[i])
                                  : localMatrices[i];
    }
    return world;
}

float wrapTime(float t, float duration, bool loop) {
    if (duration <= 0.0f) return 0.0f;
    if (!loop) return std::clamp(t, 0.0f, duration);
    float wrapped = std::fmod(t, duration);
    if (wrapped < 0.0f) wrapped += duration;
    return wrapped;
}

Vector3 lerpVector3(const Vector3& a, const Vector3& b, float t) {
    return a + ((b - a) * t);
}

struct LocalTransform {
    Vector3 translation{};
    Quaternion rotation{};
    Vector3 scale{1.0f, 1.0f, 1.0f};
};

// Evaluates one channel's keys at time t. Outside the keyed range, either
// clamps to the nearest key (non-looping) or blends across the loop boundary
// from the last key back to the first (looping) so a looped clip's last and
// first keyframes read as one continuous cycle.
template <typename Key, typename Value, typename LerpFn>
Value evalTrack(const std::vector<Key>& keys, float t, float duration, bool loop,
                 const Value& bindValue, LerpFn lerpFn) {
    if (keys.empty()) return bindValue;
    if (keys.size() == 1) return keys[0].value;

    if (t < keys.front().time) {
        if (!loop) return keys.front().value;
        const Key& k0 = keys.back();
        const Key& k1 = keys.front();
        const float span = (duration - k0.time) + k1.time;
        const float frac = span > 0.0f ? (t + (duration - k0.time)) / span : 0.0f;
        return lerpFn(k0.value, k1.value, frac);
    }
    if (t > keys.back().time) {
        if (!loop) return keys.back().value;
        const Key& k0 = keys.back();
        const Key& k1 = keys.front();
        const float span = (duration - k0.time) + k1.time;
        const float frac = span > 0.0f ? (t - k0.time) / span : 0.0f;
        return lerpFn(k0.value, k1.value, frac);
    }
    for (std::size_t i = 0; i + 1 < keys.size(); ++i) {
        if (t >= keys[i].time && t <= keys[i + 1].time) {
            const float span = keys[i + 1].time - keys[i].time;
            const float frac = span > 0.0f ? (t - keys[i].time) / span : 0.0f;
            return lerpFn(keys[i].value, keys[i + 1].value, frac);
        }
    }
    return keys.back().value;
}

std::vector<LocalTransform> sampleLocalTransforms(
    const Skeleton& skeleton, const AnimationClip& clip, float timeSeconds) {
    const float t = wrapTime(timeSeconds, clip.duration, clip.loop);
    std::vector<int> trackForBone(skeleton.bones.size(), -1);
    for (std::size_t index = 0; index < clip.tracks.size(); ++index) {
        const int bone = clip.tracks[index].boneIndex;
        if (bone >= 0 && static_cast<std::size_t>(bone) < skeleton.bones.size()) {
            trackForBone[static_cast<std::size_t>(bone)] = static_cast<int>(index);
        }
    }
    std::vector<LocalTransform> result(skeleton.bones.size());
    for (std::size_t index = 0; index < skeleton.bones.size(); ++index) {
        const Bone& bone = skeleton.bones[index];
        result[index] = {bone.localTranslation, bone.localRotation, bone.localScale};
        const int trackIndex = trackForBone[index];
        if (trackIndex < 0) continue;
        const BoneTrack& track = clip.tracks[static_cast<std::size_t>(trackIndex)];
        result[index].translation = evalTrack(
            track.translationKeys, t, clip.duration, clip.loop,
            bone.localTranslation, lerpVector3);
        result[index].rotation = evalTrack(
            track.rotationKeys, t, clip.duration, clip.loop,
            bone.localRotation, odai::math::slerp);
        result[index].scale = evalTrack(
            track.scaleKeys, t, clip.duration, clip.loop,
            bone.localScale, lerpVector3);
    }
    return result;
}

}  // namespace

void AnimationSampler::bindSkeleton(const Skeleton& skeleton) {
    std::vector<Matrix4> localMatrices(skeleton.bones.size());
    for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
        const Bone& bone = skeleton.bones[i];
        localMatrices[i] = composeLocal(bone.localTranslation, bone.localRotation, bone.localScale);
    }
    const std::vector<Matrix4> worldMatrices = composeWorldMatrices(skeleton, localMatrices);

    inverseBindMatrices_.resize(worldMatrices.size());
    for (std::size_t i = 0; i < worldMatrices.size(); ++i) {
        inverseBindMatrices_[i] = odai::math::inverse(worldMatrices[i]);
    }
}

void AnimationSampler::bindSkeleton(const Skeleton& skeleton,
                                     std::vector<Matrix4> inverseBindMatrices) {
    (void)skeleton;
    inverseBindMatrices_ = std::move(inverseBindMatrices);
}

void AnimationSampler::sample(const Skeleton& skeleton, const AnimationClip& clip, float timeSeconds,
                               std::vector<Matrix4>& outMatrices) const {
    const std::vector<LocalTransform> local =
        sampleLocalTransforms(skeleton, clip, timeSeconds);
    std::vector<Matrix4> localMatrices(skeleton.bones.size());
    for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
        localMatrices[i] = composeLocal(
            local[i].translation, local[i].rotation, local[i].scale);
    }

    const std::vector<Matrix4> worldMatrices = composeWorldMatrices(skeleton, localMatrices);

    outMatrices.resize(skeleton.bones.size());
    for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
        const Matrix4& inverseBind = (i < inverseBindMatrices_.size()) ? inverseBindMatrices_[i]
                                                                        : Matrix4::identity();
        outMatrices[i] = worldMatrices[i] * inverseBind;
    }
}

void AnimationSampler::sampleBlended(
    const Skeleton& skeleton,
    const AnimationClip& fromClip, float fromTimeSeconds,
    const AnimationClip& toClip, float toTimeSeconds,
    float alpha,
    std::vector<Matrix4>& outMatrices) const {
    const std::vector<LocalTransform> from =
        sampleLocalTransforms(skeleton, fromClip, fromTimeSeconds);
    const std::vector<LocalTransform> to =
        sampleLocalTransforms(skeleton, toClip, toTimeSeconds);
    const float blend = odai::math::saturate(alpha);
    std::vector<Matrix4> localMatrices(skeleton.bones.size());
    for (std::size_t index = 0; index < skeleton.bones.size(); ++index) {
        localMatrices[index] = composeLocal(
            lerpVector3(from[index].translation, to[index].translation, blend),
            odai::math::slerp(from[index].rotation, to[index].rotation, blend),
            lerpVector3(from[index].scale, to[index].scale, blend));
    }
    const std::vector<Matrix4> worldMatrices = composeWorldMatrices(skeleton, localMatrices);
    outMatrices.resize(skeleton.bones.size());
    for (std::size_t index = 0; index < skeleton.bones.size(); ++index) {
        const Matrix4& inverseBind = index < inverseBindMatrices_.size()
            ? inverseBindMatrices_[index] : Matrix4::identity();
        outMatrices[index] = worldMatrices[index] * inverseBind;
    }
}

}  // namespace odai::anim
