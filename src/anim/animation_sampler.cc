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

void AnimationSampler::sample(const Skeleton& skeleton, const AnimationClip& clip, float timeSeconds,
                               std::vector<Matrix4>& outMatrices) const {
    const float t = wrapTime(timeSeconds, clip.duration, clip.loop);

    std::vector<int> trackForBone(skeleton.bones.size(), -1);
    for (std::size_t i = 0; i < clip.tracks.size(); ++i) {
        const int boneIndex = clip.tracks[i].boneIndex;
        if (boneIndex >= 0 && static_cast<std::size_t>(boneIndex) < skeleton.bones.size()) {
            trackForBone[static_cast<std::size_t>(boneIndex)] = static_cast<int>(i);
        }
    }

    std::vector<Matrix4> localMatrices(skeleton.bones.size());
    for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
        const Bone& bone = skeleton.bones[i];
        const int trackIndex = trackForBone[i];
        if (trackIndex < 0) {
            localMatrices[i] = composeLocal(bone.localTranslation, bone.localRotation, bone.localScale);
            continue;
        }
        const BoneTrack& track = clip.tracks[static_cast<std::size_t>(trackIndex)];
        const Vector3 translation = evalTrack(track.translationKeys, t, clip.duration, clip.loop,
                                               bone.localTranslation, lerpVector3);
        const Quaternion rotation = evalTrack(track.rotationKeys, t, clip.duration, clip.loop,
                                               bone.localRotation, odai::math::slerp);
        const Vector3 scale = evalTrack(track.scaleKeys, t, clip.duration, clip.loop,
                                         bone.localScale, lerpVector3);
        localMatrices[i] = composeLocal(translation, rotation, scale);
    }

    const std::vector<Matrix4> worldMatrices = composeWorldMatrices(skeleton, localMatrices);

    outMatrices.resize(skeleton.bones.size());
    for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
        const Matrix4& inverseBind = (i < inverseBindMatrices_.size()) ? inverseBindMatrices_[i]
                                                                        : Matrix4::identity();
        outMatrices[i] = worldMatrices[i] * inverseBind;
    }
}

}  // namespace odai::anim
