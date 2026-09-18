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

using LocalTransform = LocalPoseTransform;

Quaternion multiply(const Quaternion& a, const Quaternion& b) {
    return odai::math::normalize(Quaternion{
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z});
}

LocalTransform addLocal(const LocalTransform& base, const LocalTransform& delta, float weight) {
    const auto scale = lerpVector3({1, 1, 1}, delta.scale, weight);
    return {base.translation + delta.translation * weight,
        multiply(base.rotation, odai::math::slerp(Quaternion{}, delta.rotation, weight)),
        {base.scale.x * scale.x, base.scale.y * scale.y, base.scale.z * scale.z}};
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

std::vector<LocalTransform> sampleLocalTransforms(
    const Skeleton& skeleton, const AnimationClip& clip, float timeSeconds, bool rawAdditive = false) {
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
        result[index] = clip.additive ? LocalTransform{} :
            LocalTransform{bone.localTranslation, bone.localRotation, bone.localScale};
        const int trackIndex = trackForBone[index];
        if (trackIndex < 0) continue;
        const BoneTrack& track = clip.tracks[static_cast<std::size_t>(trackIndex)];
        result[index].translation = evalTrack(
            track.translationKeys, t, clip.duration, clip.loop,
            result[index].translation, lerpVector3);
        if (static_cast<int>(index) == clip.extractedMotionBone && !track.translationKeys.empty())
            result[index].translation = track.translationKeys.front().value;
        result[index].rotation = evalTrack(
            track.rotationKeys, t, clip.duration, clip.loop,
            result[index].rotation, odai::math::slerp);
        result[index].scale = evalTrack(
            track.scaleKeys, t, clip.duration, clip.loop,
            result[index].scale, lerpVector3);
    }
    if (clip.additive && !rawAdditive) {
        for (std::size_t i = 0; i < result.size(); ++i) {
            const auto& bone = skeleton.bones[i];
            result[i] = addLocal({bone.localTranslation, bone.localRotation, bone.localScale}, result[i], 1);
        }
    }
    return result;
}

}  // namespace

LocalPose sampleLocalPose(const Skeleton& skeleton, const AnimationClip& clip, float time) {
    return sampleLocalTransforms(skeleton, clip, time);
}
LocalPose blendLocalPoses(const LocalPose& from, const LocalPose& to, float weight, std::span<const float> mask) {
    if (from.size() != to.size()) return from;
    LocalPose out = from;
    for (std::size_t i=0; i<out.size(); ++i) {
        const float w = weight * (mask.empty() ? 1.f : i < mask.size() ? mask[i] : 0.f);
        const float a = std::isfinite(w) ? std::clamp(w, 0.f, 1.f) : 0.f;
        out[i] = {lerpVector3(from[i].translation, to[i].translation, a),
            odai::math::slerp(from[i].rotation, to[i].rotation, a), lerpVector3(from[i].scale, to[i].scale, a)};
    }
    return out;
}
LocalPose addLocalPoses(const LocalPose& base, const LocalPose& target, const LocalPose& reference,
    float weight, std::span<const float> mask) {
    if (base.size() != target.size() || base.size() != reference.size()) return base;
    LocalPose out = base;
    for (std::size_t i=0; i<out.size(); ++i) {
        const auto& r = reference[i]; const auto& t = target[i];
        const auto ratio = [](float a, float b) { return std::abs(b)>1.e-6f ? a/b : 1.f; };
        const LocalTransform delta{t.translation-r.translation,
            multiply({-r.rotation.x,-r.rotation.y,-r.rotation.z,r.rotation.w},t.rotation),
            {ratio(t.scale.x,r.scale.x),ratio(t.scale.y,r.scale.y),ratio(t.scale.z,r.scale.z)}};
        const float w = weight * (mask.empty() ? 1.f : i < mask.size() ? mask[i] : 0.f);
        out[i] = addLocal(base[i],delta,std::isfinite(w) ? std::clamp(w,0.f,1.f) : 0.f);
    }
    return out;
}
std::vector<Matrix4> composePoseWorld(const Skeleton& skeleton, const LocalPose& pose) {
    if (pose.size() != skeleton.bones.size()) return {};
    std::vector<Matrix4> local; local.reserve(pose.size());
    for (const auto& value : pose) local.push_back(composeLocal(value.translation,value.rotation,value.scale));
    return composeWorldMatrices(skeleton,local);
}
void AnimationSampler::paletteFromLocal(const Skeleton& skeleton, const LocalPose& pose, std::vector<Matrix4>& out) const {
    out = composePoseWorld(skeleton,pose);
    for (std::size_t i=0; i<out.size(); ++i)
        out[i] = out[i] * (i < inverseBindMatrices_.size() ? inverseBindMatrices_[i] : Matrix4::identity());
}

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

void AnimationSampler::sampleLayered(const Skeleton& skeleton, const AnimationClip& base, float baseTime,
    const AnimationClip& layer, float layerTime, float weight, std::span<const float> boneMask,
    std::vector<Matrix4>& outMatrices) const {
    const auto basePose = sampleLocalTransforms(skeleton, base, baseTime);
    const auto layerPose = sampleLocalTransforms(skeleton, layer, layerTime, true);
    std::vector<Matrix4> local(skeleton.bones.size());
    for (std::size_t i = 0; i < local.size(); ++i) {
        const float mask = boneMask.empty() ? 1.0f : i < boneMask.size() ? boneMask[i] : 0.0f;
        const float alpha = std::isfinite(weight * mask) ? odai::math::saturate(weight * mask) : 0;
        const auto value = layer.additive ? addLocal(basePose[i], layerPose[i], alpha) :
            LocalTransform{lerpVector3(basePose[i].translation, layerPose[i].translation, alpha),
                odai::math::slerp(basePose[i].rotation, layerPose[i].rotation, alpha),
                lerpVector3(basePose[i].scale, layerPose[i].scale, alpha)};
        local[i] = composeLocal(value.translation, value.rotation, value.scale);
    }
    const auto world = composeWorldMatrices(skeleton, local);
    outMatrices.resize(world.size());
    for (std::size_t i = 0; i < world.size(); ++i)
        outMatrices[i] = world[i] * (i < inverseBindMatrices_.size() ? inverseBindMatrices_[i] : Matrix4::identity());
}

void AnimationSampler::sampleComposed(const Skeleton& skeleton, std::span<const WeightedAnimationPose> base,
    std::span<const AnimationPoseLayer> layers, std::vector<Matrix4>& outMatrices) const {
    std::vector<LocalTransform> pose;
    for (const auto& bone : skeleton.bones) pose.push_back({bone.localTranslation, bone.localRotation, bone.localScale});
    float accumulated = 0;
    for (const auto& sample : base) {
        if (!sample.clip || !std::isfinite(sample.weight) || sample.weight <= 0) continue;
        const auto sampled = sampleLocalTransforms(skeleton, *sample.clip, sample.time);
        const float alpha = sample.weight / (accumulated + sample.weight);
        accumulated += sample.weight;
        for (std::size_t i = 0; i < pose.size(); ++i) {
            pose[i] = {lerpVector3(pose[i].translation, sampled[i].translation, alpha),
                odai::math::slerp(pose[i].rotation, sampled[i].rotation, alpha),
                lerpVector3(pose[i].scale, sampled[i].scale, alpha)};
        }
    }
    for (const auto& layer : layers) {
        if (!layer.clip || (layer.additive && !layer.reference)) continue;
        const auto sampled = sampleLocalTransforms(skeleton, *layer.clip, layer.time);
        const auto reference = layer.additive ? sampleLocalTransforms(skeleton, *layer.reference, 0) : std::vector<LocalTransform>{};
        for (std::size_t i = 0; i < pose.size(); ++i) {
            const float weighted = layer.weight * (i < layer.mask.size() ? layer.mask[i] : 0);
            const float alpha = std::isfinite(weighted) ? std::clamp(weighted, 0.f, 1.f) : 0;
            if (layer.additive) {
                const auto& r = reference[i];
                const auto& target = sampled[i];
                const auto ratio = [](float a, float b) { return std::abs(b) > 1.e-6f ? a / b : 1.f; };
                const LocalTransform delta{target.translation - r.translation,
                    multiply({-r.rotation.x, -r.rotation.y, -r.rotation.z, r.rotation.w}, target.rotation),
                    {ratio(target.scale.x, r.scale.x), ratio(target.scale.y, r.scale.y), ratio(target.scale.z, r.scale.z)}};
                pose[i] = addLocal(pose[i], delta, alpha);
            } else pose[i] = {lerpVector3(pose[i].translation, sampled[i].translation, alpha),
                odai::math::slerp(pose[i].rotation, sampled[i].rotation, alpha),
                lerpVector3(pose[i].scale, sampled[i].scale, alpha)};
        }
    }
    std::vector<Matrix4> local;
    for (const auto& value : pose) local.push_back(composeLocal(value.translation, value.rotation, value.scale));
    const auto world = composeWorldMatrices(skeleton, local);
    outMatrices.resize(world.size());
    for (std::size_t i = 0; i < world.size(); ++i)
        outMatrices[i] = world[i] * (i < inverseBindMatrices_.size() ? inverseBindMatrices_[i] : Matrix4::identity());
}

}  // namespace odai::anim
