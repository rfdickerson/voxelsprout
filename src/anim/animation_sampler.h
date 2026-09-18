#pragma once

#include "anim/animation_clip.h"
#include "anim/skeleton.h"
#include "math/math.h"

#include <vector>
#include <span>

// Evaluates an AnimationClip against a Skeleton to produce final GPU-ready
// skinning matrices. Pure CPU math, no Vulkan — the renderer only ever
// consumes the resulting flat odai::math::Matrix4 array.
namespace odai::anim {
struct LocalPoseTransform {
    odai::math::Vector3 translation{};
    odai::math::Quaternion rotation{};
    odai::math::Vector3 scale{1, 1, 1};
};
using LocalPose = std::vector<LocalPoseTransform>;
LocalPose sampleLocalPose(const Skeleton& skeleton, const AnimationClip& clip, float timeSeconds);
LocalPose blendLocalPoses(const LocalPose& from, const LocalPose& to, float weight,
    std::span<const float> mask = {});
LocalPose addLocalPoses(const LocalPose& base, const LocalPose& target,
    const LocalPose& reference, float weight, std::span<const float> mask = {});
std::vector<odai::math::Matrix4> composePoseWorld(const Skeleton& skeleton, const LocalPose& pose);

struct WeightedAnimationPose {
    const AnimationClip* clip = nullptr;
    float time = 0, weight = 0;
};
struct AnimationPoseLayer {
    const AnimationClip* clip = nullptr;
    const AnimationClip* reference = nullptr; // reference sampled at t=0
    float time = 0, weight = 1;
    bool additive = false;
    std::vector<float> mask;
};

class AnimationSampler {
public:
    void paletteFromLocal(const Skeleton& skeleton, const LocalPose& pose,
        std::vector<odai::math::Matrix4>& out) const;
    // Precomputes bind-pose world transforms and their inverses. Call again
    // whenever the bound skeleton changes; sample() may be called with any
    // clip authored against that same skeleton without rebinding.
    void bindSkeleton(const Skeleton& skeleton);

    // Binds with inverse bind matrices the caller already holds, instead of
    // deriving them from the skeleton's own bind pose.
    //
    // The two are not interchangeable wherever an importer records them
    // explicitly. A Fallout skinned NIF stores its vertices in "skin space" and
    // NiSkinData carries the only record of how that space relates to each bone
    // (see FalloutCharacter::inverseBindMatrices); deriving them from the
    // skeleton instead assumes skin space is the skeleton root's space, which is
    // close enough to look nearly right and be consistently wrong. Sized
    // shorter than the skeleton, the missing tail falls back to identity, same
    // as sample() already does.
    void bindSkeleton(const Skeleton& skeleton, std::vector<odai::math::Matrix4> inverseBindMatrices);

    // Evaluates clip at timeSeconds (looped or clamped per clip.loop) and
    // writes skeleton.bones.size() skinning matrices into outMatrices, one
    // per bone in Skeleton::bones order: worldBoneTransform * inverseBindPose.
    // skeleton must be the same one passed to the most recent bindSkeleton().
    void sample(const Skeleton& skeleton, const AnimationClip& clip, float timeSeconds,
                std::vector<odai::math::Matrix4>& outMatrices) const;

    // Samples both clips into local translation/rotation/scale, blends those
    // channels, and only then composes hierarchy/skin matrices. Matrix-wise
    // interpolation shears rotating limbs and is not a valid transition pose.
    void sampleBlended(const Skeleton& skeleton,
                       const AnimationClip& fromClip, float fromTimeSeconds,
                       const AnimationClip& toClip, float toTimeSeconds,
                       float alpha,
                       std::vector<odai::math::Matrix4>& outMatrices) const;

    // Layer in local TRS space. Additive layers use identity defaults for
    // unkeyed channels; absolute layers blend with the base. Empty mask = all bones.
    void sampleLayered(const Skeleton& skeleton, const AnimationClip& base, float baseTime,
        const AnimationClip& layer, float layerTime, float weight, std::span<const float> boneMask,
        std::vector<odai::math::Matrix4>& outMatrices) const;

    void sampleComposed(const Skeleton& skeleton, std::span<const WeightedAnimationPose> base,
        std::span<const AnimationPoseLayer> layers, std::vector<odai::math::Matrix4>& outMatrices) const;

private:
    std::vector<odai::math::Matrix4> inverseBindMatrices_;
};

}  // namespace odai::anim
