#pragma once

#include "anim/animation_clip.h"
#include "anim/skeleton.h"
#include "math/math.h"

#include <vector>

// Evaluates an AnimationClip against a Skeleton to produce final GPU-ready
// skinning matrices. Pure CPU math, no Vulkan — the renderer only ever
// consumes the resulting flat odai::math::Matrix4 array.
namespace odai::anim {

class AnimationSampler {
public:
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

private:
    std::vector<odai::math::Matrix4> inverseBindMatrices_;
};

}  // namespace odai::anim
