#pragma once

#include "math/math.h"

#include <string>
#include <vector>

// Keyframe data for one AnimationClip (see animation_sampler.h for how a clip
// gets evaluated against a Skeleton). Pure flat data, no Vulkan.
namespace odai::anim {

struct Vector3Key {
    float time = 0.0f;
    odai::math::Vector3 value{};
};

struct QuaternionKey {
    float time = 0.0f;
    odai::math::Quaternion value{};
};

// One bone's animated channels. Any of the three key arrays may be empty, in
// which case that channel holds the bind-pose value from Skeleton::bones for
// the whole clip (e.g. a clip that only animates rotation leaves translation/
// scale at bind pose).
struct BoneTrack {
    int boneIndex = -1;
    std::vector<Vector3Key> translationKeys;
    std::vector<QuaternionKey> rotationKeys;
    std::vector<Vector3Key> scaleKeys;
};

struct AnimationAnnotation {
    float time = 0.0f;
    std::string name;
};

struct AnimationClip {
    std::string name;
    float duration = 0.0f;
    bool loop = true;
    std::vector<BoneTrack> tracks;
    // Authored timeline markers (footsteps, sounds, effects). BehaviorGraph
    // consumes crossings at the fixed simulation tick, never render time.
    std::vector<AnimationAnnotation> annotations;
};

}  // namespace odai::anim
