#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include "anim/pose_graph.h"

namespace odai::anim {

enum class AnimationExecutionMode : std::uint8_t { Native, Havok };
using AnimationContextValue = std::variant<bool, double, std::string>;
struct AnimationSelectorContext {
    std::map<std::string, AnimationContextValue> values;
    std::set<std::string> tags;
};
struct NativeAnimationCondition {
    std::string field;
    AnimationContextValue value;
    enum class Op { Equal, Minimum, Maximum, Tag } op = Op::Equal;
};
struct NativeAnimationVariant {
    std::string clip;
    double weight = 1.0;
    bool loop = true;
};
struct NativeBlendSample {
    std::string clip;
    float x = 0, z = 0; // actor-local Bethesda units/second
};
struct NativeAnimationLayer {
    std::string id, clip, referenceClip;
    int order = 0;
    float weight = 1;
    bool additive = false;
    std::vector<std::string> bones;
};
struct NativeMotionWarpWindow {
    std::string target;
    float start = 0, end = 0;
    float maxTranslation = 32; // Bethesda units over the authored window
    float maxYawRadians = .5f;
};
struct NativeAnimationRule {
    std::shared_ptr<const PoseGraphProgram> graph;
    std::string id, state, provider;
    int priority = 0;
    int layer = 0;
    float blendSeconds = 0.16f;
    bool animationDriven = false;
    bool holdUntilLanding = false;
    std::vector<NativeAnimationCondition> conditions;
    std::vector<NativeAnimationVariant> variants;
    std::vector<NativeBlendSample> blendSamples;
    std::vector<NativeAnimationLayer> layers;
    std::vector<NativeMotionWarpWindow> motionWarps;
};
struct NativeAnimationProgram {
    std::vector<NativeAnimationRule> rules;
    [[nodiscard]] const NativeAnimationRule* select(std::string_view state,
        const AnimationSelectorContext& context) const;
};

// Atomic, bounded compilation: failure leaves out unchanged. No script evaluation.
bool compileNativeAnimationPack(std::string_view json, std::string_view provider,
    int layer, NativeAnimationProgram& out, std::string& error);
std::size_t chooseNativeVariant(const NativeAnimationRule& rule, std::uint64_t& randomState);

} // namespace odai::anim
