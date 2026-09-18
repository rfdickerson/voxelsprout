#pragma once
#include "anim/animation_sampler.h"
#include "anim/humanoid_rig.h"
#include "anim/pose_modifiers.h"
#include <map>
#include <memory>
#include <string_view>
#include <variant>

namespace odai::anim {
using PoseParameter = std::variant<bool, double, std::string>;
struct PoseGraphSample {
  std::string node;
  float x = 0, y = 0;
};
struct PoseGraphTransition {
  std::string from, to, parameter;
  PoseParameter equals;
  float duration = .16f;
};
struct PoseGraphNode {
  enum class Kind {
    Clip,
    Blend,
    Blend1D,
    Blend2D,
    Layer,
    Cache,
    StateMachine,
    Modifier
  } kind = Kind::Clip;
  std::string id, clip, reference, parameter, parameterY, initial;
  std::vector<std::string> inputs, bones;
  std::vector<PoseGraphSample> samples;
  std::vector<PoseGraphTransition> transitions;
  float weight = 1, speed = 1;
  bool additive = false;
  PoseModifier modifier;
  std::string role, targetX, targetY, targetZ;
};
struct PoseGraphProgram {
  std::string root;
  std::map<std::string, PoseParameter> parameters;
  std::map<std::string, PoseGraphNode> nodes;
};
bool compilePoseGraph(std::string_view json, PoseGraphProgram &out,
                      std::string &error);
struct PoseGraphInstruction {
  std::shared_ptr<const PosePostProcess> postProcess;
  PoseGraphNode::Kind kind = PoseGraphNode::Kind::Clip;
  std::vector<std::uint32_t> inputs;
  std::vector<float> weights, mask;
  std::string clip, reference;
  float time = 0, previousTime = 0, weight = 1;
  bool additive = false;
};
// Backend-neutral evaluation program for one fixed tick. Contains no pointers
// into mutable instances and can be retained by a render frame.
struct PoseEvaluationPacket {
  std::vector<PoseGraphInstruction> instructions;
  std::uint32_t output = 0;
  std::string eventClip;
  float eventBefore = 0, eventAfter = 0;
  bool discontinuity = false;
  float transitionDuration = -1;
};
class PoseGraphInstance {
public:
  bool advance(const PoseGraphProgram &graph, const Skeleton &skeleton,
               const HumanoidRigMapping *rig,
               const std::map<std::string, PoseParameter> &parameters,
               std::span<const AnimationClip> clips, float delta,
               PoseEvaluationPacket &out, std::string &error);
  std::string save() const;
  bool restore(std::string_view saved, std::string &error);
  void reset();

private:
  struct Clock {
    double time = 0;
    std::string state, previous;
    float elapsed = 0, duration = 0;
  };
  std::map<std::string, Clock> clocks_;
  std::string lastEventClip_;
};
bool evaluatePosePacket(const PoseEvaluationPacket &packet,
                        const Skeleton &skeleton,
                        std::span<const AnimationClip> clips, LocalPose &out,
                        std::string &error);
// Marker pairs interpolate the follower's authored phase; absent/incompatible
// markers use normalized time. Neither path emits gameplay events.
float synchronizedClipTime(const AnimationClip &leader, float time,
                           const AnimationClip &follower);
std::vector<float> blendSpaceWeights(std::span<const PoseGraphSample> samples,
                                     float x, float y, bool twoDimensional);
} // namespace odai::anim
