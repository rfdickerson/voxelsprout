#pragma once
#include "anim/animation_sampler.h"
#include "anim/humanoid_rig.h"

namespace odai::anim {
struct LimbTarget {
  odai::math::Vector3 position{}, pole{0, 0, 1}, normal{0, 1, 0};
  float weight = 1;
  bool alignNormal = false;
};
struct PoseModifier {
  enum class Kind { Translate, LimbIk, Aim } kind = Kind::Translate;
  HumanoidLimbChain chain;
  LimbTarget target;
  odai::math::Vector3 forward{0, 0, 1};
  float maxRadians = 0;
};
struct PosePostProcess {
  LocalPose offsets, velocities;
  float offsetWeight = 0, velocityWeight = 0;
  std::vector<PoseModifier> modifiers;
};
bool applyPosePostProcess(const Skeleton &skeleton, const PosePostProcess &post,
                          LocalPose &pose);
// Targets are in skeleton model space. Solves rotations, preserving authored
// translations/segment lengths and every unmodified local transform.
bool solveTwoBoneIk(const Skeleton &skeleton, HumanoidLimbChain chain,
                    const LimbTarget &target, LocalPose &pose);
bool aimBone(const Skeleton &skeleton, int bone, odai::math::Vector3 target,
             odai::math::Vector3 forward, float maxRadians, float weight,
             LocalPose &pose);
class PoseInertializer {
public:
  LocalPose evaluate(const LocalPose &target, bool interrupted, float duration,
                     float delta, const LocalPose *targetPrevious = nullptr);
  void reset();
  std::string save() const;
  bool restore(std::string_view saved, std::string &error);
  PosePostProcess evaluationParameters() const;

private:
  LocalPose previous_, older_, offsets_, velocities_;
  float elapsed_ = 0, duration_ = 0, previousDelta_ = 0;
  float offsetWeight_ = 0, velocityWeight_ = 0;
};
} // namespace odai::anim
