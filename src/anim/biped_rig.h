#pragma once

#include "anim/animation_clip.h"
#include "anim/skeleton.h"

// A hand-authored procedural biped rig -- 21 bones, bind pose standing at
// attention with the pelvis (the skeleton root) 0.95m above the ground at
// local origin, feet reaching back down to roughly y=0. No traditional DCC
// asset pipeline exists in this engine (see docs/ROADMAP.md's NIF-import
// notes), so this rig and its clips are authored directly as flat
// anim::Skeleton/AnimationClip data rather than imported. Built for the
// Dragon Age: Origins touchstone's small-party (4-8 character) real-time
// combat, not a mass-battle crowd system (see docs/ROADMAP.md's explicit
// out-of-scope note on RTS-scale rendering).
//
// Bone naming follows a joint-chain convention: each non-leaf bone's local
// translation is the offset from its parent joint to *this* joint, so a
// mesh generator can draw one cylinder per parent bone spanning from the
// parent's world position to each child's world position (see
// tests/animation_tests.cc's makeTwoBoneChain for the same convention at
// toy scale, and procgen/humanoid_generator.h for the real mesh built on
// this rig).
namespace odai::anim {

inline constexpr const char* kBipedBonePelvis = "pelvis";
inline constexpr const char* kBipedBoneSpine = "spine";
inline constexpr const char* kBipedBoneChest = "chest";
inline constexpr const char* kBipedBoneNeck = "neck";
inline constexpr const char* kBipedBoneHead = "head";
inline constexpr const char* kBipedBoneShoulderL = "shoulder_l";
inline constexpr const char* kBipedBoneElbowL = "elbow_l";
inline constexpr const char* kBipedBoneWristL = "wrist_l";
inline constexpr const char* kBipedBoneHandL = "hand_l";
inline constexpr const char* kBipedBoneShoulderR = "shoulder_r";
inline constexpr const char* kBipedBoneElbowR = "elbow_r";
inline constexpr const char* kBipedBoneWristR = "wrist_r";
inline constexpr const char* kBipedBoneHandR = "hand_r";
inline constexpr const char* kBipedBoneHipL = "hip_l";
inline constexpr const char* kBipedBoneKneeL = "knee_l";
inline constexpr const char* kBipedBoneAnkleL = "ankle_l";
inline constexpr const char* kBipedBoneFootL = "foot_l";
inline constexpr const char* kBipedBoneHipR = "hip_r";
inline constexpr const char* kBipedBoneKneeR = "knee_r";
inline constexpr const char* kBipedBoneAnkleR = "ankle_r";
inline constexpr const char* kBipedBoneFootR = "foot_r";

[[nodiscard]] Skeleton makeBipedSkeleton();

// A gentle idle breathing/weight-shift loop. Only animates the pelvis/chest
// translation slightly -- every other bone stays at bind pose.
[[nodiscard]] AnimationClip makeBipedIdleClip(const Skeleton& skeleton);

// A looped marching walk cycle: opposing hip swings, knee bend on the
// lifting leg, opposing arm counter-swing.
[[nodiscard]] AnimationClip makeBipedWalkClip(const Skeleton& skeleton);

// A one-shot (non-looping) overhead sword-chop attack driven from the right
// shoulder/elbow: windup, then swing, then return to bind pose.
[[nodiscard]] AnimationClip makeBipedAttackClip(const Skeleton& skeleton);

}  // namespace odai::anim
