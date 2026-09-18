#pragma once
#include "anim/skeleton.h"
#include "anim/animation_clip.h"
#include <map>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace odai::anim {
struct HumanoidLimbChain {
    int upper = -1, lower = -1, end = -1;
};
// Canonical names follow the Skyrim/XPMSSE humanoid contract. Extra authored
// bones remain in the rig; actor proportions and local bind transforms survive.
struct HumanoidRigMapping {
    static constexpr unsigned version = 1;
    Skeleton skeleton;
    std::vector<int> sourceToCanonical;
    std::map<std::string, int> roles;
    std::map<std::string, int> sockets;
    std::vector<float> firstPersonMask;
    std::map<std::string, HumanoidLimbChain> limbs;
    std::string fingerprint;
    // Each bucket contains independent bones, for compute hierarchy evaluation.
    std::vector<std::vector<std::uint32_t>> depthLevels;
};
enum class TranslationScalePolicy { Preserve, LimbLength };
struct ClipRigBoneBinding {
    int source = -1, target = -1;
    Bone sourceRest, targetRest;
    float translationScale = 1;
};
// Construct once and share as const with clip resources. The binding owns its
// rest transforms, so changing outfits cannot invalidate it.
struct ClipRigBinding {
    std::size_t sourceBoneCount = 0, targetBoneCount = 0;
    std::string sourceFingerprint, targetFingerprint;
    std::vector<ClipRigBoneBinding> bones;
    bool direct = false;
};
bool bindHumanoidClipRig(const Skeleton& source, const Skeleton& target,
    TranslationScalePolicy policy, ClipRigBinding& out, std::string& error);
bool retargetHumanoidClip(const AnimationClip& source, const ClipRigBinding& binding,
    AnimationClip& out, std::string& error);
bool canonicalizeHumanoidRig(const Skeleton& source, HumanoidRigMapping& out, std::string& error);
bool remapHumanoidClips(std::span<AnimationClip> clips, const HumanoidRigMapping& mapping, std::string& error);
} // namespace odai::anim
