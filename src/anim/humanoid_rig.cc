#include "anim/humanoid_rig.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <bit>
#include <sstream>
#include <iomanip>

namespace odai::anim {
namespace {
struct Role { const char* role; const char* name; const char* ancestor; };
constexpr Role roles[] = {
    {"root", "NPC Root [Root]", ""},
    {"pelvis", "NPC Pelvis [Pelv]", "root"},
    {"spine", "NPC Spine [Spn0]", "root"},
    {"head", "NPC Head [Head]", "spine"},
    {"left_upper_arm", "NPC L UpperArm [LUar]", "spine"},
    {"left_forearm", "NPC L Forearm [LLar]", "left_upper_arm"},
    {"left_hand", "NPC L Hand [LHnd]", "left_forearm"},
    {"right_upper_arm", "NPC R UpperArm [RUar]", "spine"},
    {"right_forearm", "NPC R Forearm [RLar]", "right_upper_arm"},
    {"right_hand", "NPC R Hand [RHnd]", "right_forearm"},
    {"left_thigh", "NPC L Thigh [LThg]", "pelvis"},
    {"left_calf", "NPC L Calf [LClf]", "left_thigh"},
    {"left_foot", "NPC L Foot [Lft ]", "left_calf"},
    {"right_thigh", "NPC R Thigh [RThg]", "pelvis"},
    {"right_calf", "NPC R Calf [RClf]", "right_thigh"},
    {"right_foot", "NPC R Foot [Rft ]", "right_calf"},
};
bool finite(const Bone& bone) {
    for (float value : {bone.localTranslation.x, bone.localTranslation.y, bone.localTranslation.z,
        bone.localRotation.x, bone.localRotation.y, bone.localRotation.z, bone.localRotation.w,
        bone.localScale.x, bone.localScale.y, bone.localScale.z}) if (!std::isfinite(value)) return false;
    const auto& q = bone.localRotation;
    if (q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w < 1.e-8f) return false;
    return bone.localScale.x != 0 && bone.localScale.y != 0 && bone.localScale.z != 0;
}
odai::math::Quaternion product(const odai::math::Quaternion& a, const odai::math::Quaternion& b) {
    return odai::math::normalize({a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
        a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x, a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
        a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z});
}
odai::math::Quaternion conjugate(odai::math::Quaternion q) {
    q = odai::math::normalize(q);
    return {-q.x, -q.y, -q.z, q.w};
}
std::string fingerprint(const Skeleton& skeleton) {
    std::uint64_t hash = 14695981039346656037ull;
    const auto byte = [&](std::uint8_t b) { hash = (hash ^ b) * 1099511628211ull; };
    const auto word = [&](std::uint32_t w) { for (int i=0; i<4; ++i) byte(static_cast<std::uint8_t>(w >> (i*8))); };
    word(static_cast<std::uint32_t>(skeleton.bones.size()));
    for (const auto& bone : skeleton.bones) {
        word(static_cast<std::uint32_t>(bone.name.size()));
        for (unsigned char c : bone.name) byte(c);
        word(static_cast<std::uint32_t>(bone.parentIndex));
        for (float v : {bone.localTranslation.x, bone.localTranslation.y, bone.localTranslation.z,
            bone.localRotation.x, bone.localRotation.y, bone.localRotation.z, bone.localRotation.w,
            bone.localScale.x, bone.localScale.y, bone.localScale.z}) word(std::bit_cast<std::uint32_t>(v == 0 ? 0.f : v));
    }
    std::ostringstream text; text << std::hex << std::setfill('0') << std::setw(16) << hash;
    return text.str();
}
}
bool canonicalizeHumanoidRig(const Skeleton& source, HumanoidRigMapping& out, std::string& error) {
    error.clear();
    if (source.bones.empty() || source.bones.size() > std::numeric_limits<std::uint16_t>::max()) {
        error = "humanoid rig has invalid bone count"; return false;
    }
    std::map<std::string, int> named;
    for (std::size_t i = 0; i < source.bones.size(); ++i) {
        const auto& bone = source.bones[i];
        if (bone.name.empty() || !named.emplace(bone.name, static_cast<int>(i)).second ||
            bone.parentIndex < -1 || bone.parentIndex >= static_cast<int>(i) || !finite(bone)) {
            error = "invalid humanoid bone: " + bone.name; return false;
        }
    }
    std::map<std::string, int> sourceRoles;
    for (const auto& role : roles) {
        const auto found = named.find(role.name);
        if (found == named.end()) { error = "required humanoid bone missing: " + std::string(role.name); return false; }
        sourceRoles[role.role] = found->second;
        if (*role.ancestor) {
            const int ancestor = sourceRoles.at(role.ancestor);
            int parent = source.bones[found->second].parentIndex;
            while (parent >= 0 && parent != ancestor) parent = source.bones[parent].parentIndex;
            if (parent != ancestor) { error = "incompatible humanoid ancestry: " + std::string(role.name); return false; }
        }
    }
    HumanoidRigMapping result;
    result.sourceToCanonical.assign(source.bones.size(), -1);
    // A stable topological order, independent of the source's sibling order.
    // Never reparent a bone: doing so without resampling every track changes motion.
    std::set<std::pair<std::string, int>> ready;
    std::vector<std::vector<int>> children(source.bones.size());
    for (std::size_t i = 0; i < source.bones.size(); ++i) {
        const auto& bone = source.bones[i];
        if (bone.parentIndex < 0) ready.emplace(bone.name, static_cast<int>(i));
        else children[bone.parentIndex].push_back(static_cast<int>(i));
    }
    while (!ready.empty()) {
        const int old = ready.begin()->second;
        ready.erase(ready.begin());
        auto bone = source.bones[old];
        if (bone.parentIndex >= 0) bone.parentIndex = result.sourceToCanonical[bone.parentIndex];
        result.sourceToCanonical[old] = static_cast<int>(result.skeleton.bones.size());
        result.skeleton.bones.push_back(std::move(bone));
        for (int child : children[old]) ready.emplace(source.bones[child].name, child);
    }
    for (const auto& [role, bone] : sourceRoles) result.roles[role] = result.sourceToCanonical[bone];
    for (const auto& [role, name] : std::initializer_list<std::pair<const char*, const char*>>{
        {"left_toe", "NPC L Toe0 [LToe]"}, {"right_toe", "NPC R Toe0 [RToe]"},
        {"left_arm_twist", "NPC L UpperarmTwist1 [LUt1]"}, {"right_arm_twist", "NPC R UpperarmTwist1 [RUt1]"},
        {"left_forearm_twist", "NPC L ForearmTwist1 [LLt1]"}, {"right_forearm_twist", "NPC R ForearmTwist1 [RLt1]"}})
        if (const auto found = named.find(name); found != named.end()) result.roles[role] = result.sourceToCanonical[found->second];
    for (const auto* side : {"left", "right"}) {
        const std::string prefix = std::string(side) + "_";
        result.limbs[prefix + "arm"] = {result.roles.at(prefix + "upper_arm"), result.roles.at(prefix + "forearm"), result.roles.at(prefix + "hand")};
        result.limbs[prefix + "leg"] = {result.roles.at(prefix + "thigh"), result.roles.at(prefix + "calf"), result.roles.at(prefix + "foot")};
    }
    result.fingerprint = fingerprint(result.skeleton);
    std::vector<std::size_t> depths(result.skeleton.bones.size());
    for (std::size_t i = 0; i < result.skeleton.bones.size(); ++i) {
        const int parent = result.skeleton.bones[i].parentIndex;
        const auto depth = parent < 0 ? 0 : depths[parent] + 1;
        depths[i] = depth;
        if (result.depthLevels.size() <= depth) result.depthLevels.resize(depth + 1);
        result.depthLevels[depth].push_back(static_cast<std::uint32_t>(i));
    }
    result.sockets = {{"left_hand", result.roles.at("left_hand")}, {"right_hand", result.roles.at("right_hand")}};
    for (const auto* name : {"WeaponSword", "WeaponDagger", "WeaponAxe", "WeaponMace", "WeaponBack", "WeaponBow", "SHIELD", "QUIVER"})
        if (const auto found = named.find(name); found != named.end()) result.sockets[name] = result.sourceToCanonical[found->second];
    result.firstPersonMask.resize(source.bones.size());
    for (std::size_t i = 0; i < result.skeleton.bones.size(); ++i) {
        int ancestor = static_cast<int>(i);
        while (ancestor >= 0) {
            if (ancestor == result.roles.at("left_upper_arm") || ancestor == result.roles.at("right_upper_arm")) {
                result.firstPersonMask[i] = 1; break;
            }
            ancestor = result.skeleton.bones[ancestor].parentIndex;
        }
    }
    out = std::move(result);
    return true;
}

bool remapHumanoidClips(std::span<AnimationClip> clips, const HumanoidRigMapping& mapping, std::string& error) {
    error.clear();
    for (const auto& clip : clips)
        if (clip.extractedMotionBone < -1 || (clip.extractedMotionBone >= 0 &&
                static_cast<std::size_t>(clip.extractedMotionBone) >= mapping.sourceToCanonical.size())) {
            error = "root motion bone is outside source humanoid rig"; return false;
        }
    for (const auto& clip : clips) for (const auto& track : clip.tracks)
        if (track.boneIndex < 0 || static_cast<std::size_t>(track.boneIndex) >= mapping.sourceToCanonical.size()) {
            error = "animation track is outside source humanoid rig"; return false;
        }
    for (auto& clip : clips) if (clip.extractedMotionBone >= 0)
        clip.extractedMotionBone = mapping.sourceToCanonical[clip.extractedMotionBone];
    for (auto& clip : clips) for (auto& track : clip.tracks)
        track.boneIndex = mapping.sourceToCanonical[track.boneIndex];
    return true;
}

bool bindHumanoidClipRig(const Skeleton& source, const Skeleton& target,
    TranslationScalePolicy policy, ClipRigBinding& out, std::string& error) {
    HumanoidRigMapping from, to;
    if (!canonicalizeHumanoidRig(source, from, error) || !canonicalizeHumanoidRig(target, to, error)) return false;
    ClipRigBinding result;
    result.sourceBoneCount = source.bones.size();
    result.targetBoneCount = target.bones.size();
    result.sourceFingerprint = fingerprint(source);
    result.targetFingerprint = fingerprint(target);
    result.direct = result.sourceFingerprint == result.targetFingerprint;
    for (std::size_t i = 0; i < source.bones.size(); ++i) {
        const auto& rest = source.bones[i];
        const int targetIndex = target.findBone(rest.name);
        if (targetIndex < 0) continue;
        const auto& targetRest = target.bones[targetIndex];
        // Local rest correction is valid only when both local frames have the
        // same parent. Never silently treat a hierarchy change as retargeting.
        const auto parentName = [](const Skeleton& rig, const Bone& bone) -> std::string {
            return bone.parentIndex < 0 ? std::string{} : rig.bones[bone.parentIndex].name;
        };
        if (parentName(source, rest) != parentName(target, targetRest)) {
            error = "retarget hierarchy mismatch: " + rest.name; return false;
        }
        float scale = 1;
        if (policy == TranslationScalePolicy::LimbLength) {
            const float length = odai::math::length(rest.localTranslation);
            if (length > 1.e-6f) scale = odai::math::length(targetRest.localTranslation) / length;
        }
        if (!std::isfinite(scale) || scale <= 0) { error = "invalid retarget scale: " + rest.name; return false; }
        result.bones.push_back({static_cast<int>(i), targetIndex, rest, targetRest, scale});
    }
    out = std::move(result); error.clear(); return true;
}

bool retargetHumanoidClip(const AnimationClip& source, const ClipRigBinding& binding,
    AnimationClip& out, std::string& error) {
    AnimationClip result = source;
    result.tracks.clear(); result.extractedMotionBone = -1;
    if (!std::isfinite(source.duration) || source.duration < 0 || source.extractedMotionBone < -1 ||
        (source.extractedMotionBone >= 0 && static_cast<std::size_t>(source.extractedMotionBone) >= binding.sourceBoneCount)) {
        error = "invalid retarget clip duration or root"; return false;
    }
    for (const auto& bone : binding.bones) {
        if (bone.source < 0 || static_cast<std::size_t>(bone.source) >= binding.sourceBoneCount ||
            bone.target < 0 || static_cast<std::size_t>(bone.target) >= binding.targetBoneCount ||
            !finite(bone.sourceRest) || !finite(bone.targetRest) || !std::isfinite(bone.translationScale) || bone.translationScale <= 0) {
            error = "invalid clip rig binding"; return false;
        }
        if (bone.source == source.extractedMotionBone) result.extractedMotionBone = bone.target;
    }
    std::set<int> seen;
    for (const auto& track : source.tracks) {
        if (track.boneIndex < 0 || static_cast<std::size_t>(track.boneIndex) >= binding.sourceBoneCount || !seen.insert(track.boneIndex).second) {
            error = "invalid or duplicate retarget track"; return false;
        }
        const auto validKeys = [&](const auto& keys) {
            float previous = -1;
            for (const auto& key : keys) {
                if (!std::isfinite(key.time) || key.time < previous || key.time < 0 || key.time > source.duration ||
                    !std::isfinite(key.value.x) || !std::isfinite(key.value.y) || !std::isfinite(key.value.z)) return false;
                if constexpr (requires { key.value.w; })
                    if (!std::isfinite(key.value.w) || odai::math::length(key.value) < 1.e-6f) return false;
                previous = key.time;
            }
            return true;
        };
        if (!validKeys(track.translationKeys) || !validKeys(track.rotationKeys) || !validKeys(track.scaleKeys)) {
            error = "invalid retarget keyframe"; return false;
        }
        const auto found = std::find_if(binding.bones.begin(), binding.bones.end(),
            [&](const auto& bone) { return bone.source == track.boneIndex; });
        if (found == binding.bones.end()) continue;
        auto remapped = track; remapped.boneIndex = found->target;
        if (!binding.direct) {
            const auto correction = product(found->targetRest.localRotation, conjugate(found->sourceRest.localRotation));
            for (auto& key : remapped.rotationKeys)
                key.value = source.additive ? product(product(correction, key.value), conjugate(correction)) : product(correction, key.value);
            for (auto& key : remapped.translationKeys) {
                const auto delta = source.additive ? key.value : key.value - found->sourceRest.localTranslation;
                const auto rotated = odai::math::transformPoint(odai::math::toMatrix(correction), delta * found->translationScale);
                key.value = source.additive ? rotated : found->targetRest.localTranslation + rotated;
            }
            if (!source.additive) for (auto& key : remapped.scaleKeys) {
                key.value = {key.value.x * found->targetRest.localScale.x / found->sourceRest.localScale.x,
                    key.value.y * found->targetRest.localScale.y / found->sourceRest.localScale.y,
                    key.value.z * found->targetRest.localScale.z / found->sourceRest.localScale.z};
            }
        }
        if (track.boneIndex == source.extractedMotionBone) result.extractedMotionBone = found->target;
        result.tracks.push_back(std::move(remapped));
    }
    if (source.extractedMotionBone >= 0 && result.extractedMotionBone < 0) {
        error = "unmapped root motion bone"; return false;
    }
    out = std::move(result); error.clear(); return true;
}
} // namespace odai::anim
