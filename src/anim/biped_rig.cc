#include "anim/biped_rig.h"

#include "math/math.h"

#include <cmath>
#include <initializer_list>
#include <utility>

namespace odai::anim {

namespace {

using odai::math::Quaternion;
using odai::math::Vector3;

Quaternion quatAroundX(float radiansValue) {
    return Quaternion{std::sin(radiansValue * 0.5f), 0.0f, 0.0f, std::cos(radiansValue * 0.5f)};
}

int addBone(Skeleton& skeleton, const char* name, int parentIndex, const Vector3& localTranslation) {
    Bone bone;
    bone.name = name;
    bone.parentIndex = parentIndex;
    bone.localTranslation = localTranslation;
    skeleton.bones.push_back(bone);
    return static_cast<int>(skeleton.bones.size()) - 1;
}

// Appends a rotation-only BoneTrack (translation/scale stay at bind pose via
// AnimationSampler::sample's per-channel fallback) with `keys` as {time,
// degrees} pairs rotated around the local X axis (this rig's hinge axis for
// every shoulder/elbow/hip/knee joint).
void addRotationXTrack(
    AnimationClip& clip, const Skeleton& skeleton, const char* boneName,
    std::initializer_list<std::pair<float, float>> keysDegrees
) {
    const int boneIndex = skeleton.findBone(boneName);
    if (boneIndex < 0) {
        return;
    }
    BoneTrack track;
    track.boneIndex = boneIndex;
    track.rotationKeys.reserve(keysDegrees.size());
    for (const auto& [time, degrees] : keysDegrees) {
        track.rotationKeys.push_back(QuaternionKey{time, quatAroundX(odai::math::radians(degrees))});
    }
    clip.tracks.push_back(std::move(track));
}

}  // namespace

Skeleton makeBipedSkeleton() {
    Skeleton skeleton;

    const int pelvis = addBone(skeleton, kBipedBonePelvis, -1, Vector3{0.0f, 0.95f, 0.0f});
    const int spine = addBone(skeleton, kBipedBoneSpine, pelvis, Vector3{0.0f, 0.22f, 0.0f});
    const int chest = addBone(skeleton, kBipedBoneChest, spine, Vector3{0.0f, 0.22f, 0.0f});
    const int neck = addBone(skeleton, kBipedBoneNeck, chest, Vector3{0.0f, 0.14f, 0.0f});
    addBone(skeleton, kBipedBoneHead, neck, Vector3{0.0f, 0.12f, 0.0f});

    const int shoulderL = addBone(skeleton, kBipedBoneShoulderL, chest, Vector3{-0.19f, 0.08f, 0.0f});
    const int elbowL = addBone(skeleton, kBipedBoneElbowL, shoulderL, Vector3{-0.27f, 0.0f, 0.0f});
    const int wristL = addBone(skeleton, kBipedBoneWristL, elbowL, Vector3{-0.25f, 0.0f, 0.0f});
    addBone(skeleton, kBipedBoneHandL, wristL, Vector3{-0.10f, 0.0f, 0.0f});

    const int shoulderR = addBone(skeleton, kBipedBoneShoulderR, chest, Vector3{0.19f, 0.08f, 0.0f});
    const int elbowR = addBone(skeleton, kBipedBoneElbowR, shoulderR, Vector3{0.27f, 0.0f, 0.0f});
    const int wristR = addBone(skeleton, kBipedBoneWristR, elbowR, Vector3{0.25f, 0.0f, 0.0f});
    addBone(skeleton, kBipedBoneHandR, wristR, Vector3{0.10f, 0.0f, 0.0f});

    const int hipL = addBone(skeleton, kBipedBoneHipL, pelvis, Vector3{-0.10f, -0.02f, 0.0f});
    const int kneeL = addBone(skeleton, kBipedBoneKneeL, hipL, Vector3{0.0f, -0.45f, 0.0f});
    const int ankleL = addBone(skeleton, kBipedBoneAnkleL, kneeL, Vector3{0.0f, -0.42f, 0.0f});
    addBone(skeleton, kBipedBoneFootL, ankleL, Vector3{0.0f, -0.05f, 0.12f});

    const int hipR = addBone(skeleton, kBipedBoneHipR, pelvis, Vector3{0.10f, -0.02f, 0.0f});
    const int kneeR = addBone(skeleton, kBipedBoneKneeR, hipR, Vector3{0.0f, -0.45f, 0.0f});
    const int ankleR = addBone(skeleton, kBipedBoneAnkleR, kneeR, Vector3{0.0f, -0.42f, 0.0f});
    addBone(skeleton, kBipedBoneFootR, ankleR, Vector3{0.0f, -0.05f, 0.12f});

    return skeleton;
}

AnimationClip makeBipedIdleClip(const Skeleton& skeleton) {
    AnimationClip clip;
    clip.name = "idle";
    clip.duration = 2.0f;
    clip.loop = true;

    const int pelvisIndex = skeleton.findBone(kBipedBonePelvis);
    if (pelvisIndex >= 0) {
        const Vector3 bindTranslation = skeleton.bones[static_cast<std::size_t>(pelvisIndex)].localTranslation;
        BoneTrack track;
        track.boneIndex = pelvisIndex;
        track.translationKeys = {
            Vector3Key{0.0f, bindTranslation},
            Vector3Key{1.0f, bindTranslation + Vector3{0.0f, 0.015f, 0.0f}},
            Vector3Key{2.0f, bindTranslation},
        };
        clip.tracks.push_back(std::move(track));
    }
    return clip;
}

AnimationClip makeBipedWalkClip(const Skeleton& skeleton) {
    AnimationClip clip;
    clip.name = "walk";
    clip.duration = 0.8f;
    clip.loop = true;

    addRotationXTrack(clip, skeleton, kBipedBoneHipL,
        {{0.0f, -25.0f}, {0.2f, 0.0f}, {0.4f, 25.0f}, {0.6f, 0.0f}, {0.8f, -25.0f}});
    addRotationXTrack(clip, skeleton, kBipedBoneHipR,
        {{0.0f, 25.0f}, {0.2f, 0.0f}, {0.4f, -25.0f}, {0.6f, 0.0f}, {0.8f, 25.0f}});
    addRotationXTrack(clip, skeleton, kBipedBoneKneeL,
        {{0.0f, 0.0f}, {0.2f, 35.0f}, {0.4f, 0.0f}, {0.6f, 0.0f}, {0.8f, 0.0f}});
    addRotationXTrack(clip, skeleton, kBipedBoneKneeR,
        {{0.0f, 0.0f}, {0.2f, 0.0f}, {0.4f, 0.0f}, {0.6f, 35.0f}, {0.8f, 0.0f}});
    addRotationXTrack(clip, skeleton, kBipedBoneShoulderL,
        {{0.0f, 15.0f}, {0.2f, 0.0f}, {0.4f, -15.0f}, {0.6f, 0.0f}, {0.8f, 15.0f}});
    addRotationXTrack(clip, skeleton, kBipedBoneShoulderR,
        {{0.0f, -15.0f}, {0.2f, 0.0f}, {0.4f, 15.0f}, {0.6f, 0.0f}, {0.8f, -15.0f}});

    return clip;
}

AnimationClip makeBipedAttackClip(const Skeleton& skeleton) {
    AnimationClip clip;
    clip.name = "attack";
    clip.duration = 0.6f;
    clip.loop = false;

    // Windup (arm raises back/up), then the chop swing, then return to bind.
    addRotationXTrack(clip, skeleton, kBipedBoneShoulderR,
        {{0.0f, 0.0f}, {0.15f, -60.0f}, {0.3f, 70.0f}, {0.6f, 0.0f}});
    addRotationXTrack(clip, skeleton, kBipedBoneElbowR,
        {{0.0f, 0.0f}, {0.15f, 40.0f}, {0.3f, 0.0f}, {0.6f, 0.0f}});

    return clip;
}

}  // namespace odai::anim
