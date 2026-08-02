// Correctness tests for the skeleton/animation-clip sampler (Dragon Age:
// Origins touchstone, see docs/ROADMAP.md's Party RPG / Narrative section):
// bind-pose sanity, hierarchy composition, keyframe interpolation (lerp and
// slerp), loop-vs-clamp time handling across the loop boundary, and JSON
// loading. No Vulkan, no GTest -- same lightweight harness style as
// tests/dialogue_tests.cc.

#include "anim/animation_clip.h"
#include "anim/animation_sampler.h"
#include "anim/skeleton.h"
#include "anim/skeleton_io.h"
#include "math/math.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace odai::anim;
using odai::math::Matrix4;
using odai::math::Quaternion;
using odai::math::Vector3;

int g_failures = 0;

void expectTrue(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[animation test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

bool approxEqual(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) <= eps;
}

bool approxEqualVec3(const Vector3& a, const Vector3& b, float eps = 1e-3f) {
    return approxEqual(a.x, b.x, eps) && approxEqual(a.y, b.y, eps) && approxEqual(a.z, b.z, eps);
}

void expectVec3Near(const Vector3& actual, const Vector3& expected, const std::string& message) {
    if (!approxEqualVec3(actual, expected)) {
        std::cerr << "[animation test] FAIL: " << message << " (expected (" << expected.x << ", "
                   << expected.y << ", " << expected.z << "), got (" << actual.x << ", " << actual.y << ", "
                   << actual.z << "))\n";
        ++g_failures;
    }
}

Quaternion quaternionAroundZ(float radians) {
    return Quaternion{0.0f, 0.0f, std::sin(radians * 0.5f), std::cos(radians * 0.5f)};
}

// root -> child, child offset (0,1,0) from root in bind pose.
Skeleton makeTwoBoneChain() {
    Skeleton skeleton;
    Bone root;
    root.name = "root";
    root.parentIndex = -1;
    skeleton.bones.push_back(root);

    Bone child;
    child.name = "child";
    child.parentIndex = 0;
    child.localTranslation = Vector3{0.0f, 1.0f, 0.0f};
    skeleton.bones.push_back(child);
    return skeleton;
}

void testBindPoseReproducesIdentitySkinning() {
    const Skeleton skeleton = makeTwoBoneChain();
    AnimationSampler sampler;
    sampler.bindSkeleton(skeleton);

    const AnimationClip emptyClip;  // no tracks -> every bone falls back to bind pose
    std::vector<Matrix4> matrices;
    sampler.sample(skeleton, emptyClip, 0.0f, matrices);

    expectTrue(matrices.size() == 2, "one skinning matrix per bone");
    for (std::size_t i = 0; i < matrices.size(); ++i) {
        const Vector3 probe = odai::math::transformPoint(matrices[i], Vector3{3.0f, -2.0f, 5.0f});
        expectVec3Near(probe, Vector3{3.0f, -2.0f, 5.0f},
                        "bone " + std::to_string(i) + "'s skinning matrix is identity at bind pose");
    }
}

void testHierarchyCompositionCarriesParentRotationToChild() {
    const Skeleton skeleton = makeTwoBoneChain();
    AnimationSampler sampler;
    sampler.bindSkeleton(skeleton);

    AnimationClip clip;
    clip.duration = 1.0f;
    clip.loop = false;
    BoneTrack rootTrack;
    rootTrack.boneIndex = 0;
    rootTrack.rotationKeys.push_back(QuaternionKey{0.0f, quaternionAroundZ(odai::math::radians(90.0f))});
    clip.tracks.push_back(rootTrack);

    std::vector<Matrix4> matrices;
    sampler.sample(skeleton, clip, 0.0f, matrices);

    // The child's own bind-pose world position (0,1,0) run through its
    // skinning matrix should land wherever a 90-degree parent rotation
    // carries it: (0,1,0) -> (-1,0,0).
    const Vector3 childBindWorld{0.0f, 1.0f, 0.0f};
    const Vector3 skinned = odai::math::transformPoint(matrices[1], childBindWorld);
    expectVec3Near(skinned, Vector3{-1.0f, 0.0f, 0.0f},
                   "child bone inherits the animated parent's rotation");
}

void testLerpInterpolatesTranslationMidway() {
    Skeleton skeleton;
    Bone root;
    root.name = "root";
    skeleton.bones.push_back(root);
    AnimationSampler sampler;
    sampler.bindSkeleton(skeleton);

    AnimationClip clip;
    clip.duration = 1.0f;
    clip.loop = false;
    BoneTrack track;
    track.boneIndex = 0;
    track.translationKeys = {
        Vector3Key{0.0f, Vector3{0.0f, 0.0f, 0.0f}},
        Vector3Key{1.0f, Vector3{2.0f, 0.0f, 0.0f}},
    };
    clip.tracks.push_back(track);

    std::vector<Matrix4> matrices;
    sampler.sample(skeleton, clip, 0.5f, matrices);
    const Vector3 origin = odai::math::transformPoint(matrices[0], Vector3{});
    expectVec3Near(origin, Vector3{1.0f, 0.0f, 0.0f}, "translation lerp at t=0.5 is the midpoint");
}

void testSlerpInterpolatesRotationMidway() {
    Skeleton skeleton;
    Bone root;
    root.name = "root";
    skeleton.bones.push_back(root);
    AnimationSampler sampler;
    sampler.bindSkeleton(skeleton);

    AnimationClip clip;
    clip.duration = 1.0f;
    clip.loop = false;
    BoneTrack track;
    track.boneIndex = 0;
    track.rotationKeys = {
        QuaternionKey{0.0f, Quaternion{}},
        QuaternionKey{1.0f, quaternionAroundZ(odai::math::radians(90.0f))},
    };
    clip.tracks.push_back(track);

    std::vector<Matrix4> matrices;
    sampler.sample(skeleton, clip, 0.5f, matrices);
    const Vector3 rotated = odai::math::transformPoint(matrices[0], Vector3{1.0f, 0.0f, 0.0f});
    const float c45 = std::cos(odai::math::radians(45.0f));
    const float s45 = std::sin(odai::math::radians(45.0f));
    expectVec3Near(rotated, Vector3{c45, s45, 0.0f}, "rotation slerp at t=0.5 is the 45-degree midpoint");
}

void testLoopBlendsAcrossTheDurationBoundary() {
    Skeleton skeleton;
    Bone root;
    root.name = "root";
    skeleton.bones.push_back(root);
    AnimationSampler sampler;
    sampler.bindSkeleton(skeleton);

    AnimationClip clip;
    clip.duration = 1.0f;
    clip.loop = true;
    BoneTrack track;
    track.boneIndex = 0;
    track.translationKeys = {
        Vector3Key{0.2f, Vector3{1.0f, 0.0f, 0.0f}},
        Vector3Key{0.8f, Vector3{3.0f, 0.0f, 0.0f}},
    };
    clip.tracks.push_back(track);

    std::vector<Matrix4> matrices;
    // t=0.9 is past the last key (0.8) but the clip loops, so it should blend
    // toward the first key (0.2) across the wrap: frac=(0.9-0.8)/0.4=0.25.
    sampler.sample(skeleton, clip, 0.9f, matrices);
    Vector3 wrapped = odai::math::transformPoint(matrices[0], Vector3{});
    expectVec3Near(wrapped, Vector3{2.5f, 0.0f, 0.0f}, "looping clip blends from the last key toward the first across the boundary");

    AnimationClip clampedClip = clip;
    clampedClip.loop = false;
    sampler.sample(skeleton, clampedClip, 2.0f, matrices);
    const Vector3 clamped = odai::math::transformPoint(matrices[0], Vector3{});
    expectVec3Near(clamped, Vector3{3.0f, 0.0f, 0.0f}, "a non-looping clip clamps to the last key past its duration");
}

constexpr const char* kSampleSkeletonJson = R"json({
  "bones": [
    { "name": "root", "parent": "", "translation": [0,0,0], "rotation": [0,0,0,1], "scale": [1,1,1] },
    { "name": "child", "parent": "root", "translation": [0,1,0], "rotation": [0,0,0,1], "scale": [1,1,1] }
  ]
})json";

void testSkeletonLoadsFromJson() {
    const SkeletonLoadResult result = loadSkeletonFromJson(kSampleSkeletonJson, "kSampleSkeletonJson");
    expectTrue(result.ok(), "sample skeleton loads without errors");
    expectTrue(result.skeleton.bones.size() == 2, "both bones parsed");
    expectTrue(result.skeleton.findBone("child") == 1, "child bone resolved to index 1");
    expectTrue(result.skeleton.bones[1].parentIndex == 0, "child's parent resolved to root's index");
}

void testSkeletonUnknownParentRecordsError() {
    constexpr const char* kBroken = R"json({
      "bones": [ { "name": "a", "parent": "does-not-exist" } ]
    })json";
    const SkeletonLoadResult result = loadSkeletonFromJson(kBroken, "kBroken");
    expectTrue(!result.ok(), "a bone referencing an unknown parent records an error");
}

void testAnimationClipLoadsAndSortsKeys() {
    const SkeletonLoadResult skeletonResult = loadSkeletonFromJson(kSampleSkeletonJson, "kSampleSkeletonJson");
    constexpr const char* kClipJson = R"json({
      "name": "test_clip", "duration": 1.0, "loop": true,
      "tracks": [
        { "bone": "child",
          "translationKeys": [ {"time":1.0,"value":[2,0,0]}, {"time":0.0,"value":[0,0,0]} ] }
      ]
    })json";
    const AnimationClipLoadResult result = loadAnimationClipFromJson(kClipJson, skeletonResult.skeleton, "kClipJson");
    expectTrue(result.ok(), "sample clip loads without errors");
    expectTrue(result.clip.tracks.size() == 1, "one track parsed");
    if (!result.clip.tracks.empty()) {
        expectTrue(result.clip.tracks[0].boneIndex == 1, "track's bone name resolved to child's index");
        const std::vector<Vector3Key>& keys = result.clip.tracks[0].translationKeys;
        expectTrue(keys.size() == 2 && keys[0].time < keys[1].time,
                   "out-of-order authored keys are sorted ascending by time");
    }
}

void testAnimationClipUnknownBoneRecordsErrorAndSkipsTrack() {
    const SkeletonLoadResult skeletonResult = loadSkeletonFromJson(kSampleSkeletonJson, "kSampleSkeletonJson");
    constexpr const char* kBrokenClip = R"json({
      "name": "broken", "tracks": [ { "bone": "nonexistent" } ]
    })json";
    const AnimationClipLoadResult result =
        loadAnimationClipFromJson(kBrokenClip, skeletonResult.skeleton, "kBrokenClip");
    expectTrue(!result.ok(), "a track referencing an unknown bone records an error");
    expectTrue(result.clip.tracks.empty(), "the unresolvable track is skipped, not added with boneIndex=-1");
}

}  // namespace

int main() {
    testBindPoseReproducesIdentitySkinning();
    testHierarchyCompositionCarriesParentRotationToChild();
    testLerpInterpolatesTranslationMidway();
    testSlerpInterpolatesRotationMidway();
    testLoopBlendsAcrossTheDurationBoundary();
    testSkeletonLoadsFromJson();
    testSkeletonUnknownParentRecordsError();
    testAnimationClipLoadsAndSortsKeys();
    testAnimationClipUnknownBoneRecordsErrorAndSkipsTrack();

    if (g_failures != 0) {
        std::cerr << "[animation test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[animation test] all checks passed\n";
    return 0;
}
