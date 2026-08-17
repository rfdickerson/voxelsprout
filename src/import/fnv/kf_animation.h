#pragma once

// Reader for Gamebryo .kf animation files (Fallout 3 / Fallout: New Vegas).
//
// A .kf is a NIF by format -- same header, same block table, same string table
// -- carrying no geometry at all. What it holds is one NiControllerSequence: a
// named clip (start/stop time, cycle type) plus a list of CONTROLLED BLOCKS,
// each pairing a target node NAME with an interpolator holding that node's
// keyframes. Nothing in the file references the skeleton it animates; the bond
// is the node name alone, which is why this reader emits tracks keyed by name
// and leaves resolving them to a skeleton to buildFalloutAnimationClip
// (character_builder.h).
//
// Everything here stays in BETHESDA coordinates (Z-up), exactly as stored. The
// basis change into engine space belongs with the skeleton conversion that has
// to agree with it, not here -- see character_builder.cc's changePointBasis /
// changeRotationBasis.
//
// The ControlledBlock layout is the one genuinely version-conditional part of
// the format, and it is MEASURED, not assumed (odai_bethesda_probe --kf dumps
// it). At New Vegas's NIF 20.2.0.7 a ControlledBlock is 29 bytes and NOT
// 4-byte aligned, because a one-byte priority sits between two string indices:
//
//   interpolator(i32) controller(i32) priority(u8)
//   nodeName(i32) propertyType(i32) controllerType(i32)
//   controllerID(i32) interpolatorID(i32)
//
// Older files (<= 20.0.0.5) put string-PALETTE offsets there instead, and
// reading this file with that layout yields node names that are plausible
// garbage rather than an obvious failure.

#include "math/math.h"

#include <cstdint>
#include <string>
#include <vector>

namespace odai::importer::fnv {

struct KfVector3Key {
    float time = 0.0f;
    odai::math::Vector3 value{};
};

struct KfQuaternionKey {
    float time = 0.0f;
    odai::math::Quaternion value{};
};

// One target node's channels. Any array may be empty, meaning that channel is
// not animated and the bone keeps its bind-pose value (which is exactly how
// anim::AnimationSampler already treats an empty channel).
struct KfBoneTrack {
    std::string nodeName;
    std::vector<KfVector3Key> translationKeys;
    std::vector<KfQuaternionKey> rotationKeys;
    std::vector<KfVector3Key> scaleKeys;
};

struct KfAnimationStats {
    std::size_t controlledBlocks = 0;
    // Interpolators this reader understands (NiTransformInterpolator).
    std::size_t transformInterpolators = 0;
    // NiBSplineCompTransformInterpolator: cubic B-splines with 16-bit
    // quantized control points, decoded and sampled into ordinary keys. These
    // are the bones that carry a human animation -- both arms, both legs and
    // the head -- so a clip reporting zero of them on a townsperson is a clip
    // that will render a T-pose.
    std::size_t bSplineInterpolators = 0;
    // NiBSplineCompTransformInterpolator and friends: B-spline-compressed
    // curves this reader does not decode. The bones they target are simply
    // absent from `tracks` and hold their bind pose, which is a visibly
    // stiffer joint rather than a broken one.
    std::size_t unsupportedInterpolators = 0;
    // WHICH nodes those were. The count alone cannot distinguish "a few finger
    // joints are stiff" from "both arms never move", and on a human idle the
    // difference is the whole pose.
    std::vector<std::string> unsupportedNodes;
};

struct KfAnimation {
    std::string name;
    float startTime = 0.0f;
    float stopTime = 0.0f;
    // NiControllerSequence cycle type: 0 = loop, 1 = reverse, 2 = clamp.
    std::uint32_t cycleType = 0;
    std::vector<KfBoneTrack> tracks;
    KfAnimationStats stats;

    [[nodiscard]] float duration() const { return stopTime > startTime ? (stopTime - startTime) : 0.0f; }
    [[nodiscard]] bool loops() const { return cycleType == 0; }
};

// Parses one .kf. Returns false with outError set when the file is not a
// readable NIF 20.2.0.7, holds no NiControllerSequence, or fails the internal
// consistency checks (block refs and string indices must all be in range --
// these are what catch a wrong ControlledBlock stride, which otherwise
// produces confident nonsense).
//
// Key times are rebased so the clip starts at 0.
bool parseKfAnimation(
    const std::vector<std::uint8_t>& bytes, KfAnimation& outAnimation, std::string& outError);

}  // namespace odai::importer::fnv
