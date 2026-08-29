#pragma once

#include "anim/animation_clip.h"
#include "anim/skeleton.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace odai::anim {

struct HkxSectionView {
    std::string name;
    std::uint32_t dataStart = 0;
    std::uint32_t localFixups = 0;
    std::uint32_t globalFixups = 0;
    std::uint32_t virtualFixups = 0;
    std::uint32_t exports = 0;
    std::uint32_t imports = 0;
    std::uint32_t end = 0;
};

struct HkxObjectReference {
    std::uint32_t sourceSection = 0;
    std::uint32_t sourceOffset = 0;
    std::uint32_t targetSection = 0;
    std::uint32_t targetOffset = 0;
};

struct HkxObjectRecord {
    std::uint32_t section = 0;
    std::uint32_t offset = 0;
    std::string className;
};

enum class HkxGeneratorIdentity : std::uint8_t {
    Unknown,
    Vanilla,
    Fnis,
    Nemesis,
};

struct HkxPackfileSummary {
    std::string contentsVersion;
    std::uint8_t pointerSize = 0;
    bool littleEndian = false;
    std::vector<HkxSectionView> sections;
    std::vector<std::string> classNames;
    std::vector<std::string> unsupportedBehaviorClasses;
    std::vector<HkxObjectReference> objectReferences;
    std::vector<HkxObjectRecord> objects;
    std::size_t localFixupCount = 0;
    std::size_t globalFixupCount = 0;
    std::size_t virtualFixupCount = 0;
    bool containsSkeleton = false;
    bool containsAnimation = false;
    bool containsBehaviorGraph = false;
    bool containsPhysicsMetadata = false;
    HkxGeneratorIdentity generator = HkxGeneratorIdentity::Unknown;
};

struct HkxReadLimits {
    std::size_t maxFileBytes = 256u * 1024u * 1024u;
    std::size_t maxSections = 32u;
    std::size_t maxClassNames = 16384u;
    std::size_t maxStringBytes = 512u;
    std::size_t maxAnimationTracks = 4096u;
    std::size_t maxAnimationFrames = 65536u;
    std::size_t maxDecodedTransformKeys = 16u * 1024u * 1024u;
    std::size_t maxBehaviorNodes = 65536u;
    std::size_t maxBehaviorEdges = 262144u;
};

struct HkxAnimationAnnotation {
    float time = 0.0f;
    std::string text;
};

struct HkxDecodedClipMetadata {
    std::uint32_t frameCount = 0;
    std::uint32_t transformTrackCount = 0;
    std::uint32_t floatTrackCount = 0;
    std::uint32_t blockCount = 0;
    float frameDuration = 0.0f;
    std::string originalSkeletonName;
    std::vector<std::string> trackNames;
    std::vector<std::int16_t> transformTrackToBoneIndices;
    std::vector<HkxAnimationAnnotation> annotations;
    std::uint32_t blendHint = 0;
    std::size_t boundTracks = 0;
    std::size_t missingTracks = 0;
};

struct HkxDecodedSkeleton {
    std::string name;
    std::vector<std::string> boneNames;
    std::vector<std::int16_t> parentIndices;
    std::vector<bool> translationLocked;
};

enum class HkxBehaviorNodeKind : std::uint8_t {
    Graph,
    StateMachine,
    State,
    Clip,
    BehaviorReference,
    Blender,
    BlenderChild,
    ManualSelector,
    ModifierGenerator,
    TransitionEffect,
};

// A fixup-backed, immutable description of the authored generator topology.
// Runtime graph instances compile this catalog into their own state and never
// retain pointers into the source packfile.
struct HkxBehaviorNode {
    HkxBehaviorNodeKind kind = HkxBehaviorNodeKind::Graph;
    std::string name;
    std::string assetPath;
    std::vector<std::uint32_t> children;
    std::int32_t stateId = -1;
    std::int32_t startStateId = -1;
    float weight = 1.0f;
    float playbackSpeed = 1.0f;
    float transitionDuration = 0.0f;
};

struct HkxDecodedBehaviorGraph {
    std::string name;
    std::uint32_t rootNode = 0;
    std::vector<HkxBehaviorNode> nodes;
    std::size_t clipGeneratorCount = 0;
    std::size_t behaviorReferenceCount = 0;
    std::size_t stateMachineCount = 0;
    std::size_t transitionEffectCount = 0;
};

// Clean-room structural reader for the legacy x64 little-endian Havok
// packfiles shipped by Skyrim SE. It validates section and fixup ranges and
// inventories reflected class names, but never constructs a Havok object.
bool inspectHkxPackfile(
    std::span<const std::uint8_t> bytes, HkxPackfileSummary& out,
    std::string& outError, const HkxReadLimits& limits = {});

// Decodes Skyrim SE's reflected hkaSplineCompressedAnimation into the engine's
// immutable, per-bone AnimationClip representation. The packfile fixups remain
// the only pointer authority; serialized pointer bytes are never dereferenced.
// Tracks are matched to the NIF-derived runtime skeleton by annotation-track
// name, then rebased from Bethesda's Z-up coordinates to engine Y-up.
bool decodeHkxAnimationClip(
    std::span<const std::uint8_t> bytes, const Skeleton& targetSkeleton,
    std::string clipName, AnimationClip& outClip,
    HkxDecodedClipMetadata& outMetadata, std::string& outError,
    const HkxDecodedSkeleton* sourceSkeleton = nullptr,
    const HkxReadLimits& limits = {});

// Reads the animation (largest) hkaSkeleton from a Skyrim SE skeleton HKX.
// Ragdoll skeletons in the same packfile are intentionally ignored.
bool decodeHkxAnimationSkeleton(
    std::span<const std::uint8_t> bytes, HkxDecodedSkeleton& outSkeleton,
    std::string& outError, const HkxReadLimits& limits = {});

// Decodes the vanilla generator topology used by Skyrim locomotion graphs.
// Supported typed nodes include behavior graphs/references, state machines,
// selectors, blenders, modifier generators, clip generators, and blending
// transition effects. Conditions and gameplay variables remain authored data;
// this function inventories topology and assets without executing them.
bool decodeHkxBehaviorGraph(
    std::span<const std::uint8_t> bytes, HkxDecodedBehaviorGraph& outGraph,
    std::string& outError, const HkxReadLimits& limits = {});

const char* hkxGeneratorName(HkxGeneratorIdentity generator);

}  // namespace odai::anim
