#include "anim/hkx_packfile.h"
#include "anim/skyrim_animation.h"
#include "bethesda/bethesda_physics_world.h"
#include "bethesda/bethesda_session.h"
#include "bethesda/runtime_ids.h"
#include "bethesda/save_game.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <Jolt/Jolt.h>

namespace {

void writeU32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void writeF32(std::vector<std::uint8_t>& bytes, std::size_t offset, float value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void writeSection(std::vector<std::uint8_t>& bytes, std::size_t header,
                  const char* name, std::uint32_t dataStart,
                  std::uint32_t localFixups, std::uint32_t globalFixups,
                  std::uint32_t virtualFixups, std::uint32_t exports,
                  std::uint32_t imports, std::uint32_t end) {
    std::memcpy(bytes.data() + header, name, std::strlen(name));
    bytes[header + 19u] = 0xffu;
    writeU32(bytes, header + 20u, dataStart);
    const std::uint32_t fields[]{localFixups, globalFixups, virtualFixups,
        exports, imports, end};
    for (std::size_t index = 0; index < std::size(fields); ++index) {
        writeU32(bytes, header + 24u + index * 4u, fields[index] - dataStart);
    }
}

std::vector<std::uint8_t> syntheticAnimationPackfile(std::size_t* outBlob = nullptr) {
    constexpr std::size_t classStart = 160u;
    const std::string className = "hkaSplineCompressedAnimation";
    const std::size_t classEnd = (classStart + className.size() + 1u + 15u) & ~15u;
    const std::size_t dataStart = classEnd;
    constexpr std::size_t object = 0u;
    constexpr std::size_t annotationTrack = 0xb0u;
    constexpr std::size_t annotationEvent = 0xc8u;
    constexpr std::size_t trackName = 0xd8u;
    constexpr std::size_t eventText = 0xddu;
    constexpr std::size_t blockOffsets = 0xf0u;
    constexpr std::size_t blob = 0x100u;
    constexpr std::size_t blobBytes = 28u;
    constexpr std::size_t localFixups = 0x120u;
    constexpr std::size_t globalFixups = localFixups + 6u * 8u;
    constexpr std::size_t virtualFixups = globalFixups;
    constexpr std::size_t exports = virtualFixups + 20u;
    constexpr std::size_t end = exports;
    std::vector<std::uint8_t> bytes(dataStart + end, 0xffu);
    std::fill(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(dataStart + localFixups), 0u);
    const std::uint8_t magic[] = {0x57u, 0xe0u, 0xe0u, 0x57u, 0x10u, 0xc0u, 0xc0u, 0x10u};
    std::memcpy(bytes.data(), magic, sizeof(magic));
    writeU32(bytes, 12u, 8u);
    bytes[16] = 8u;
    bytes[17] = 1u;
    writeU32(bytes, 20u, 2u);
    writeU32(bytes, 24u, 1u);
    writeU32(bytes, 28u, 0u);
    writeU32(bytes, 32u, 0u);
    writeU32(bytes, 36u, 0u);
    const char version[] = "hk_2010.2.0-r1";
    std::memcpy(bytes.data() + 40u, version, sizeof(version));
    writeSection(bytes, 64u, "__classnames__", classStart,
        static_cast<std::uint32_t>(classEnd), static_cast<std::uint32_t>(classEnd),
        static_cast<std::uint32_t>(classEnd), static_cast<std::uint32_t>(classEnd),
        static_cast<std::uint32_t>(classEnd), static_cast<std::uint32_t>(classEnd));
    writeSection(bytes, 112u, "__data__", static_cast<std::uint32_t>(dataStart),
        static_cast<std::uint32_t>(dataStart + localFixups),
        static_cast<std::uint32_t>(dataStart + globalFixups),
        static_cast<std::uint32_t>(dataStart + virtualFixups),
        static_cast<std::uint32_t>(dataStart + exports),
        static_cast<std::uint32_t>(dataStart + exports),
        static_cast<std::uint32_t>(dataStart + end));
    std::memcpy(bytes.data() + classStart, className.c_str(), className.size() + 1u);
    const std::size_t base = dataStart + object;
    writeU32(bytes, base + 0x10u, 5u);
    writeF32(bytes, base + 0x14u, 1.0f / 30.0f);
    writeU32(bytes, base + 0x18u, 1u);
    writeU32(bytes, base + 0x30u, 1u);
    writeU32(bytes, base + 0x34u, 0x80000001u);
    writeU32(bytes, base + 0x38u, 2u);
    writeU32(bytes, base + 0x3cu, 1u);
    writeU32(bytes, base + 0x40u, 256u);
    writeU32(bytes, base + 0x44u, 4u);
    writeF32(bytes, base + 0x48u, 8.5f);
    writeF32(bytes, base + 0x4cu, 1.0f / 8.5f);
    writeF32(bytes, base + 0x50u, 1.0f / 30.0f);
    writeU32(bytes, base + 0x60u, 1u);
    writeU32(bytes, base + 0x64u, 0x80000001u);
    writeU32(bytes, base + 0xa0u, blobBytes);
    writeU32(bytes, base + 0xa4u, 0x80000000u | blobBytes);
    writeU32(bytes, dataStart + annotationTrack + 0x10u, 1u);
    writeU32(bytes, dataStart + annotationTrack + 0x14u, 0x80000001u);
    writeF32(bytes, dataStart + annotationEvent, 1.0f / 60.0f);
    std::memcpy(bytes.data() + dataStart + trackName, "Bone", 5u);
    std::memcpy(bytes.data() + dataStart + eventText, "FootLeft", 9u);
    writeU32(bytes, dataStart + blockOffsets, 0u);
    const std::size_t b = dataStart + blob;
    bytes[b + 0u] = 0u;     // 8-bit scalar, polar32 rotation, 8-bit scale
    bytes[b + 1u] = 0x12u;  // X spline, Y static, Z identity
    bytes[b + 4u] = 1u;     // max control-point index
    bytes[b + 6u] = 1u;     // degree
    bytes[b + 7u] = 0u; bytes[b + 8u] = 0u;
    bytes[b + 9u] = 1u; bytes[b + 10u] = 1u;
    writeF32(bytes, b + 12u, 0.0f);
    writeF32(bytes, b + 16u, 10.0f);
    writeF32(bytes, b + 20u, 2.0f);
    bytes[b + 24u] = 0u;
    bytes[b + 25u] = 255u;
    const std::pair<std::uint32_t, std::uint32_t> fixups[]{
        {0x28u, annotationTrack}, {annotationTrack, trackName},
        {annotationTrack + 8u, annotationEvent}, {annotationEvent + 8u, eventText},
        {0x58u, blockOffsets}, {0x98u, blob}};
    for (std::size_t index = 0; index < std::size(fixups); ++index) {
        writeU32(bytes, dataStart + localFixups + index * 8u, fixups[index].first);
        writeU32(bytes, dataStart + localFixups + index * 8u + 4u, fixups[index].second);
    }
    writeU32(bytes, dataStart + virtualFixups, 0u);
    writeU32(bytes, dataStart + virtualFixups + 4u, 0u);
    writeU32(bytes, dataStart + virtualFixups + 8u, 0u);
    if (outBlob != nullptr) *outBlob = b;
    return bytes;
}

std::vector<std::uint8_t> syntheticSkeletonPackfile() {
    constexpr std::size_t classStart = 160u;
    const std::string className = "hkaSkeleton";
    const std::size_t classEnd = (classStart + className.size() + 1u + 15u) & ~15u;
    const std::size_t dataStart = classEnd;
    constexpr std::size_t skeletonName = 0x90u;
    constexpr std::size_t parents = 0xa0u;
    constexpr std::size_t bones = 0xb0u;
    constexpr std::size_t rootName = 0xd0u;
    constexpr std::size_t childName = 0xd5u;
    constexpr std::size_t localFixups = 0xe0u;
    constexpr std::size_t globalFixups = localFixups + 5u * 8u;
    constexpr std::size_t virtualFixups = globalFixups;
    constexpr std::size_t exports = virtualFixups + 20u;
    constexpr std::size_t end = exports;
    std::vector<std::uint8_t> bytes(dataStart + end, 0xffu);
    std::fill(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(dataStart + localFixups), 0u);
    const std::uint8_t magic[] = {0x57u, 0xe0u, 0xe0u, 0x57u, 0x10u, 0xc0u, 0xc0u, 0x10u};
    std::memcpy(bytes.data(), magic, sizeof(magic));
    writeU32(bytes, 12u, 8u);
    bytes[16] = 8u; bytes[17] = 1u;
    writeU32(bytes, 20u, 2u);
    writeU32(bytes, 24u, 1u); writeU32(bytes, 28u, 0u);
    writeU32(bytes, 32u, 0u); writeU32(bytes, 36u, 0u);
    const char version[] = "hk_2010.2.0-r1";
    std::memcpy(bytes.data() + 40u, version, sizeof(version));
    writeSection(bytes, 64u, "__classnames__", classStart,
        static_cast<std::uint32_t>(classEnd), static_cast<std::uint32_t>(classEnd),
        static_cast<std::uint32_t>(classEnd), static_cast<std::uint32_t>(classEnd),
        static_cast<std::uint32_t>(classEnd), static_cast<std::uint32_t>(classEnd));
    writeSection(bytes, 112u, "__data__", static_cast<std::uint32_t>(dataStart),
        static_cast<std::uint32_t>(dataStart + localFixups),
        static_cast<std::uint32_t>(dataStart + globalFixups),
        static_cast<std::uint32_t>(dataStart + virtualFixups),
        static_cast<std::uint32_t>(dataStart + exports),
        static_cast<std::uint32_t>(dataStart + exports),
        static_cast<std::uint32_t>(dataStart + end));
    std::memcpy(bytes.data() + classStart, className.c_str(), className.size() + 1u);
    writeU32(bytes, dataStart + 0x20u, 2u);
    writeU32(bytes, dataStart + 0x24u, 0x80000002u);
    writeU32(bytes, dataStart + 0x30u, 2u);
    writeU32(bytes, dataStart + 0x34u, 0x80000002u);
    std::memcpy(bytes.data() + dataStart + skeletonName, "Rig", 4u);
    std::int16_t parentValues[]{-1, 0};
    std::memcpy(bytes.data() + dataStart + parents, parentValues, sizeof(parentValues));
    bytes[dataStart + bones + 8u] = 0u;
    bytes[dataStart + bones + 16u + 8u] = 1u;
    std::memcpy(bytes.data() + dataStart + rootName, "Root", 5u);
    std::memcpy(bytes.data() + dataStart + childName, "Child", 6u);
    const std::pair<std::uint32_t, std::uint32_t> fixups[]{
        {0x10u, skeletonName}, {0x18u, parents}, {0x28u, bones},
        {bones, rootName}, {bones + 16u, childName}};
    for (std::size_t index = 0; index < std::size(fixups); ++index) {
        writeU32(bytes, dataStart + localFixups + index * 8u, fixups[index].first);
        writeU32(bytes, dataStart + localFixups + index * 8u + 4u, fixups[index].second);
    }
    writeU32(bytes, dataStart + virtualFixups, 0u);
    writeU32(bytes, dataStart + virtualFixups + 4u, 0u);
    writeU32(bytes, dataStart + virtualFixups + 8u, 0u);
    return bytes;
}

std::vector<std::uint8_t> syntheticBehaviorGraphPackfile(
    std::size_t* outDataStart = nullptr) {
    constexpr std::size_t classStart = 160u;
    const std::vector<std::string> classNames{
        "hkbBehaviorGraph", "hkbStateMachine", "hkbStateMachineStateInfo",
        "hkbManualSelectorGenerator", "hkbBlenderGenerator",
        "hkbBlenderGeneratorChild", "hkbClipGenerator",
        "hkbBlendingTransitionEffect"};
    std::vector<std::uint32_t> classOffsets;
    std::size_t classBytes = 0u;
    for (const std::string& name : classNames) {
        classOffsets.push_back(static_cast<std::uint32_t>(classBytes));
        classBytes += name.size() + 1u;
    }
    const std::size_t classEnd = (classStart + classBytes + 15u) & ~15u;
    const std::size_t dataStart = classEnd;
    constexpr std::uint32_t graph = 0x000u;
    constexpr std::uint32_t machine = 0x100u;
    constexpr std::uint32_t state = 0x200u;
    constexpr std::uint32_t selector = 0x300u;
    constexpr std::uint32_t blender = 0x400u;
    constexpr std::uint32_t child = 0x500u;
    constexpr std::uint32_t clip = 0x600u;
    constexpr std::uint32_t transition = 0x700u;
    constexpr std::uint32_t stateArray = 0x800u;
    constexpr std::uint32_t selectorArray = 0x810u;
    constexpr std::uint32_t blenderArray = 0x820u;
    constexpr std::uint32_t strings = 0x900u;
    const std::vector<std::string> text{
        "SyntheticGraph", "LocomotionSM", "Moving", "MovementSelector",
        "SpeedBlend", "WalkClip", "Animations\\male\\MT_WalkForward.hkx",
        "QuickBlend"};
    std::vector<std::uint32_t> textOffsets;
    std::size_t stringCursor = strings;
    for (const std::string& value : text) {
        textOffsets.push_back(static_cast<std::uint32_t>(stringCursor));
        stringCursor += value.size() + 1u;
    }
    const std::size_t localFixups = (stringCursor + 15u) & ~15u;
    const std::pair<std::uint32_t, std::uint32_t> fixups[]{
        {graph + 0x38u, textOffsets[0]}, {graph + 0x80u, machine},
        {machine + 0x38u, textOffsets[1]}, {machine + 0x90u, stateArray},
        {stateArray, state}, {state + 0x58u, selector}, {state + 0x60u, textOffsets[2]},
        {selector + 0x38u, textOffsets[3]}, {selector + 0x48u, selectorArray},
        {selectorArray, blender}, {selectorArray + 8u, clip},
        {blender + 0x38u, textOffsets[4]}, {blender + 0x60u, blenderArray},
        {blenderArray, child}, {child + 0x30u, clip},
        {clip + 0x38u, textOffsets[5]}, {clip + 0x48u, textOffsets[6]},
        {transition + 0x38u, textOffsets[7]}};
    const std::size_t globalFixups = localFixups + std::size(fixups) * 8u;
    const std::size_t virtualFixups = globalFixups;
    const std::size_t exports = virtualFixups + classNames.size() * 12u;
    const std::size_t end = exports;
    std::vector<std::uint8_t> bytes(dataStart + end, 0xffu);
    std::fill(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(dataStart + localFixups), 0u);
    const std::uint8_t magic[] = {0x57u, 0xe0u, 0xe0u, 0x57u, 0x10u, 0xc0u, 0xc0u, 0x10u};
    std::memcpy(bytes.data(), magic, sizeof(magic));
    writeU32(bytes, 12u, 8u); bytes[16] = 8u; bytes[17] = 1u;
    writeU32(bytes, 20u, 2u);
    writeU32(bytes, 24u, 1u); writeU32(bytes, 28u, graph);
    writeU32(bytes, 32u, 0u); writeU32(bytes, 36u, classOffsets[0]);
    const char version[] = "hk_2010.2.0-r1";
    std::memcpy(bytes.data() + 40u, version, sizeof(version));
    writeSection(bytes, 64u, "__classnames__", classStart,
        static_cast<std::uint32_t>(classEnd), static_cast<std::uint32_t>(classEnd),
        static_cast<std::uint32_t>(classEnd), static_cast<std::uint32_t>(classEnd),
        static_cast<std::uint32_t>(classEnd), static_cast<std::uint32_t>(classEnd));
    writeSection(bytes, 112u, "__data__", static_cast<std::uint32_t>(dataStart),
        static_cast<std::uint32_t>(dataStart + localFixups),
        static_cast<std::uint32_t>(dataStart + globalFixups),
        static_cast<std::uint32_t>(dataStart + virtualFixups),
        static_cast<std::uint32_t>(dataStart + exports),
        static_cast<std::uint32_t>(dataStart + exports),
        static_cast<std::uint32_t>(dataStart + end));
    std::size_t classCursor = classStart;
    for (const std::string& name : classNames) {
        std::memcpy(bytes.data() + classCursor, name.c_str(), name.size() + 1u);
        classCursor += name.size() + 1u;
    }
    for (std::size_t index = 0; index < text.size(); ++index) {
        std::memcpy(bytes.data() + dataStart + textOffsets[index],
            text[index].c_str(), text[index].size() + 1u);
    }
    writeU32(bytes, dataStart + machine + 0x68u, 7u);
    writeU32(bytes, dataStart + machine + 0x98u, 1u);
    writeU32(bytes, dataStart + machine + 0x9cu, 0x80000001u);
    writeU32(bytes, dataStart + state + 0x68u, 7u);
    writeU32(bytes, dataStart + selector + 0x50u, 2u);
    writeU32(bytes, dataStart + selector + 0x54u, 0x80000002u);
    writeU32(bytes, dataStart + blender + 0x68u, 1u);
    writeU32(bytes, dataStart + blender + 0x6cu, 0x80000001u);
    writeF32(bytes, dataStart + child + 0x40u, 0.75f);
    writeF32(bytes, dataStart + clip + 0x60u, 1.25f);
    writeF32(bytes, dataStart + transition + 0x50u, 0.20f);
    for (std::size_t index = 0; index < std::size(fixups); ++index) {
        writeU32(bytes, dataStart + localFixups + index * 8u, fixups[index].first);
        writeU32(bytes, dataStart + localFixups + index * 8u + 4u, fixups[index].second);
    }
    const std::uint32_t objectOffsets[]{
        graph, machine, state, selector, blender, child, clip, transition};
    for (std::size_t index = 0; index < classNames.size(); ++index) {
        const std::size_t entry = dataStart + virtualFixups + index * 12u;
        writeU32(bytes, entry, objectOffsets[index]);
        writeU32(bytes, entry + 4u, 0u);
        writeU32(bytes, entry + 8u, classOffsets[index]);
    }
    if (outDataStart != nullptr) *outDataStart = dataStart;
    return bytes;
}

std::vector<std::uint8_t> syntheticPackfile() {
    constexpr std::size_t dataStart = 112u;
    const std::string classes = std::string("hkaSkeleton\0hkbBehaviorGraph\0", 29u);
    std::vector<std::uint8_t> bytes(dataStart + classes.size(), 0u);
    const std::uint8_t magic[] = {0x57u, 0xe0u, 0xe0u, 0x57u, 0x10u, 0xc0u, 0xc0u, 0x10u};
    std::memcpy(bytes.data(), magic, sizeof(magic));
    writeU32(bytes, 12u, 8u);
    bytes[16] = 8u;
    bytes[17] = 1u;
    writeU32(bytes, 20u, 1u);
    const char version[] = "hk_2010.2.0-r1";
    std::memcpy(bytes.data() + 40u, version, sizeof(version));
    const char section[] = "__classnames__";
    std::memcpy(bytes.data() + 64u, section, sizeof(section));
    bytes[83] = 0xffu;
    for (std::size_t field = 0; field < 7u; ++field) {
        writeU32(bytes, 84u + field * 4u,
            field == 0u ? static_cast<std::uint32_t>(dataStart) :
                static_cast<std::uint32_t>(bytes.size() - dataStart));
    }
    std::memcpy(bytes.data() + dataStart, classes.data(), classes.size());
    return bytes;
}

void testHkxInspection() {
    auto bytes = syntheticPackfile();
    odai::anim::HkxPackfileSummary summary;
    std::string error;
    assert(odai::anim::inspectHkxPackfile(bytes, summary, error));
    assert(summary.pointerSize == 8u && summary.littleEndian);
    assert(summary.containsSkeleton && summary.containsBehaviorGraph);
    bytes[111] = 0xffu;
    assert(!odai::anim::inspectHkxPackfile(bytes, summary, error));
}

void testHkxClipDecoding() {
    std::size_t blob = 0u;
    const auto bytes = syntheticAnimationPackfile(&blob);
    odai::anim::Skeleton skeleton;
    skeleton.bones.push_back({"Bone", -1});
    odai::anim::AnimationClip clip;
    odai::anim::HkxDecodedClipMetadata metadata;
    std::string error;
    assert(odai::anim::decodeHkxAnimationClip(
        bytes, skeleton, "synthetic spline", clip, metadata, error));
    assert(clip.name == "synthetic spline" && clip.tracks.size() == 1u);
    assert(metadata.frameCount == 2u && metadata.boundTracks == 1u);
    assert(metadata.trackNames == std::vector<std::string>{"Bone"});
    assert(metadata.annotations.size() == 1u);
    assert(metadata.annotations.front().text == "FootLeft");
    const auto& keys = clip.tracks.front().translationKeys;
    assert(keys.size() == 2u);
    assert(std::fabs(keys.front().value.x) < 1.0e-5f);
    assert(std::fabs(keys.back().value.x - 10.0f) < 1.0e-4f);
    assert(std::fabs(keys.front().value.y) < 1.0e-5f);
    assert(std::fabs(keys.front().value.z + 2.0f) < 1.0e-5f);

    auto unsupported = bytes;
    unsupported[blob] = static_cast<std::uint8_t>(3u << 2u);
    unsupported[blob + 2u] = 1u;
    assert(!odai::anim::decodeHkxAnimationClip(
        unsupported, skeleton, "unsupported", clip, metadata, error));
    assert(error.find("threecomp24") != std::string::npos);

    auto malformed = bytes;
    malformed.resize(malformed.size() - 1u);
    assert(!odai::anim::decodeHkxAnimationClip(
        malformed, skeleton, "malformed", clip, metadata, error));
}

void testHkxSkeletonDecoding() {
    auto bytes = syntheticSkeletonPackfile();
    odai::anim::HkxDecodedSkeleton skeleton;
    std::string error;
    assert(odai::anim::decodeHkxAnimationSkeleton(bytes, skeleton, error));
    assert(skeleton.name == "Rig");
    assert(skeleton.boneNames == std::vector<std::string>({"Root", "Child"}));
    assert(skeleton.parentIndices == std::vector<std::int16_t>({-1, 0}));
    assert(!skeleton.translationLocked[0] && skeleton.translationLocked[1]);
}

void testHkxBehaviorGraphDecoding() {
    std::size_t dataStart = 0u;
    auto bytes = syntheticBehaviorGraphPackfile(&dataStart);
    odai::anim::HkxDecodedBehaviorGraph graph;
    std::string error;
    assert(odai::anim::decodeHkxBehaviorGraph(bytes, graph, error));
    assert(graph.name == "SyntheticGraph" && graph.nodes.size() == 8u);
    assert(graph.stateMachineCount == 1u && graph.clipGeneratorCount == 1u);
    assert(graph.transitionEffectCount == 1u && graph.behaviorReferenceCount == 0u);
    const auto findKind = [&](odai::anim::HkxBehaviorNodeKind kind) -> const auto& {
        const auto found = std::find_if(graph.nodes.begin(), graph.nodes.end(),
            [&](const auto& node) { return node.kind == kind; });
        assert(found != graph.nodes.end());
        return *found;
    };
    const auto& machine = findKind(odai::anim::HkxBehaviorNodeKind::StateMachine);
    assert(machine.startStateId == 7 && machine.children.size() == 1u);
    const auto& state = findKind(odai::anim::HkxBehaviorNodeKind::State);
    assert(state.stateId == 7 && state.name == "Moving" && state.children.size() == 1u);
    const auto& selector = findKind(odai::anim::HkxBehaviorNodeKind::ManualSelector);
    assert(selector.children.size() == 2u);
    const auto& clip = findKind(odai::anim::HkxBehaviorNodeKind::Clip);
    assert(clip.assetPath == "Animations\\male\\MT_WalkForward.hkx");
    assert(std::fabs(clip.playbackSpeed - 1.25f) < 1.0e-6f);
    const auto& child = findKind(odai::anim::HkxBehaviorNodeKind::BlenderChild);
    assert(std::fabs(child.weight - 0.75f) < 1.0e-6f && child.children.size() == 1u);
    const auto& transition = findKind(odai::anim::HkxBehaviorNodeKind::TransitionEffect);
    assert(std::fabs(transition.transitionDuration - 0.20f) < 1.0e-6f);

    writeU32(bytes, dataStart + 0x100u + 0x98u, 0xffffffffu);
    assert(!odai::anim::decodeHkxBehaviorGraph(bytes, graph, error));
    assert(error.find("state array") != std::string::npos);
}

odai::anim::Skeleton makeRig() {
    odai::anim::Skeleton rig;
    rig.bones.push_back({"NPC Root [Root]", -1});
    rig.bones.push_back({"WeaponSword", 0, {0.0f, 0.0f, 10.0f}});
    rig.bones.push_back({"QUIVER", 0, {0.0f, -5.0f, 20.0f}});
    return rig;
}

void testRigBindingAndGraphSnapshot() {
    const auto rig = std::make_shared<odai::anim::Skeleton>(makeRig());
    const std::vector<std::string> names{"NPC Root [Root]", "weaponsword", "missing"};
    const auto binding = odai::anim::bindTracksByName(names, *rig);
    assert(binding.exactMatches == 1u && binding.caseInsensitiveMatches == 1u);
    assert(binding.missingTracks.size() == 1u && binding.coverage() > 0.66f);

    odai::anim::AnimationClip idle;
    idle.name = "idle";
    idle.duration = 1.0f;
    odai::anim::AnimationClip walk = idle;
    walk.name = "walk";
    walk.annotations.push_back({0.01f, "FootLeft"});
    odai::anim::BoneTrack root;
    root.boneIndex = 0;
    root.translationKeys = {{0.0f, {}}, {1.0f, {100.0f, 0.0f, 0.0f}}};
    walk.tracks.push_back(root);
    odai::anim::AnimationView view;
    view.skeleton = rig;
    view.clips = {idle, walk};
    view.stateClips = {
        {"idle", "idle"}, {"locomotion", "walk"}, {"sprint", "walk"}};
    view.socketBoneNames = {"WeaponSword", "QUIVER"};
    view.supportedBehaviorGraph = true;

    odai::anim::BehaviorGraphInstance first;
    std::string error;
    assert(first.bind(view, error));
    odai::anim::AnimationInputState input;
    input.movementSpeed = 120.0f;
    input.animationDriven = true;
    auto output = first.step(input, 1.0f / 60.0f);
    assert(output.activeState == "locomotion" && output.desiredRootMotion.x > 1.0f);
    assert(std::find(output.events.begin(), output.events.end(),
        odai::anim::AnimationEvent{"FootLeft", {}}) != output.events.end());
    assert(output.socketTransforms.size() == 2u);
    const auto saved = first.snapshot();
    assert(saved.previousState == "idle" && saved.transitionDuration > 0.0f);
    output = first.step(input, 1.0f / 60.0f);

    odai::anim::BehaviorGraphInstance restored;
    assert(restored.bind(view, error));
    assert(restored.restore(saved, error));
    const auto replay = restored.step(input, 1.0f / 60.0f);
    assert(replay.activeState == output.activeState);
    assert(std::fabs(replay.desiredRootMotion.x - output.desiredRootMotion.x) < 1.0e-4f);
    input.sprinting = true;
    const auto sprint = restored.step(input, 1.0f / 60.0f);
    assert(sprint.activeState == "sprint");
    for (const auto& matrix : sprint.pose) {
        for (const float value : matrix.m) assert(std::isfinite(value));
    }
}

void testJoltCharacterGroundingAndSnapshot() {
    using namespace odai::bethesda;
    BethesdaPhysicsWorld world;
    std::string error;
    assert(world.initialize(error));
    const std::vector<odai::math::Vector3> vertices{
        {-500.0f, 0.0f, -500.0f}, {500.0f, 0.0f, -500.0f},
        {500.0f, 0.0f, 500.0f}, {-500.0f, 0.0f, 500.0f}};
    const std::vector<std::uint32_t> indices{0u, 1u, 2u, 0u, 2u, 3u};
    assert(world.addStreamedStaticCollision(17u, vertices, indices, error));
    const auto floorHit = world.castDown({0.0f, 100.0f, 0.0f}, 200.0f);
    assert(floorHit.has_value());
    assert(std::fabs(floorHit->position.y) < 1.0e-3f);
    assert(std::fabs(floorHit->distance - 100.0f) < 1.0e-3f);
    assert(floorHit->normal.y > 0.99f);
    const auto footHit = world.castDown({0.0f, 40.0f, 0.0f}, 80.0f);
    assert(footHit.has_value() && std::fabs(footHit->position.y) < 1.0e-3f);
    assert(!footHit->object.has_value());
    const ObjectId cameraBlocker = ObjectId::runtime(6u);
    PhysicsDynamicBodyConfig blocker;
    blocker.position = {0.0f, 100.0f, 0.0f};
    blocker.boundsHalfExtents = {10.0f, 10.0f, 10.0f};
    assert(world.addDynamicBody(cameraBlocker, blocker, error));
    const auto boomHit = world.castSphere(
        {-100.0f, 100.0f, 0.0f}, {100.0f, 100.0f, 0.0f}, 12.0f);
    assert(boomHit.has_value());
    assert(boomHit->distance > 70.0f && boomHit->distance < 90.0f);
    assert(boomHit->object == cameraBlocker);
    assert(!world.castSphere(
        {-100.0f, 100.0f, 0.0f}, {100.0f, 100.0f, 0.0f}, 12.0f,
        cameraBlocker).has_value());
    assert(world.removeDynamicBody(cameraBlocker));
    const ObjectId actor = ObjectId::runtime(7u);
    PhysicsCharacterConfig config;
    config.position = {0.0f, 100.0f, 0.0f};
    assert(world.addCharacter(actor, config, error));
    for (int tick = 0; tick < 180; ++tick) world.step(1.0f / 60.0f);
    const auto state = world.characterState(actor);
    assert(state.has_value());
    assert(state->grounded);
    assert(state->position.y > -1.0f && state->position.y < 1.0f);
    const auto saved = world.snapshot();
    assert(saved.size() == 1u && saved.front().object == actor);
    PhysicsCharacterInput input;
    input.desiredVelocity = {200.0f, 0.0f, 0.0f};
    assert(world.setCharacterInput(actor, input));
    world.step(1.0f / 60.0f);
    assert(world.restore(saved, error));
    assert(std::fabs(world.characterState(actor)->position.x - saved.front().position.x) < 1.0e-4f);
    assert(world.removeStreamedStaticCollision(17u));
    for (int tick = 0; tick < 60; ++tick) world.step(1.0f / 60.0f);
    assert(!world.characterState(actor)->grounded);
    assert(world.characterState(actor)->position.y < 40.0f);
}

void testJoltCharactersBlockEachOther() {
    using namespace odai::bethesda;
    BethesdaPhysicsWorld world;
    std::string error;
    assert(world.initialize(error));
    const std::vector<odai::math::Vector3> vertices{
        {-500.0f, 0.0f, -500.0f}, {500.0f, 0.0f, -500.0f},
        {500.0f, 0.0f, 500.0f}, {-500.0f, 0.0f, 500.0f}};
    const std::vector<std::uint32_t> indices{0u, 1u, 2u, 0u, 2u, 3u};
    assert(world.addStreamedStaticCollision(18u, vertices, indices, error));

    const ObjectId player = ObjectId::runtime(70u);
    const ObjectId actor = ObjectId::runtime(71u);
    PhysicsCharacterConfig config;
    config.position = {0.0f, 0.0f, 0.0f};
    assert(world.addCharacter(player, config, error));
    config.position = {100.0f, 0.0f, 0.0f};
    assert(world.addCharacter(actor, config, error));
    for (int tick = 0; tick < 30; ++tick) world.step(1.0f / 60.0f);

    PhysicsCharacterInput input;
    input.desiredVelocity = {200.0f, 0.0f, 0.0f};
    assert(world.setCharacterInput(player, input));
    for (int tick = 0; tick < 60; ++tick) world.step(1.0f / 60.0f);
    const auto playerState = world.characterState(player);
    const auto actorState = world.characterState(actor);
    assert(playerState.has_value() && actorState.has_value());
    assert(playerState->position.x < actorState->position.x);
    assert((actorState->position.x - playerState->position.x) > 40.0f);
}

void testJoltImpulseCanCarryCharacterOffLedge() {
    using namespace odai::bethesda;
    BethesdaPhysicsWorld world;
    std::string error;
    assert(world.initialize(error));
    const std::vector<odai::math::Vector3> ledge{
        {-200.0f, 0.0f, -200.0f}, {80.0f, 0.0f, -200.0f},
        {80.0f, 0.0f, 200.0f}, {-200.0f, 0.0f, 200.0f}};
    const std::vector<std::uint32_t> indices{0u, 1u, 2u, 0u, 2u, 3u};
    assert(world.addStreamedStaticCollision(19u, ledge, indices, error));
    const ObjectId actor = ObjectId::runtime(72u);
    PhysicsCharacterConfig config;
    config.position = {0.0f, 0.0f, 0.0f};
    assert(world.addCharacter(actor, config, error));
    for (int tick = 0; tick < 30; ++tick) world.step(1.0f / 60.0f);
    assert(world.characterState(actor)->grounded);

    assert(world.addCharacterImpulse(actor, {500.0f, 150.0f, 0.0f}));
    for (int tick = 0; tick < 120; ++tick) world.step(1.0f / 60.0f);
    const auto state = world.characterState(actor);
    assert(state.has_value());
    assert(state->position.x > 80.0f);
    assert(state->position.y < -20.0f);
    assert(state->falling);
}

void testJoltOverlappingRetailCollisionWarningIsRecoverable() {
    using namespace odai::bethesda;
    BethesdaPhysicsWorld world;
    std::string error;
    assert(world.initialize(error));

    // The host callback itself is the regression boundary: Jolt's debug-build
    // default intentionally breaks here, while ODAI must treat Trace as a
    // recoverable compatibility diagnostic.
    JPH::Trace("ODAI recoverable trace fixture");

    // More coincident triangles than Jolt permits in one leaf force its AABB
    // builder down the documented random-split warning path. That warning must
    // remain diagnostic: the default Jolt callback traps if the host forgets
    // to install one, which used to crash exterior streaming on retail meshes.
    const std::vector<odai::math::Vector3> vertices{
        {-100.0f, 0.0f, -100.0f}, {100.0f, 0.0f, -100.0f},
        {0.0f, 0.0f, 100.0f}};
    std::vector<std::uint32_t> indices;
    for (int triangle = 0; triangle < 16; ++triangle) {
        indices.insert(indices.end(), {0u, 1u, 2u});
    }
    assert(world.addStreamedStaticCollision(23u, vertices, indices, error));
    const auto hit = world.castDown({0.0f, 100.0f, 0.0f}, 200.0f);
    assert(hit.has_value());
    assert(std::fabs(hit->position.y) < 1.0e-3f);
}

void testSessionFixedTickAndSaveContinuation() {
    using namespace odai::bethesda;
    const auto rig = std::make_shared<odai::anim::Skeleton>(makeRig());
    auto third = std::make_shared<odai::anim::AnimationView>();
    third->skeleton = rig;
    odai::anim::AnimationClip idle;
    idle.name = "idle";
    idle.duration = 1.0f;
    third->clips.push_back(idle);
    third->supportedBehaviorGraph = true;
    auto first = std::make_shared<odai::anim::AnimationView>(*third);

    BethesdaSessionConfig config;
    config.game = odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition;
    config.contentFingerprint = "animation-save-fixture";
    std::string error;
    BethesdaSession session;
    assert(session.configure(config, error));
    RuntimeObject actor;
    actor.id = ObjectId::runtime(99u);
    actor.base = makeRecordKey("skyrim.esm", 0x7u);
    actor.kind = RuntimeObjectKind::Actor;
    actor.actorValues = ActorValues{};
    actor.transform.position = {0.0, 0.0, 120.0};
    assert(session.world().addInitialObject(actor, error));
    PhysicsCharacterConfig physical;
    physical.position = {0.0f, 0.0f, 120.0f};
    assert(session.registerActorAnimation(actor.id, third, first, physical, error));
    odai::anim::AnimationInputState input;
    input.movementSpeed = 80.0f;
    assert(session.setActorAnimationInput(actor.id, input));
    const auto advanced = session.advance(1.0 / 30.0);
    assert(advanced.clock.steps == 2u);
    const auto snapshots = session.animationSnapshots();
    assert(snapshots.size() == 1u && snapshots.front().firstPerson.has_value());
    assert(snapshots.front().thirdPerson.fixedTick == 2u &&
        snapshots.front().firstPerson->fixedTick == 2u);
    assert(snapshots.front().thirdPerson.previousState == "idle");
    assert(snapshots.front().thirdPerson.transitionDuration > 0.0f);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "odai-animation-save-v2.odai";
    assert(saveOdaiGameAtomic(path, session, error));
    const std::uint64_t expectedHash = session.deterministicHash();
    BethesdaSession restored;
    assert(restored.configure(config, error));
    assert(restored.world().addInitialObject(actor, error));
    assert(restored.registerActorAnimation(actor.id, third, first, physical, error));
    SaveLoadReport report;
    assert(loadOdaiGame(path, restored, {}, report, error));
    assert(restored.deterministicHash() == expectedHash);
    std::error_code removeError;
    std::filesystem::remove(path, removeError);
}

}  // namespace

int main() {
    testHkxInspection();
    testHkxClipDecoding();
    testHkxSkeletonDecoding();
    testHkxBehaviorGraphDecoding();
    testRigBindingAndGraphSnapshot();
    testJoltCharacterGroundingAndSnapshot();
    testJoltCharactersBlockEachOther();
    testJoltImpulseCanCarryCharacterOffLedge();
    testJoltOverlappingRetailCollisionWarningIsRecoverable();
    testSessionFixedTickAndSaveContinuation();
    std::cout << "Skyrim animation/Jolt tests passed\n";
    return 0;
}
