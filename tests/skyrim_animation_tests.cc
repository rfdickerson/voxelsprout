#include "import/fnv/character_asset_manifest.h"
#include "anim/hkx_packfile.h"
#include "games/newvegas/npc_demo_locomotion.h"
#include "import/fnv/skyrim_animation_assets.h"
#include <fstream>
#include <chrono>
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
#include <limits>
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

std::vector<std::uint8_t> syntheticSkeletonPackfile(bool referencePose = false) {
    constexpr std::size_t classStart = 160u;
    const std::string className = "hkaSkeleton";
    const std::size_t classEnd = (classStart + className.size() + 1u + 15u) & ~15u;
    const std::size_t dataStart = classEnd;
    constexpr std::size_t skeletonName = 0x90u;
    constexpr std::size_t parents = 0xa0u;
    constexpr std::size_t bones = 0xb0u;
    constexpr std::size_t rootName = 0xd0u;
    constexpr std::size_t childName = 0xd5u;
    const std::size_t localFixups = referencePose ? 0x140u : 0xe0u;
    const std::size_t globalFixups = localFixups + (referencePose ? 6u : 5u) * 8u;
    const std::size_t virtualFixups = globalFixups;
    const std::size_t exports = virtualFixups + 20u;
    const std::size_t end = exports;
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
    if (referencePose) {
        writeU32(bytes,dataStart+0x40u,2u);
        writeU32(bytes,dataStart+0x44u,0x80000002u);
        writeU32(bytes,dataStart+localFixups+5u*8u,0x38u);
        writeU32(bytes,dataStart+localFixups+5u*8u+4u,0xe0u);
        for (std::size_t bone=0;bone<2;++bone) {
            const float transform[]{1,2,3,0,0,0,0,1,1,2,3,0};
            std::memcpy(bytes.data()+dataStart+0xe0u+bone*48u,transform,sizeof(transform));
        }
    }
    writeU32(bytes, dataStart + virtualFixups, 0u);
    writeU32(bytes, dataStart + virtualFixups + 4u, 0u);
    writeU32(bytes, dataStart + virtualFixups + 8u, 0u);
    return bytes;
}

std::vector<std::uint8_t> syntheticBehaviorGraphPackfile(
    std::size_t* outDataStart = nullptr, bool expressionCondition = false) {
    constexpr std::size_t classStart = 160u;
    const std::vector<std::string> classNames{
        "hkbBehaviorGraph", "hkbStateMachine", "hkbStateMachineStateInfo",
        "hkbManualSelectorGenerator", "hkbBlenderGenerator",
        "hkbBlenderGeneratorChild", "hkbClipGenerator",
        "hkbBlendingTransitionEffect", "hkbStateMachineTransitionInfoArray", "hkbExpressionCondition", "hkbCharacterStringData"};
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
    constexpr std::uint32_t strings = 0xe00u;
    const std::vector<std::string> text{
        "SyntheticGraph", "LocomotionSM", "Moving", "MovementSelector",
        "SpeedBlend", "WalkClip", "Animations\\male\\MT_WalkForward.hkx",
        "QuickBlend", "Speed > 1.5", "Speed", "selectedGeneratorIndex", "custom\\skeleton.hkx", "behaviors\\custom.hkx", "animations\\custom.hkx"};
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
        {transition + 0x38u, textOffsets[7]},
        {state + 0x50u, 0x830u}, {machine + 0xa0u, 0x830u},
        {0x840u, 0x850u}, {0x870u, transition},
        {0x878u, expressionCondition ? 0x8c0u : selector}, {0x8d0u, textOffsets[8]},
        {graph + 0x88u, 0xa00u}, {0xa20u, 0xb50u}, {0xa70u, 0xb00u}, {0xa78u, 0xa80u},
        {0xab0u, 0xb70u}, {0xb70u, textOffsets[9]}, {0xb10u, 0xb60u},
        {selector + 0x10u, 0xb80u}, {0xb90u, 0xbb0u}, {0xbb0u, textOffsets[10]}, {0xda0u, textOffsets[0]}, {0xda8u, textOffsets[11]},
        {0xdb8u, textOffsets[12]}, {0xd40u, 0xdc0u}, {0xdc0u, textOffsets[13]}};
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
    writeU32(bytes, dataStart + 0xd48u, 1u);
    writeU32(bytes, dataStart + 0xa28u, 1u);
    writeU32(bytes, dataStart + 0xab8u, 1u);
    writeU32(bytes, dataStart + 0xb18u, 1u);
    bytes[dataStart + 0xb54u] = 4; // real variable
    writeF32(bytes, dataStart + 0xb60u, 2.5f);
    writeU32(bytes, dataStart + 0xb98u, 1u);
    writeU32(bytes, dataStart + 0xba0u, 0xffffffffu);
    writeU32(bytes, dataStart + 0xbccu, 0u);
    bytes[dataStart + 0xbd0u] = 0xffu;
    writeU32(bytes, dataStart + machine + 0x68u, 7u);
    writeU32(bytes, dataStart + machine + 0x98u, 1u);
    writeU32(bytes, dataStart + machine + 0x9cu, 0x80000001u);
    writeU32(bytes, dataStart + state + 0x68u, 7u);
    writeU32(bytes, dataStart + selector + 0x50u, 2u);
    writeU32(bytes, dataStart + selector + 0x54u, 0x80000002u);
    writeU32(bytes, dataStart + blender + 0x68u, 1u);
    writeU32(bytes, dataStart + blender + 0x6cu, 0x80000001u);
    writeF32(bytes, dataStart + child + 0x40u, 0.75f);
    writeF32(bytes, dataStart + clip + 0x64u, 1.25f);
    writeF32(bytes, dataStart + transition + 0x50u, 0.20f);
    writeU32(bytes, dataStart + 0x848u, 1u);
    writeU32(bytes, dataStart + 0x880u, 42u);
    writeU32(bytes, dataStart + 0x884u, 7u);
    writeU32(bytes, dataStart + 0x888u, 0xffffffffu);
    writeU32(bytes, dataStart + 0x88cu, 0xffffffffu);
    writeU32(bytes, dataStart + 0x890u, 0x12340003u);
    writeF32(bytes, dataStart + 0x858u, 0.25f);
    for (std::size_t index = 0; index < std::size(fixups); ++index) {
        writeU32(bytes, dataStart + localFixups + index * 8u, fixups[index].first);
        writeU32(bytes, dataStart + localFixups + index * 8u + 4u, fixups[index].second);
    }
    const std::uint32_t objectOffsets[]{
        graph, machine, state, selector, blender, child, clip, transition, 0x830u, 0x8c0u, 0xd00u};
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

    odai::anim::HkxDecodedSkeleton source;
    source.boneNames = {"CustomBone"}; source.parentIndices = {-1}; source.translationLocked = {false};
    assert(!odai::anim::decodeHkxAnimationClip(bytes, skeleton, "missing custom bone", clip, metadata, error, &source));
    assert(metadata.boundTracks == 0 && metadata.missingTracks == 1);
    auto customRig = skeleton;
    customRig.bones[0].name = "CustomBone";
    assert(odai::anim::decodeHkxAnimationClip(bytes, customRig, "custom bone", clip, metadata, error, &source));
    assert(metadata.boundTracks == 1 && metadata.missingTracks == 0);

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
    // A signature byte that resembles text must not hide a fixup-backed class.
    auto prefixed = bytes;
    std::memmove(prefixed.data() + 161, prefixed.data() + 160, 12);
    prefixed[160] = 'Q';
    writeU32(prefixed, 176 + 0xe0 + 5 * 8 + 8, 1);
    odai::anim::HkxPackfileSummary summary;
    std::string inspectionError;
    assert(odai::anim::inspectHkxPackfile(prefixed, summary, inspectionError));
    assert(summary.containsSkeleton);
    odai::anim::HkxDecodedSkeleton skeleton;
    std::string error;
    assert(odai::anim::decodeHkxAnimationSkeleton(bytes, skeleton, error));
    assert(skeleton.name == "Rig");
    assert(skeleton.boneNames == std::vector<std::string>({"Root", "Child"}));
    assert(skeleton.parentIndices == std::vector<std::int16_t>({-1, 0}));
    assert(!skeleton.translationLocked[0] && skeleton.translationLocked[1]);
    assert(skeleton.referenceSkeleton.bones.empty());
    bytes=syntheticSkeletonPackfile(true);
    assert(odai::anim::decodeHkxAnimationSkeleton(bytes,skeleton,error));
    assert(skeleton.referenceSkeleton.bones.size()==2);
    const auto& rest=skeleton.referenceSkeleton.bones[1];
    assert(rest.localTranslation.x==1 && rest.localTranslation.y==3 && rest.localTranslation.z==-2);
    assert(rest.localScale.x==1 && rest.localScale.y==3 && rest.localScale.z==2);

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

    assert(state.transitions.size() == 1u && machine.transitions.size() == 1u);
    const auto& rule = state.transitions.front();
    assert(rule.eventId == 42 && rule.toStateId == 7 && rule.fromNestedStateId == -1);
    assert(rule.priority == 3 && rule.flags == 0x1234u);
    assert(rule.hasEffect && rule.effectNode >= 0 && rule.hasCondition);
    assert(rule.conditionClass == "hkbManualSelectorGenerator");
    assert(rule.triggerInterval.enterTime == 0.25f);
    auto corrupt = bytes;
    writeU32(corrupt, dataStart + 0x848u, 0xffffffffu);
    assert(!odai::anim::decodeHkxBehaviorGraph(corrupt, graph, error));
    corrupt = bytes;
    writeU32(corrupt, dataStart + 0x858u, 0x7fc00000u);
    assert(!odai::anim::decodeHkxBehaviorGraph(corrupt, graph, error));
    odai::anim::HkxReadLimits limits;
    limits.maxBehaviorEdges = 1;
    assert(!odai::anim::decodeHkxBehaviorGraph(bytes, graph, error, limits));

    writeU32(bytes, dataStart + 0x100u + 0x98u, 0xffffffffu);
    assert(!odai::anim::decodeHkxBehaviorGraph(bytes, graph, error));
    assert(error.find("state array") != std::string::npos);
    assert(graph.nodes.empty());
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

void testAuthoredGraphExecutionAndAdmission() {
    using namespace odai::anim;
    HkxDecodedBehaviorGraph graph;
    graph.eventNames = {"attackStart", "attackEnd"};
    graph.nodes.resize(6);
    graph.rootNode = 0;
    graph.nodes[0].kind = HkxBehaviorNodeKind::Graph;
    graph.nodes[0].children = {1};
    graph.nodes[1].kind = HkxBehaviorNodeKind::StateMachine;
    graph.nodes[1].children = {2, 4};
    graph.nodes[1].startStateId = 10;
    graph.nodes[2].kind = HkxBehaviorNodeKind::State;
    graph.nodes[2].stateId = 10;
    graph.nodes[2].children = {3};
    graph.nodes[3].kind = HkxBehaviorNodeKind::Clip;
    graph.nodes[3].assetPath = "idle.hkx";
    graph.nodes[4].kind = HkxBehaviorNodeKind::State;
    graph.nodes[4].stateId = 20;
    graph.nodes[4].children = {5};
    graph.nodes[5].kind = HkxBehaviorNodeKind::Clip;
    graph.nodes[5].assetPath = "attack.hkx";
    graph.nodes[5].playbackMode = 0;
    HkxBehaviorTransition attack;
    attack.eventId = 0;
    attack.toStateId = 20;
    attack.priority = 2;
    graph.nodes[2].transitions = {attack};
    HkxBehaviorTransition done;
    done.eventId = 1;
    done.toStateId = 10;
    graph.nodes[4].transitions = {done};
    auto lowerPriority = attack;
    lowerPriority.toStateId = 10;
    lowerPriority.priority = 1;
    graph.nodes[1].transitions = {lowerPriority};
    const auto program = compileBehaviorProgram(graph);
    assert(program->executable());
    AnimationView view;
    view.skeleton = std::make_shared<Skeleton>(makeRig());
    view.executionMode = AnimationExecutionMode::Havok;
    view.behavior = program;
    view.sourceFingerprint = "fixture-v1";
    AnimationClip idle, strike;
    idle.name = "idle.hkx"; idle.duration = 1;
    strike.name = "attack.hkx"; strike.duration = 1; strike.loop = false;
    view.clips = {idle, strike};
    BehaviorGraphInstance instance;
    std::string error;
    assert(instance.bind(view, error));
    auto output = instance.step({}, 0.1f);
    assert(output.authoredGraphExecuted && !output.proceduralFallback && output.activeClip == idle.name);
    instance.queueEvent({"attackStart", {}});
    output = instance.step({}, 0.1f);
    assert(output.activeClip == strike.name && output.actionActive);
    const auto saved = instance.snapshot();
    assert(saved.activeStates.at(1) == 20);
    output = instance.step({}, 0.1f);
    assert(output.activeClip == strike.name); // no one-tick return to idle
    BehaviorGraphInstance restored;
    assert(restored.bind(view, error));
    assert(restored.restore(saved, error));
    assert(restored.step({}, 0.1f).activeClip == output.activeClip);
    assert(restored.snapshot() == instance.snapshot());
    auto invalid = saved;
    invalid.activeStates[1] = 99;
    assert(!restored.restore(invalid, error));
    invalid = saved; invalid.graphFingerprint = "other-provider";
    assert(!restored.restore(invalid, error));
    restored.queueEvent({"attackEnd", {}});
    assert(restored.step({}, 0.1f).activeClip == idle.name);
    graph.nodes[2].transitions[0].hasCondition = true;
    assert(!compileBehaviorProgram(graph)->executable());
    graph.nodes[2].transitions[0].hasCondition = false;
    graph.variableNames = {"Speed", "CanAttack"};
    graph.variableDefaults = {{"Speed", 0.f}, {"CanAttack", 1.f}};
    graph.nodes[2].transitions[0].hasCondition = true;
    graph.nodes[2].transitions[0].conditionClass = "hkbExpressionCondition";
    graph.nodes[2].transitions[0].conditionExpression = "Speed >= 2.5 && CanAttack";
    view.executionMode = AnimationExecutionMode::Havok;
    view.behavior = compileBehaviorProgram(graph);
    assert(view.behavior->executable());
    assert(instance.bind(view, error));
    instance.queueEvent({"attackStart", {}});
    assert(instance.step({}, 0.1f).activeClip == idle.name);
    AnimationInputState moving;
    moving.variables = {{"Speed", 3.f}};
    moving.events = {{"attackStart", {}}};
    assert(instance.step(moving, 0.1f).activeClip == strike.name);
    assert(restored.bind(view, error));
    assert(restored.restore(instance.snapshot(), error));
    assert(restored.snapshot().variables.at("CanAttack") == 1.f);
    assert(restored.step({}, 0.1f).activeClip == instance.step({}, 0.1f).activeClip);
    assert(restored.snapshot() == instance.snapshot());
    graph.nodes[2].transitions[0].hasCondition = false;
    auto& conditioned = graph.nodes[2].transitions[0];
    conditioned.hasCondition = true;
    conditioned.conditionExpression = "CanAttack / Speed";
    view.executionMode = AnimationExecutionMode::Havok;
    view.behavior = compileBehaviorProgram(graph);
    assert(instance.bind(view, error));
    instance.queueEvent({"attackStart", {}});
    const auto failedCondition = instance.step({}, .1f);
    assert(!failedCondition.authoredGraphExecuted && failedCondition.proceduralFallback);
    assert(!failedCondition.diagnostics.empty() && failedCondition.activeClip == idle.name);
    conditioned.flags = HkxBehaviorTransition::DisableCondition;
    conditioned.conditionClass = "unknown";
    view.executionMode = AnimationExecutionMode::Havok;
    view.behavior = compileBehaviorProgram(graph);
    assert(view.behavior->executable() && instance.bind(view, error));
    instance.queueEvent({"attackStart", {}});
    assert(instance.step({}, .1f).activeClip == strike.name);
    conditioned.flags = HkxBehaviorTransition::Disabled;
    view.executionMode = AnimationExecutionMode::Havok;
    view.behavior = compileBehaviorProgram(graph);
    assert(view.behavior->executable() && instance.bind(view, error));
    instance.queueEvent({"attackStart", {}});
    assert(instance.step({}, .1f).activeClip == idle.name);
    conditioned.flags = 0;
    conditioned.hasCondition = false;
    graph.nodes[3].hasUndecodedChildren = true;
    assert(!compileBehaviorProgram(graph)->executable());
    graph.nodes[3].hasUndecodedChildren = false;
    graph.nodes[0].children = {0};
    assert(!compileBehaviorProgram(graph)->executable());
}

void testBehaviorExpressions() {
    using namespace odai::anim;
    std::string error;
    const std::vector<std::string> names{"Speed", "Ready"};
    const auto evaluate = [&](std::string_view source) {
        const auto expression = compileBehaviorExpression(source, names, error);
        assert(expression && error.empty());
        return expression->evaluate({{"Speed", 3.f}, {"Ready", 0.f}});
    };
    assert(evaluate("Speed + 2 * 4") == 11.f);
    assert(evaluate("(Speed + 2) * 4") == 20.f);
    assert(evaluate("10 - 3 - 2") == 5.f);
    assert(evaluate("!Ready && Speed >= 3 || false") == 1.f);
    assert(evaluate("-(Speed + 1) / 2") == -2.f);
    assert(evaluate(".25 + 1e-2") == .26f);
    assert(evaluate("false && (1 / 0)") == 0.f);
    assert(evaluate("true || (1 / 0)") == 1.f);
    assert(!evaluate("1 / Ready"));
    assert(!evaluate("3e38 * 3e38"));
    for (const auto* invalid : {"", "Speed = 2", "Unknown", "sin(Speed)", "Speed +", "(Speed", "1; 2", "1e999"})
        assert(!compileBehaviorExpression(invalid, names, error) && !error.empty());
    assert(!compileBehaviorExpression(std::string(1000, '!') + "true", names, error));
    assert(!compileBehaviorExpression(std::string(5000, ' '), names, error));
    auto variable = compileBehaviorExpression("Speed", names, error);
    assert(variable && !variable->evaluate({}));
    assert(!variable->evaluate({{"Speed", std::numeric_limits<float>::infinity()}}));
}

void testHkxConditionAndVariables() {
    using namespace odai::anim;
    std::size_t data = 0;
    auto bytes = syntheticBehaviorGraphPackfile(&data, true);
    HkxDecodedBehaviorGraph graph;
    std::string error;
    assert(decodeHkxBehaviorGraph(bytes, graph, error));
    assert(graph.variableNames == std::vector<std::string>{"Speed"});
    assert(graph.variableDefaults.at("Speed") == 2.5f);
    for (const auto& node : graph.nodes) {
        for (const auto& rule : node.transitions) {
            assert(rule.conditionClass == "hkbExpressionCondition");
            assert(rule.conditionExpression == "Speed > 1.5");
        }
        if (node.kind == HkxBehaviorNodeKind::ManualSelector) {
            assert(node.bindings.size() == 1);
            assert(node.bindings[0].memberPath == "selectedGeneratorIndex");
            assert(node.bindings[0].variableIndex == 0 && node.bindings[0].bitIndex == -1);
        }
    }
    auto corrupt = bytes;
    writeU32(corrupt, data + 0xb98u, 0xffffffffu);
    assert(!decodeHkxBehaviorGraph(corrupt, graph, error));
    corrupt = bytes;
    writeU32(corrupt, data + 0xb18u, 0u);
    assert(!decodeHkxBehaviorGraph(corrupt, graph, error));
    corrupt = bytes;
    writeF32(corrupt, data + 0xb60u, std::numeric_limits<float>::infinity());
    assert(!decodeHkxBehaviorGraph(corrupt, graph, error));
}

void testCharacterAssetReferences() {
    using namespace odai::anim;
    std::size_t data = 0;
    auto bytes = syntheticBehaviorGraphPackfile(&data);
    HkxCharacterAssets character;
    std::string error;
    assert(decodeHkxCharacterAssets(bytes, character, error));
    assert(character.skeletonPath == "custom\\skeleton.hkx");
    assert(character.behaviorPath == "behaviors\\custom.hkx");
    assert(character.animationPaths == std::vector<std::string>{"animations\\custom.hkx"});
    writeU32(bytes, data + 0xd48u, 0xffffffffu);
    assert(!decodeHkxCharacterAssets(bytes, character, error));
    assert(character.skeletonPath.empty() && character.animationPaths.empty());
}

void testLayeredCharacterDependencies() {
    using namespace odai::importer::fnv;
    const auto root = std::filesystem::temp_directory_path() /
        ("odai-fnis-layers-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto base = root / "base";
    const auto generated = root / "generated";
    const auto skeletons = root / "skeletons";
    const auto write = [](const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        assert(stream.good());
    };
    const std::string prefix = "meshes/actors/character/";
    const auto character = syntheticBehaviorGraphPackfile();
    write(base / (prefix + "characters/defaultmale.hkx"), character);
    write(base / (prefix + "characters female/defaultfemale.hkx"), character);
    write(base / (prefix + "behaviors/custom.hkx"), character);
    write(generated / (prefix + "behaviors/0_master.hkx"), character);
    write(skeletons / (prefix + "custom/skeleton.hkx"), syntheticSkeletonPackfile());
    FalloutAssetSource assets;
    assert(assets.open(base));
    assert(assets.addModDirectory(base));
    assert(assets.addModDirectory(generated));
    assert(assets.addModDirectory(skeletons));
    SkyrimAnimationAssetReport report;
    std::string error;
    assert(inspectSkyrimAnimationBundle(assets, report, false, error));
    assert(report.coherent && report.missingAssets.empty());
    // Discovery must retain catalogs even when the graph cannot execute. This
    // fixture includes both character data and a deliberately unsupported graph.
    const auto manifest = discoverCharacterAssets(assets, {
        {"meshes/actors/character/characters/defaultmale.hkx", {"NPC_:test"}}});
    assert(manifest.complete);
    assert(manifest.assets.contains("meshes\\actors\\character\\animations\\custom.hkx"));
    assert(manifest.assets.contains("meshes\\actors\\character\\animations\\male\\mt_walkforward.hkx"));
    assert(manifest.assets.at("meshes\\actors\\character\\animations\\custom.hkx").status == "missing_dependency");
    assert(!manifest.diagnostics.empty()); // custom graph references its own character behavior
    const auto bounded = discoverCharacterAssets(assets, {
        {"meshes/actors/character/characters/defaultmale.hkx", {}}}, 2);
    assert(!bounded.complete && bounded.assets.size() == 2);

    std::filesystem::remove(skeletons / (prefix + "custom/skeleton.hkx"));
    FalloutAssetSource missing;
    assert(missing.open(base));
    assert(missing.addModDirectory(base));
    assert(missing.addModDirectory(generated));
    assert(inspectSkyrimAnimationBundle(missing, report, false, error));
    assert(!report.coherent && !report.missingAssets.empty());
    std::filesystem::remove_all(root);
}

void testBoundManualSelector() {
    using namespace odai::anim;
    HkxDecodedBehaviorGraph graph;
    graph.variableNames = {"Selected"};
    graph.variableDefaults = {{"Selected", 0.f}};
    graph.nodes.resize(3);
    graph.nodes[0].kind = HkxBehaviorNodeKind::ManualSelector;
    graph.nodes[0].children = {1, 2};
    graph.nodes[0].hasBindings = true;
    graph.nodes[0].bindings = {{"selectedGeneratorIndex", 0, -1, 0}};
    graph.nodes[1].kind = graph.nodes[2].kind = HkxBehaviorNodeKind::Clip;
    graph.nodes[1].assetPath = "a";
    graph.nodes[2].assetPath = "b";
    AnimationView view;
    view.skeleton = std::make_shared<Skeleton>(makeRig());
    view.executionMode = AnimationExecutionMode::Havok;
    view.behavior = compileBehaviorProgram(graph);
    assert(view.behavior->executable());
    AnimationClip a, b;
    a.name = "a"; b.name = "b"; a.duration = b.duration = 1.f;
    view.clips = {a, b};
    BehaviorGraphInstance instance, restored;
    std::string error;
    assert(instance.bind(view, error));
    assert(instance.step({}, .1f).activeClip == "a");
    AnimationInputState input;
    input.variables = {{"Selected", 1.f}};
    assert(instance.step(input, .1f).activeClip == "b");
    assert(instance.snapshot().stateTime == .1f);
    assert(restored.bind(view, error));
    assert(restored.restore(instance.snapshot(), error));
    restored.step({}, .1f); instance.step({}, .1f);
    assert(restored.snapshot() == instance.snapshot());
    input.variables["Selected"] = 99.f;
    const auto invalid = instance.step(input, .1f);
    assert(invalid.activeClip == "b" && !invalid.diagnostics.empty());
    auto snapshot = restored.snapshot(); snapshot.activeStates[0] = -1;
    assert(!restored.restore(snapshot, error));
    graph.nodes[0].bindings[0].memberPath = "unknown";
    assert(!compileBehaviorProgram(graph)->executable());
}

void testLocomotionDirectionAndPhase() {
    using namespace odai::anim;
    AnimationView view;
    view.skeleton = std::make_shared<Skeleton>(makeRig());
    for (const auto* name : {"idle", "walk_forward", "walk_left", "run_left", "walk_left_1hm", "combat_idle_1hm"}) {
        AnimationClip clip;
        clip.name = name;
        clip.duration = std::string(name) == "run_left" ? .5f : 1.f;
        view.clips.push_back(clip);
    }
    BehaviorGraphInstance instance;
    std::string error;
    assert(instance.bind(view, error));
    AnimationInputState input;
    input.movementSpeed = 80.f;
    input.localVelocity = {0, 0, -80};
    assert(instance.step(input, .2f).activeClip == "walk_forward");
    input.localVelocity = {-80, 0, 0};
    assert(instance.step(input, .1f).activeClip == "walk_left");
    assert(std::abs(instance.snapshot().stateTime - .3f) < 1e-6f);
    input.running = true;
    assert(instance.step(input, .1f).activeClip == "run_left");
    assert(std::abs(instance.snapshot().stateTime - .25f) < 1e-6f);
    input.running = false;
    input.weaponDrawn = true;
    input.weaponStyle = "1hm";
    assert(instance.step(input, .1f).activeClip == "walk_left_1hm");
    input.weaponStyle = "bow";
    assert(instance.step(input, .1f).activeClip == "walk_left");
    input.weaponStyle = "1hm";
    input.movementSpeed = 0;
    assert(instance.step(input, .1f).activeClip == "combat_idle_1hm");
}

void testAdditiveLayerMask() {
    using namespace odai::anim;
    Skeleton rig;
    Bone root; root.name = "Root"; root.parentIndex = -1;
    Bone child; child.name = "Child"; child.parentIndex = 0; child.localTranslation = {0, 5, 0};
    rig.bones = {root, child};
    AnimationClip base, layer;
    base.duration = layer.duration = 1;
    BoneTrack baseRoot, delta;
    baseRoot.boneIndex = delta.boneIndex = 0;
    baseRoot.translationKeys = {{0, {1, 0, 0}}};
    delta.translationKeys = {{0, {2, 0, 0}}};
    base.tracks = {baseRoot}; layer.tracks = {delta}; layer.additive = true;
    AnimationSampler sampler;
    sampler.bindSkeleton(rig);
    std::vector<odai::math::Matrix4> pose;
    sampler.sampleLayered(rig, base, 0, layer, 0, 0.5f, {}, pose);
    assert(std::fabs(pose[0](0, 3) - 2.0f) < 0.001f);
    assert(std::fabs(pose[1](1, 3)) < 0.001f); // unkeyed child keeps its bind translation
    const std::array<float, 2> mask{0, 1};
    sampler.sampleLayered(rig, base, 0, layer, 0, 1, mask, pose);
    assert(std::fabs(pose[0](0, 3) - 1.0f) < 0.001f);
    sampler.sample(rig, layer, 0, pose);
    assert(std::fabs(pose[0](0, 3) - 2.0f) < 0.001f);
    assert(std::fabs(pose[1](1, 3)) < 0.001f);
}

void testActionLifetimeAndRootLoop() {
    using namespace odai::anim;
    AnimationView view;
    view.skeleton = std::make_shared<Skeleton>(makeRig());
    AnimationClip idle, attack, walk;
    idle.name = "idle"; idle.duration = 1;
    attack.name = "attack"; attack.duration = 0.5f; attack.loop = false;
    attack.annotations = {{0.25f, "HitFrame"}};
    walk.name = "locomotion"; walk.duration = 1;
    BoneTrack root;
    root.boneIndex = 0;
    root.translationKeys = {{0, {0, 0, 0}}, {1, {100, 0, 0}}};
    walk.tracks = {root};
    view.clips = {idle, attack, walk};
    BehaviorGraphInstance instance;
    std::string error;
    assert(instance.bind(view, error));
    AnimationInputState input;
    input.attacking = true;
    assert(instance.step(input, 0.1f).activeState == "attack");
    input.attacking = false;
    unsigned hits = 0;
    for (int i = 0; i < 5; ++i) {
        auto output = instance.step(input, 0.1f);
        for (const auto& event : output.events) if (event.name == "HitFrame") ++hits;
    }
    assert(hits == 1);
    assert(instance.step(input, 0.1f).activeState == "idle");
    input.movementSpeed = 100;
    input.animationDriven = true;
    instance.step(input, 0.25f);
    auto saved = instance.snapshot();
    saved.stateTime = 0.9f;
    assert(instance.restore(saved, error));
    const auto output = instance.step(input, 0.2f);
    assert(std::fabs(output.desiredRootMotion.x - 20.0f) < 0.001f);
    saved.stateTime = 0.9f;
    assert(instance.restore(saved, error));
    input.actorYawRadians = 1.57079632679f;
    const auto turned = instance.step(input, 0.2f);
    assert(std::fabs(turned.desiredRootMotion.x) < 0.001f);
    assert(std::fabs(turned.desiredRootMotion.z + 20.0f) < 0.001f);
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
    PhysicsCharacterConfig placement;
    placement.position = {0, .1f, 0};
    assert(world.isCharacterPlacementClear(placement));
    assert(world.addDynamicBody(cameraBlocker, blocker, error));
    assert(!world.isCharacterPlacementClear(placement));
    placement.position.x = 100;
    assert(world.isCharacterPlacementClear(placement));
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

void testJoltRagdollActivationAndValidatedRecovery() {
    using namespace odai::bethesda;
    BethesdaPhysicsWorld world;
    std::string error;
    assert(world.initialize(error));
    const std::vector<odai::math::Vector3> floor{
        {-500.0f, 0.0f, -500.0f}, {500.0f, 0.0f, -500.0f},
        {500.0f, 0.0f, 500.0f}, {-500.0f, 0.0f, 500.0f}};
    const std::vector<std::uint32_t> triangles{0u, 1u, 2u, 0u, 2u, 3u};
    assert(world.addStreamedStaticCollision(91u, floor, triangles, error));
    const ObjectId actor = ObjectId::runtime(901u);
    PhysicsCharacterConfig character;
    character.position = {0.0f, 80.0f, 0.0f};
    assert(world.addCharacter(actor, character, error));
    const std::vector<PhysicsRagdollJointConfig> joints{
        {"pelvis", -1, {0, 80, 0}, {}, 10, 16, 12},
        {"spine", 0, {0, 108, 0}, {}, 9, 15, 8},
        {"head", 1, {0, 136, 0}, {}, 9, 9, 5},
        {"left_thigh", 0, {-9, 54, 0}, {}, 7, 16, 7},
        {"right_thigh", 0, {9, 54, 0}, {}, 7, 16, 7}};
    assert(world.activateRagdoll(actor, joints, {30, 0, 0}, error));
    assert(world.hasActiveRagdoll(actor));
    assert(world.step(1.0f / 60.0f).empty());
    for (int tick = 0; tick < 120; ++tick) world.step(1.0f / 60.0f);
    const auto fallen = world.ragdollSnapshot(actor);
    assert(fallen && fallen->active && fallen->joints.size() == joints.size());
    odai::math::Vector3 placement;
    assert(world.recoverRagdoll(actor, 200.0f, placement, error));
    assert(!world.hasActiveRagdoll(actor));
    assert(std::fabs(placement.y) < 1.0f);
    assert(world.characterState(actor)->grounded);
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

void testMeleeTimelineAndSaveContinuation() {
    using namespace odai::bethesda;
    using namespace odai::anim;
    BethesdaSessionConfig config;
    config.game = odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition;
    config.contentFingerprint = "timeline-fixture";
    BethesdaSession session;
    std::string error;
    assert(session.configure(config, error));
    auto view = std::make_shared<AnimationView>();
    view->skeleton = std::make_shared<Skeleton>(makeRig());
    AnimationClip idle, attack;
    idle.name = "idle"; idle.duration = 1;
    attack.name = "attack"; attack.duration = 0.5f; attack.loop = false;
    attack.annotations = {{0.15f, "HitFrame"}, {0.3f, "HitFrame"}};
    view->clips = {idle, attack};
    RuntimeObject attacker, target, distant;
    attacker.id = ObjectId::runtime(201);
    attacker.base = makeRecordKey("skyrim.esm", 7);
    attacker.kind = RuntimeObjectKind::Actor;
    attacker.actorValues = ActorValues{};
    target = attacker; target.id = ObjectId::runtime(202);
    target.transform.position = {100, 0, 0};
    distant = attacker; distant.id = ObjectId::runtime(203);
    distant.transform.position = {5000, 0, 0};
    assert(session.world().addInitialObject(attacker, error));
    assert(session.world().addInitialObject(target, error));
    assert(session.world().addInitialObject(distant, error));
    PhysicsCharacterConfig physics;
    assert(session.registerActorAnimation(attacker.id, view, nullptr, physics, error));
    physics.position = {100, 0, 0};
    assert(session.registerActorController(target.id, physics, error));
    assert(session.registerActorAnimation(distant.id, view, nullptr, physics, error, false));
    assert(!session.physics().hasCharacter(distant.id));
    const std::vector<odai::math::Vector3> floor{{-1000, 0, -1000}, {1000, 0, -1000},
        {1000, 0, 1000}, {-1000, 0, 1000}};
    const std::vector<std::uint32_t> triangles{0, 1, 2, 0, 2, 3};
    assert(session.physics().addStreamedStaticCollision(1, floor, triangles, error));
    for (int i = 0; i < 60; ++i) session.advance(1.0 / 60.0);
    MeleeAttackResult requested;
    session.advance(1.0 / 60.0, [&](std::uint64_t, double) {
        requested = session.performMeleeAttack(attacker.id, {1, 0, 0}, 10, 170);
        assert(!session.performMeleeAttack(attacker.id, {1, 0, 0}, 10, 170).accepted);
    });
    assert(requested.accepted && !requested.hit);
    assert(session.world().find(target.id)->actorValues->health == 100);
    assert(session.world().find(attacker.id)->combatState->pendingMelee);
    // A script request with the same name must not masquerade as clip contact.
    session.queueActorAnimationEvent(attacker.id, {"HitFrame", {}});
    session.advance(1.0 / 60.0);
    assert(session.world().find(target.id)->actorValues->health == 100);
    const auto path = std::filesystem::temp_directory_path() / "odai-timed-melee.odai";
    assert(saveOdaiGameAtomic(path, session, error));
    for (int i = 0; i < 25; ++i) session.advance(1.0 / 60.0);
    assert(session.world().find(target.id)->actorValues->health == 90);
    assert(session.world().find(attacker.id)->combatState->hitsLanded == 1);
    SaveLoadReport report;
    assert(loadOdaiGame(path, session, {}, report, error));
    assert(session.world().find(target.id)->actorValues->health == 100);
    for (int i = 0; i < 25; ++i) session.advance(1.0 / 60.0);
    assert(session.world().find(target.id)->actorValues->health == 90);
    assert(session.world().find(attacker.id)->combatState->hitsLanded == 1);
    std::filesystem::remove(path);
}

void testEquipmentTimelinePersistence() {
    using namespace odai::bethesda;
    using namespace odai::anim;
    BethesdaSession session;
    BethesdaSessionConfig config;
    config.game = odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition;
    config.contentFingerprint = "equipment-timeline";
    std::string error;
    assert(session.configure(config, error));
    auto view = std::make_shared<AnimationView>();
    view->skeleton = std::make_shared<Skeleton>(makeRig());
    AnimationClip idle, draw, sheath;
    idle.name = "idle"; idle.duration = 1;
    draw.name = "equip"; draw.duration = .5f; draw.loop = false;
    draw.annotations = {{.2f, "weaponDraw"}, {.3f, "weaponDraw"}};
    sheath = draw; sheath.name = "unequip"; sheath.annotations = {{.2f, "weaponSheathe"}};
    view->clips = {idle, draw, sheath};
    RuntimeObject actor;
    actor.id = ObjectId::runtime(501); actor.base = makeRecordKey("skyrim.esm", 7);
    actor.kind = RuntimeObjectKind::Actor; actor.actorValues.emplace();
    assert(session.world().addInitialObject(actor, error));
    PhysicsCharacterConfig physical;
    assert(session.registerActorAnimation(actor.id, view, nullptr, physical, error, false));
    assert(session.requestActorWeaponDraw(actor.id, true, error));
    for (int i = 0; i < 6; ++i) session.advance(1. / 60.);
    assert(!session.world().find(actor.id)->equipment.drawn);
    assert(session.world().find(actor.id)->equipment.transitioning);
    const auto path = std::filesystem::temp_directory_path() / "odai-equipment-timeline.json";
    assert(saveOdaiGameAtomic(path, session, error));
    assert(session.unregisterActorAnimation(actor.id));
    assert(session.registerActorAnimation(actor.id, view, nullptr, physical, error, false));
    SaveLoadReport report;
    assert(loadOdaiGame(path, session, {}, report, error));
    for (int i = 0; i < 10; ++i) session.advance(1. / 60.);
    assert(session.world().find(actor.id)->equipment.drawn);
    assert(!session.world().find(actor.id)->equipment.transitioning);
    assert(session.requestActorWeaponDraw(actor.id, false, error));
    for (int i = 0; i < 6; ++i) session.advance(1. / 60.);
    // The old draw clip's duplicate notification cannot undo a sheath request.
    assert(session.world().find(actor.id)->equipment.drawn);
    for (int i = 0; i < 12; ++i) session.advance(1. / 60.);
    assert(!session.world().find(actor.id)->equipment.drawn);
    assert(!session.world().find(actor.id)->equipment.transitioning);
    const auto axe = makeRecordKey("fixture.esm", 0x601);
    session.setSkyrimItems({{axe, {axe, "Axe", {}, {}, 20, 0, "WEAP", 3}}});
    auto* owner = session.world().find(actor.id);
    owner->inventory = {{axe, 1, true, kEquipmentRightHand}};
    owner->equipment.initialized = true;
    assert(session.registerActorController(actor.id, physical, error));
    const auto premature = session.performEquippedMeleeAttack(actor.id, {1, 0, 0});
    assert(!premature.accepted && !premature.hit);
    assert(premature.diagnostic.find("finish drawing") != std::string::npos);
    for (int i = 0; i < 40; ++i) session.advance(1. / 60.);
    assert(session.world().find(actor.id)->equipment.drawn);
    assert(!session.world().find(actor.id)->equipment.combatDraw);
    // A manual draw remains ready outside combat. An automatic combat draw
    // returns to its sheath after the combat target clears.
    session.world().find(actor.id)->equipment.combatDraw = true;
    for (int i = 0; i < 30; ++i) session.advance(1. / 60.);
    assert(!session.world().find(actor.id)->equipment.drawn);
    assert(!session.world().find(actor.id)->equipment.combatDraw);
    std::filesystem::remove(path);
}

void testStudioJoltLocomotionTransitions() {
    using namespace odai::bethesda;
    using namespace odai::anim;
    BethesdaSession session;
    BethesdaSessionConfig config;
    config.game = odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition;
    config.contentFingerprint = "studio-jolt-test";
    config.livingWorldEnabled = false;
    std::string error;
    assert(session.configure(config, error));
    const std::vector<odai::math::Vector3> floor{
        {-700,0,-700}, {700,0,-700}, {700,0,700}, {-700,0,700}};
    const std::vector<std::uint32_t> indices{0,1,2,0,2,3};
    assert(session.physics().addStaticCollision(ObjectId::runtime(900), floor, indices, error));
    RuntimeObject actor;
    actor.id = ObjectId::runtime(901);
    actor.base = makeRecordKey("skyrim.esm", 0x7);
    actor.kind = RuntimeObjectKind::Actor;
    actor.actorValues = ActorValues{};
    assert(session.world().addInitialObject(actor, error));
    auto view = std::make_shared<AnimationView>();
    view->skeleton = std::make_shared<Skeleton>(makeRig());
    for (const std::string name : {"idle", "run_forward", "run_left", "jump", "fall", "landing"}) {
        AnimationClip clip;
        clip.name = name; clip.duration = 0.25f;
        clip.loop = name != "jump" && name != "landing";
        view->clips.push_back(clip); view->stateClips[name] = name;
    }
    assert(session.registerActorAnimation(actor.id, view, nullptr, {}, error));
    const auto tick = [&] { session.advance(1.0 / 60.0); };
    for (int i=0; i<20; ++i) tick();
    assert(session.physics().characterState(actor.id)->grounded);
    AnimationInputState animation;
    animation.running = true;
    assert(session.setActorAnimationInput(actor.id, animation));
    PhysicsCharacterInput movement;
    movement.desiredVelocity = {0,0,-230};
    animation.requestedVelocity = movement.desiredVelocity;
    assert(session.setActorAnimationInput(actor.id, animation));
    assert(session.setActorControllerInput(actor.id, movement));
    for (int i=0; i<20; ++i) tick();
    assert(session.actorAnimationOutput(actor.id)->activeState == "run_forward");
    movement.desiredVelocity = {-230,0,0};
    animation.requestedVelocity = movement.desiredVelocity;
    assert(session.setActorAnimationInput(actor.id, animation));
    assert(session.setActorControllerInput(actor.id, movement));
    // Animation observes the completed previous physics step.
    tick(); tick();
    assert(session.actorAnimationOutput(actor.id)->activeState == "run_left");
    assert(session.animationSnapshots().front().thirdPerson.transitionDuration > 0.0f);
    movement.desiredVelocity = {};
    movement.jumpRequested = true;
    animation.jumpRequested = true;
    animation.requestedVelocity = movement.desiredVelocity;
    assert(session.setActorAnimationInput(actor.id, animation));
    assert(session.setActorControllerInput(actor.id, movement));
    tick();
    movement.desiredVelocity = {};
    movement.jumpRequested = false;
    animation.jumpRequested = false;
    animation.requestedVelocity = movement.desiredVelocity;
    assert(session.setActorAnimationInput(actor.id, animation));
    assert(session.setActorControllerInput(actor.id, movement));
    bool jump=false, fall=false, landing=false;
    float highest = 0;
    for (int i=0; i<120; ++i) {
        tick();
        const auto* output = session.actorAnimationOutput(actor.id);
        jump |= output->activeState == "jump";
        fall |= output->activeState == "fall";
        landing |= output->activeState == "landing";
        highest = std::max(highest, session.physics().characterState(actor.id)->position.y);
    }
    assert(jump && fall && landing && highest > 20.0f);
    assert(session.physics().characterState(actor.id)->grounded);
    assert(std::abs(session.physics().characterState(actor.id)->position.y) < 1.0f);
    assert(session.actorAnimationOutput(actor.id)->activeState == "idle");

    odai::newvegas::NpcDemoLocomotion route;
    {
        odai::newvegas::NpcDemoLocomotion stopping;
        stopping.input(false, false, {1,0,0});
        odai::math::Vector3 velocity{230,0,0};
        stopping.input(false, false, {});
        for (int tick = 0; tick < 7; ++tick)
            velocity = stopping.step({}, velocity, 1.0f / 60.0f);
        assert(odai::math::length(velocity) == 0.0f);
    }
    assert(odai::math::length(route.target({})) == 0.0f);
    route.input(false, true, {}); // A press of 2, with no arrow input.
    route.input(true, false, {});
    assert(odai::math::length(route.target({})) == 0.0f);
    route.input(false, true, {});
    assert(odai::math::length(route.target({})) > 0.0f);
    odai::math::Vector3 velocity{};
    float angleTravel = 0.0f;
    bool circleJump = false, circleLanding = false;
    float lastAngle = 0.0f;
    for (int i = 0; i < 1440; ++i) {
        const auto before = session.physics().characterState(actor.id).value();
        const bool impulse = route.advanceJump(i == 600, before.grounded, 1.0f / 60.0f, velocity);
        if (!before.grounded || impulse) velocity = route.airborneVelocity;
        else {
            velocity = route.step(before.position, velocity, 1.0f / 60.0f);
            route.airborneVelocity = velocity;
        }
        animation.jumpPreparing = route.preparing || impulse;
        animation.requestedVelocity = velocity;
        animation.actorYawRadians = std::atan2(-velocity.x, -velocity.z);
        animation.running = true;
        animation.jumpRequested = impulse;
        if (i == 600) assert(before.grounded && impulse && !route.preparing);
        assert(session.setActorAnimationInput(actor.id, animation));
        tick();
        const auto physical = session.physics().characterState(actor.id).value();
        const auto state = session.actorAnimationOutput(actor.id)->activeState;
        const float angle = std::atan2(physical.position.z, physical.position.x);
        if (i > 240) {
            if (i < 600 || i > 780)
                assert(std::abs(std::hypot(physical.position.x, physical.position.z) - 140.0f) < 15.0f);
            const float turn = std::remainder(angle - lastAngle, 6.2831853f);
            assert(turn < 0.0f && turn > -0.06f);
            angleTravel += turn;
            if (i < 600 || i > 780) {
                assert(physical.grounded);
                assert(std::abs(physical.position.y) < 1.0f);
                assert(state.starts_with("run"));
            }
        }
        circleJump |= i >= 600 && state == "jump";
        circleLanding |= i >= 600 && state == "landing";
        lastAngle = angle;
    }
    assert(angleTravel < -4.0f * 6.2831853f);
    assert(circleJump && circleLanding);
    // A four-metre platform drop selects fall and landing from Jolt state.
    const std::vector<odai::math::Vector3> boxTop{
        {-100,280,-100}, {100,280,-100}, {100,280,100}, {-100,280,100}};
    assert(session.physics().addStaticCollision(ObjectId::runtime(902), boxTop, indices, error));
    PhysicsCharacterSnapshot elevated;
    elevated.object = actor.id; elevated.position = {0,280,0};
    elevated.groundNormal = {0,1,0};
    assert(session.physics().restoreCharacter(elevated, error));
    animation = {};
    assert(session.setActorAnimationInput(actor.id, animation));
    assert(session.setActorControllerInput(actor.id, {}));
    for (int i = 0; i < 20; ++i) tick();
    assert(session.physics().characterState(actor.id)->grounded);
    animation.requestedVelocity = {230,0,0}; animation.running = true;
    animation.jumpRequested = true;
    assert(session.setActorAnimationInput(actor.id, animation));
    tick();
    animation.jumpRequested = false;
    assert(session.setActorAnimationInput(actor.id, animation));
    bool dropped = false, dropLanded = false;
    float impact = 0;
    for (int i = 0; i < 150; ++i) {
        tick();
        const auto body = session.physics().characterState(actor.id).value();
        const auto* out = session.actorAnimationOutput(actor.id);
        dropped |= body.position.y < 250 && body.falling && out->activeState == "fall";
        if (body.landed) impact = std::max(impact, body.landingImpactMetres);
        dropLanded |= dropped && out->activeState == "landing";
    }
    assert(dropped && dropLanded && impact > 8.0f);
    {
        BehaviorGraphInstance graph;
        assert(graph.bind(*view, error));
        AnimationInputState input;
        input.running = true;
        input.movementSpeed = 230.0f;
        input.localVelocity = {0,0,-230};
        input.jumpPreparing = true;
        assert(graph.step(input, 1.0f/60.0f).activeState == "jump");
        input.jumpPreparing = false;
        input.landed = true;
        assert(graph.step(input, 1.0f/60.0f).activeState == "landing");
        input.landed = false;
        for (int i=0; i<12; ++i) graph.step(input, 1.0f/60.0f);
        assert(graph.snapshot().state == "run_forward");
    }

    route.input(true, false, {});
    for (int i = 0; i < 120; ++i) {
        velocity = route.step(session.physics().characterState(actor.id)->position, velocity, 1.0f / 60.0f);
        animation.requestedVelocity = velocity;
        assert(session.setActorAnimationInput(actor.id, animation));
        tick();
    }
    assert(odai::math::length(velocity) == 0.0f);
    assert(session.actorAnimationOutput(actor.id)->activeState == "idle");
    route.input(false, true, {});
    assert(odai::math::length(route.target({140,0,0})) > 200.0f);
    route.input(false, false, {1,0,0});
    assert(route.mode == odai::newvegas::NpcDemoLocomotion::Mode::Manual);
    assert(route.target({140,0,0}).x == 230.0f);
    route.input(false, false, {});
    assert(odai::math::length(route.target({140,0,0})) == 0.0f);
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
    auto native = std::make_shared<odai::anim::NativeAnimationProgram>();
    odai::anim::NativeAnimationRule rule;
    rule.id = "save-variant"; rule.state = "fall"; rule.provider = "fixture";
    rule.variants = {{"idle", 1, true}};
    native->rules.push_back(rule);
    third->nativeProgram = native;
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
    assert(snapshots.front().thirdPerson.selectedRule == "save-variant");
    assert(snapshots.front().thirdPerson.randomState != 0);
    assert(snapshots.front().firstPerson->actionIdentity == snapshots.front().thirdPerson.actionIdentity);
    assert(snapshots.front().thirdPerson.fixedTick == 2u &&
        snapshots.front().firstPerson->fixedTick == 2u);
    assert(snapshots.front().thirdPerson.previousState == "idle");
    assert(snapshots.front().thirdPerson.transitionDuration > 0.0f);

    const std::vector<PhysicsRagdollJointConfig> savedRagdoll{
        {"pelvis", -1, {0, 120, 0}, {}, 10, 16, 12},
        {"spine", 0, {0, 148, 0}, {}, 9, 15, 8},
        {"head", 1, {0, 176, 0}, {}, 9, 9, 5}};
    assert(session.physics().activateRagdoll(
        actor.id, savedRagdoll, {12, -3, 0}, error));

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
    assert(restored.physics().hasActiveRagdoll(actor.id));
    assert(restored.ragdollSnapshots().front().joints.size() == 3u);
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
    testAuthoredGraphExecutionAndAdmission();
    testBehaviorExpressions();
    testBoundManualSelector();
    testLocomotionDirectionAndPhase();
    testHkxConditionAndVariables();
    testCharacterAssetReferences();
    testLayeredCharacterDependencies();
    testActionLifetimeAndRootLoop();
    testAdditiveLayerMask();
    testJoltCharacterGroundingAndSnapshot();
    testJoltRagdollActivationAndValidatedRecovery();
    testJoltCharactersBlockEachOther();
    testJoltImpulseCanCarryCharacterOffLedge();
    testJoltOverlappingRetailCollisionWarningIsRecoverable();
    testStudioJoltLocomotionTransitions();
    testSessionFixedTickAndSaveContinuation();
    testEquipmentTimelinePersistence();
    testMeleeTimelineAndSaveContinuation();
    std::cout << "Skyrim animation/Jolt tests passed\n";
    return 0;
}
