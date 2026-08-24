#include "anim/hkx_packfile.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <set>
#include <string_view>

namespace odai::anim {
namespace {

constexpr std::array<std::uint8_t, 8> kPackfileMagic{
    0x57u, 0xe0u, 0xe0u, 0x57u, 0x10u, 0xc0u, 0xc0u, 0x10u};
constexpr std::size_t kHeaderBytes = 64u;
constexpr std::size_t kSectionHeaderBytes = 48u;

bool readU32(std::span<const std::uint8_t> bytes, std::size_t offset, std::uint32_t& out) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(out)) return false;
    std::memcpy(&out, bytes.data() + offset, sizeof(out));
    return true;
}

std::string fixedString(std::span<const std::uint8_t> bytes, std::size_t offset, std::size_t count) {
    if (offset > bytes.size() || bytes.size() - offset < count) return {};
    std::size_t length = 0;
    while (length < count && bytes[offset + length] != 0u && bytes[offset + length] != 0xffu) ++length;
    return std::string(reinterpret_cast<const char*>(bytes.data() + offset), length);
}

bool validClassName(std::string_view text) {
    if (text.size() < 3u || text.size() > 180u) return false;
    if (!(text.starts_with("hk") || text.starts_with("BS") || text.starts_with("FNIS") ||
          text.starts_with("Nemesis"))) return false;
    return std::all_of(text.begin(), text.end(), [](unsigned char ch) {
        return std::isalnum(ch) != 0 || ch == '_' || ch == ':' || ch == '-';
    });
}

bool contains(const std::vector<std::string>& names, std::string_view wanted) {
    return std::find(names.begin(), names.end(), wanted) != names.end();
}

bool isSupportedBehaviorClass(std::string_view name) {
    static constexpr std::array supported{
        "hkbBehaviorGraph", "hkbBehaviorGraphData", "hkbBehaviorGraphStringData",
        "hkbBehaviorReferenceGenerator", "hkbBlenderGenerator", "hkbBlenderGeneratorChild",
        "hkbBlendingTransitionEffect", "hkbBoneIndexArray", "hkbBoneWeightArray",
        "hkbCharacterData", "hkbCharacterStringData", "hkbClipGenerator",
        "hkbClipTriggerArray", "hkbDampingModifier", "hkbEvaluateExpressionModifier",
        "hkbEventDrivenModifier", "hkbEventRangeDataArray", "hkbEventsFromRangeModifier",
        "hkbExpressionCondition", "hkbExpressionDataArray", "hkbFootIkControlsModifier",
        "hkbFootIkDriverInfo", "hkbKeyframeBonesModifier",
        "hkbManualSelectorGenerator", "hkbMirroredSkeletonInfo", "hkbModifierGenerator",
        "hkbModifierList", "hkbPoseMatchingGenerator", "hkbRotateCharacterModifier",
        "hkbStateMachine", "hkbStateMachineEventPropertyArray", "hkbStateMachineStateInfo",
        "hkbStateMachineTransitionInfoArray", "hkbStringCondition", "hkbStringEventPayload",
        "hkbTimerModifier", "hkbTwistModifier", "hkbVariableBindingSet",
        "hkbVariableValueSet"};
    return std::find(supported.begin(), supported.end(), name) != supported.end();
}

}  // namespace

bool inspectHkxPackfile(std::span<const std::uint8_t> bytes, HkxPackfileSummary& out,
                        std::string& outError, const HkxReadLimits& limits) {
    out = HkxPackfileSummary{};
    outError.clear();
    if (bytes.size() > limits.maxFileBytes) {
        outError = "HKX exceeds configured byte limit";
        return false;
    }
    if (bytes.size() < kHeaderBytes ||
        !std::equal(kPackfileMagic.begin(), kPackfileMagic.end(), bytes.begin())) {
        outError = "not a Havok packfile (bad or truncated magic)";
        return false;
    }
    out.pointerSize = bytes[16];
    out.littleEndian = bytes[17] != 0u;
    if (out.pointerSize != 8u || !out.littleEndian) {
        outError = "only x64 little-endian HKX packfiles are supported";
        return false;
    }
    std::uint32_t sectionCount = 0;
    if (!readU32(bytes, 20u, sectionCount) || sectionCount == 0u ||
        sectionCount > limits.maxSections) {
        outError = "invalid HKX section count";
        return false;
    }
    if (sectionCount > (bytes.size() - kHeaderBytes) / kSectionHeaderBytes) {
        outError = "truncated HKX section table";
        return false;
    }
    out.contentsVersion = fixedString(bytes, 40u, 16u);
    if (out.contentsVersion != "hk_2010.2.0-r1") {
        outError = "unsupported HKX contents version: " + out.contentsVersion;
        return false;
    }

    out.sections.reserve(sectionCount);
    for (std::uint32_t index = 0; index < sectionCount; ++index) {
        const std::size_t base = kHeaderBytes + index * kSectionHeaderBytes;
        HkxSectionView section;
        section.name = fixedString(bytes, base, 20u);
        std::uint32_t* fields[] = {&section.dataStart, &section.localFixups, &section.globalFixups,
            &section.virtualFixups, &section.exports, &section.imports, &section.end};
        for (std::size_t field = 0; field < std::size(fields); ++field) {
            if (!readU32(bytes, base + 20u + field * 4u, *fields[field])) {
                outError = "truncated HKX section header";
                return false;
            }
        }
        // Only absoluteDataStart is absolute. Every remaining section-header
        // offset is relative to that start (which is why the final __data__
        // end plus its 0x6c0 start equals the physical file size).
        for (std::size_t fieldIndex = 1u; fieldIndex < std::size(fields); ++fieldIndex) {
            std::uint32_t* field = fields[fieldIndex];
            if (*field > bytes.size() || section.dataStart > bytes.size() - *field) {
                outError = "HKX relative section offset overflows the file";
                return false;
            }
            *field += section.dataStart;
        }
        if (section.dataStart > section.localFixups ||
            section.localFixups > section.globalFixups ||
            section.globalFixups > section.virtualFixups ||
            section.virtualFixups > section.exports || section.exports > section.imports ||
            section.imports > section.end || section.end > bytes.size()) {
            outError = "HKX section/fixup offsets are out of bounds or unordered";
            return false;
        }
        out.sections.push_back(std::move(section));
    }
    std::uint32_t contentsSection = 0u, contentsOffset = 0u;
    std::uint32_t classSection = 0u, classOffset = 0u;
    readU32(bytes, 24u, contentsSection); readU32(bytes, 28u, contentsOffset);
    readU32(bytes, 32u, classSection); readU32(bytes, 36u, classOffset);
    if (contentsSection >= out.sections.size() || classSection >= out.sections.size() ||
        contentsOffset >= out.sections[contentsSection].end - out.sections[contentsSection].dataStart ||
        classOffset >= out.sections[classSection].end - out.sections[classSection].dataStart) {
        outError = "HKX root object/class reference is out of bounds";
        return false;
    }

    // Fixup entries are section-relative object references. Validate every
    // source, destination section, and destination offset before any future
    // typed reader is allowed to follow it.
    for (std::size_t index = 0; index < out.sections.size(); ++index) {
        const HkxSectionView& section = out.sections[index];
        const std::uint32_t sectionBytes = section.end - section.dataStart;
        const auto crossFixupEnd = [&](std::size_t begin, std::size_t end) {
            const std::size_t remainder = (end - begin) % 12u;
            if (remainder == 0u) return end;
            std::uint32_t first = 0u, second = 0u;
            if (remainder == 8u && readU32(bytes, end - 8u, first) &&
                readU32(bytes, end - 4u, second) && first == 0xffffffffu &&
                second == 0xffffffffu) return end - 8u;
            return std::size_t{0u};
        };
        const std::size_t globalEntriesEnd = crossFixupEnd(
            section.globalFixups, section.virtualFixups);
        const std::size_t virtualEntriesEnd = crossFixupEnd(
            section.virtualFixups, section.exports);
        if ((section.globalFixups - section.localFixups) % 8u != 0u ||
            globalEntriesEnd == 0u || virtualEntriesEnd == 0u) {
            outError = "HKX fixup table has a truncated entry";
            return false;
        }
        for (std::size_t cursor = section.localFixups; cursor < section.globalFixups; cursor += 8u) {
            std::uint32_t source = 0, destination = 0;
            readU32(bytes, cursor, source); readU32(bytes, cursor + 4u, destination);
            if (source == 0xffffffffu) continue;
            if (source >= sectionBytes || destination >= sectionBytes) {
                outError = "HKX local fixup points outside its section";
                return false;
            }
            ++out.localFixupCount;
        }
        const auto validateCrossSection = [&](std::size_t begin, std::size_t end,
                                              bool classNameFixup) {
            for (std::size_t cursor = begin; cursor < end; cursor += 12u) {
                std::uint32_t source = 0, targetSection = 0, destination = 0;
                readU32(bytes, cursor, source); readU32(bytes, cursor + 4u, targetSection);
                readU32(bytes, cursor + 8u, destination);
                if (source == 0xffffffffu) continue;
                if (source >= sectionBytes || targetSection >= out.sections.size()) return false;
                const HkxSectionView& target = out.sections[targetSection];
                if (destination >= target.end - target.dataStart) return false;
                if (classNameFixup) {
                    const std::string className = fixedString(
                        bytes, target.dataStart + destination,
                        std::min<std::size_t>(limits.maxStringBytes,
                            target.end - target.dataStart - destination));
                    if (!validClassName(className)) return false;
                    out.objects.push_back(HkxObjectRecord{static_cast<std::uint32_t>(index),
                        source, className});
                    ++out.virtualFixupCount;
                } else {
                    out.objectReferences.push_back(HkxObjectReference{
                        static_cast<std::uint32_t>(index), source, targetSection, destination});
                    ++out.globalFixupCount;
                }
            }
            return true;
        };
        if (!validateCrossSection(section.globalFixups, globalEntriesEnd, false)) {
            outError = "HKX global fixup points outside a section";
            return false;
        }
        if (!validateCrossSection(section.virtualFixups, virtualEntriesEnd, true)) {
            outError = "HKX virtual fixup/class reference is out of bounds";
            return false;
        }
    }

    std::set<std::string> uniqueNames;
    for (const HkxSectionView& section : out.sections) {
        for (std::size_t cursor = section.dataStart; cursor < section.end;) {
            if (bytes[cursor] == 0u || bytes[cursor] == 0xffu ||
                std::isalpha(static_cast<unsigned char>(bytes[cursor])) == 0) {
                ++cursor;
                continue;
            }
            std::size_t end = cursor;
            while (end < section.end && bytes[end] != 0u &&
                   end - cursor <= limits.maxStringBytes) ++end;
            if (end < section.end) {
                const std::string_view candidate(
                    reinterpret_cast<const char*>(bytes.data() + cursor), end - cursor);
                if (validClassName(candidate)) uniqueNames.emplace(candidate);
            }
            cursor = end > cursor ? end + 1u : cursor + 1u;
        }
    }
    if (uniqueNames.size() > limits.maxClassNames) {
        outError = "HKX class-name limit exceeded";
        return false;
    }
    out.classNames.assign(uniqueNames.begin(), uniqueNames.end());
    out.containsSkeleton = contains(out.classNames, "hkaSkeleton");
    out.containsAnimation = contains(out.classNames, "hkaSplineCompressedAnimation") ||
        contains(out.classNames, "hkaAnimationBinding");
    out.containsBehaviorGraph = contains(out.classNames, "hkbBehaviorGraph");
    out.containsPhysicsMetadata = contains(out.classNames, "hkaRagdollInstance") ||
        contains(out.classNames, "hkpPhysicsSystem") ||
        contains(out.classNames, "hkbRigidBodyRagdollControlsModifier") ||
        contains(out.classNames, "hkbPoweredRagdollControlsModifier");
    for (const std::string& name : out.classNames) {
        if (name.starts_with("hkb") && !isSupportedBehaviorClass(name)) {
            out.unsupportedBehaviorClasses.push_back(name);
        }
        if (name.find("FNIS") != std::string::npos) out.generator = HkxGeneratorIdentity::Fnis;
        if (name.find("Nemesis") != std::string::npos) out.generator = HkxGeneratorIdentity::Nemesis;
    }
    if (out.generator == HkxGeneratorIdentity::Unknown &&
        (out.containsBehaviorGraph || out.containsSkeleton || out.containsAnimation)) {
        out.generator = HkxGeneratorIdentity::Vanilla;
    }
    return true;
}

const char* hkxGeneratorName(HkxGeneratorIdentity generator) {
    switch (generator) {
        case HkxGeneratorIdentity::Vanilla: return "vanilla";
        case HkxGeneratorIdentity::Fnis: return "FNIS";
        case HkxGeneratorIdentity::Nemesis: return "Nemesis";
        default: return "unknown";
    }
}

}  // namespace odai::anim
