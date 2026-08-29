#include "anim/hkx_packfile.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>
#include <tuple>
#include <unordered_map>

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
        // These vanilla master-graph branches are recognized but intentionally
        // inert for the locomotion-only slice; their presence must not make an
        // otherwise coherent retail animation provider look mod-incompatible.
        "hkbRigidBodyRagdollControlsModifier", "hkbPoweredRagdollControlsModifier",
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
            // Havok pads cross-section tables with between 0 and 16 bytes of
            // 0xff. Some writers emit a two-word terminator; Skyrim SE retail
            // clips also use a partial final word after complete 12-byte rows.
            // Parse every complete row and require all non-row bytes to be
            // explicit padding instead of assuming one exact terminator size.
            const std::size_t entriesEnd = begin + ((end - begin) / 12u) * 12u;
            if (std::all_of(bytes.begin() + static_cast<std::ptrdiff_t>(entriesEnd),
                    bytes.begin() + static_cast<std::ptrdiff_t>(end),
                    [](std::uint8_t value) { return value == 0xffu; })) {
                return entriesEnd;
            }
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
            out.objectReferences.push_back(HkxObjectReference{
                static_cast<std::uint32_t>(index), source,
                static_cast<std::uint32_t>(index), destination});
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

namespace {

std::uint64_t referenceKey(std::uint32_t section, std::uint32_t offset) {
    return (static_cast<std::uint64_t>(section) << 32u) | offset;
}

std::size_t alignUp(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

struct ResolvedOffset {
    std::uint32_t section = 0;
    std::uint32_t offset = 0;
};

class TypedPackfileReader {
public:
    TypedPackfileReader(std::span<const std::uint8_t> bytes,
                        const HkxPackfileSummary& summary)
        : m_bytes(bytes), m_summary(summary) {
        for (const HkxObjectReference& reference : summary.objectReferences) {
            m_references.emplace(referenceKey(reference.sourceSection, reference.sourceOffset),
                ResolvedOffset{reference.targetSection, reference.targetOffset});
        }
    }

    template <typename T>
    bool read(std::uint32_t section, std::size_t offset, T& out) const {
        if (section >= m_summary.sections.size()) return false;
        const HkxSectionView& view = m_summary.sections[section];
        const std::size_t contentBytes = view.localFixups - view.dataStart;
        if (offset > contentBytes || contentBytes - offset < sizeof(T)) return false;
        std::memcpy(&out, m_bytes.data() + view.dataStart + offset, sizeof(T));
        return true;
    }

    bool bytes(std::uint32_t section, std::size_t offset, std::size_t count,
               std::span<const std::uint8_t>& out) const {
        if (section >= m_summary.sections.size()) return false;
        const HkxSectionView& view = m_summary.sections[section];
        const std::size_t contentBytes = view.localFixups - view.dataStart;
        if (offset > contentBytes || contentBytes - offset < count) return false;
        out = m_bytes.subspan(view.dataStart + offset, count);
        return true;
    }

    std::optional<ResolvedOffset> resolve(std::uint32_t section, std::size_t source) const {
        if (source > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;
        const auto found = m_references.find(
            referenceKey(section, static_cast<std::uint32_t>(source)));
        if (found == m_references.end()) return std::nullopt;
        return found->second;
    }

    bool stringAtPointer(std::uint32_t section, std::size_t pointerOffset,
                         std::size_t maxBytes, std::string& out) const {
        const auto target = resolve(section, pointerOffset);
        if (!target.has_value() || target->section >= m_summary.sections.size()) return false;
        const HkxSectionView& view = m_summary.sections[target->section];
        const std::size_t contentBytes = view.localFixups - view.dataStart;
        if (target->offset >= contentBytes) return false;
        const std::size_t available = std::min(maxBytes, contentBytes - target->offset);
        const char* begin = reinterpret_cast<const char*>(
            m_bytes.data() + view.dataStart + target->offset);
        const void* terminator = std::memchr(begin, 0, available);
        if (terminator == nullptr) return false;
        out.assign(begin, static_cast<const char*>(terminator));
        return true;
    }

    template <typename T>
    bool array(std::uint32_t section, std::size_t fieldOffset, std::uint8_t pointerSize,
               std::size_t maxCount, std::vector<T>& out) const {
        std::uint32_t count = 0;
        if (!read(section, fieldOffset + pointerSize, count) || count > maxCount) return false;
        out.clear();
        if (count == 0u) return true;
        const auto target = resolve(section, fieldOffset);
        if (!target.has_value() || count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            return false;
        }
        std::span<const std::uint8_t> raw;
        if (!bytes(target->section, target->offset, static_cast<std::size_t>(count) * sizeof(T), raw)) {
            return false;
        }
        out.resize(count);
        std::memcpy(out.data(), raw.data(), raw.size());
        return true;
    }

    bool byteArray(std::uint32_t section, std::size_t fieldOffset, std::uint8_t pointerSize,
                   std::size_t maxCount, std::span<const std::uint8_t>& out) const {
        std::uint32_t count = 0;
        if (!read(section, fieldOffset + pointerSize, count) || count > maxCount) return false;
        if (count == 0u) { out = {}; return true; }
        const auto target = resolve(section, fieldOffset);
        return target.has_value() && bytes(target->section, target->offset, count, out);
    }

    bool pointerArray(std::uint32_t section, std::size_t fieldOffset,
                      std::uint8_t pointerSize, std::size_t maxCount,
                      std::vector<ResolvedOffset>& out) const {
        std::uint32_t count = 0;
        if (!read(section, fieldOffset + pointerSize, count) || count > maxCount) return false;
        out.clear();
        if (count == 0u) return true;
        const auto target = resolve(section, fieldOffset);
        if (!target.has_value() || count > std::numeric_limits<std::size_t>::max() / pointerSize) {
            return false;
        }
        std::span<const std::uint8_t> storage;
        if (!bytes(target->section, target->offset,
                static_cast<std::size_t>(count) * pointerSize, storage)) {
            return false;
        }
        out.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            const auto item = resolve(target->section,
                static_cast<std::size_t>(target->offset) + index * pointerSize);
            if (!item.has_value()) return false;
            out.push_back(*item);
        }
        return true;
    }

private:
    std::span<const std::uint8_t> m_bytes;
    const HkxPackfileSummary& m_summary;
    std::unordered_map<std::uint64_t, ResolvedOffset> m_references;
};

class BlobCursor {
public:
    BlobCursor(std::span<const std::uint8_t> bytes, std::size_t begin, std::size_t end)
        : m_bytes(bytes), m_offset(begin), m_end(std::min(end, bytes.size())) {}

    template <typename T>
    bool read(T& out) {
        if (m_offset > m_end || m_end - m_offset < sizeof(T)) return false;
        std::memcpy(&out, m_bytes.data() + m_offset, sizeof(T));
        m_offset += sizeof(T);
        return true;
    }

    bool readBytes(std::size_t count, std::span<const std::uint8_t>& out) {
        if (m_offset > m_end || m_end - m_offset < count) return false;
        out = m_bytes.subspan(m_offset, count);
        m_offset += count;
        return true;
    }

    bool align(std::size_t alignment) {
        const std::size_t next = alignUp(m_offset, alignment);
        if (next > m_end) return false;
        m_offset = next;
        return true;
    }

    [[nodiscard]] std::size_t offset() const { return m_offset; }

private:
    std::span<const std::uint8_t> m_bytes;
    std::size_t m_offset = 0;
    std::size_t m_end = 0;
};

enum class ChannelStorage : std::uint8_t { Identity, Static, Spline };

ChannelStorage scalarStorage(std::uint8_t mask, int axis) {
    if ((mask & (1u << (axis + 4))) != 0u) return ChannelStorage::Spline;
    if ((mask & (1u << axis)) != 0u) return ChannelStorage::Static;
    return ChannelStorage::Identity;
}

ChannelStorage rotationStorage(std::uint8_t mask) {
    if ((mask & 0xf0u) != 0u) return ChannelStorage::Spline;
    if ((mask & 0x0fu) != 0u) return ChannelStorage::Static;
    return ChannelStorage::Identity;
}

struct SplineHeader {
    std::uint16_t maxControlPoint = 0;
    std::uint8_t degree = 0;
    std::vector<float> knots;
    [[nodiscard]] std::size_t controlPointCount() const {
        return static_cast<std::size_t>(maxControlPoint) + 1u;
    }
};

bool readSplineHeader(BlobCursor& cursor, SplineHeader& out, std::string& error) {
    if (!cursor.read(out.maxControlPoint) || !cursor.read(out.degree)) {
        error = "truncated HKX spline header";
        return false;
    }
    if (out.degree < 1u || out.degree > 3u || out.maxControlPoint < out.degree ||
        out.maxControlPoint > 4095u) {
        error = "invalid HKX spline degree/control-point count";
        return false;
    }
    const std::size_t knotCount = static_cast<std::size_t>(out.maxControlPoint) +
        out.degree + 2u;
    std::span<const std::uint8_t> rawKnots;
    if (!cursor.readBytes(knotCount, rawKnots)) {
        error = "truncated HKX spline knot vector";
        return false;
    }
    out.knots.assign(rawKnots.begin(), rawKnots.end());
    if (!std::is_sorted(out.knots.begin(), out.knots.end())) {
        error = "HKX spline knots are not monotonic";
        return false;
    }
    return true;
}

template <typename T, typename Mix>
T evaluateSpline(const SplineHeader& spline, float time,
                 const std::vector<T>& controlPoints, Mix&& mix) {
    const int n = static_cast<int>(controlPoints.size()) - 1;
    const int degree = spline.degree;
    if (n <= 0) return controlPoints.front();
    int span = degree;
    if (time >= spline.knots[static_cast<std::size_t>(n + 1)]) {
        span = n;
    } else if (time > spline.knots[static_cast<std::size_t>(degree)]) {
        int low = degree;
        int high = n + 1;
        while (high - low > 1) {
            const int mid = (low + high) / 2;
            if (time < spline.knots[static_cast<std::size_t>(mid)]) high = mid;
            else low = mid;
        }
        span = low;
    }
    std::array<T, 4> work{};
    for (int j = 0; j <= degree; ++j) {
        work[static_cast<std::size_t>(j)] =
            controlPoints[static_cast<std::size_t>(span - degree + j)];
    }
    for (int level = 1; level <= degree; ++level) {
        for (int j = degree; j >= level; --j) {
            const int knotIndex = span - degree + j;
            const float low = spline.knots[static_cast<std::size_t>(knotIndex)];
            const float high = spline.knots[static_cast<std::size_t>(span + 1 + j - level)];
            const float alpha = high > low ? std::clamp((time - low) / (high - low), 0.0f, 1.0f)
                                           : 0.0f;
            work[static_cast<std::size_t>(j)] = mix(
                work[static_cast<std::size_t>(j - 1)],
                work[static_cast<std::size_t>(j)], alpha);
        }
    }
    return work[static_cast<std::size_t>(degree)];
}

float mixFloat(float a, float b, float t) { return a + (b - a) * t; }

odai::math::Quaternion mixQuaternion(const odai::math::Quaternion& a,
                                     odai::math::Quaternion b, float t) {
    const float dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    if (dot < 0.0f) b = {-b.x, -b.y, -b.z, -b.w};
    return odai::math::normalize(odai::math::Quaternion{
        mixFloat(a.x, b.x, t), mixFloat(a.y, b.y, t),
        mixFloat(a.z, b.z, t), mixFloat(a.w, b.w, t)});
}

bool readQuantizedScalar(BlobCursor& cursor, std::uint8_t quantization,
                         float minimum, float maximum, float& out) {
    if (quantization == 0u) {
        std::uint8_t packed = 0;
        if (!cursor.read(packed)) return false;
        out = minimum + (maximum - minimum) * (static_cast<float>(packed) / 255.0f);
        return true;
    }
    if (quantization == 1u) {
        std::uint16_t packed = 0;
        if (!cursor.read(packed)) return false;
        out = minimum + (maximum - minimum) * (static_cast<float>(packed) / 65535.0f);
        return true;
    }
    return false;
}

odai::math::Quaternion smallestThreeQuaternion(float a, float b, float c,
                                               int omitted, bool negative) {
    const float omittedValue = (negative ? -1.0f : 1.0f) *
        std::sqrt(std::max(0.0f, 1.0f - (a * a + b * b + c * c)));
    std::array<float, 4> values{};
    int source = 0;
    for (int component = 0; component < 4; ++component) {
        values[static_cast<std::size_t>(component)] = component == omitted
            ? omittedValue : std::array<float, 3>{a, b, c}[static_cast<std::size_t>(source++)];
    }
    return odai::math::normalize(odai::math::Quaternion{
        values[0], values[1], values[2], values[3]});
}

bool readQuaternion(BlobCursor& cursor, std::uint8_t format,
                    odai::math::Quaternion& out, std::string& error) {
    static constexpr std::array<std::size_t, 6> alignment{4u, 1u, 2u, 1u, 2u, 4u};
    if (format >= alignment.size() || !cursor.align(alignment[format])) {
        error = "invalid or truncated HKX quaternion alignment";
        return false;
    }
    if (format == 0u) {
        std::uint32_t packed = 0;
        if (!cursor.read(packed)) { error = "truncated HKX polar32 quaternion"; return false; }
        constexpr std::uint32_t mask = (1u << 10u) - 1u;
        float r = static_cast<float>((packed >> 18u) & mask) / static_cast<float>(mask);
        r = 1.0f - r * r;
        const float phiTheta = static_cast<float>(packed & 0x3ffffu);
        float phi = std::floor(std::sqrt(phiTheta));
        float theta = 0.0f;
        if (phi > 0.0f) {
            theta = (3.14159265358979323846f * 0.25f) * (phiTheta - phi * phi) / phi;
            phi *= (3.14159265358979323846f * 0.5f) / 511.0f;
        }
        const float magnitude = std::sqrt(std::max(0.0f, 1.0f - r * r));
        std::array<float, 4> q{std::sin(phi) * std::cos(theta) * magnitude,
            std::sin(phi) * std::sin(theta) * magnitude, std::cos(phi) * magnitude, r};
        for (std::size_t i = 0; i < q.size(); ++i) {
            if ((packed & (0x10000000u << i)) != 0u) q[i] = -q[i];
        }
        out = odai::math::normalize(odai::math::Quaternion{q[0], q[1], q[2], q[3]});
        return true;
    }
    if (format == 1u) {
        std::span<const std::uint8_t> raw;
        if (!cursor.readBytes(5u, raw)) { error = "truncated HKX threecomp40 quaternion"; return false; }
        std::uint64_t packed = 0;
        for (std::size_t i = 0; i < raw.size(); ++i) packed |= static_cast<std::uint64_t>(raw[i]) << (i * 8u);
        constexpr float scale = 0.000345436f;
        const float a = (static_cast<float>((packed >> 0u) & 0xfffu) - 2049.0f) * scale;
        const float b = (static_cast<float>((packed >> 12u) & 0xfffu) - 2049.0f) * scale;
        const float c = (static_cast<float>((packed >> 24u) & 0xfffu) - 2049.0f) * scale;
        out = smallestThreeQuaternion(a, b, c, static_cast<int>((packed >> 36u) & 3u),
            ((packed >> 38u) & 1u) != 0u);
        return true;
    }
    if (format == 2u) {
        std::uint16_t aRaw = 0, bRaw = 0, cRaw = 0;
        if (!cursor.read(aRaw) || !cursor.read(bRaw) || !cursor.read(cRaw)) {
            error = "truncated HKX threecomp48 quaternion";
            return false;
        }
        constexpr std::uint16_t mask = 0x7fffu;
        constexpr float scale = 0.000043161f;
        const float a = (static_cast<float>(aRaw & mask) - 16383.0f) * scale;
        const float b = (static_cast<float>(bRaw & mask) - 16383.0f) * scale;
        const float c = (static_cast<float>(cRaw & mask) - 16383.0f) * scale;
        const int omitted = static_cast<int>(((bRaw >> 14u) & 2u) | ((aRaw >> 15u) & 1u));
        out = smallestThreeQuaternion(a, b, c, omitted, (cRaw & 0x8000u) != 0u);
        return true;
    }
    if (format == 5u) {
        if (!cursor.read(out.x) || !cursor.read(out.y) || !cursor.read(out.z) ||
            !cursor.read(out.w)) {
            error = "truncated HKX uncompressed quaternion";
            return false;
        }
        out = odai::math::normalize(out);
        return true;
    }
    error = "unsupported HKX rotation quantization " + std::to_string(format) +
        " (threecomp24/straight16 are not accepted)";
    return false;
}

struct DecodedTransform {
    odai::math::Vector3 translation{};
    odai::math::Quaternion rotation{};
    odai::math::Vector3 scale{1.0f, 1.0f, 1.0f};
};

bool finiteTransform(const DecodedTransform& transform) {
    return std::isfinite(transform.translation.x) && std::isfinite(transform.translation.y) &&
        std::isfinite(transform.translation.z) && std::isfinite(transform.rotation.x) &&
        std::isfinite(transform.rotation.y) && std::isfinite(transform.rotation.z) &&
        std::isfinite(transform.rotation.w) && std::isfinite(transform.scale.x) &&
        std::isfinite(transform.scale.y) && std::isfinite(transform.scale.z);
}

bool decodeVectorChannel(BlobCursor& cursor, std::uint8_t mask, std::uint8_t quantization,
                         const odai::math::Vector3& identity, std::size_t frameCount,
                         std::vector<odai::math::Vector3>& out, std::string& error) {
    if (quantization > 1u) {
        error = "unsupported HKX scalar quantization " + std::to_string(quantization);
        return false;
    }
    const bool hasSpline = scalarStorage(mask, 0) == ChannelStorage::Spline ||
        scalarStorage(mask, 1) == ChannelStorage::Spline ||
        scalarStorage(mask, 2) == ChannelStorage::Spline;
    SplineHeader spline;
    if (hasSpline && (!readSplineHeader(cursor, spline, error) || !cursor.align(4u))) return false;
    std::array<ChannelStorage, 3> storage{};
    std::array<float, 3> minimum{identity.x, identity.y, identity.z};
    std::array<float, 3> maximum = minimum;
    for (int axis = 0; axis < 3; ++axis) {
        storage[static_cast<std::size_t>(axis)] = scalarStorage(mask, axis);
        if (storage[static_cast<std::size_t>(axis)] == ChannelStorage::Spline) {
            if (!cursor.read(minimum[static_cast<std::size_t>(axis)]) ||
                !cursor.read(maximum[static_cast<std::size_t>(axis)])) {
                error = "truncated HKX scalar spline bounds";
                return false;
            }
        } else if (storage[static_cast<std::size_t>(axis)] == ChannelStorage::Static) {
            if (!cursor.read(minimum[static_cast<std::size_t>(axis)])) {
                error = "truncated HKX static scalar channel";
                return false;
            }
            maximum[static_cast<std::size_t>(axis)] = minimum[static_cast<std::size_t>(axis)];
        }
        if (!std::isfinite(minimum[static_cast<std::size_t>(axis)]) ||
            !std::isfinite(maximum[static_cast<std::size_t>(axis)])) {
            error = "non-finite HKX scalar bounds";
            return false;
        }
    }
    std::array<std::vector<float>, 3> controlPoints;
    if (hasSpline) {
        for (std::size_t point = 0; point < spline.controlPointCount(); ++point) {
            for (int axis = 0; axis < 3; ++axis) {
                if (storage[static_cast<std::size_t>(axis)] != ChannelStorage::Spline) continue;
                float value = 0.0f;
                if (!readQuantizedScalar(cursor, quantization,
                        minimum[static_cast<std::size_t>(axis)],
                        maximum[static_cast<std::size_t>(axis)], value)) {
                    error = "truncated HKX quantized scalar control points";
                    return false;
                }
                controlPoints[static_cast<std::size_t>(axis)].push_back(value);
            }
        }
    }
    if (!cursor.align(4u)) { error = "truncated HKX scalar alignment"; return false; }
    out.resize(frameCount, identity);
    for (std::size_t frame = 0; frame < frameCount; ++frame) {
        std::array<float, 3> value{identity.x, identity.y, identity.z};
        for (int axis = 0; axis < 3; ++axis) {
            if (storage[static_cast<std::size_t>(axis)] == ChannelStorage::Static) {
                value[static_cast<std::size_t>(axis)] = minimum[static_cast<std::size_t>(axis)];
            } else if (storage[static_cast<std::size_t>(axis)] == ChannelStorage::Spline) {
                value[static_cast<std::size_t>(axis)] = evaluateSpline(
                    spline, static_cast<float>(frame),
                    controlPoints[static_cast<std::size_t>(axis)], mixFloat);
            }
        }
        out[frame] = {value[0], value[1], value[2]};
    }
    return true;
}

bool decodeRotationChannel(BlobCursor& cursor, std::uint8_t mask,
                           std::uint8_t quantization, std::size_t frameCount,
                           std::vector<odai::math::Quaternion>& out,
                           std::string& error) {
    const ChannelStorage storage = rotationStorage(mask);
    out.assign(frameCount, odai::math::Quaternion{});
    if (storage == ChannelStorage::Identity) return cursor.align(4u);
    if (storage == ChannelStorage::Static) {
        odai::math::Quaternion value;
        if (!readQuaternion(cursor, quantization, value, error) || !cursor.align(4u)) return false;
        std::fill(out.begin(), out.end(), value);
        return true;
    }
    SplineHeader spline;
    if (!readSplineHeader(cursor, spline, error)) return false;
    std::vector<odai::math::Quaternion> controlPoints;
    controlPoints.reserve(spline.controlPointCount());
    for (std::size_t point = 0; point < spline.controlPointCount(); ++point) {
        odai::math::Quaternion value;
        if (!readQuaternion(cursor, quantization, value, error)) return false;
        if (!controlPoints.empty()) {
            const auto& previous = controlPoints.back();
            const float dot = previous.x * value.x + previous.y * value.y +
                previous.z * value.z + previous.w * value.w;
            if (dot < 0.0f) value = {-value.x, -value.y, -value.z, -value.w};
        }
        controlPoints.push_back(value);
    }
    if (!cursor.align(4u)) { error = "truncated HKX rotation alignment"; return false; }
    for (std::size_t frame = 0; frame < frameCount; ++frame) {
        out[frame] = evaluateSpline(spline, static_cast<float>(frame),
            controlPoints, mixQuaternion);
    }
    return true;
}

odai::math::Vector3 rebaseTranslation(const odai::math::Vector3& value) {
    return {value.x, value.z, -value.y};
}

odai::math::Vector3 rebaseScale(const odai::math::Vector3& value) {
    return {value.x, value.z, value.y};
}

odai::math::Quaternion rebaseRotation(const odai::math::Quaternion& value) {
    return odai::math::normalize(
        odai::math::Quaternion{value.x, value.z, -value.y, value.w});
}

}  // namespace

bool decodeHkxAnimationSkeleton(std::span<const std::uint8_t> bytes,
                                HkxDecodedSkeleton& outSkeleton,
                                std::string& outError,
                                const HkxReadLimits& limits) {
    outSkeleton = HkxDecodedSkeleton{};
    outError.clear();
    HkxPackfileSummary summary;
    if (!inspectHkxPackfile(bytes, summary, outError, limits)) return false;
    if (summary.pointerSize != 8u) {
        outError = "HKX skeleton decoding requires Skyrim SE x64 packfiles";
        return false;
    }
    TypedPackfileReader reader(bytes, summary);
    const HkxObjectRecord* selected = nullptr;
    std::uint32_t selectedCount = 0u;
    for (const HkxObjectRecord& object : summary.objects) {
        if (object.className != "hkaSkeleton") continue;
        std::uint32_t count = 0u;
        if (reader.read(object.section, object.offset + 0x20u, count) &&
            count > selectedCount && count <= limits.maxAnimationTracks) {
            selected = &object;
            selectedCount = count;
        }
    }
    if (selected == nullptr || selectedCount == 0u) {
        outError = "HKX has no usable hkaSkeleton object";
        return false;
    }
    (void)reader.stringAtPointer(selected->section, selected->offset + 0x10u,
        limits.maxStringBytes, outSkeleton.name);
    if (!reader.array(selected->section, selected->offset + 0x18u, 8u,
            limits.maxAnimationTracks, outSkeleton.parentIndices) ||
        outSkeleton.parentIndices.size() != selectedCount) {
        outError = "invalid hkaSkeleton parent-index array";
        return false;
    }
    std::uint32_t boneCount = 0u;
    if (!reader.read(selected->section, selected->offset + 0x30u, boneCount) ||
        boneCount != selectedCount) {
        outError = "hkaSkeleton bone count does not match parent indices";
        return false;
    }
    const auto bones = reader.resolve(selected->section, selected->offset + 0x28u);
    if (!bones.has_value()) {
        outError = "hkaSkeleton bone array has no fixup";
        return false;
    }
    constexpr std::size_t boneStride = 16u;
    outSkeleton.boneNames.resize(boneCount);
    outSkeleton.translationLocked.resize(boneCount);
    for (std::size_t index = 0; index < boneCount; ++index) {
        const std::size_t bone = bones->offset + index * boneStride;
        std::uint8_t locked = 0u;
        if (!reader.stringAtPointer(bones->section, bone, limits.maxStringBytes,
                outSkeleton.boneNames[index]) ||
            !reader.read(bones->section, bone + 8u, locked) || locked > 1u) {
            outError = "invalid hkaSkeleton bone entry " + std::to_string(index);
            return false;
        }
        const int parent = outSkeleton.parentIndices[index];
        if (parent >= static_cast<int>(index) || parent < -1) {
            outError = "hkaSkeleton parents are not stored parent-before-child";
            return false;
        }
        outSkeleton.translationLocked[index] = locked != 0u;
    }
    return true;
}

bool decodeHkxBehaviorGraph(std::span<const std::uint8_t> bytes,
                            HkxDecodedBehaviorGraph& outGraph,
                            std::string& outError,
                            const HkxReadLimits& limits) {
    outGraph = HkxDecodedBehaviorGraph{};
    outError.clear();
    HkxPackfileSummary summary;
    if (!inspectHkxPackfile(bytes, summary, outError, limits)) return false;
    if (!summary.containsBehaviorGraph || summary.pointerSize != 8u) {
        outError = "HKX has no Skyrim SE hkbBehaviorGraph";
        return false;
    }
    const auto nodeKind = [](std::string_view className)
        -> std::optional<HkxBehaviorNodeKind> {
        if (className == "hkbBehaviorGraph") return HkxBehaviorNodeKind::Graph;
        if (className == "hkbStateMachine") return HkxBehaviorNodeKind::StateMachine;
        if (className == "hkbStateMachineStateInfo") return HkxBehaviorNodeKind::State;
        if (className == "hkbClipGenerator") return HkxBehaviorNodeKind::Clip;
        if (className == "hkbBehaviorReferenceGenerator") {
            return HkxBehaviorNodeKind::BehaviorReference;
        }
        if (className == "hkbBlenderGenerator") return HkxBehaviorNodeKind::Blender;
        if (className == "hkbBlenderGeneratorChild") {
            return HkxBehaviorNodeKind::BlenderChild;
        }
        if (className == "hkbManualSelectorGenerator") {
            return HkxBehaviorNodeKind::ManualSelector;
        }
        if (className == "hkbModifierGenerator") {
            return HkxBehaviorNodeKind::ModifierGenerator;
        }
        if (className == "hkbBlendingTransitionEffect") {
            return HkxBehaviorNodeKind::TransitionEffect;
        }
        return std::nullopt;
    };

    std::vector<const HkxObjectRecord*> objects;
    objects.reserve(summary.objects.size());
    for (const HkxObjectRecord& object : summary.objects) {
        if (nodeKind(object.className).has_value()) objects.push_back(&object);
    }
    std::sort(objects.begin(), objects.end(), [](const auto* left, const auto* right) {
        return std::tie(left->section, left->offset, left->className) <
            std::tie(right->section, right->offset, right->className);
    });
    objects.erase(std::unique(objects.begin(), objects.end(), [](const auto* left,
                                                                 const auto* right) {
        return left->section == right->section && left->offset == right->offset;
    }), objects.end());
    if (objects.empty() || objects.size() > limits.maxBehaviorNodes) {
        outError = "HKX behavior node limit exceeded or graph is empty";
        return false;
    }

    TypedPackfileReader reader(bytes, summary);
    std::unordered_map<std::uint64_t, std::uint32_t> nodeByObject;
    outGraph.nodes.resize(objects.size());
    for (std::size_t index = 0; index < objects.size(); ++index) {
        nodeByObject.emplace(referenceKey(objects[index]->section, objects[index]->offset),
            static_cast<std::uint32_t>(index));
        outGraph.nodes[index].kind = *nodeKind(objects[index]->className);
    }
    std::size_t edgeCount = 0u;
    const auto appendChild = [&](HkxBehaviorNode& node, const ResolvedOffset& target) {
        const auto found = nodeByObject.find(referenceKey(target.section, target.offset));
        if (found == nodeByObject.end()) return;
        node.children.push_back(found->second);
        ++edgeCount;
    };
    const auto readName = [&](const HkxObjectRecord& object, std::size_t offset,
                              std::string& out) {
        return reader.stringAtPointer(object.section, object.offset + offset,
            limits.maxStringBytes, out);
    };
    const auto readChild = [&](const HkxObjectRecord& object, std::size_t offset,
                               HkxBehaviorNode& node) {
        const auto target = reader.resolve(object.section, object.offset + offset);
        if (target.has_value()) appendChild(node, *target);
    };
    const auto readChildren = [&](const HkxObjectRecord& object, std::size_t offset,
                                  HkxBehaviorNode& node) {
        std::vector<ResolvedOffset> targets;
        if (!reader.pointerArray(object.section, object.offset + offset, 8u,
                limits.maxBehaviorEdges, targets)) {
            return false;
        }
        for (const ResolvedOffset& target : targets) appendChild(node, target);
        return true;
    };

    bool foundGraph = false;
    for (std::size_t index = 0; index < objects.size(); ++index) {
        const HkxObjectRecord& object = *objects[index];
        HkxBehaviorNode& node = outGraph.nodes[index];
        switch (node.kind) {
            case HkxBehaviorNodeKind::Graph:
                if (!readName(object, 0x38u, node.name)) {
                    outError = "hkbBehaviorGraph has no valid name fixup";
                    return false;
                }
                readChild(object, 0x80u, node);
                if (!foundGraph) {
                    foundGraph = true;
                    outGraph.name = node.name;
                    outGraph.rootNode = static_cast<std::uint32_t>(index);
                }
                break;
            case HkxBehaviorNodeKind::StateMachine:
                (void)readName(object, 0x38u, node.name);
                if (!reader.read(object.section, object.offset + 0x68u, node.startStateId) ||
                    !readChildren(object, 0x90u, node)) {
                    outError = "invalid hkbStateMachine state array";
                    return false;
                }
                ++outGraph.stateMachineCount;
                break;
            case HkxBehaviorNodeKind::State:
                if (!readName(object, 0x60u, node.name) ||
                    !reader.read(object.section, object.offset + 0x68u, node.stateId)) {
                    outError = "invalid hkbStateMachineStateInfo";
                    return false;
                }
                readChild(object, 0x58u, node);
                break;
            case HkxBehaviorNodeKind::Clip:
                if (!readName(object, 0x38u, node.name) ||
                    !readName(object, 0x48u, node.assetPath) ||
                    !reader.read(object.section, object.offset + 0x60u, node.playbackSpeed) ||
                    !std::isfinite(node.playbackSpeed)) {
                    outError = "invalid hkbClipGenerator";
                    return false;
                }
                ++outGraph.clipGeneratorCount;
                break;
            case HkxBehaviorNodeKind::BehaviorReference:
                if (!readName(object, 0x38u, node.name) ||
                    !readName(object, 0x48u, node.assetPath)) {
                    outError = "invalid hkbBehaviorReferenceGenerator";
                    return false;
                }
                ++outGraph.behaviorReferenceCount;
                break;
            case HkxBehaviorNodeKind::Blender:
                (void)readName(object, 0x38u, node.name);
                if (!readChildren(object, 0x60u, node)) {
                    outError = "invalid hkbBlenderGenerator child array";
                    return false;
                }
                break;
            case HkxBehaviorNodeKind::BlenderChild:
                if (!reader.read(object.section, object.offset + 0x40u, node.weight) ||
                    !std::isfinite(node.weight)) {
                    outError = "invalid hkbBlenderGeneratorChild weight";
                    return false;
                }
                readChild(object, 0x30u, node);
                break;
            case HkxBehaviorNodeKind::ManualSelector:
                (void)readName(object, 0x38u, node.name);
                if (!readChildren(object, 0x48u, node)) {
                    outError = "invalid hkbManualSelectorGenerator child array";
                    return false;
                }
                break;
            case HkxBehaviorNodeKind::ModifierGenerator:
                (void)readName(object, 0x38u, node.name);
                readChild(object, 0x50u, node);
                break;
            case HkxBehaviorNodeKind::TransitionEffect:
                (void)readName(object, 0x38u, node.name);
                if (!reader.read(object.section, object.offset + 0x50u,
                        node.transitionDuration) ||
                    !std::isfinite(node.transitionDuration) || node.transitionDuration < 0.0f) {
                    outError = "invalid hkbBlendingTransitionEffect duration";
                    return false;
                }
                ++outGraph.transitionEffectCount;
                break;
        }
        if (edgeCount > limits.maxBehaviorEdges) {
            outError = "HKX behavior edge limit exceeded";
            return false;
        }
    }
    if (!foundGraph || outGraph.nodes[outGraph.rootNode].children.empty()) {
        outError = "hkbBehaviorGraph has no decodable root generator";
        outGraph = HkxDecodedBehaviorGraph{};
        return false;
    }
    return true;
}

bool decodeHkxAnimationClip(std::span<const std::uint8_t> bytes,
                            const Skeleton& targetSkeleton, std::string clipName,
                            AnimationClip& outClip, HkxDecodedClipMetadata& outMetadata,
                            std::string& outError,
                            const HkxDecodedSkeleton* sourceSkeleton,
                            const HkxReadLimits& limits) {
    outClip = AnimationClip{};
    outMetadata = HkxDecodedClipMetadata{};
    outError.clear();
    HkxPackfileSummary summary;
    if (!inspectHkxPackfile(bytes, summary, outError, limits)) return false;
    if (summary.pointerSize != 8u) {
        outError = "HKX clip decoding requires Skyrim SE x64 packfiles";
        return false;
    }
    const auto animationObject = std::find_if(summary.objects.begin(), summary.objects.end(),
        [](const HkxObjectRecord& object) {
            return object.className == "hkaSplineCompressedAnimation";
        });
    if (animationObject == summary.objects.end()) {
        outError = "HKX has no hkaSplineCompressedAnimation object";
        return false;
    }
    const std::uint32_t section = animationObject->section;
    const std::size_t object = animationObject->offset;
    const std::uint8_t pointerSize = summary.pointerSize;
    constexpr std::size_t arraySize = 16u;
    constexpr std::size_t animationBase = 16u;
    constexpr std::size_t annotationArrayOffset = animationBase + 16u + 8u;
    constexpr std::size_t splineBase = annotationArrayOffset + arraySize;
    constexpr std::size_t blockOffsetsOffset = 88u;
    constexpr std::size_t floatBlockOffsetsOffset = blockOffsetsOffset + arraySize;
    constexpr std::size_t dataOffset = blockOffsetsOffset + arraySize * 4u;
    TypedPackfileReader reader(bytes, summary);
    float duration = 0.0f;
    std::uint32_t trackCount = 0, floatTrackCount = 0, frameCount = 0, blockCount = 0,
        maxFramesPerBlock = 0, maskAndQuantizationSize = 0;
    float blockDuration = 0.0f, frameDuration = 0.0f;
    if (!reader.read(section, object + animationBase + 4u, duration) ||
        !reader.read(section, object + animationBase + 8u, trackCount) ||
        !reader.read(section, object + animationBase + 12u, floatTrackCount) ||
        !reader.read(section, object + splineBase, frameCount) ||
        !reader.read(section, object + splineBase + 4u, blockCount) ||
        !reader.read(section, object + splineBase + 8u, maxFramesPerBlock) ||
        !reader.read(section, object + splineBase + 12u, maskAndQuantizationSize) ||
        !reader.read(section, object + splineBase + 16u, blockDuration) ||
        !reader.read(section, object + splineBase + 24u, frameDuration)) {
        outError = "truncated hkaSplineCompressedAnimation header";
        return false;
    }
    if (!std::isfinite(duration) || duration <= 0.0f || !std::isfinite(blockDuration) ||
        blockDuration <= 0.0f || !std::isfinite(frameDuration) || frameDuration <= 0.0f ||
        trackCount == 0u || trackCount > limits.maxAnimationTracks || frameCount < 2u ||
        frameCount > limits.maxAnimationFrames || blockCount == 0u ||
        maxFramesPerBlock < 2u || maskAndQuantizationSize < trackCount * 4u ||
        static_cast<std::uint64_t>(trackCount) * frameCount > limits.maxDecodedTransformKeys) {
        outError = "invalid or over-limit hkaSplineCompressedAnimation dimensions";
        return false;
    }
    const std::uint64_t requiredBlocks =
        (static_cast<std::uint64_t>(frameCount) - 2u) / (maxFramesPerBlock - 1u) + 1u;
    if (requiredBlocks != blockCount) {
        outError = "HKX spline block count does not cover its declared frames";
        return false;
    }
    std::vector<std::uint32_t> blockOffsets;
    std::vector<std::uint32_t> floatBlockOffsets;
    if (!reader.array(section, object + blockOffsetsOffset, pointerSize,
            limits.maxAnimationFrames, blockOffsets) || blockOffsets.size() != blockCount ||
        !reader.array(section, object + floatBlockOffsetsOffset, pointerSize,
            limits.maxAnimationFrames, floatBlockOffsets) ||
        (!floatBlockOffsets.empty() && floatBlockOffsets.size() != blockCount)) {
        outError = "invalid HKX spline block-offset arrays";
        return false;
    }
    std::span<const std::uint8_t> blob;
    if (!reader.byteArray(section, object + dataOffset, pointerSize,
            limits.maxFileBytes, blob) || blob.empty()) {
        outError = "missing HKX spline data blob";
        return false;
    }
    if (!std::is_sorted(blockOffsets.begin(), blockOffsets.end()) ||
        blockOffsets.front() >= blob.size()) {
        outError = "HKX spline block offsets are out of order or out of bounds";
        return false;
    }

    outMetadata.frameCount = frameCount;
    outMetadata.transformTrackCount = trackCount;
    outMetadata.floatTrackCount = floatTrackCount;
    outMetadata.blockCount = blockCount;
    outMetadata.frameDuration = frameDuration;

    std::uint32_t annotationCount = 0;
    if (!reader.read(section, object + annotationArrayOffset + pointerSize, annotationCount) ||
        annotationCount > limits.maxAnimationTracks) {
        outError = "invalid HKX annotation-track array";
        return false;
    }
    outMetadata.trackNames.assign(trackCount, {});
    if (annotationCount > 0u) {
        const auto annotations = reader.resolve(section, object + annotationArrayOffset);
        if (!annotations.has_value()) {
            outError = "HKX annotation-track pointer has no fixup";
            return false;
        }
        constexpr std::size_t annotationTrackStride = 24u;
        constexpr std::size_t annotationStride = 16u;
        for (std::size_t index = 0; index < annotationCount; ++index) {
            const std::size_t track = annotations->offset + index * annotationTrackStride;
            if (index < outMetadata.trackNames.size()) {
                (void)reader.stringAtPointer(annotations->section, track,
                    limits.maxStringBytes, outMetadata.trackNames[index]);
            }
            std::uint32_t eventCount = 0;
            if (!reader.read(annotations->section, track + pointerSize + pointerSize, eventCount) ||
                eventCount > limits.maxAnimationFrames) {
                outError = "invalid HKX annotation event array";
                return false;
            }
            if (eventCount == 0u) continue;
            const auto events = reader.resolve(annotations->section, track + pointerSize);
            if (!events.has_value()) {
                outError = "HKX annotation event pointer has no fixup";
                return false;
            }
            for (std::size_t eventIndex = 0; eventIndex < eventCount; ++eventIndex) {
                const std::size_t eventOffset = events->offset + eventIndex * annotationStride;
                HkxAnimationAnnotation event;
                if (!reader.read(events->section, eventOffset, event.time) ||
                    !std::isfinite(event.time) || event.time < 0.0f ||
                    event.time > duration + frameDuration ||
                    !reader.stringAtPointer(events->section, eventOffset + pointerSize,
                        limits.maxStringBytes, event.text)) {
                    outError = "invalid HKX animation annotation";
                    return false;
                }
                outMetadata.annotations.push_back(std::move(event));
            }
        }
    }
    std::sort(outMetadata.annotations.begin(), outMetadata.annotations.end(),
        [](const HkxAnimationAnnotation& a, const HkxAnimationAnnotation& b) {
            if (a.time != b.time) return a.time < b.time;
            return a.text < b.text;
        });

    const auto bindingObject = std::find_if(summary.objects.begin(), summary.objects.end(),
        [&](const HkxObjectRecord& candidate) {
            if (candidate.className != "hkaAnimationBinding") return false;
            const auto target = reader.resolve(candidate.section, candidate.offset + 24u);
            return target.has_value() && target->section == section && target->offset == object;
        });
    if (bindingObject != summary.objects.end()) {
        (void)reader.stringAtPointer(bindingObject->section, bindingObject->offset + 16u,
            limits.maxStringBytes, outMetadata.originalSkeletonName);
        if (!reader.array(bindingObject->section, bindingObject->offset + 32u, pointerSize,
                limits.maxAnimationTracks, outMetadata.transformTrackToBoneIndices)) {
            outError = "invalid hkaAnimationBinding transform mapping";
            return false;
        }
        if (!reader.read(bindingObject->section, bindingObject->offset + 64u,
                outMetadata.blendHint) || outMetadata.blendHint > 1u) {
            outError = "invalid hkaAnimationBinding blend hint";
            return false;
        }
        if (!outMetadata.transformTrackToBoneIndices.empty() &&
            outMetadata.transformTrackToBoneIndices.size() != trackCount) {
            outError = "hkaAnimationBinding transform mapping size does not match clip";
            return false;
        }
    }

    std::vector<int> sourceBoneForTrack(trackCount, -1);
    if (sourceSkeleton != nullptr) {
        if (sourceSkeleton->boneNames.size() != sourceSkeleton->parentIndices.size() ||
            sourceSkeleton->boneNames.size() != sourceSkeleton->translationLocked.size()) {
            outError = "source HKX skeleton metadata arrays do not agree";
            return false;
        }
        for (std::size_t track = 0; track < trackCount; ++track) {
            int sourceBone = static_cast<int>(track);
            if (!outMetadata.transformTrackToBoneIndices.empty()) {
                sourceBone = outMetadata.transformTrackToBoneIndices[track];
            }
            if (sourceBone < 0 ||
                static_cast<std::size_t>(sourceBone) >= sourceSkeleton->boneNames.size()) {
                outError = "HKX clip track mapping exceeds the source skeleton";
                return false;
            }
            sourceBoneForTrack[track] = sourceBone;
            if (outMetadata.trackNames[track].empty()) {
                outMetadata.trackNames[track] =
                    sourceSkeleton->boneNames[static_cast<std::size_t>(sourceBone)];
            }
        }
    }

    std::vector<std::vector<DecodedTransform>> decoded(
        trackCount, std::vector<DecodedTransform>(frameCount));
    for (std::size_t block = 0; block < blockCount; ++block) {
        const std::size_t blockStart = blockOffsets[block];
        std::size_t blockEnd = block + 1u < blockOffsets.size()
            ? blockOffsets[block + 1u] : blob.size();
        if (!floatBlockOffsets.empty() && floatBlockOffsets[block] != 0u) {
            const std::size_t floatStart = blockStart + floatBlockOffsets[block];
            blockEnd = std::min(blockEnd, floatStart);
        }
        if (blockStart > blockEnd || blockEnd > blob.size() ||
            maskAndQuantizationSize > blockEnd - blockStart) {
            outError = "HKX spline block range is out of bounds";
            return false;
        }
        const std::size_t firstFrame = block * (static_cast<std::size_t>(maxFramesPerBlock) - 1u);
        const std::size_t framesInBlock = std::min<std::size_t>(
            maxFramesPerBlock, static_cast<std::size_t>(frameCount) - firstFrame);
        BlobCursor cursor(blob, blockStart + maskAndQuantizationSize, blockEnd);
        for (std::size_t track = 0; track < trackCount; ++track) {
            const std::size_t maskOffset = blockStart + track * 4u;
            if (maskOffset + 4u > blockStart + maskAndQuantizationSize) {
                outError = "HKX transform masks exceed mask-and-quantization header";
                return false;
            }
            const std::uint8_t packedQuantization = blob[maskOffset];
            const std::uint8_t translationMask = blob[maskOffset + 1u];
            const std::uint8_t rotationMask = blob[maskOffset + 2u];
            const std::uint8_t scaleMask = blob[maskOffset + 3u];
            const std::uint8_t translationQuantization = packedQuantization & 0x03u;
            const std::uint8_t rotationQuantization = (packedQuantization >> 2u) & 0x0fu;
            const std::uint8_t scaleQuantization = (packedQuantization >> 6u) & 0x03u;
            std::vector<odai::math::Vector3> translations;
            std::vector<odai::math::Quaternion> rotations;
            std::vector<odai::math::Vector3> scales;
            if (!decodeVectorChannel(cursor, translationMask, translationQuantization,
                    {}, framesInBlock, translations, outError) ||
                !decodeRotationChannel(cursor, rotationMask, rotationQuantization,
                    framesInBlock, rotations, outError) ||
                !decodeVectorChannel(cursor, scaleMask, scaleQuantization,
                    {1.0f, 1.0f, 1.0f}, framesInBlock, scales, outError)) {
                outError = "track " + std::to_string(track) + ": " + outError;
                return false;
            }
            for (std::size_t localFrame = 0; localFrame < framesInBlock; ++localFrame) {
                DecodedTransform& transform = decoded[track][firstFrame + localFrame];
                transform.translation = rebaseTranslation(translations[localFrame]);
                transform.rotation = rebaseRotation(rotations[localFrame]);
                transform.scale = rebaseScale(scales[localFrame]);
                if (!finiteTransform(transform)) {
                    outError = "HKX decoded a non-finite transform on track " +
                        std::to_string(track);
                    return false;
                }
            }
        }
        if (blockEnd - cursor.offset() > 15u) {
            outError = "HKX spline transform block has " +
                std::to_string(blockEnd - cursor.offset()) +
                " unexplained bytes after decoding";
            return false;
        }
    }

    std::unordered_map<std::string, int> exactBones;
    std::unordered_map<std::string, int> foldedBones;
    for (std::size_t index = 0; index < targetSkeleton.bones.size(); ++index) {
        exactBones.emplace(targetSkeleton.bones[index].name, static_cast<int>(index));
        std::string folded = targetSkeleton.bones[index].name;
        std::transform(folded.begin(), folded.end(), folded.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        foldedBones.emplace(std::move(folded), static_cast<int>(index));
    }
    outClip.name = clipName.empty() ? "retail Skyrim HKX clip" : std::move(clipName);
    outClip.duration = duration;
    outClip.loop = true;
    outClip.annotations.reserve(outMetadata.annotations.size());
    for (const HkxAnimationAnnotation& annotation : outMetadata.annotations) {
        outClip.annotations.push_back({annotation.time, annotation.text});
    }
    outClip.tracks.reserve(trackCount);
    for (std::size_t trackIndex = 0; trackIndex < trackCount; ++trackIndex) {
        int boneIndex = -1;
        if (trackIndex < outMetadata.trackNames.size() &&
            !outMetadata.trackNames[trackIndex].empty()) {
            const std::string& trackName = outMetadata.trackNames[trackIndex];
            if (const auto exact = exactBones.find(trackName); exact != exactBones.end()) {
                boneIndex = exact->second;
            } else {
                std::string folded = trackName;
                std::transform(folded.begin(), folded.end(), folded.begin(), [](unsigned char ch) {
                    return static_cast<char>(std::tolower(ch));
                });
                if (const auto match = foldedBones.find(folded); match != foldedBones.end()) {
                    boneIndex = match->second;
                }
            }
        }
        if (boneIndex < 0 && trackIndex < outMetadata.transformTrackToBoneIndices.size()) {
            const int mapped = outMetadata.transformTrackToBoneIndices[trackIndex];
            if (mapped >= 0 && static_cast<std::size_t>(mapped) < targetSkeleton.bones.size()) {
                boneIndex = mapped;
            }
        }
        if (boneIndex < 0 && outMetadata.trackNames[trackIndex].empty() &&
            trackIndex < targetSkeleton.bones.size()) {
            boneIndex = static_cast<int>(trackIndex);
        }
        if (boneIndex < 0) { ++outMetadata.missingTracks; continue; }
        BoneTrack track;
        track.boneIndex = boneIndex;
        track.translationKeys.reserve(frameCount);
        track.rotationKeys.reserve(frameCount);
        track.scaleKeys.reserve(frameCount);
        for (std::size_t frame = 0; frame < frameCount; ++frame) {
            const float time = std::min(duration, static_cast<float>(frame) * frameDuration);
            const DecodedTransform& transform = decoded[trackIndex][frame];
            odai::math::Vector3 translation = transform.translation;
            const int sourceBone = sourceBoneForTrack[trackIndex];
            if (sourceBone >= 0 && sourceSkeleton->translationLocked[
                    static_cast<std::size_t>(sourceBone)]) {
                translation = targetSkeleton.bones[
                    static_cast<std::size_t>(boneIndex)].localTranslation;
            }
            track.translationKeys.push_back({time, translation});
            track.rotationKeys.push_back({time, transform.rotation});
            track.scaleKeys.push_back({time, transform.scale});
        }
        outClip.tracks.push_back(std::move(track));
        ++outMetadata.boundTracks;
    }
    if (outClip.tracks.empty()) {
        outError = "HKX clip decoded, but none of its tracks bind to the runtime skeleton";
        outClip = AnimationClip{};
        return false;
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
