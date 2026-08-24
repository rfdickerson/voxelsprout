#include "bethesda/vmad_reader.h"

#include <bit>
#include <limits>

namespace odai::bethesda {
namespace {

class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> bytes, std::size_t offset = 0u)
        : m_bytes(bytes), m_at(offset) {}
    bool u8(std::uint8_t& out) {
        if (m_at >= m_bytes.size()) return false;
        out = m_bytes[m_at++]; return true;
    }
    bool u16(std::uint16_t& out) {
        if (m_at + 2u > m_bytes.size()) return false;
        out = static_cast<std::uint16_t>(m_bytes[m_at] | (m_bytes[m_at + 1u] << 8u));
        m_at += 2u; return true;
    }
    bool i8(std::int8_t& out) {
        std::uint8_t raw = 0u;
        if (!u8(raw)) return false;
        out = std::bit_cast<std::int8_t>(raw);
        return true;
    }
    bool i16(std::int16_t& out) {
        std::uint16_t raw = 0u;
        if (!u16(raw)) return false;
        out = std::bit_cast<std::int16_t>(raw);
        return true;
    }
    bool u32(std::uint32_t& out) {
        if (m_at + 4u > m_bytes.size()) return false;
        out = 0u;
        for (std::size_t index = 0u; index < 4u; ++index)
            out |= static_cast<std::uint32_t>(m_bytes[m_at + index]) << (index * 8u);
        m_at += 4u; return true;
    }
    bool i32(std::int32_t& out) {
        std::uint32_t raw = 0u; if (!u32(raw)) return false;
        out = std::bit_cast<std::int32_t>(raw); return true;
    }
    bool f32(float& out) {
        std::uint32_t raw = 0u; if (!u32(raw)) return false;
        out = std::bit_cast<float>(raw); return true;
    }
    bool string16(std::string& out) {
        std::uint16_t size = 0u;
        if (!u16(size) || m_at + size > m_bytes.size()) return false;
        out.assign(reinterpret_cast<const char*>(m_bytes.data() + m_at), size);
        m_at += size; return true;
    }
    [[nodiscard]] std::size_t at() const { return m_at; }

private:
    std::span<const std::uint8_t> m_bytes;
    std::size_t m_at = 0u;
};

bool scalar(Reader& reader, std::uint16_t objectFormat, VmadValueType type, VmadValue& out) {
    out.type = type;
    switch (type) {
        case VmadValueType::Object: {
            std::uint16_t padding = 0u;
            if (objectFormat == 1u) {
                return reader.u32(out.object.formId) && reader.u16(out.object.alias) &&
                    reader.u16(padding);
            }
            return reader.u16(padding) && reader.u16(out.object.alias) &&
                reader.u32(out.object.formId);
        }
        case VmadValueType::String: return reader.string16(out.string);
        case VmadValueType::Integer: return reader.i32(out.integer);
        case VmadValueType::Float: return reader.f32(out.real);
        case VmadValueType::Boolean: {
            std::uint8_t value = 0u;
            if (!reader.u8(value) || value > 1u) return false;
            out.boolean = value != 0u; return true;
        }
        default: return false;
    }
}

bool value(Reader& reader, std::uint16_t objectFormat, std::uint8_t rawType, VmadValue& out) {
    if (rawType >= 1u && rawType <= 5u) {
        return scalar(reader, objectFormat, static_cast<VmadValueType>(rawType), out);
    }
    if (rawType < 11u || rawType > 15u) return false;
    out.type = static_cast<VmadValueType>(rawType);
    std::uint32_t count = 0u;
    if (!reader.u32(count) || count > 1'000'000u) return false;
    out.array.reserve(count);
    const auto elementType = static_cast<VmadValueType>(rawType - 10u);
    for (std::uint32_t index = 0u; index < count; ++index) {
        VmadValue element;
        if (!scalar(reader, objectFormat, elementType, element)) return false;
        out.array.push_back(std::move(element));
    }
    return true;
}

bool scriptAttachment(
    Reader& reader,
    std::uint16_t version,
    std::uint16_t objectFormat,
    std::uint16_t scriptIndex,
    VmadScriptAttachment& out,
    std::string& outError) {
    VmadScriptAttachment script;
    std::uint16_t propertyCount = 0u;
    if (!reader.string16(script.className) || script.className.empty() ||
        (version >= 4u && !reader.u8(script.status)) || !reader.u16(propertyCount)) {
        outError = "malformed VMAD script attachment at index " +
            std::to_string(scriptIndex);
        return false;
    }
    script.properties.reserve(propertyCount);
    for (std::uint16_t propertyIndex = 0u; propertyIndex < propertyCount; ++propertyIndex) {
        VmadProperty property;
        std::uint8_t rawType = 0u;
        const std::size_t propertyOffset = reader.at();
        if (!reader.string16(property.name) || property.name.empty() ||
            !reader.u8(rawType) || (version >= 4u && !reader.u8(property.status)) ||
            !value(reader, objectFormat, rawType, property.value)) {
            outError = "malformed VMAD property " + std::to_string(propertyIndex) +
                " on " + script.className + " at byte " +
                std::to_string(propertyOffset) + " (type " + std::to_string(rawType) + ")";
            return false;
        }
        script.properties.push_back(std::move(property));
    }
    out = std::move(script);
    return true;
}

}  // namespace

bool readVmadAttachments(
    std::span<const std::uint8_t> bytes, VmadAttachments& out, std::string& outError) {
    Reader reader(bytes);
    VmadAttachments parsed;
    std::uint16_t scriptCount = 0u;
    if (!reader.u16(parsed.version) || !reader.u16(parsed.objectFormat) ||
        !reader.u16(scriptCount) || parsed.objectFormat < 1u || parsed.objectFormat > 2u) {
        outError = "malformed VMAD header"; return false;
    }
    parsed.scripts.reserve(scriptCount);
    for (std::uint16_t scriptIndex = 0u; scriptIndex < scriptCount; ++scriptIndex) {
        VmadScriptAttachment script;
        if (!scriptAttachment(reader, parsed.version, parsed.objectFormat,
                scriptIndex, script, outError)) return false;
        parsed.scripts.push_back(std::move(script));
    }
    parsed.trailingOffset = reader.at();
    out = std::move(parsed); outError.clear(); return true;
}

bool readVmadQuestAttachments(
    std::span<const std::uint8_t> bytes,
    VmadQuestAttachments& out,
    std::string& outError) {
    VmadQuestAttachments parsed;
    if (!readVmadAttachments(bytes, parsed.common, outError)) return false;
    // QUST records are permitted to contain only the common script section;
    // in that case there is no fragment or alias count tail at all.
    if (parsed.common.trailingOffset == bytes.size()) {
        out = std::move(parsed);
        outError.clear();
        return true;
    }
    Reader reader(bytes, parsed.common.trailingOffset);
    std::uint16_t fragmentCount = 0u;
    if (!reader.i8(parsed.unknown) || !reader.u16(fragmentCount) ||
        !reader.string16(parsed.fragmentFile)) {
        outError = "malformed QUST VMAD fragment header at byte " +
            std::to_string(parsed.common.trailingOffset);
        return false;
    }
    parsed.fragments.reserve(fragmentCount);
    for (std::uint16_t index = 0u; index < fragmentCount; ++index) {
        VmadQuestFragment fragment;
        if (!reader.u16(fragment.stage) || !reader.i16(fragment.unknown1) ||
            !reader.i32(fragment.logEntry) || !reader.i8(fragment.unknown2) ||
            !reader.string16(fragment.scriptClass) || fragment.scriptClass.empty() ||
            !reader.string16(fragment.function) || fragment.function.empty()) {
            outError = "malformed QUST VMAD stage fragment at index " +
                std::to_string(index) + " byte " + std::to_string(reader.at());
            return false;
        }
        parsed.fragments.push_back(std::move(fragment));
    }
    std::uint16_t aliasCount = 0u;
    if (!reader.u16(aliasCount)) {
        outError = "truncated QUST VMAD alias count";
        return false;
    }
    parsed.aliases.reserve(aliasCount);
    for (std::uint16_t aliasIndex = 0u; aliasIndex < aliasCount; ++aliasIndex) {
        VmadQuestAliasAttachment alias;
        VmadValue object;
        std::uint16_t scriptCount = 0u;
        if (!scalar(reader, parsed.common.objectFormat, VmadValueType::Object, object) ||
            !reader.u16(alias.version) || !reader.u16(alias.objectFormat) ||
            !reader.u16(scriptCount) || alias.objectFormat < 1u || alias.objectFormat > 2u) {
            outError = "malformed QUST VMAD alias attachment at index " +
                std::to_string(aliasIndex);
            return false;
        }
        alias.object = object.object;
        alias.scripts.reserve(scriptCount);
        for (std::uint16_t scriptIndex = 0u; scriptIndex < scriptCount; ++scriptIndex) {
            VmadScriptAttachment script;
            if (!scriptAttachment(reader, alias.version, alias.objectFormat,
                    scriptIndex, script, outError)) {
                outError = "QUST alias " + std::to_string(aliasIndex) + ": " + outError;
                return false;
            }
            alias.scripts.push_back(std::move(script));
        }
        parsed.aliases.push_back(std::move(alias));
    }
    if (reader.at() != bytes.size()) {
        outError = "QUST VMAD has " + std::to_string(bytes.size() - reader.at()) +
            " unparsed trailing bytes";
        return false;
    }
    out = std::move(parsed);
    outError.clear();
    return true;
}

bool readVmadInfoAttachments(
    std::span<const std::uint8_t> bytes,
    VmadInfoAttachments& out,
    std::string& outError) {
    VmadInfoAttachments parsed;
    if (!readVmadAttachments(bytes, parsed.common, outError)) return false;
    // A common-only VMAD is valid on INFO records that attach a reusable
    // script without an inline fragment.
    if (parsed.common.trailingOffset == bytes.size()) {
        out = std::move(parsed);
        outError.clear();
        return true;
    }
    Reader reader(bytes, parsed.common.trailingOffset);
    if (!reader.i8(parsed.unknown) || !reader.u8(parsed.flags) ||
        !reader.string16(parsed.fragmentFile)) {
        outError = "malformed INFO VMAD fragment header at byte " +
            std::to_string(parsed.common.trailingOffset);
        return false;
    }
    const std::uint32_t fragmentCount = std::popcount(parsed.flags);
    parsed.fragments.reserve(fragmentCount);
    for (std::uint32_t index = 0u; index < fragmentCount; ++index) {
        VmadInfoFragment fragment;
        if (!reader.i8(fragment.unknown) ||
            !reader.string16(fragment.scriptClass) || fragment.scriptClass.empty() ||
            !reader.string16(fragment.function) || fragment.function.empty()) {
            outError = "malformed INFO VMAD fragment at index " +
                std::to_string(index) + " byte " + std::to_string(reader.at());
            return false;
        }
        parsed.fragments.push_back(std::move(fragment));
    }
    if (reader.at() != bytes.size()) {
        outError = "INFO VMAD has " + std::to_string(bytes.size() - reader.at()) +
            " unparsed trailing bytes";
        return false;
    }
    out = std::move(parsed);
    outError.clear();
    return true;
}

}  // namespace odai::bethesda
