#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace odai::bethesda {

enum class VmadValueType : std::uint8_t {
    Object = 1u,
    String = 2u,
    Integer = 3u,
    Float = 4u,
    Boolean = 5u,
    ObjectArray = 11u,
    StringArray = 12u,
    IntegerArray = 13u,
    FloatArray = 14u,
    BooleanArray = 15u,
};

struct VmadObjectValue {
    std::uint32_t formId = 0u;
    std::uint16_t alias = 0xffffu;
};

struct VmadValue {
    VmadValueType type = VmadValueType::Object;
    VmadObjectValue object;
    std::string string;
    std::int32_t integer = 0;
    float real = 0.0f;
    bool boolean = false;
    std::vector<VmadValue> array;
};

struct VmadProperty {
    std::string name;
    std::uint8_t status = 0u;
    VmadValue value;
};

struct VmadScriptAttachment {
    std::string className;
    std::uint8_t status = 0u;
    std::vector<VmadProperty> properties;
};

struct VmadAttachments {
    std::uint16_t version = 0u;
    std::uint16_t objectFormat = 0u;
    std::vector<VmadScriptAttachment> scripts;
    std::size_t trailingOffset = 0u;
};

struct VmadQuestFragment {
    std::uint16_t stage = 0u;
    std::int16_t unknown1 = 0;
    std::int32_t logEntry = 0;
    std::int8_t unknown2 = 0;
    std::string scriptClass;
    std::string function;
};

struct VmadQuestAliasAttachment {
    VmadObjectValue object;
    std::uint16_t version = 0u;
    std::uint16_t objectFormat = 0u;
    std::vector<VmadScriptAttachment> scripts;
};

struct VmadQuestAttachments {
    VmadAttachments common;
    std::int8_t unknown = 0;
    std::string fragmentFile;
    std::vector<VmadQuestFragment> fragments;
    std::vector<VmadQuestAliasAttachment> aliases;
};

// TES5 INFO VMADs add a flags-selected fragment list after the common script
// section. Bit 0 is the begin/result fragment and bit 1 is the end fragment;
// the format itself stays generic by preserving the authored order and flags.
struct VmadInfoFragment {
    std::int8_t unknown = 0;
    std::string scriptClass;
    std::string function;
};

struct VmadInfoAttachments {
    VmadAttachments common;
    std::int8_t unknown = 0;
    std::uint8_t flags = 0u;
    std::string fragmentFile;
    std::vector<VmadInfoFragment> fragments;
};

// Parses the common VMAD attachment prefix. Record-specific quest fragments,
// aliases, scenes, and package payloads remain at trailingOffset for their
// compiled generation adapter to decode.
bool readVmadAttachments(
    std::span<const std::uint8_t> bytes,
    VmadAttachments& out,
    std::string& outError);

// Parses the complete TES5 QUST VMAD payload: common scripts, stage
// fragments, and alias script attachments. Any unknown tail is rejected so a
// malformed or newer layout cannot silently become missing quest behavior.
bool readVmadQuestAttachments(
    std::span<const std::uint8_t> bytes,
    VmadQuestAttachments& out,
    std::string& outError);

// Parses the complete TES5 INFO VMAD payload and rejects unknown trailing
// bytes. This is the authored effect path for dialogue choices and responses.
bool readVmadInfoAttachments(
    std::span<const std::uint8_t> bytes,
    VmadInfoAttachments& out,
    std::string& outError);

}  // namespace odai::bethesda
