#include "bethesda/pex_reader.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <set>

namespace odai::bethesda {
namespace {

std::string lowerAscii(std::string value) {
    for (char& ch : value) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return value;
}

class Reader {
public:
    Reader(std::span<const std::uint8_t> bytes, bool bigEndian)
        : m_bytes(bytes), m_bigEndian(bigEndian) {}

    bool u8(std::uint8_t& out) {
        if (m_at >= m_bytes.size()) return false;
        out = m_bytes[m_at++]; return true;
    }
    bool u16(std::uint16_t& out) {
        if (m_at + 2u > m_bytes.size()) return false;
        out = m_bigEndian
            ? static_cast<std::uint16_t>((m_bytes[m_at] << 8u) | m_bytes[m_at + 1u])
            : static_cast<std::uint16_t>(m_bytes[m_at] | (m_bytes[m_at + 1u] << 8u));
        m_at += 2u; return true;
    }
    bool u32(std::uint32_t& out) {
        if (m_at + 4u > m_bytes.size()) return false;
        out = 0u;
        for (std::size_t index = 0u; index < 4u; ++index) {
            const std::size_t shift = (m_bigEndian ? 3u - index : index) * 8u;
            out |= static_cast<std::uint32_t>(m_bytes[m_at + index]) << shift;
        }
        m_at += 4u; return true;
    }
    bool u64(std::uint64_t& out) {
        if (m_at + 8u > m_bytes.size()) return false;
        out = 0u;
        for (std::size_t index = 0u; index < 8u; ++index) {
            const std::size_t shift = (m_bigEndian ? 7u - index : index) * 8u;
            out |= static_cast<std::uint64_t>(m_bytes[m_at + index]) << shift;
        }
        m_at += 8u; return true;
    }
    bool i32(std::int32_t& out) {
        std::uint32_t value = 0u; if (!u32(value)) return false;
        out = std::bit_cast<std::int32_t>(value); return true;
    }
    bool f32(float& out) {
        std::uint32_t value = 0u; if (!u32(value)) return false;
        out = std::bit_cast<float>(value); return true;
    }
    bool string16(std::string& out) {
        std::uint16_t size = 0u;
        if (!u16(size) || m_at + size > m_bytes.size()) return false;
        out.assign(reinterpret_cast<const char*>(m_bytes.data() + m_at), size);
        m_at += size; return true;
    }
    [[nodiscard]] std::size_t at() const { return m_at; }
    [[nodiscard]] std::size_t remaining() const { return m_bytes.size() - m_at; }

private:
    std::span<const std::uint8_t> m_bytes;
    std::size_t m_at = 0u;
    bool m_bigEndian = false;
};

bool byteOrder(std::span<const std::uint8_t> bytes, bool& bigEndian) {
    if (bytes.size() < 4u) return false;
    bigEndian = bytes[0] == 0xfau && bytes[1] == 0x57u &&
        bytes[2] == 0xc0u && bytes[3] == 0xdeu;
    return bigEndian || (bytes[0] == 0xdeu && bytes[1] == 0xc0u &&
        bytes[2] == 0x57u && bytes[3] == 0xfau);
}

bool stringRef(Reader& reader, const PexModuleInfo& module, std::string& out) {
    std::uint16_t index = 0u;
    if (!reader.u16(index) || index >= module.strings.size()) return false;
    out = module.strings[index]; return true;
}

bool value(Reader& reader, const PexModuleInfo& module, PexValue& out) {
    std::uint8_t kind = 0u;
    if (!reader.u8(kind) || kind > static_cast<std::uint8_t>(PexValueKind::Boolean)) return false;
    out = {}; out.kind = static_cast<PexValueKind>(kind);
    switch (out.kind) {
        case PexValueKind::None: return true;
        case PexValueKind::Identifier:
        case PexValueKind::String: return stringRef(reader, module, out.text);
        case PexValueKind::Integer: return reader.i32(out.integer);
        case PexValueKind::Float: return reader.f32(out.real);
        case PexValueKind::Boolean: {
            std::uint8_t boolean = 0u;
            if (!reader.u8(boolean) || boolean > 1u) return false;
            out.boolean = boolean != 0u; return true;
        }
    }
    return false;
}

std::uint8_t fixedArgumentCount(std::uint8_t opcode) {
    if (opcode == 0u) return 0u;
    if (opcode == 20u || opcode == 26u) return 1u;
    if ((opcode >= 10u && opcode <= 14u) || opcode == 21u || opcode == 22u ||
        opcode == 30u || opcode == 31u) return 2u;
    if ((opcode >= 1u && opcode <= 9u) || (opcode >= 15u && opcode <= 19u) ||
        (opcode >= 27u && opcode <= 29u) || opcode == 32u || opcode == 33u) return 3u;
    if (opcode == 34u || opcode == 35u) return 4u;
    if (opcode == 23u) return 3u;
    if (opcode == 24u) return 2u;
    if (opcode == 25u) return 3u;
    return 0xffu;
}

bool instruction(Reader& reader, const PexModuleInfo& module, PexInstructionInfo& out) {
    if (!reader.u8(out.opcode)) return false;
    const std::uint8_t fixed = fixedArgumentCount(out.opcode);
    if (fixed == 0xffu) return false;
    for (std::uint8_t index = 0u; index < fixed; ++index) {
        PexValue argument;
        if (!value(reader, module, argument)) return false;
        out.arguments.push_back(std::move(argument));
    }
    if (out.opcode == 23u || out.opcode == 24u || out.opcode == 25u) {
        PexValue count;
        if (!value(reader, module, count) || count.kind != PexValueKind::Integer ||
            count.integer < 0 || count.integer > 4096) return false;
        out.arguments.push_back(count);
        for (std::int32_t index = 0; index < count.integer; ++index) {
            PexValue argument;
            if (!value(reader, module, argument)) return false;
            out.arguments.push_back(std::move(argument));
        }
    }
    return true;
}

bool typedNames(
    Reader& reader, const PexModuleInfo& module,
    std::vector<std::string>& outNames, std::vector<std::string>& outTypes) {
    std::uint16_t count = 0u;
    if (!reader.u16(count)) return false;
    outNames.reserve(count);
    outTypes.reserve(count);
    for (std::uint16_t index = 0u; index < count; ++index) {
        std::string name, type;
        if (!stringRef(reader, module, name) || !stringRef(reader, module, type)) return false;
        outNames.push_back(std::move(name));
        outTypes.push_back(std::move(type));
    }
    return true;
}

bool function(
    Reader& reader, const PexModuleInfo& module,
    std::string objectName, std::string stateName, std::string functionName,
    PexFunctionInfo& out) {
    out.objectName = std::move(objectName);
    out.stateName = std::move(stateName);
    out.name = std::move(functionName);
    std::string doc;
    std::uint32_t userFlags = 0u;
    if (!stringRef(reader, module, out.returnType) || !stringRef(reader, module, doc) ||
        !reader.u32(userFlags) || !reader.u8(out.flags) ||
        !typedNames(reader, module, out.parameters, out.parameterTypes) ||
        !typedNames(reader, module, out.locals, out.localTypes)) {
        return false;
    }
    std::uint16_t instructionCount = 0u;
    if (!reader.u16(instructionCount)) return false;
    out.instructions.reserve(instructionCount);
    for (std::uint16_t index = 0u; index < instructionCount; ++index) {
        PexInstructionInfo decoded;
        if (!instruction(reader, module, decoded)) return false;
        out.instructions.push_back(std::move(decoded));
    }
    return true;
}

bool variables(
    Reader& reader, const PexModuleInfo& module, std::vector<PexVariableInfo>& out) {
    std::uint16_t count = 0u;
    if (!reader.u16(count)) return false;
    out.reserve(count);
    for (std::uint16_t index = 0u; index < count; ++index) {
        PexVariableInfo decoded;
        std::uint32_t flags = 0u;
        if (!stringRef(reader, module, decoded.name) ||
            !stringRef(reader, module, decoded.type) ||
            !reader.u32(flags) || !value(reader, module, decoded.initialValue)) return false;
        out.push_back(std::move(decoded));
    }
    return true;
}

bool skipDebugAndFlags(Reader& reader, const PexModuleInfo& module) {
    std::uint8_t debug = 0u;
    if (!reader.u8(debug) || debug > 1u) return false;
    if (debug != 0u) {
        std::uint64_t time = 0u; std::uint16_t count = 0u;
        if (!reader.u64(time) || !reader.u16(count)) return false;
        for (std::uint16_t index = 0u; index < count; ++index) {
            std::string ignored; std::uint8_t type = 0u; std::uint16_t lines = 0u;
            if (!stringRef(reader, module, ignored) || !stringRef(reader, module, ignored) ||
                !stringRef(reader, module, ignored) || !reader.u8(type) || type > 2u ||
                !reader.u16(lines)) return false;
            for (std::uint16_t line = 0u, ignoredLine = 0u; line < lines; ++line) {
                if (!reader.u16(ignoredLine)) return false;
            }
        }
    }
    std::uint16_t flagCount = 0u;
    if (!reader.u16(flagCount)) return false;
    for (std::uint16_t index = 0u; index < flagCount; ++index) {
        std::string ignored; std::uint8_t bit = 0u;
        if (!stringRef(reader, module, ignored) || !reader.u8(bit)) return false;
    }
    return true;
}

}  // namespace

std::string PexFunctionInfo::qualifiedName() const {
    std::string result = objectName;
    if (!stateName.empty()) result += "." + stateName;
    return result + "." + name;
}

const char* pexOpcodeName(std::uint8_t opcode) {
    static constexpr std::array<const char*, 40> names = {
        "nop", "iadd", "fadd", "isub", "fsub", "imul", "fmul", "idiv", "fdiv", "imod",
        "not", "ineg", "fneg", "assign", "cast", "cmp_eq", "cmp_lt", "cmp_le", "cmp_gt",
        "cmp_ge", "jmp", "jmpt", "jmpf", "callmethod", "callparent", "callstatic", "ret",
        "strcat", "propget", "propset", "array_create", "array_length", "array_getelement",
        "array_setelement", "array_findelement", "array_rfindelement", "array_struct_create",
        "array_struct_get", "array_struct_set", "array_struct_find"};
    return opcode < names.size() ? names[opcode] : "unknown";
}

bool readPexModuleInfo(
    std::span<const std::uint8_t> bytes, PexModuleInfo& out, std::string& outError) {
    bool bigEndian = false;
    if (!byteOrder(bytes, bigEndian)) {
        outError = "not a compiled Papyrus PEX file (bad magic)"; return false;
    }
    Reader reader(bytes, bigEndian);
    std::uint32_t magic = 0u;
    PexModuleInfo parsed; parsed.bigEndian = bigEndian;
    if (!reader.u32(magic) || magic != 0xfa57c0deu ||
        !reader.u8(parsed.majorVersion) || !reader.u8(parsed.minorVersion) ||
        !reader.u16(parsed.gameId) || !reader.u64(parsed.compilationTime) ||
        !reader.string16(parsed.sourceFile) || !reader.string16(parsed.userName) ||
        !reader.string16(parsed.machineName)) {
        outError = "truncated Papyrus PEX header"; return false;
    }
    std::uint16_t count = 0u;
    if (!reader.u16(count)) { outError = "truncated Papyrus PEX string table"; return false; }
    parsed.strings.reserve(count);
    for (std::uint16_t index = 0u; index < count; ++index) {
        std::string item;
        if (!reader.string16(item)) { outError = "truncated Papyrus PEX string table"; return false; }
        parsed.strings.push_back(std::move(item));
    }
    parsed.bodyOffset = reader.at();
    out = std::move(parsed); outError.clear(); return true;
}

bool readPexScript(
    std::span<const std::uint8_t> bytes, PexScript& out, std::string& outError) {
    PexScript parsed;
    if (!readPexModuleInfo(bytes, parsed.module, outError)) return false;
    if (!parsed.module.bigEndian || parsed.module.gameId != 1u) {
        outError = "full PEX decoding currently supports Skyrim game ID 1 only";
        return false;
    }
    Reader reader(bytes.subspan(parsed.module.bodyOffset), parsed.module.bigEndian);
    if (!skipDebugAndFlags(reader, parsed.module)) {
        outError = "malformed PEX debug/user-flag tables"; return false;
    }
    std::uint16_t objectCount = 0u;
    if (!reader.u16(objectCount)) { outError = "truncated PEX object table"; return false; }
    for (std::uint16_t objectIndex = 0u; objectIndex < objectCount; ++objectIndex) {
        PexObjectInfo object;
        std::string ignored;
        std::uint32_t objectSize = 0u, userFlags = 0u;
        if (!stringRef(reader, parsed.module, object.name) || !reader.u32(objectSize) ||
            !stringRef(reader, parsed.module, object.parentClass) ||
            !stringRef(reader, parsed.module, ignored) ||
            !reader.u32(userFlags) || !stringRef(reader, parsed.module, object.autoState) ||
            !variables(reader, parsed.module, object.variables)) {
            outError = "malformed PEX object/variable table"; return false;
        }
        (void)objectSize;
        (void)userFlags;
        parsed.objects.push_back(object.name);
        std::uint16_t propertyCount = 0u;
        if (!reader.u16(propertyCount)) { outError = "truncated PEX property table"; return false; }
        for (std::uint16_t propertyIndex = 0u; propertyIndex < propertyCount; ++propertyIndex) {
            PexPropertyInfo property;
            std::uint32_t flags = 0u;
            if (!stringRef(reader, parsed.module, property.name) ||
                !stringRef(reader, parsed.module, property.type) ||
                !stringRef(reader, parsed.module, ignored) ||
                !reader.u32(flags) || !reader.u8(property.flags)) {
                outError = "malformed PEX property"; return false;
            }
            (void)flags;
            if ((property.flags & 0x4u) != 0u) {
                if (!stringRef(reader, parsed.module, property.autoVariable)) {
                    outError = "malformed PEX autovar"; return false;
                }
            } else {
                if ((property.flags & 0x1u) != 0u) {
                    PexFunctionInfo getter;
                    if (!function(reader, parsed.module, object.name, {}, property.name + ".get", getter)) {
                        outError = "malformed PEX property getter"; return false;
                    }
                    parsed.functions.push_back(std::move(getter));
                }
                if ((property.flags & 0x2u) != 0u) {
                    PexFunctionInfo setter;
                    if (!function(reader, parsed.module, object.name, {}, property.name + ".set", setter)) {
                        outError = "malformed PEX property setter"; return false;
                    }
                    parsed.functions.push_back(std::move(setter));
                }
            }
            object.properties.push_back(std::move(property));
        }
        std::uint16_t stateCount = 0u;
        if (!reader.u16(stateCount)) { outError = "truncated PEX state table"; return false; }
        for (std::uint16_t stateIndex = 0u; stateIndex < stateCount; ++stateIndex) {
            std::string stateName;
            std::uint16_t functionCount = 0u;
            if (!stringRef(reader, parsed.module, stateName) || !reader.u16(functionCount)) {
                outError = "malformed PEX state"; return false;
            }
            for (std::uint16_t functionIndex = 0u; functionIndex < functionCount; ++functionIndex) {
                std::string functionName;
                if (!stringRef(reader, parsed.module, functionName)) {
                    outError = "malformed PEX function name"; return false;
                }
                PexFunctionInfo decoded;
                if (!function(reader, parsed.module, object.name, stateName, functionName, decoded)) {
                    outError = "malformed PEX function " + object.name + "." + functionName; return false;
                }
                parsed.functions.push_back(std::move(decoded));
            }
        }
        parsed.objectInfo.push_back(std::move(object));
    }
    if (reader.remaining() != 0u) {
        outError = "PEX contains " + std::to_string(reader.remaining()) + " unparsed trailing bytes";
        return false;
    }
    out = std::move(parsed); outError.clear(); return true;
}

PexCompatibilityReport inspectPexCompatibility(const PexScript& script) {
    PexCompatibilityReport report;
    std::set<std::string> natives, calls, unresolved, errors;
    std::set<std::string> definedFunctions, definedMethods;
    for (const PexFunctionInfo& function : script.functions) {
        if (!function.native()) {
            definedFunctions.insert(lowerAscii(function.qualifiedName()));
            definedMethods.insert(lowerAscii(function.name));
        }
    }
    static const std::set<std::uint8_t> executable = {
        0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u,
        10u, 11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u, 19u,
        20u, 21u, 22u, 23u, 24u, 25u, 26u, 27u, 28u, 29u,
        30u, 31u, 32u, 33u, 34u, 35u};
    for (const PexFunctionInfo& function : script.functions) {
        if (function.native()) {
            natives.insert(function.qualifiedName());
            unresolved.insert("native:" + function.qualifiedName());
        }
        for (const PexInstructionInfo& instruction : function.instructions) {
            const std::string opcodeName = pexOpcodeName(instruction.opcode);
            ++report.opcodeHistogram[opcodeName];
            if (!executable.contains(instruction.opcode)) {
                errors.insert("VM does not execute PEX opcode " + opcodeName);
            }
            if (instruction.opcode == 23u && !instruction.arguments.empty()) {
                const std::string call = "method:" + instruction.arguments[0].text;
                calls.insert(call);
                if (!definedMethods.contains(lowerAscii(instruction.arguments[0].text))) {
                    unresolved.insert(call);
                }
            } else if (instruction.opcode == 24u && !instruction.arguments.empty()) {
                const std::string call = "parent:" + instruction.arguments[0].text;
                calls.insert(call);
                unresolved.insert(call);
            } else if (instruction.opcode == 25u && instruction.arguments.size() >= 2u) {
                const std::string call = instruction.arguments[0].text + "." + instruction.arguments[1].text;
                calls.insert(call);
                if (!definedFunctions.contains(lowerAscii(call))) unresolved.insert(call);
            }
        }
    }
    report.declaredNatives.assign(natives.begin(), natives.end());
    report.calledFunctions.assign(calls.begin(), calls.end());
    report.unresolvedCalls.assign(unresolved.begin(), unresolved.end());
    report.compatibilityErrors.assign(errors.begin(), errors.end());
    return report;
}

}  // namespace odai::bethesda
