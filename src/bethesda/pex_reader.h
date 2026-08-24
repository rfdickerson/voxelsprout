#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace odai::bethesda {

struct PexModuleInfo {
    std::uint8_t majorVersion = 0u;
    std::uint8_t minorVersion = 0u;
    std::uint16_t gameId = 0u;
    std::uint64_t compilationTime = 0u;
    std::string sourceFile;
    std::string userName;
    std::string machineName;
    std::vector<std::string> strings;
    std::size_t bodyOffset = 0u;
    bool bigEndian = false;
};

enum class PexValueKind : std::uint8_t {
    None = 0u,
    Identifier = 1u,
    String = 2u,
    Integer = 3u,
    Float = 4u,
    Boolean = 5u,
};

struct PexValue {
    PexValueKind kind = PexValueKind::None;
    std::string text;
    std::int32_t integer = 0;
    float real = 0.0f;
    bool boolean = false;
};

struct PexInstructionInfo {
    std::uint8_t opcode = 0u;
    std::vector<PexValue> arguments;
};

struct PexVariableInfo {
    std::string name;
    std::string type;
    PexValue initialValue;
};

struct PexPropertyInfo {
    std::string name;
    std::string type;
    std::string autoVariable;
    std::uint8_t flags = 0u;
};

struct PexObjectInfo {
    std::string name;
    std::string parentClass;
    std::string autoState;
    std::vector<PexVariableInfo> variables;
    std::vector<PexPropertyInfo> properties;
};

struct PexFunctionInfo {
    std::string objectName;
    std::string stateName;
    std::string name;
    std::string returnType;
    std::uint8_t flags = 0u;
    std::vector<std::string> parameters;
    std::vector<std::string> parameterTypes;
    std::vector<std::string> locals;
    std::vector<std::string> localTypes;
    std::vector<PexInstructionInfo> instructions;

    [[nodiscard]] bool native() const { return (flags & 0x2u) != 0u; }
    [[nodiscard]] std::string qualifiedName() const;
};

struct PexScript {
    PexModuleInfo module;
    std::vector<std::string> objects;
    std::vector<PexObjectInfo> objectInfo;
    std::vector<PexFunctionInfo> functions;
};

struct PexCompatibilityReport {
    std::map<std::string, std::size_t> opcodeHistogram;
    std::vector<std::string> declaredNatives;
    std::vector<std::string> calledFunctions;
    std::vector<std::string> unresolvedCalls;
    std::vector<std::string> compatibilityErrors;
};

bool readPexModuleInfo(
    std::span<const std::uint8_t> bytes,
    PexModuleInfo& out,
    std::string& outError);

bool readPexScript(
    std::span<const std::uint8_t> bytes,
    PexScript& out,
    std::string& outError);

[[nodiscard]] const char* pexOpcodeName(std::uint8_t opcode);
[[nodiscard]] PexCompatibilityReport inspectPexCompatibility(const PexScript& script);

}  // namespace odai::bethesda
