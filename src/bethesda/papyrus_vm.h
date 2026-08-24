#pragma once

#include "bethesda/pex_reader.h"
#include "bethesda/runtime_ids.h"

#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace odai::bethesda {

class BethesdaWorld;

enum class PapyrusValueType : std::uint8_t {
    None,
    Integer,
    Float,
    Boolean,
    String,
    Object,
    Array,
};

struct PapyrusValue {
    PapyrusValueType type = PapyrusValueType::None;
    std::int64_t integer = 0;
    double real = 0.0;
    bool boolean = false;
    std::string string;
    ObjectId object;
    std::vector<PapyrusValue> array;

    [[nodiscard]] static PapyrusValue fromInteger(std::int64_t value);
    [[nodiscard]] static PapyrusValue fromFloat(double value);
    [[nodiscard]] static PapyrusValue fromBoolean(bool value);
    [[nodiscard]] static PapyrusValue fromString(std::string value);
    [[nodiscard]] static PapyrusValue fromObject(ObjectId value);
    [[nodiscard]] static PapyrusValue fromArray(std::vector<PapyrusValue> value = {});
    [[nodiscard]] bool truthy() const;
    friend bool operator==(const PapyrusValue&, const PapyrusValue&) = default;
};

struct PapyrusOperand {
    bool local = false;
    std::string localName;
    PapyrusValue literal;

    [[nodiscard]] static PapyrusOperand fromLocal(std::string name);
    [[nodiscard]] static PapyrusOperand fromLiteral(PapyrusValue value);
};

enum class PapyrusOpcode : std::uint8_t {
    Nop,
    Assign,
    IntegerAdd,
    FloatAdd,
    IntegerSubtract,
    FloatSubtract,
    IntegerMultiply,
    FloatMultiply,
    IntegerDivide,
    FloatDivide,
    IntegerModulo,
    LogicalNot,
    IntegerNegate,
    FloatNegate,
    Cast,
    CompareEqual,
    CompareLess,
    CompareLessEqual,
    CompareGreater,
    CompareGreaterEqual,
    Jump,
    JumpIfTrue,
    JumpIfFalse,
    CallMethod,
    CallParent,
    CallStatic,
    CallNative,
    StringConcat,
    PropertyGet,
    PropertySet,
    ArrayCreate,
    ArrayLength,
    ArrayGetElement,
    ArraySetElement,
    ArrayFindElement,
    ArrayReverseFindElement,
    WaitTicks,
    Return,
};

struct PapyrusInstruction {
    PapyrusOpcode opcode = PapyrusOpcode::Nop;
    std::string destination;
    std::string name;
    std::string targetType;
    std::vector<PapyrusOperand> operands;
    std::int32_t jumpOffset = 0;
};

struct PapyrusFunction {
    std::string name;
    std::string scriptClass;
    std::string parentClass;
    std::vector<std::string> parameters;
    std::vector<std::string> parameterTypes;
    std::unordered_map<std::string, std::string> localTypes;
    std::vector<PapyrusInstruction> instructions;
};

struct NativeCallResult {
    bool completed = true;
    PapyrusValue value;
    std::uint64_t resumeTick = 0u;
    std::string error;
};

struct PapyrusNativeContext {
    std::uint64_t currentTick = 0u;
    ObjectId self;
    std::string scriptClass;
    std::string callerFunction;
};

using NativeFunction = std::function<NativeCallResult(
    std::span<const PapyrusValue> arguments,
    std::uint64_t currentTick,
    BethesdaWorld& world)>;
using ContextNativeFunction = std::function<NativeCallResult(
    const PapyrusNativeContext& context,
    std::span<const PapyrusValue> arguments,
    BethesdaWorld& world)>;

struct PapyrusUpdateRegistrationSnapshot {
    ObjectId object;
    std::string scriptClass;
    std::string eventFunction = "onupdate";
    std::uint64_t intervalTicks = 1u;
    std::uint64_t nextTick = 0u;
    bool repeating = true;
    friend bool operator==(const PapyrusUpdateRegistrationSnapshot&,
                           const PapyrusUpdateRegistrationSnapshot&) = default;
};

struct PapyrusCallFrameSnapshot {
    std::string function;
    std::size_t instruction = 0u;
    std::string returnDestination;
    ObjectId self;
    std::string scriptClass;
    std::unordered_map<std::string, PapyrusValue> locals;
};

struct PapyrusThreadSnapshot : PapyrusCallFrameSnapshot {
    std::uint64_t id = 0u;
    std::uint64_t resumeTick = 0u;
    bool failed = false;
    std::vector<PapyrusCallFrameSnapshot> callStack;
};

struct PapyrusScriptInstanceSnapshot {
    ObjectId object;
    std::string scriptClass;
    std::string activeState;
    std::unordered_map<std::string, PapyrusValue> properties;
};

struct PapyrusVmSnapshot {
    std::uint64_t nextThreadId = 1u;
    std::vector<PapyrusThreadSnapshot> threads;
    std::unordered_map<std::string, PapyrusValue> globals;
    std::vector<PapyrusScriptInstanceSnapshot> instances;
    std::vector<PapyrusUpdateRegistrationSnapshot> updates;
};

struct PapyrusAdvanceResult {
    std::uint32_t instructions = 0u;
    std::uint32_t completedThreads = 0u;
    std::vector<std::string> diagnostics;
};

class PapyrusVm {
public:
    // Translates the executable subset of a decoded Skyrim PEX into VM
    // functions atomically. Strict mode rejects the whole module when any
    // opcode/native declaration lies outside that subset.
    bool loadPexScript(
        const PexScript& script,
        bool strict,
        PexCompatibilityReport& outReport,
        std::string& outError);
    bool registerFunction(PapyrusFunction function, std::string& outError);
    void registerNative(std::string name, NativeFunction function);
    void registerContextNative(std::string name, ContextNativeFunction function);
    void registerClassParent(std::string scriptClass, std::string parentClass);
    [[nodiscard]] std::uint64_t startFunction(
        const std::string& name,
        std::span<const PapyrusValue> arguments,
        std::string& outError);
    [[nodiscard]] std::uint64_t startFunctionOnObject(
        ObjectId self,
        const std::string& scriptClass,
        const std::string& function,
        std::span<const PapyrusValue> arguments,
        std::string& outError);
    bool attachScript(
        ObjectId object,
        std::string scriptClass,
        std::unordered_map<std::string, PapyrusValue> properties,
        std::string& outError);
    bool setProperty(
        const ObjectId& object, const std::string& scriptClass,
        const std::string& property, PapyrusValue value, std::string& outError);
    [[nodiscard]] const PapyrusValue* findProperty(
        const ObjectId& object, const std::string& scriptClass,
        const std::string& property) const;
    // CTDA GetVMQuestVariable names a property but not its attached script
    // class. Quest fragments normally have one owning instance; if several
    // expose the name, deterministic class-map order selects the first.
    [[nodiscard]] const PapyrusValue* findProperty(
        const ObjectId& object, const std::string& property) const;
    bool registerForUpdate(
        ObjectId object, std::string scriptClass, double seconds,
        std::uint64_t currentTick, bool repeating, std::string& outError,
        std::string eventFunction = "OnUpdate");
    void unregisterForUpdate(
        const ObjectId& object, const std::string& scriptClass,
        std::string eventFunction = {});
    [[nodiscard]] std::uint64_t postEvent(
        const std::string& eventFunction,
        std::span<const PapyrusValue> arguments,
        std::string& outError) {
        return startFunction(eventFunction, arguments, outError);
    }
    PapyrusAdvanceResult advance(
        std::uint64_t currentTick,
        std::uint32_t instructionBudget,
        BethesdaWorld& world);

    [[nodiscard]] bool hasFunction(const std::string& name) const;
    [[nodiscard]] bool hasScriptClass(const std::string& name) const;
    [[nodiscard]] bool hasNative(const std::string& name) const;
    [[nodiscard]] std::vector<std::string> functionsForClass(const std::string& scriptClass) const;
    [[nodiscard]] std::vector<std::string> scriptClassesForObject(ObjectId object) const;
    [[nodiscard]] std::vector<std::string> unresolvedCallBindings() const;
    [[nodiscard]] std::vector<std::string> unresolvedCallBindings(
        std::span<const std::string> rootFunctions) const;
    [[nodiscard]] std::size_t activeThreadCount() const {
        return m_threads.size() + m_pendingThreads.size();
    }
    [[nodiscard]] PapyrusVmSnapshot snapshot() const;
    bool restore(const PapyrusVmSnapshot& snapshot, std::string& outError);
    void clearRuntimeState();

private:
    [[nodiscard]] PapyrusValue operandValue(
        const PapyrusThreadSnapshot& thread, const PapyrusOperand& operand) const;
    [[nodiscard]] std::string operandDeclaredType(
        const PapyrusFunction& function, const PapyrusOperand& operand) const;
    bool enterFunction(
        PapyrusThreadSnapshot& thread, const std::string& function,
        ObjectId self, std::span<const PapyrusValue> arguments,
        std::string returnDestination, std::string& outError);

    std::unordered_map<std::string, PapyrusFunction> m_functions;
    std::unordered_map<std::string, NativeFunction> m_natives;
    std::unordered_map<std::string, ContextNativeFunction> m_contextNatives;
    std::unordered_map<std::string, PapyrusValue> m_globals;
    std::unordered_map<std::string, std::unordered_map<std::string, PapyrusValue>> m_classDefaults;
    // PEX auto-properties compile ordinary accesses to a hidden backing
    // variable (for example ::MS13_var). Map that variable back to the
    // canonical property stored on each attached instance.
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
        m_classAutoVariableProperties;
    std::unordered_map<std::string, std::string> m_classParents;
    std::unordered_map<std::string, std::string> m_classAutoStates;
    std::map<ObjectId, std::map<std::string, PapyrusScriptInstanceSnapshot>> m_instances;
    std::vector<PapyrusThreadSnapshot> m_threads;
    // Native calls may post events while advance() is iterating m_threads.
    // Stage those starts until the iteration ends to avoid iterator
    // invalidation while retaining deterministic monotonically assigned IDs.
    std::vector<PapyrusThreadSnapshot> m_pendingThreads;
    std::vector<PapyrusUpdateRegistrationSnapshot> m_updates;
    std::uint64_t m_nextThreadId = 1u;
    bool m_advancing = false;
};

}  // namespace odai::bethesda
