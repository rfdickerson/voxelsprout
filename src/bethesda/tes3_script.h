#pragma once

#include "bethesda/runtime_ids.h"

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace odai::bethesda {

enum class Tes3ValueType : std::uint8_t { None, Number, String, Object };

struct Tes3Value {
    Tes3ValueType type = Tes3ValueType::None;
    double number = 0.0;
    std::string string;
    ObjectId object;

    static Tes3Value fromNumber(double value);
    static Tes3Value fromString(std::string value);
    static Tes3Value fromObject(ObjectId value);
    [[nodiscard]] bool truthy() const;
    friend bool operator==(const Tes3Value&, const Tes3Value&) = default;
};

enum class Tes3LocalType : std::uint8_t { Short, Long, Float, Reference };

enum class Tes3OpCode : std::uint8_t {
    Assign,
    Call,
    BranchIfFalse,
    Jump,
    Return,
};

struct Tes3Instruction {
    Tes3OpCode op = Tes3OpCode::Return;
    std::uint32_t sourceLine = 0u;
    std::string destination;
    std::string target;
    std::string command;
    std::vector<std::string> arguments;
    std::string expression;
    std::size_t jump = 0u;
};

struct Tes3ScriptProgram {
    std::string id;
    std::map<std::string, Tes3LocalType> locals;
    std::vector<Tes3Instruction> instructions;
    std::set<std::string> commands;
    std::uint64_t sourceHash = 0u;
};

struct Tes3CompileDiagnostic {
    std::uint32_t line = 0u;
    bool error = true;
    std::string message;
};

struct Tes3CompileResult {
    Tes3ScriptProgram program;
    std::vector<Tes3CompileDiagnostic> diagnostics;
    [[nodiscard]] bool success() const;
};

class Tes3ScriptCompiler {
public:
    [[nodiscard]] Tes3CompileResult compile(
        std::string_view source, std::string scriptId = {}) const;
};

enum class Tes3NativeDisposition : std::uint8_t {
    Implemented,
    PresentationOnly,
    Unsupported,
};

struct Tes3NativeDefinition {
    std::string name;
    Tes3NativeDisposition disposition = Tes3NativeDisposition::Unsupported;
    bool gameplayAffecting = true;
};

class Tes3NativeRegistry {
public:
    void registerNative(Tes3NativeDefinition definition);
    [[nodiscard]] const Tes3NativeDefinition* find(std::string_view command) const;
    [[nodiscard]] const std::map<std::string, Tes3NativeDefinition>& definitions() const {
        return m_definitions;
    }
    [[nodiscard]] static Tes3NativeRegistry coreRuntimeRegistry();

private:
    std::map<std::string, Tes3NativeDefinition> m_definitions;
};

enum class Tes3ThreadState : std::uint8_t {
    Running,
    Suspended,
    Completed,
    Failed,
};

struct Tes3ScriptThread {
    std::uint64_t id = 0u;
    std::string program;
    ObjectId owner;
    std::size_t instruction = 0u;
    std::map<std::string, Tes3Value> locals;
    std::map<std::string, Tes3Value> eventVariables;
    Tes3ThreadState state = Tes3ThreadState::Running;
    std::string suspensionReason;
    std::string error;
    friend bool operator==(const Tes3ScriptThread&, const Tes3ScriptThread&) = default;
};

struct Tes3NativeCall {
    std::string target;
    std::string command;
    std::vector<Tes3Value> arguments;
    std::uint64_t tick = 0u;
    ObjectId owner;
};

struct Tes3NativeResult {
    Tes3Value value;
    bool suspend = false;
    std::string suspensionReason;
    std::string error;
};

using Tes3NativeExecutor = std::function<Tes3NativeResult(const Tes3NativeCall&)>;

struct Tes3VmStepResult {
    std::uint32_t instructions = 0u;
    std::uint32_t completedThreads = 0u;
    std::vector<std::string> diagnostics;
};

class Tes3ScriptVm {
public:
    bool registerProgram(Tes3ScriptProgram program, std::string& outError);
    [[nodiscard]] std::uint64_t start(
        std::string_view program, ObjectId owner, std::string& outError);
    [[nodiscard]] Tes3VmStepResult step(
        std::uint64_t tick, std::uint32_t instructionBudget,
        const Tes3NativeExecutor& execute);
    bool resume(std::uint64_t threadId, std::string& outError);
    void clear();

    [[nodiscard]] const std::map<std::string, Tes3ScriptProgram>& programs() const {
        return m_programs;
    }
    [[nodiscard]] const std::map<std::uint64_t, Tes3ScriptThread>& threads() const {
        return m_threads;
    }
    [[nodiscard]] std::map<std::uint64_t, Tes3ScriptThread>& threadsForRestore() {
        return m_threads;
    }
    [[nodiscard]] std::map<std::string, Tes3Value>& globals() { return m_globals; }
    [[nodiscard]] const std::map<std::string, Tes3Value>& globals() const { return m_globals; }
    [[nodiscard]] std::uint64_t nextThreadId() const { return m_nextThreadId; }
    void setNextThreadId(std::uint64_t value) { m_nextThreadId = value == 0u ? 1u : value; }

private:
    std::optional<Tes3Value> evaluate(
        std::string_view expression, Tes3ScriptThread& thread,
        std::uint64_t tick, const Tes3NativeExecutor& execute,
        std::string& outError) const;
    Tes3Value lookup(std::string_view name, const Tes3ScriptThread& thread) const;

    std::map<std::string, Tes3ScriptProgram> m_programs;
    std::map<std::uint64_t, Tes3ScriptThread> m_threads;
    std::map<std::string, Tes3Value> m_globals;
    std::uint64_t m_nextThreadId = 1u;
};

[[nodiscard]] std::string normalizeTes3Symbol(std::string_view symbol);

}  // namespace odai::bethesda
