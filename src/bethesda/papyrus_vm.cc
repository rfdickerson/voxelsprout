#include "bethesda/papyrus_vm.h"

#include "bethesda/runtime_world.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>

namespace odai::bethesda {
namespace {

constexpr std::size_t kMaximumArrayElements = 1u << 20u;
constexpr std::size_t kMaximumCallDepth = 256u;

std::string normalizedSymbol(std::string name) {
    for (char& ch : name) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return name;
}

PapyrusValue pexLiteral(const PexValue& value) {
    switch (value.kind) {
        case PexValueKind::String: return PapyrusValue::fromString(value.text);
        case PexValueKind::Integer: return PapyrusValue::fromInteger(value.integer);
        case PexValueKind::Float: return PapyrusValue::fromFloat(value.real);
        case PexValueKind::Boolean: return PapyrusValue::fromBoolean(value.boolean);
        case PexValueKind::None:
        case PexValueKind::Identifier: return {};
    }
    return {};
}

PapyrusOperand translatedOperand(const PexValue& value) {
    if (value.kind == PexValueKind::Identifier) {
        return PapyrusOperand::fromLocal(normalizedSymbol(value.text));
    }
    return PapyrusOperand::fromLiteral(pexLiteral(value));
}

bool numeric(const PapyrusValue& value, double& out) {
    if (value.type == PapyrusValueType::Integer) {
        out = static_cast<double>(value.integer);
        return true;
    }
    if (value.type == PapyrusValueType::Float) {
        out = value.real;
        return true;
    }
    return false;
}

std::string valueString(const PapyrusValue& value) {
    switch (value.type) {
        case PapyrusValueType::None: return "None";
        case PapyrusValueType::Integer: return std::to_string(value.integer);
        case PapyrusValueType::Float: {
            std::ostringstream out;
            out.precision(9);
            out << value.real;
            return out.str();
        }
        case PapyrusValueType::Boolean: return value.boolean ? "True" : "False";
        case PapyrusValueType::String: return value.string;
        case PapyrusValueType::Object: return value.object.toString();
        case PapyrusValueType::Array: return "[array:" + std::to_string(value.array.size()) + "]";
    }
    return {};
}

bool comparable(
    const PapyrusValue& left, const PapyrusValue& right,
    int& comparison, std::string& error) {
    double leftNumber = 0.0;
    double rightNumber = 0.0;
    if (numeric(left, leftNumber) && numeric(right, rightNumber)) {
        comparison = leftNumber < rightNumber ? -1 : leftNumber > rightNumber ? 1 : 0;
        return true;
    }
    if (left.type == PapyrusValueType::String && right.type == PapyrusValueType::String) {
        comparison = left.string < right.string ? -1 : left.string > right.string ? 1 : 0;
        return true;
    }
    error = "values are not order-comparable";
    return false;
}

PapyrusValue castValue(
    const PapyrusValue& source, std::string targetType,
    const std::map<ObjectId, std::map<std::string, PapyrusScriptInstanceSnapshot>>& instances,
    bool& ok) {
    ok = true;
    targetType = normalizedSymbol(std::move(targetType));
    if (targetType == "none" || targetType.empty()) return {};
    if (targetType == "int") {
        if (source.type == PapyrusValueType::Integer) return source;
        if (source.type == PapyrusValueType::Float) {
            return PapyrusValue::fromInteger(static_cast<std::int64_t>(source.real));
        }
        if (source.type == PapyrusValueType::Boolean) {
            return PapyrusValue::fromInteger(source.boolean ? 1 : 0);
        }
        ok = false;
        return {};
    }
    if (targetType == "float") {
        if (source.type == PapyrusValueType::Float) return source;
        if (source.type == PapyrusValueType::Integer) {
            return PapyrusValue::fromFloat(static_cast<double>(source.integer));
        }
        if (source.type == PapyrusValueType::Boolean) {
            return PapyrusValue::fromFloat(source.boolean ? 1.0 : 0.0);
        }
        ok = false;
        return {};
    }
    if (targetType == "bool") return PapyrusValue::fromBoolean(source.truthy());
    if (targetType == "string") return PapyrusValue::fromString(valueString(source));
    if (targetType.ends_with("[]")) {
        if (source.type == PapyrusValueType::None || source.type == PapyrusValueType::Array) return source;
        ok = false;
        return {};
    }
    if (source.type == PapyrusValueType::None) return {};
    if (source.type != PapyrusValueType::Object) {
        ok = false;
        return {};
    }
    static const std::vector<std::string> builtInObjectTypes = {
        "form", "objectreference", "actor", "quest", "referencealias", "locationalias",
        "package", "faction", "scene", "globalvariable", "message", "keyword"};
    if (std::find(builtInObjectTypes.begin(), builtInObjectTypes.end(), targetType) !=
        builtInObjectTypes.end()) {
        return source;
    }
    const auto object = instances.find(source.object);
    if (object != instances.end() && object->second.contains(targetType)) return source;
    return {};
}

}  // namespace

PapyrusValue PapyrusValue::fromInteger(std::int64_t value) {
    PapyrusValue out; out.type = PapyrusValueType::Integer; out.integer = value; return out;
}

PapyrusValue PapyrusValue::fromFloat(double value) {
    PapyrusValue out; out.type = PapyrusValueType::Float; out.real = value; return out;
}

PapyrusValue PapyrusValue::fromBoolean(bool value) {
    PapyrusValue out; out.type = PapyrusValueType::Boolean; out.boolean = value; return out;
}

PapyrusValue PapyrusValue::fromString(std::string value) {
    PapyrusValue out; out.type = PapyrusValueType::String; out.string = std::move(value); return out;
}

PapyrusValue PapyrusValue::fromObject(ObjectId value) {
    PapyrusValue out; out.type = PapyrusValueType::Object; out.object = std::move(value); return out;
}

PapyrusValue PapyrusValue::fromArray(std::vector<PapyrusValue> value) {
    PapyrusValue out; out.type = PapyrusValueType::Array; out.array = std::move(value); return out;
}

bool PapyrusValue::truthy() const {
    switch (type) {
        case PapyrusValueType::None: return false;
        case PapyrusValueType::Integer: return integer != 0;
        case PapyrusValueType::Float: return real != 0.0 && std::isfinite(real);
        case PapyrusValueType::Boolean: return boolean;
        case PapyrusValueType::String: return !string.empty();
        case PapyrusValueType::Object: return object.valid();
        case PapyrusValueType::Array: return !array.empty();
    }
    return false;
}

PapyrusOperand PapyrusOperand::fromLocal(std::string name) {
    PapyrusOperand out; out.local = true; out.localName = normalizedSymbol(std::move(name)); return out;
}

PapyrusOperand PapyrusOperand::fromLiteral(PapyrusValue value) {
    PapyrusOperand out; out.literal = std::move(value); return out;
}

bool PapyrusVm::loadPexScript(
    const PexScript& script,
    bool strict,
    PexCompatibilityReport& outReport,
    std::string& outError) {
    outReport = inspectPexCompatibility(script);
    if (strict && !outReport.compatibilityErrors.empty()) {
        outError = "PEX module is not executable in strict mode: " +
            outReport.compatibilityErrors.front();
        return false;
    }

    auto stagedFunctions = m_functions;
    auto stagedDefaults = m_classDefaults;
    auto stagedAutoVariableProperties = m_classAutoVariableProperties;
    auto stagedParents = m_classParents;
    auto stagedAutoStates = m_classAutoStates;
    for (const PexObjectInfo& object : script.objectInfo) {
        const std::string objectName = normalizedSymbol(object.name);
        stagedParents.insert_or_assign(objectName, normalizedSymbol(object.parentClass));
        stagedAutoStates.insert_or_assign(objectName, normalizedSymbol(object.autoState));
        std::unordered_map<std::string, PapyrusValue>& defaults = stagedDefaults[objectName];
        std::unordered_map<std::string, PapyrusValue> variables;
        for (const PexVariableInfo& variable : object.variables) {
            variables.insert_or_assign(normalizedSymbol(variable.name), pexLiteral(variable.initialValue));
        }
        for (const PexPropertyInfo& property : object.properties) {
            PapyrusValue initial;
            if (!property.autoVariable.empty()) {
                const auto found = variables.find(normalizedSymbol(property.autoVariable));
                if (found != variables.end()) initial = found->second;
            }
            defaults.insert_or_assign(normalizedSymbol(property.name), std::move(initial));
            if (!property.autoVariable.empty()) {
                stagedAutoVariableProperties[objectName].insert_or_assign(
                    normalizedSymbol(property.autoVariable),
                    normalizedSymbol(property.name));
            }
        }
    }

    const auto declaredType = [](const PexFunctionInfo& source, const PexValue& value) {
        if (value.kind != PexValueKind::Identifier) return std::string{};
        const std::string identifier = normalizedSymbol(value.text);
        if (identifier == "self") return normalizedSymbol(source.objectName);
        for (std::size_t index = 0u; index < source.parameters.size(); ++index) {
            if (normalizedSymbol(source.parameters[index]) == identifier &&
                index < source.parameterTypes.size()) {
                return normalizedSymbol(source.parameterTypes[index]);
            }
        }
        for (std::size_t index = 0u; index < source.locals.size(); ++index) {
            if (normalizedSymbol(source.locals[index]) == identifier && index < source.localTypes.size()) {
                return normalizedSymbol(source.localTypes[index]);
            }
        }
        return std::string{};
    };

    std::set<std::string> loadedFunctions;
    for (const PexFunctionInfo& source : script.functions) {
        if (source.native()) continue;
        PapyrusFunction function;
        function.name = normalizedSymbol(source.qualifiedName());
        function.scriptClass = normalizedSymbol(source.objectName);
        function.parentClass = stagedParents[function.scriptClass];
        function.parameters = source.parameters;
        function.parameterTypes = source.parameterTypes;
        for (std::string& parameter : function.parameters) parameter = normalizedSymbol(std::move(parameter));
        for (std::string& type : function.parameterTypes) type = normalizedSymbol(std::move(type));
        for (std::size_t index = 0u; index < source.locals.size(); ++index) {
            function.localTypes.emplace(
                normalizedSymbol(source.locals[index]),
                index < source.localTypes.size() ? normalizedSymbol(source.localTypes[index]) : "");
        }
        for (std::size_t index = 0u; index < function.parameters.size(); ++index) {
            function.localTypes.insert_or_assign(
                function.parameters[index],
                index < function.parameterTypes.size() ? function.parameterTypes[index] : "");
        }
        const auto object = std::find_if(script.objectInfo.begin(), script.objectInfo.end(),
            [&](const PexObjectInfo& candidate) {
                return normalizedSymbol(candidate.name) == function.scriptClass;
            });
        if (object != script.objectInfo.end()) {
            for (const PexVariableInfo& variable : object->variables) {
                function.localTypes.insert_or_assign(
                    normalizedSymbol(variable.name), normalizedSymbol(variable.type));
            }
        }
        const auto valueType = [&](const PexValue& value) {
            std::string type = declaredType(source, value);
            if (!type.empty() || value.kind != PexValueKind::Identifier) return type;
            const auto found = function.localTypes.find(normalizedSymbol(value.text));
            return found == function.localTypes.end() ? std::string{} : found->second;
        };

        bool functionSupported = true;
        for (const PexInstructionInfo& sourceInstruction : source.instructions) {
            PapyrusInstruction instruction;
            bool supported = true;
            const auto require = [&](std::size_t count) {
                return sourceInstruction.arguments.size() == count;
            };
            const auto destination = [&](std::size_t index) {
                return index < sourceInstruction.arguments.size() &&
                    sourceInstruction.arguments[index].kind == PexValueKind::Identifier;
            };
            const auto binary = [&](PapyrusOpcode opcode) {
                supported = require(3u) && destination(0u);
                if (supported) {
                    instruction.opcode = opcode;
                    instruction.destination = normalizedSymbol(sourceInstruction.arguments[0].text);
                    instruction.operands = {translatedOperand(sourceInstruction.arguments[1]),
                                            translatedOperand(sourceInstruction.arguments[2])};
                }
            };
            const auto unary = [&](PapyrusOpcode opcode) {
                supported = require(2u) && destination(0u);
                if (supported) {
                    instruction.opcode = opcode;
                    instruction.destination = normalizedSymbol(sourceInstruction.arguments[0].text);
                    instruction.operands = {translatedOperand(sourceInstruction.arguments[1])};
                }
            };
            switch (sourceInstruction.opcode) {
                case 0u: instruction.opcode = PapyrusOpcode::Nop; supported = require(0u); break;
                case 1u: binary(PapyrusOpcode::IntegerAdd); break;
                case 2u: binary(PapyrusOpcode::FloatAdd); break;
                case 3u: binary(PapyrusOpcode::IntegerSubtract); break;
                case 4u: binary(PapyrusOpcode::FloatSubtract); break;
                case 5u: binary(PapyrusOpcode::IntegerMultiply); break;
                case 6u: binary(PapyrusOpcode::FloatMultiply); break;
                case 7u: binary(PapyrusOpcode::IntegerDivide); break;
                case 8u: binary(PapyrusOpcode::FloatDivide); break;
                case 9u: binary(PapyrusOpcode::IntegerModulo); break;
                case 10u: unary(PapyrusOpcode::LogicalNot); break;
                case 11u: unary(PapyrusOpcode::IntegerNegate); break;
                case 12u: unary(PapyrusOpcode::FloatNegate); break;
                case 13u: unary(PapyrusOpcode::Assign); break;
                case 14u:
                    unary(PapyrusOpcode::Cast);
                    if (supported) instruction.targetType = function.localTypes[instruction.destination];
                    break;
                case 15u: binary(PapyrusOpcode::CompareEqual); break;
                case 16u: binary(PapyrusOpcode::CompareLess); break;
                case 17u: binary(PapyrusOpcode::CompareLessEqual); break;
                case 18u: binary(PapyrusOpcode::CompareGreater); break;
                case 19u: binary(PapyrusOpcode::CompareGreaterEqual); break;
                case 20u:
                    supported = require(1u) &&
                        sourceInstruction.arguments[0].kind == PexValueKind::Integer;
                    if (supported) {
                        instruction.opcode = PapyrusOpcode::Jump;
                        instruction.jumpOffset = sourceInstruction.arguments[0].integer;
                    }
                    break;
                case 21u:
                case 22u:
                    supported = require(2u) &&
                        sourceInstruction.arguments[1].kind == PexValueKind::Integer;
                    if (supported) {
                        instruction.opcode = sourceInstruction.opcode == 21u
                            ? PapyrusOpcode::JumpIfTrue : PapyrusOpcode::JumpIfFalse;
                        instruction.operands = {translatedOperand(sourceInstruction.arguments[0])};
                        instruction.jumpOffset = sourceInstruction.arguments[1].integer;
                    }
                    break;
                case 23u: {
                    supported = sourceInstruction.arguments.size() >= 4u && destination(2u) &&
                        sourceInstruction.arguments[3].kind == PexValueKind::Integer &&
                        sourceInstruction.arguments[3].integer >= 0 &&
                        sourceInstruction.arguments.size() ==
                            4u + static_cast<std::size_t>(sourceInstruction.arguments[3].integer);
                    if (supported) {
                        instruction.opcode = PapyrusOpcode::CallMethod;
                        instruction.name = normalizedSymbol(sourceInstruction.arguments[0].text);
                        instruction.destination = normalizedSymbol(sourceInstruction.arguments[2].text);
                        instruction.targetType = valueType(sourceInstruction.arguments[1]);
                        instruction.operands.push_back(translatedOperand(sourceInstruction.arguments[1]));
                        for (std::size_t index = 4u; index < sourceInstruction.arguments.size(); ++index) {
                            instruction.operands.push_back(translatedOperand(sourceInstruction.arguments[index]));
                        }
                    }
                    break;
                }
                case 24u: {
                    supported = sourceInstruction.arguments.size() >= 3u && destination(1u) &&
                        sourceInstruction.arguments[2].kind == PexValueKind::Integer &&
                        sourceInstruction.arguments[2].integer >= 0 &&
                        sourceInstruction.arguments.size() ==
                            3u + static_cast<std::size_t>(sourceInstruction.arguments[2].integer);
                    if (supported) {
                        instruction.opcode = PapyrusOpcode::CallParent;
                        instruction.name = normalizedSymbol(sourceInstruction.arguments[0].text);
                        instruction.destination = normalizedSymbol(sourceInstruction.arguments[1].text);
                        instruction.targetType = function.parentClass;
                        for (std::size_t index = 3u; index < sourceInstruction.arguments.size(); ++index) {
                            instruction.operands.push_back(translatedOperand(sourceInstruction.arguments[index]));
                        }
                    }
                    break;
                }
                case 25u: {
                    supported = sourceInstruction.arguments.size() >= 4u && destination(2u) &&
                        sourceInstruction.arguments[3].kind == PexValueKind::Integer &&
                        sourceInstruction.arguments[3].integer >= 0 &&
                        sourceInstruction.arguments.size() ==
                            4u + static_cast<std::size_t>(sourceInstruction.arguments[3].integer);
                    if (supported) {
                        instruction.opcode = PapyrusOpcode::CallStatic;
                        instruction.targetType = normalizedSymbol(sourceInstruction.arguments[0].text);
                        instruction.name = instruction.targetType + "." +
                            normalizedSymbol(sourceInstruction.arguments[1].text);
                        instruction.destination = normalizedSymbol(sourceInstruction.arguments[2].text);
                        for (std::size_t index = 4u; index < sourceInstruction.arguments.size(); ++index) {
                            instruction.operands.push_back(translatedOperand(sourceInstruction.arguments[index]));
                        }
                    }
                    break;
                }
                case 26u:
                    instruction.opcode = PapyrusOpcode::Return;
                    supported = require(1u);
                    if (supported) instruction.operands = {translatedOperand(sourceInstruction.arguments[0])};
                    break;
                case 27u: binary(PapyrusOpcode::StringConcat); break;
                case 28u:
                    supported = require(3u) && destination(2u);
                    if (supported) {
                        instruction.opcode = PapyrusOpcode::PropertyGet;
                        instruction.name = normalizedSymbol(sourceInstruction.arguments[0].text);
                        instruction.operands = {translatedOperand(sourceInstruction.arguments[1])};
                        instruction.destination = normalizedSymbol(sourceInstruction.arguments[2].text);
                        instruction.targetType = valueType(sourceInstruction.arguments[1]);
                    }
                    break;
                case 29u:
                    supported = require(3u);
                    if (supported) {
                        instruction.opcode = PapyrusOpcode::PropertySet;
                        instruction.name = normalizedSymbol(sourceInstruction.arguments[0].text);
                        instruction.operands = {translatedOperand(sourceInstruction.arguments[1]),
                                                translatedOperand(sourceInstruction.arguments[2])};
                        instruction.targetType = valueType(sourceInstruction.arguments[1]);
                    }
                    break;
                case 30u: unary(PapyrusOpcode::ArrayCreate); break;
                case 31u: unary(PapyrusOpcode::ArrayLength); break;
                case 32u: binary(PapyrusOpcode::ArrayGetElement); break;
                case 33u:
                    supported = require(3u);
                    if (supported) {
                        instruction.opcode = PapyrusOpcode::ArraySetElement;
                        instruction.operands = {translatedOperand(sourceInstruction.arguments[0]),
                                                translatedOperand(sourceInstruction.arguments[1]),
                                                translatedOperand(sourceInstruction.arguments[2])};
                    }
                    break;
                case 34u:
                case 35u:
                    supported = require(4u) && destination(0u);
                    if (supported) {
                        instruction.opcode = sourceInstruction.opcode == 34u
                            ? PapyrusOpcode::ArrayFindElement : PapyrusOpcode::ArrayReverseFindElement;
                        instruction.destination = normalizedSymbol(sourceInstruction.arguments[0].text);
                        instruction.operands = {translatedOperand(sourceInstruction.arguments[1]),
                                                translatedOperand(sourceInstruction.arguments[2]),
                                                translatedOperand(sourceInstruction.arguments[3])};
                    }
                    break;
                default: supported = false; break;
            }
            if (!supported) {
                if (strict) {
                    outError = "cannot translate " + std::string(pexOpcodeName(sourceInstruction.opcode)) +
                        " in " + source.qualifiedName();
                    return false;
                }
                functionSupported = false;
                break;
            }
            function.instructions.push_back(std::move(instruction));
        }
        if (functionSupported) {
            if (function.name.empty() || stagedFunctions.contains(function.name)) {
                outError = "duplicate or empty Papyrus function " + function.name;
                return false;
            }
            stagedFunctions.emplace(function.name, std::move(function));
            loadedFunctions.insert(normalizedSymbol(source.qualifiedName()));
        }
    }

    {
        std::set<std::string> unresolved;
        const auto resolveMethod = [&](std::string type, const std::string& method) {
            type = normalizedSymbol(std::move(type));
            std::set<std::string> visited;
            while (!type.empty() && visited.insert(type).second) {
                const std::string candidate = type + "." + normalizedSymbol(method);
                if (stagedFunctions.contains(candidate) || m_natives.contains(candidate) ||
                    m_contextNatives.contains(candidate)) {
                    return candidate;
                }
                const std::string stateSuffix = "." + normalizedSymbol(method);
                const auto stateMethod = std::find_if(
                    stagedFunctions.begin(), stagedFunctions.end(), [&](const auto& entry) {
                        return entry.second.scriptClass == type && entry.first.ends_with(stateSuffix);
                    });
                if (stateMethod != stagedFunctions.end()) return stateMethod->first;
                const auto parent = stagedParents.find(type);
                type = parent == stagedParents.end() ? std::string{} : parent->second;
            }
            return std::string{};
        };
        for (const std::string& functionName : loadedFunctions) {
            const PapyrusFunction& function = stagedFunctions.at(functionName);
            for (const PapyrusInstruction& instruction : function.instructions) {
                std::string target;
                if (instruction.opcode == PapyrusOpcode::CallStatic) {
                    target = normalizedSymbol(instruction.name);
                } else if (instruction.opcode == PapyrusOpcode::CallParent) {
                    target = normalizedSymbol(instruction.targetType) + "." +
                        normalizedSymbol(instruction.name);
                } else if (instruction.opcode == PapyrusOpcode::CallMethod) {
                    target = resolveMethod(instruction.targetType, instruction.name);
                    if (target.empty()) target = resolveMethod(function.scriptClass, instruction.name);
                } else {
                    continue;
                }
                if (target.empty() || target.starts_with('.') ||
                    (!stagedFunctions.contains(target) && !m_natives.contains(target) &&
                     !m_contextNatives.contains(target))) {
                    unresolved.insert(target.empty()
                        ? normalizedSymbol(instruction.targetType) + "." +
                            normalizedSymbol(instruction.name)
                        : target);
                }
            }
        }
        outReport.unresolvedCalls.assign(unresolved.begin(), unresolved.end());
        if (strict && !unresolved.empty()) {
            outError = "PEX module requires unresolved call binding " + *unresolved.begin();
            return false;
        }
    }

    m_functions = std::move(stagedFunctions);
    m_classDefaults = std::move(stagedDefaults);
    m_classAutoVariableProperties = std::move(stagedAutoVariableProperties);
    m_classParents = std::move(stagedParents);
    m_classAutoStates = std::move(stagedAutoStates);
    outError.clear();
    return true;
}

bool PapyrusVm::registerFunction(PapyrusFunction function, std::string& outError) {
    function.name = normalizedSymbol(std::move(function.name));
    function.scriptClass = normalizedSymbol(std::move(function.scriptClass));
    function.parentClass = normalizedSymbol(std::move(function.parentClass));
    if (function.scriptClass.empty()) {
        const std::size_t separator = function.name.find('.');
        if (separator != std::string::npos) function.scriptClass = function.name.substr(0u, separator);
    }
    for (std::string& parameter : function.parameters) parameter = normalizedSymbol(std::move(parameter));
    if (function.name.empty()) {
        outError = "Papyrus function has no name";
        return false;
    }
    if (m_functions.contains(function.name)) {
        outError = "duplicate Papyrus function " + function.name;
        return false;
    }
    m_functions.emplace(function.name, std::move(function));
    outError.clear();
    return true;
}

void PapyrusVm::registerNative(std::string name, NativeFunction function) {
    m_natives.insert_or_assign(normalizedSymbol(std::move(name)), std::move(function));
}

void PapyrusVm::registerContextNative(std::string name, ContextNativeFunction function) {
    m_contextNatives.insert_or_assign(normalizedSymbol(std::move(name)), std::move(function));
}

void PapyrusVm::registerClassParent(std::string scriptClass, std::string parentClass) {
    m_classParents.insert_or_assign(
        normalizedSymbol(std::move(scriptClass)), normalizedSymbol(std::move(parentClass)));
}

bool PapyrusVm::enterFunction(
    PapyrusThreadSnapshot& thread,
    const std::string& functionName,
    ObjectId self,
    std::span<const PapyrusValue> arguments,
    std::string returnDestination,
    std::string& outError) {
    const std::string key = normalizedSymbol(functionName);
    const auto found = m_functions.find(key);
    if (found == m_functions.end()) {
        outError = "unknown Papyrus function " + functionName;
        return false;
    }
    if (arguments.size() != found->second.parameters.size()) {
        outError = "Papyrus function " + functionName + " expected " +
            std::to_string(found->second.parameters.size()) + " arguments, got " +
            std::to_string(arguments.size());
        return false;
    }
    thread.function = key;
    thread.instruction = 0u;
    thread.returnDestination = normalizedSymbol(std::move(returnDestination));
    thread.self = std::move(self);
    thread.scriptClass = found->second.scriptClass;
    thread.locals.clear();
    for (std::size_t index = 0u; index < arguments.size(); ++index) {
        thread.locals.emplace(found->second.parameters[index], arguments[index]);
    }
    outError.clear();
    return true;
}

std::uint64_t PapyrusVm::startFunction(
    const std::string& name,
    std::span<const PapyrusValue> arguments,
    std::string& outError) {
    PapyrusThreadSnapshot thread;
    thread.id = m_nextThreadId++;
    if (!enterFunction(thread, name, {}, arguments, {}, outError)) return 0u;
    const std::uint64_t id = thread.id;
    (m_advancing ? m_pendingThreads : m_threads).push_back(std::move(thread));
    return id;
}

std::uint64_t PapyrusVm::startFunctionOnObject(
    ObjectId self,
    const std::string& scriptClass,
    const std::string& function,
    std::span<const PapyrusValue> arguments,
    std::string& outError) {
    if (!self.valid()) {
        outError = "Papyrus object event requires a valid ObjectId";
        return 0u;
    }
    const std::string className = normalizedSymbol(scriptClass);
    const auto attached = m_instances.find(self);
    if (attached == m_instances.end() || !attached->second.contains(className)) {
        outError = "object does not have attached Papyrus script " + scriptClass;
        return 0u;
    }
    PapyrusThreadSnapshot thread;
    thread.id = m_nextThreadId++;
    std::string target = className + "." + normalizedSymbol(function);
    const auto instance = attached->second.find(className);
    if (instance != attached->second.end() && !instance->second.activeState.empty()) {
        const std::string stateTarget = className + "." + instance->second.activeState + "." +
            normalizedSymbol(function);
        if (m_functions.contains(stateTarget)) target = stateTarget;
    }
    if (!enterFunction(
            thread, target,
            std::move(self), arguments, {}, outError)) {
        return 0u;
    }
    const std::uint64_t id = thread.id;
    (m_advancing ? m_pendingThreads : m_threads).push_back(std::move(thread));
    return id;
}

bool PapyrusVm::attachScript(
    ObjectId object,
    std::string scriptClass,
    std::unordered_map<std::string, PapyrusValue> properties,
    std::string& outError) {
    if (!object.valid()) {
        outError = "cannot attach a Papyrus script to an invalid ObjectId";
        return false;
    }
    scriptClass = normalizedSymbol(std::move(scriptClass));
    if (scriptClass.empty()) {
        outError = "cannot attach an unnamed Papyrus script";
        return false;
    }
    PapyrusScriptInstanceSnapshot instance;
    instance.object = object;
    instance.scriptClass = scriptClass;
    const auto autoState = m_classAutoStates.find(scriptClass);
    if (autoState != m_classAutoStates.end()) instance.activeState = autoState->second;
    const auto defaults = m_classDefaults.find(scriptClass);
    if (defaults != m_classDefaults.end()) instance.properties = defaults->second;
    for (auto& [name, value] : properties) {
        instance.properties.insert_or_assign(normalizedSymbol(name), std::move(value));
    }
    m_instances[object].insert_or_assign(scriptClass, std::move(instance));
    outError.clear();
    return true;
}

bool PapyrusVm::setProperty(
    const ObjectId& object,
    const std::string& scriptClass,
    const std::string& property,
    PapyrusValue value,
    std::string& outError) {
    auto objectIt = m_instances.find(object);
    const std::string className = normalizedSymbol(scriptClass);
    if (objectIt == m_instances.end() || !objectIt->second.contains(className)) {
        outError = "missing Papyrus instance " + scriptClass + " on " + object.toString();
        return false;
    }
    objectIt->second.at(className).properties.insert_or_assign(
        normalizedSymbol(property), std::move(value));
    outError.clear();
    return true;
}

const PapyrusValue* PapyrusVm::findProperty(
    const ObjectId& object,
    const std::string& scriptClass,
    const std::string& property) const {
    const auto objectIt = m_instances.find(object);
    if (objectIt == m_instances.end()) return nullptr;
    const std::string className = normalizedSymbol(scriptClass);
    const auto instance = objectIt->second.find(className);
    if (instance == objectIt->second.end()) return nullptr;
    const auto found = instance->second.properties.find(normalizedSymbol(property));
    return found == instance->second.properties.end() ? nullptr : &found->second;
}

const PapyrusValue* PapyrusVm::findProperty(
    const ObjectId& object, const std::string& property) const {
    const auto objectIt = m_instances.find(object);
    if (objectIt == m_instances.end()) return nullptr;
    const std::string propertyName = normalizedSymbol(property);
    for (const auto& [scriptClass, instance] : objectIt->second) {
        (void)scriptClass;
        const auto found = instance.properties.find(propertyName);
        if (found != instance.properties.end()) return &found->second;
    }
    return nullptr;
}

bool PapyrusVm::registerForUpdate(
    ObjectId object,
    std::string scriptClass,
    double seconds,
    std::uint64_t currentTick,
    bool repeating,
    std::string& outError,
    std::string eventFunction) {
    scriptClass = normalizedSymbol(std::move(scriptClass));
    eventFunction = normalizedSymbol(std::move(eventFunction));
    const auto attached = m_instances.find(object);
    if (!object.valid() || attached == m_instances.end() ||
        !attached->second.contains(scriptClass)) {
        outError = "update registration requires an attached script instance";
        return false;
    }
    if (!std::isfinite(seconds) || seconds < 0.0) {
        outError = "update registration interval is invalid";
        return false;
    }
    if (eventFunction != "onupdate" && eventFunction != "onupdategametime") {
        outError = "unsupported update event " + eventFunction;
        return false;
    }
    const std::uint64_t interval = std::max<std::uint64_t>(1u,
        static_cast<std::uint64_t>(std::min<double>(
            std::ceil(seconds * 60.0),
            static_cast<double>(std::numeric_limits<std::uint32_t>::max()))));
    const auto existing = std::find_if(m_updates.begin(), m_updates.end(),
        [&](const PapyrusUpdateRegistrationSnapshot& update) {
            return update.object == object && update.scriptClass == scriptClass &&
                update.eventFunction == eventFunction;
        });
    PapyrusUpdateRegistrationSnapshot update{
        object, scriptClass, eventFunction, interval, currentTick + interval, repeating};
    if (existing == m_updates.end()) m_updates.push_back(std::move(update));
    else *existing = std::move(update);
    std::sort(m_updates.begin(), m_updates.end(), [](const auto& left, const auto& right) {
        if (left.object != right.object) return left.object < right.object;
        if (left.scriptClass != right.scriptClass) {
            return left.scriptClass < right.scriptClass;
        }
        return left.eventFunction < right.eventFunction;
    });
    outError.clear();
    return true;
}

void PapyrusVm::unregisterForUpdate(
    const ObjectId& object,
    const std::string& scriptClass,
    std::string eventFunction) {
    const std::string className = normalizedSymbol(scriptClass);
    eventFunction = normalizedSymbol(std::move(eventFunction));
    m_updates.erase(std::remove_if(m_updates.begin(), m_updates.end(),
        [&](const PapyrusUpdateRegistrationSnapshot& update) {
            return update.object == object && update.scriptClass == className &&
                (eventFunction.empty() || update.eventFunction == eventFunction);
        }), m_updates.end());
}

PapyrusValue PapyrusVm::operandValue(
    const PapyrusThreadSnapshot& thread, const PapyrusOperand& operand) const {
    if (!operand.local) return operand.literal;
    if (operand.localName == "self") {
        return thread.self.valid() ? PapyrusValue::fromObject(thread.self) : PapyrusValue{};
    }
    if (operand.localName == "::state" && thread.self.valid()) {
        const auto object = m_instances.find(thread.self);
        if (object != m_instances.end()) {
            const auto instance = object->second.find(thread.scriptClass);
            if (instance != object->second.end()) {
                return PapyrusValue::fromString(instance->second.activeState);
            }
        }
    }
    const auto local = thread.locals.find(operand.localName);
    if (local != thread.locals.end()) return local->second;
    if (thread.self.valid()) {
        const auto variables = m_classAutoVariableProperties.find(thread.scriptClass);
        if (variables != m_classAutoVariableProperties.end()) {
            const auto property = variables->second.find(operand.localName);
            if (property != variables->second.end()) {
                if (const PapyrusValue* value = findProperty(
                        thread.self, thread.scriptClass, property->second)) {
                    return *value;
                }
            }
        }
    }
    const auto global = m_globals.find(operand.localName);
    return global == m_globals.end() ? PapyrusValue{} : global->second;
}

std::string PapyrusVm::operandDeclaredType(
    const PapyrusFunction& function, const PapyrusOperand& operand) const {
    if (!operand.local) return {};
    if (operand.localName == "self") return function.scriptClass;
    const auto found = function.localTypes.find(operand.localName);
    return found == function.localTypes.end() ? std::string{} : found->second;
}

PapyrusAdvanceResult PapyrusVm::advance(
    std::uint64_t currentTick,
    std::uint32_t instructionBudget,
    BethesdaWorld& world) {
    PapyrusAdvanceResult result;
    for (PapyrusUpdateRegistrationSnapshot& update : m_updates) {
        if (update.nextTick > currentTick) continue;
        std::string error;
        if (startFunctionOnObject(
                update.object, update.scriptClass, update.eventFunction, {}, error) == 0u) {
            result.diagnostics.push_back(
                "Papyrus update registration failed for " + update.scriptClass + ": " + error);
            update.repeating = false;
        } else if (update.repeating) {
            update.nextTick = currentTick + update.intervalTicks;
        }
    }
    m_updates.erase(std::remove_if(m_updates.begin(), m_updates.end(),
        [&](const PapyrusUpdateRegistrationSnapshot& update) {
            return !update.repeating && update.nextTick <= currentTick;
        }), m_updates.end());
    std::sort(m_threads.begin(), m_threads.end(),
        [](const PapyrusThreadSnapshot& left, const PapyrusThreadSnapshot& right) {
            return left.id < right.id;
        });
    m_advancing = true;
    std::vector<std::uint64_t> completed;
    for (PapyrusThreadSnapshot& thread : m_threads) {
        if (result.instructions >= instructionBudget || thread.resumeTick > currentTick || thread.failed) {
            continue;
        }
        bool yielded = false;
        while (!yielded && result.instructions < instructionBudget) {
            const auto functionIt = m_functions.find(thread.function);
            if (functionIt == m_functions.end()) {
                thread.failed = true;
                result.diagnostics.push_back("active Papyrus function disappeared: " + thread.function);
                break;
            }
            const PapyrusFunction& function = functionIt->second;
            if (thread.instruction >= function.instructions.size()) {
                PapyrusValue returned;
                if (thread.callStack.empty()) {
                    completed.push_back(thread.id);
                    ++result.completedThreads;
                    break;
                }
                const std::string destination = thread.returnDestination;
                PapyrusCallFrameSnapshot caller = std::move(thread.callStack.back());
                thread.callStack.pop_back();
                static_cast<PapyrusCallFrameSnapshot&>(thread) = std::move(caller);
                if (!destination.empty()) thread.locals[destination] = std::move(returned);
                continue;
            }
            const PapyrusInstruction& instruction = function.instructions[thread.instruction];
            ++result.instructions;

            const auto fail = [&](std::string message) {
                thread.failed = true;
                result.diagnostics.push_back(
                    "Papyrus " + thread.function + " @" + std::to_string(thread.instruction) +
                    ": " + std::move(message));
                yielded = true;
            };
            const auto store = [&](PapyrusValue value) {
                if (!instruction.destination.empty()) {
                    const auto variables =
                        m_classAutoVariableProperties.find(thread.scriptClass);
                    const auto property = variables == m_classAutoVariableProperties.end()
                        ? std::unordered_map<std::string, std::string>::const_iterator{}
                        : variables->second.find(instruction.destination);
                    if (variables != m_classAutoVariableProperties.end() &&
                        property != variables->second.end() && thread.self.valid()) {
                        std::string error;
                        if (!setProperty(thread.self, thread.scriptClass,
                                property->second, std::move(value), error)) {
                            fail(std::move(error));
                        }
                    } else {
                        thread.locals[instruction.destination] = std::move(value);
                    }
                }
            };
            const auto jump = [&](std::int32_t offset) {
                const std::int64_t next = static_cast<std::int64_t>(thread.instruction) + offset;
                if (next < 0 || next > static_cast<std::int64_t>(function.instructions.size())) {
                    fail("jump outside function");
                } else {
                    thread.instruction = static_cast<std::size_t>(next);
                }
            };
            const auto twoOperands = [&]() { return instruction.operands.size() == 2u; };
            const auto integerBinary = [&](auto operation, const char* name) {
                if (!twoOperands()) { fail(std::string(name) + " needs two operands"); return; }
                const PapyrusValue left = operandValue(thread, instruction.operands[0]);
                const PapyrusValue right = operandValue(thread, instruction.operands[1]);
                if (left.type != PapyrusValueType::Integer || right.type != PapyrusValueType::Integer) {
                    fail(std::string(name) + " received non-integer operands"); return;
                }
                std::int64_t output = 0;
                std::string error;
                if (!operation(left.integer, right.integer, output, error)) { fail(std::move(error)); return; }
                store(PapyrusValue::fromInteger(output));
                ++thread.instruction;
            };
            const auto floatBinary = [&](auto operation, const char* name) {
                if (!twoOperands()) { fail(std::string(name) + " needs two operands"); return; }
                double left = 0.0, right = 0.0;
                if (!numeric(operandValue(thread, instruction.operands[0]), left) ||
                    !numeric(operandValue(thread, instruction.operands[1]), right)) {
                    fail(std::string(name) + " received non-numeric operands"); return;
                }
                double output = 0.0;
                std::string error;
                if (!operation(left, right, output, error)) { fail(std::move(error)); return; }
                store(PapyrusValue::fromFloat(output));
                ++thread.instruction;
            };

            switch (instruction.opcode) {
                case PapyrusOpcode::Nop: ++thread.instruction; break;
                case PapyrusOpcode::Assign:
                    if (instruction.operands.size() != 1u) fail("assign needs one operand");
                    else {
                        PapyrusValue assigned = operandValue(thread, instruction.operands[0]);
                        if (instruction.destination == "::state" && thread.self.valid()) {
                            if (assigned.type != PapyrusValueType::String) {
                                fail("state assignment requires a string");
                                break;
                            }
                            auto object = m_instances.find(thread.self);
                            if (object == m_instances.end() ||
                                !object->second.contains(thread.scriptClass)) {
                                fail("state assignment requires an attached script instance");
                                break;
                            }
                            object->second.at(thread.scriptClass).activeState =
                                normalizedSymbol(assigned.string);
                        } else {
                            store(std::move(assigned));
                        }
                        ++thread.instruction;
                    }
                    break;
                case PapyrusOpcode::IntegerAdd:
                    integerBinary([](auto a, auto b, auto& out, std::string&) {
                        out = std::bit_cast<std::int64_t>(
                            std::bit_cast<std::uint64_t>(a) + std::bit_cast<std::uint64_t>(b));
                        return true;
                    }, "iadd"); break;
                case PapyrusOpcode::FloatAdd:
                    floatBinary([](auto a, auto b, auto& out, std::string&) { out = a + b; return true; }, "fadd"); break;
                case PapyrusOpcode::IntegerSubtract:
                    integerBinary([](auto a, auto b, auto& out, std::string&) {
                        out = std::bit_cast<std::int64_t>(
                            std::bit_cast<std::uint64_t>(a) - std::bit_cast<std::uint64_t>(b));
                        return true;
                    }, "isub"); break;
                case PapyrusOpcode::FloatSubtract:
                    floatBinary([](auto a, auto b, auto& out, std::string&) { out = a - b; return true; }, "fsub"); break;
                case PapyrusOpcode::IntegerMultiply:
                    integerBinary([](auto a, auto b, auto& out, std::string&) {
                        out = std::bit_cast<std::int64_t>(
                            std::bit_cast<std::uint64_t>(a) * std::bit_cast<std::uint64_t>(b));
                        return true;
                    }, "imul"); break;
                case PapyrusOpcode::FloatMultiply:
                    floatBinary([](auto a, auto b, auto& out, std::string&) { out = a * b; return true; }, "fmul"); break;
                case PapyrusOpcode::IntegerDivide:
                    integerBinary([](auto a, auto b, auto& out, std::string& error) {
                        if (b == 0) { error = "integer division by zero"; return false; }
                        if (a == std::numeric_limits<std::int64_t>::min() && b == -1) {
                            error = "integer division overflow"; return false;
                        }
                        out = a / b; return true;
                    }, "idiv"); break;
                case PapyrusOpcode::FloatDivide:
                    floatBinary([](auto a, auto b, auto& out, std::string& error) {
                        if (b == 0.0) { error = "floating-point division by zero"; return false; }
                        out = a / b; return true;
                    }, "fdiv"); break;
                case PapyrusOpcode::IntegerModulo:
                    integerBinary([](auto a, auto b, auto& out, std::string& error) {
                        if (b == 0) { error = "integer modulo by zero"; return false; }
                        if (a == std::numeric_limits<std::int64_t>::min() && b == -1) out = 0;
                        else out = a % b;
                        return true;
                    }, "imod"); break;
                case PapyrusOpcode::LogicalNot:
                    if (instruction.operands.size() != 1u) fail("not needs one operand");
                    else {
                        store(PapyrusValue::fromBoolean(!operandValue(thread, instruction.operands[0]).truthy()));
                        ++thread.instruction;
                    }
                    break;
                case PapyrusOpcode::IntegerNegate: {
                    if (instruction.operands.size() != 1u) { fail("ineg needs one operand"); break; }
                    const PapyrusValue value = operandValue(thread, instruction.operands[0]);
                    if (value.type != PapyrusValueType::Integer ||
                        value.integer == std::numeric_limits<std::int64_t>::min()) {
                        fail("ineg received invalid integer operand"); break;
                    }
                    store(PapyrusValue::fromInteger(-value.integer)); ++thread.instruction; break;
                }
                case PapyrusOpcode::FloatNegate: {
                    if (instruction.operands.size() != 1u) { fail("fneg needs one operand"); break; }
                    double value = 0.0;
                    if (!numeric(operandValue(thread, instruction.operands[0]), value)) {
                        fail("fneg received non-numeric operand"); break;
                    }
                    store(PapyrusValue::fromFloat(-value)); ++thread.instruction; break;
                }
                case PapyrusOpcode::Cast: {
                    if (instruction.operands.size() != 1u) { fail("cast needs one operand"); break; }
                    bool ok = false;
                    PapyrusValue casted = castValue(
                        operandValue(thread, instruction.operands[0]),
                        instruction.targetType, m_instances, ok);
                    if (!ok) { fail("invalid cast to " + instruction.targetType); break; }
                    store(std::move(casted)); ++thread.instruction; break;
                }
                case PapyrusOpcode::CompareEqual: {
                    if (!twoOperands()) { fail("cmp_eq needs two operands"); break; }
                    const PapyrusValue left = operandValue(thread, instruction.operands[0]);
                    const PapyrusValue right = operandValue(thread, instruction.operands[1]);
                    double leftNumber = 0.0, rightNumber = 0.0;
                    const bool equal = numeric(left, leftNumber) && numeric(right, rightNumber)
                        ? leftNumber == rightNumber : left == right;
                    store(PapyrusValue::fromBoolean(equal)); ++thread.instruction; break;
                }
                case PapyrusOpcode::CompareLess:
                case PapyrusOpcode::CompareLessEqual:
                case PapyrusOpcode::CompareGreater:
                case PapyrusOpcode::CompareGreaterEqual: {
                    if (!twoOperands()) { fail("comparison needs two operands"); break; }
                    int comparison = 0;
                    std::string error;
                    if (!comparable(operandValue(thread, instruction.operands[0]),
                            operandValue(thread, instruction.operands[1]), comparison, error)) {
                        fail(std::move(error)); break;
                    }
                    bool matched = false;
                    if (instruction.opcode == PapyrusOpcode::CompareLess) matched = comparison < 0;
                    else if (instruction.opcode == PapyrusOpcode::CompareLessEqual) matched = comparison <= 0;
                    else if (instruction.opcode == PapyrusOpcode::CompareGreater) matched = comparison > 0;
                    else matched = comparison >= 0;
                    store(PapyrusValue::fromBoolean(matched)); ++thread.instruction; break;
                }
                case PapyrusOpcode::Jump: jump(instruction.jumpOffset); break;
                case PapyrusOpcode::JumpIfTrue:
                case PapyrusOpcode::JumpIfFalse:
                    if (instruction.operands.size() != 1u) fail("conditional jump needs one condition");
                    else {
                        const bool condition = operandValue(thread, instruction.operands[0]).truthy();
                        if (condition == (instruction.opcode == PapyrusOpcode::JumpIfTrue)) jump(instruction.jumpOffset);
                        else ++thread.instruction;
                    }
                    break;
                case PapyrusOpcode::CallMethod:
                case PapyrusOpcode::CallParent:
                case PapyrusOpcode::CallStatic: {
                    ObjectId receiver;
                    std::size_t argumentStart = 0u;
                    std::string targetType = normalizedSymbol(instruction.targetType);
                    if (instruction.opcode == PapyrusOpcode::CallMethod) {
                        if (instruction.operands.empty()) { fail("callmethod has no receiver"); break; }
                        const PapyrusValue owner = operandValue(thread, instruction.operands[0]);
                        if (owner.type != PapyrusValueType::Object || !owner.object.valid()) {
                            fail("callmethod receiver is None or not an object"); break;
                        }
                        receiver = owner.object; argumentStart = 1u;
                        if (targetType.empty() && receiver == thread.self) targetType = thread.scriptClass;
                    } else if (instruction.opcode == PapyrusOpcode::CallParent) {
                        receiver = thread.self;
                        if (targetType.empty()) targetType = function.parentClass;
                    }
                    std::string target = instruction.opcode == PapyrusOpcode::CallStatic
                        ? normalizedSymbol(instruction.name)
                        : targetType + "." + normalizedSymbol(instruction.name);
                    if (instruction.opcode == PapyrusOpcode::CallMethod) {
                        const auto resolveMethod = [&](std::string type) {
                            std::set<std::string> visited;
                            while (!type.empty() && visited.insert(type).second) {
                                const std::string candidate = type + "." + normalizedSymbol(instruction.name);
                                if (m_functions.contains(candidate) || m_natives.contains(candidate) ||
                                    m_contextNatives.contains(candidate)) {
                                    return candidate;
                                }
                                if (receiver.valid()) {
                                    const auto object = m_instances.find(receiver);
                                    if (object != m_instances.end()) {
                                        const auto instance = object->second.find(type);
                                        if (instance != object->second.end() &&
                                            !instance->second.activeState.empty()) {
                                            const std::string stateCandidate = type + "." +
                                                instance->second.activeState + "." +
                                                normalizedSymbol(instruction.name);
                                            if (m_functions.contains(stateCandidate)) return stateCandidate;
                                        }
                                    }
                                }
                                const auto parent = m_classParents.find(type);
                                type = parent == m_classParents.end() ? std::string{} : parent->second;
                            }
                            return std::string{};
                        };
                        const std::string resolved = resolveMethod(targetType);
                        if (!resolved.empty()) target = resolved;
                        else if (receiver == thread.self) {
                            const std::string selfResolved = resolveMethod(thread.scriptClass);
                            if (!selfResolved.empty()) target = selfResolved;
                        }
                    }
                    std::vector<PapyrusValue> arguments;
                    arguments.reserve(instruction.operands.size() - argumentStart);
                    for (std::size_t index = argumentStart; index < instruction.operands.size(); ++index) {
                        arguments.push_back(operandValue(thread, instruction.operands[index]));
                    }
                    if (m_functions.contains(target)) {
                        if (thread.callStack.size() >= kMaximumCallDepth) {
                            fail("call stack exceeded " + std::to_string(kMaximumCallDepth) + " frames"); break;
                        }
                        PapyrusCallFrameSnapshot caller = static_cast<const PapyrusCallFrameSnapshot&>(thread);
                        caller.instruction = thread.instruction + 1u;
                        thread.callStack.push_back(std::move(caller));
                        std::string error;
                        if (!enterFunction(thread, target, std::move(receiver), arguments,
                                instruction.destination, error)) {
                            thread.callStack.pop_back(); fail(std::move(error));
                        }
                        break;
                    }
                    const auto contextNative = m_contextNatives.find(target);
                    const auto native = m_natives.find(target);
                    if (contextNative == m_contextNatives.end() && native == m_natives.end()) {
                        fail("unsupported call " + target); break;
                    }
                    PapyrusNativeContext context;
                    context.currentTick = currentTick;
                    context.self = receiver;
                    context.scriptClass = normalizedSymbol(instruction.targetType);
                    if (context.scriptClass.empty()) context.scriptClass = thread.scriptClass;
                    context.callerFunction = thread.function;
                    NativeCallResult call;
                    if (contextNative != m_contextNatives.end()) {
                        call = contextNative->second(context, arguments, world);
                    } else if (instruction.opcode != PapyrusOpcode::CallStatic) {
                        arguments.insert(arguments.begin(), PapyrusValue::fromObject(receiver));
                        call = native->second(arguments, currentTick, world);
                    } else {
                        call = native->second(arguments, currentTick, world);
                    }
                    if (!call.error.empty()) fail("native " + target + " failed: " + call.error);
                    else {
                        store(std::move(call.value)); ++thread.instruction;
                        if (!call.completed) {
                            thread.resumeTick = std::max(currentTick + 1u, call.resumeTick); yielded = true;
                        }
                    }
                    break;
                }
                case PapyrusOpcode::CallNative: {
                    const std::string nativeName = normalizedSymbol(instruction.name);
                    const auto contextNative = m_contextNatives.find(nativeName);
                    const auto native = m_natives.find(nativeName);
                    if (native == m_natives.end() && contextNative == m_contextNatives.end()) {
                        fail("unsupported native " + instruction.name); break;
                    }
                    std::vector<PapyrusValue> arguments;
                    arguments.reserve(instruction.operands.size());
                    for (const PapyrusOperand& operand : instruction.operands) {
                        arguments.push_back(operandValue(thread, operand));
                    }
                    NativeCallResult call;
                    if (contextNative != m_contextNatives.end()) {
                        call = contextNative->second(PapyrusNativeContext{
                            currentTick, thread.self, thread.scriptClass, thread.function},
                            arguments, world);
                    } else {
                        call = native->second(arguments, currentTick, world);
                    }
                    if (!call.error.empty()) fail("native " + instruction.name + " failed: " + call.error);
                    else {
                        store(std::move(call.value)); ++thread.instruction;
                        if (!call.completed) {
                            thread.resumeTick = std::max(currentTick + 1u, call.resumeTick); yielded = true;
                        }
                    }
                    break;
                }
                case PapyrusOpcode::StringConcat:
                    if (!twoOperands()) fail("strcat needs two operands");
                    else {
                        store(PapyrusValue::fromString(
                            valueString(operandValue(thread, instruction.operands[0])) +
                            valueString(operandValue(thread, instruction.operands[1]))));
                        ++thread.instruction;
                    }
                    break;
                case PapyrusOpcode::PropertyGet: {
                    if (instruction.operands.size() != 1u) { fail("propget needs an owner"); break; }
                    const PapyrusValue owner = operandValue(thread, instruction.operands[0]);
                    if (owner.type != PapyrusValueType::Object || !owner.object.valid()) {
                        fail("propget owner is None or not an object"); break;
                    }
                    std::string className = normalizedSymbol(instruction.targetType);
                    if (className.empty() && owner.object == thread.self) className = thread.scriptClass;
                    const PapyrusValue* property = findProperty(owner.object, className, instruction.name);
                    if (property == nullptr) { fail("missing property " + className + "." + instruction.name); break; }
                    store(*property); ++thread.instruction; break;
                }
                case PapyrusOpcode::PropertySet: {
                    if (instruction.operands.size() != 2u) { fail("propset needs owner and value"); break; }
                    const PapyrusValue owner = operandValue(thread, instruction.operands[0]);
                    if (owner.type != PapyrusValueType::Object || !owner.object.valid()) {
                        fail("propset owner is None or not an object"); break;
                    }
                    std::string className = normalizedSymbol(instruction.targetType);
                    if (className.empty() && owner.object == thread.self) className = thread.scriptClass;
                    std::string error;
                    if (!setProperty(owner.object, className, instruction.name,
                            operandValue(thread, instruction.operands[1]), error)) {
                        fail(std::move(error)); break;
                    }
                    ++thread.instruction; break;
                }
                case PapyrusOpcode::ArrayCreate: {
                    if (instruction.operands.size() != 1u) { fail("array_create needs a size"); break; }
                    const PapyrusValue size = operandValue(thread, instruction.operands[0]);
                    if (size.type != PapyrusValueType::Integer || size.integer < 0 ||
                        static_cast<std::uint64_t>(size.integer) > kMaximumArrayElements) {
                        fail("array_create size is invalid"); break;
                    }
                    store(PapyrusValue::fromArray(
                        std::vector<PapyrusValue>(static_cast<std::size_t>(size.integer))));
                    ++thread.instruction; break;
                }
                case PapyrusOpcode::ArrayLength: {
                    if (instruction.operands.size() != 1u) { fail("array_length needs an array"); break; }
                    const PapyrusValue array = operandValue(thread, instruction.operands[0]);
                    if (array.type != PapyrusValueType::Array) { fail("array_length received non-array"); break; }
                    store(PapyrusValue::fromInteger(static_cast<std::int64_t>(array.array.size())));
                    ++thread.instruction; break;
                }
                case PapyrusOpcode::ArrayGetElement: {
                    if (!twoOperands()) { fail("array_getelement needs array and index"); break; }
                    const PapyrusValue array = operandValue(thread, instruction.operands[0]);
                    const PapyrusValue index = operandValue(thread, instruction.operands[1]);
                    if (array.type != PapyrusValueType::Array || index.type != PapyrusValueType::Integer ||
                        index.integer < 0 || static_cast<std::uint64_t>(index.integer) >= array.array.size()) {
                        fail("array_getelement index is out of range"); break;
                    }
                    store(array.array[static_cast<std::size_t>(index.integer)]);
                    ++thread.instruction; break;
                }
                case PapyrusOpcode::ArraySetElement: {
                    if (instruction.operands.size() != 3u || !instruction.operands[0].local) {
                        fail("array_setelement requires a mutable array local"); break;
                    }
                    auto found = thread.locals.find(instruction.operands[0].localName);
                    const PapyrusValue index = operandValue(thread, instruction.operands[1]);
                    if (found == thread.locals.end() || found->second.type != PapyrusValueType::Array ||
                        index.type != PapyrusValueType::Integer || index.integer < 0 ||
                        static_cast<std::uint64_t>(index.integer) >= found->second.array.size()) {
                        fail("array_setelement index is out of range"); break;
                    }
                    found->second.array[static_cast<std::size_t>(index.integer)] =
                        operandValue(thread, instruction.operands[2]);
                    ++thread.instruction; break;
                }
                case PapyrusOpcode::ArrayFindElement:
                case PapyrusOpcode::ArrayReverseFindElement: {
                    if (instruction.operands.size() != 3u) { fail("array find needs array, value, start"); break; }
                    const PapyrusValue array = operandValue(thread, instruction.operands[0]);
                    const PapyrusValue needle = operandValue(thread, instruction.operands[1]);
                    const PapyrusValue start = operandValue(thread, instruction.operands[2]);
                    if (array.type != PapyrusValueType::Array || start.type != PapyrusValueType::Integer) {
                        fail("array find received invalid operands"); break;
                    }
                    std::int64_t foundIndex = -1;
                    if (instruction.opcode == PapyrusOpcode::ArrayFindElement) {
                        const std::size_t begin = start.integer < 0 ? 0u :
                            std::min<std::size_t>(static_cast<std::size_t>(start.integer), array.array.size());
                        for (std::size_t index = begin; index < array.array.size(); ++index) {
                            if (array.array[index] == needle) { foundIndex = static_cast<std::int64_t>(index); break; }
                        }
                    } else if (!array.array.empty()) {
                        std::size_t begin = start.integer < 0 ||
                            static_cast<std::uint64_t>(start.integer) >= array.array.size()
                            ? array.array.size() - 1u : static_cast<std::size_t>(start.integer);
                        for (std::size_t index = begin + 1u; index-- > 0u;) {
                            if (array.array[index] == needle) {
                                foundIndex = static_cast<std::int64_t>(index); break;
                            }
                        }
                    }
                    store(PapyrusValue::fromInteger(foundIndex)); ++thread.instruction; break;
                }
                case PapyrusOpcode::WaitTicks: {
                    if (instruction.operands.size() != 1u) { fail("wait has no duration"); break; }
                    const PapyrusValue duration = operandValue(thread, instruction.operands[0]);
                    if (duration.type != PapyrusValueType::Integer || duration.integer < 0) {
                        fail("wait duration is not non-negative"); break;
                    }
                    thread.resumeTick = currentTick + static_cast<std::uint64_t>(duration.integer);
                    ++thread.instruction; yielded = true; break;
                }
                case PapyrusOpcode::Return: {
                    PapyrusValue returned;
                    if (!instruction.operands.empty()) returned = operandValue(thread, instruction.operands[0]);
                    if (thread.callStack.empty()) {
                        completed.push_back(thread.id); ++result.completedThreads; yielded = true;
                    } else {
                        const std::string destination = thread.returnDestination;
                        PapyrusCallFrameSnapshot caller = std::move(thread.callStack.back());
                        thread.callStack.pop_back();
                        static_cast<PapyrusCallFrameSnapshot&>(thread) = std::move(caller);
                        if (!destination.empty()) thread.locals[destination] = std::move(returned);
                    }
                    break;
                }
            }
        }
    }
    if (!completed.empty()) {
        std::sort(completed.begin(), completed.end());
        m_threads.erase(std::remove_if(m_threads.begin(), m_threads.end(),
            [&](const PapyrusThreadSnapshot& thread) {
                return std::binary_search(completed.begin(), completed.end(), thread.id);
            }), m_threads.end());
    }
    m_advancing = false;
    if (!m_pendingThreads.empty()) {
        m_threads.insert(m_threads.end(),
            std::make_move_iterator(m_pendingThreads.begin()),
            std::make_move_iterator(m_pendingThreads.end()));
        m_pendingThreads.clear();
    }
    return result;
}

bool PapyrusVm::hasFunction(const std::string& name) const {
    return m_functions.contains(normalizedSymbol(name));
}

bool PapyrusVm::hasScriptClass(const std::string& name) const {
    return m_classParents.contains(normalizedSymbol(name));
}

bool PapyrusVm::hasNative(const std::string& name) const {
    const std::string key = normalizedSymbol(name);
    return m_natives.contains(key) || m_contextNatives.contains(key);
}

std::vector<std::string> PapyrusVm::functionsForClass(const std::string& scriptClass) const {
    const std::string normalizedClass = normalizedSymbol(scriptClass);
    std::vector<std::string> functions;
    for (const auto& [name, function] : m_functions) {
        if (function.scriptClass == normalizedClass) functions.push_back(name);
    }
    std::sort(functions.begin(), functions.end());
    return functions;
}

std::vector<std::string> PapyrusVm::scriptClassesForObject(ObjectId object) const {
    std::vector<std::string> classes;
    const auto found = m_instances.find(object);
    if (found == m_instances.end()) return classes;
    classes.reserve(found->second.size());
    for (const auto& [scriptClass, instance] : found->second) {
        (void)instance;
        classes.push_back(scriptClass);
    }
    return classes;
}

std::vector<std::string> PapyrusVm::unresolvedCallBindings() const {
    std::vector<std::string> allFunctions;
    allFunctions.reserve(m_functions.size());
    for (const auto& [name, function] : m_functions) {
        (void)function;
        allFunctions.push_back(name);
    }
    return unresolvedCallBindings(allFunctions);
}

std::vector<std::string> PapyrusVm::unresolvedCallBindings(
    std::span<const std::string> rootFunctions) const {
    const auto resolveMethod = [&](std::string type, const std::string& method) {
        type = normalizedSymbol(std::move(type));
        std::set<std::string> visited;
        while (!type.empty() && visited.insert(type).second) {
            const std::string candidate = type + "." + normalizedSymbol(method);
            if (m_functions.contains(candidate) || m_natives.contains(candidate) ||
                m_contextNatives.contains(candidate)) return candidate;
            const std::string stateSuffix = "." + normalizedSymbol(method);
            const auto stateMethod = std::find_if(
                m_functions.begin(), m_functions.end(), [&](const auto& entry) {
                    return entry.second.scriptClass == type && entry.first.ends_with(stateSuffix);
                });
            if (stateMethod != m_functions.end()) return stateMethod->first;
            const auto parent = m_classParents.find(type);
            type = parent == m_classParents.end() ? std::string{} : parent->second;
        }
        return std::string{};
    };
    std::set<std::string> unresolved;
    std::set<std::string> visited;
    std::vector<std::string> pending;
    pending.reserve(rootFunctions.size());
    for (const std::string& root : rootFunctions) {
        const std::string normalizedRoot = normalizedSymbol(root);
        if (m_functions.contains(normalizedRoot)) pending.push_back(normalizedRoot);
    }
    while (!pending.empty()) {
        const std::string functionName = std::move(pending.back());
        pending.pop_back();
        if (!visited.insert(functionName).second) continue;
        const auto functionFound = m_functions.find(functionName);
        if (functionFound == m_functions.end()) continue;
        const PapyrusFunction& function = functionFound->second;
        for (const PapyrusInstruction& instruction : function.instructions) {
            std::string target;
            if (instruction.opcode == PapyrusOpcode::CallStatic) {
                target = normalizedSymbol(instruction.name);
            } else if (instruction.opcode == PapyrusOpcode::CallParent) {
                target = resolveMethod(instruction.targetType, instruction.name);
            } else if (instruction.opcode == PapyrusOpcode::CallMethod) {
                target = resolveMethod(instruction.targetType, instruction.name);
                if (target.empty()) target = resolveMethod(function.scriptClass, instruction.name);
            } else {
                continue;
            }
            if (target.empty() || (!m_functions.contains(target) && !m_natives.contains(target) &&
                                   !m_contextNatives.contains(target))) {
                const std::string requested = instruction.opcode == PapyrusOpcode::CallStatic
                    ? normalizedSymbol(instruction.name)
                    : normalizedSymbol(instruction.targetType) + "." +
                        normalizedSymbol(instruction.name);
                unresolved.insert(functionName + " -> " + requested);
            } else if (m_functions.contains(target) && !visited.contains(target)) {
                pending.push_back(target);
            }
        }
    }
    return {unresolved.begin(), unresolved.end()};
}

PapyrusVmSnapshot PapyrusVm::snapshot() const {
    PapyrusVmSnapshot snapshot;
    snapshot.nextThreadId = m_nextThreadId;
    snapshot.threads = m_threads;
    snapshot.threads.insert(snapshot.threads.end(),
        m_pendingThreads.begin(), m_pendingThreads.end());
    std::sort(snapshot.threads.begin(), snapshot.threads.end(),
        [](const PapyrusThreadSnapshot& left, const PapyrusThreadSnapshot& right) {
            return left.id < right.id;
        });
    snapshot.globals = m_globals;
    snapshot.updates = m_updates;
    for (const auto& [object, scripts] : m_instances) {
        (void)object;
        for (const auto& [className, instance] : scripts) {
            (void)className;
            snapshot.instances.push_back(instance);
        }
    }
    return snapshot;
}

bool PapyrusVm::restore(const PapyrusVmSnapshot& snapshot, std::string& outError) {
    const auto validFrame = [&](const PapyrusCallFrameSnapshot& frame) {
        const auto function = m_functions.find(frame.function);
        return function != m_functions.end() && frame.instruction <= function->second.instructions.size();
    };
    for (const PapyrusThreadSnapshot& thread : snapshot.threads) {
        if (!validFrame(thread)) {
            outError = "save requires missing Papyrus function or invalid offset " + thread.function;
            return false;
        }
        if (thread.callStack.size() > kMaximumCallDepth) {
            outError = "saved Papyrus call stack exceeds limit";
            return false;
        }
        for (const PapyrusCallFrameSnapshot& frame : thread.callStack) {
            if (!validFrame(frame)) {
                outError = "save requires missing Papyrus caller or invalid offset " + frame.function;
                return false;
            }
        }
    }
    std::map<ObjectId, std::map<std::string, PapyrusScriptInstanceSnapshot>> instances;
    for (PapyrusScriptInstanceSnapshot instance : snapshot.instances) {
        if (!instance.object.valid() || instance.scriptClass.empty()) {
            outError = "save contains an invalid Papyrus script instance";
            return false;
        }
        instance.scriptClass = normalizedSymbol(std::move(instance.scriptClass));
        instance.activeState = normalizedSymbol(std::move(instance.activeState));
        auto& scripts = instances[instance.object];
        if (scripts.contains(instance.scriptClass)) {
            outError = "save contains duplicate Papyrus script instance " + instance.scriptClass;
            return false;
        }
        scripts.emplace(instance.scriptClass, std::move(instance));
    }
    for (const PapyrusUpdateRegistrationSnapshot& update : snapshot.updates) {
        const auto object = instances.find(update.object);
        if (update.intervalTicks == 0u || object == instances.end() ||
            !object->second.contains(normalizedSymbol(update.scriptClass)) ||
            (normalizedSymbol(update.eventFunction) != "onupdate" &&
             normalizedSymbol(update.eventFunction) != "onupdategametime")) {
            outError = "save contains an invalid Papyrus update registration";
            return false;
        }
    }
    m_nextThreadId = std::max<std::uint64_t>(1u, snapshot.nextThreadId);
    m_threads = snapshot.threads;
    m_pendingThreads.clear();
    m_advancing = false;
    m_globals = snapshot.globals;
    m_instances = std::move(instances);
    m_updates = snapshot.updates;
    outError.clear();
    return true;
}

void PapyrusVm::clearRuntimeState() {
    m_globals.clear();
    m_threads.clear();
    m_pendingThreads.clear();
    m_instances.clear();
    m_updates.clear();
    m_nextThreadId = 1u;
    m_advancing = false;
}

}  // namespace odai::bethesda
