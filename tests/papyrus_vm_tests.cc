#include "bethesda/papyrus_vm.h"
#include "bethesda/runtime_world.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace odai::bethesda;

namespace {

void put16(std::vector<std::uint8_t>& bytes, std::uint16_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value >> 8u));
    bytes.push_back(static_cast<std::uint8_t>(value));
}

void put32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (int byte = 3; byte >= 0; --byte) bytes.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
}

void put64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (int byte = 7; byte >= 0; --byte) bytes.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
}

void putString(std::vector<std::uint8_t>& bytes, const std::string& value) {
    put16(bytes, static_cast<std::uint16_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

void putIdentifier(std::vector<std::uint8_t>& bytes, std::uint16_t stringIndex) {
    bytes.push_back(static_cast<std::uint8_t>(PexValueKind::Identifier));
    put16(bytes, stringIndex);
}

void putInteger(std::vector<std::uint8_t>& bytes, std::int32_t value) {
    bytes.push_back(static_cast<std::uint8_t>(PexValueKind::Integer));
    put32(bytes, static_cast<std::uint32_t>(value));
}

}  // namespace

int main() {
    PapyrusVm vm;
    BethesdaWorld world;
    std::string error;
    PapyrusFunction function;
    function.name = "Fixture.OnInit";
    function.parameters = {"amount"};
    PapyrusInstruction assign;
    assign.opcode = PapyrusOpcode::Assign;
    assign.destination = "counter";
    assign.operands = {PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(1))};
    PapyrusInstruction add;
    add.opcode = PapyrusOpcode::IntegerAdd;
    add.destination = "counter";
    add.operands = {PapyrusOperand::fromLocal("counter"), PapyrusOperand::fromLocal("amount")};
    PapyrusInstruction wait;
    wait.opcode = PapyrusOpcode::WaitTicks;
    wait.operands = {PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(2))};
    PapyrusInstruction finish;
    finish.opcode = PapyrusOpcode::Return;
    function.instructions = {assign, add, wait, finish};
    assert(vm.registerFunction(function, error));
    const std::vector<PapyrusValue> arguments{PapyrusValue::fromInteger(4)};
    assert(vm.startFunction("fixture.oninit", arguments, error) != 0u);
    PapyrusAdvanceResult first = vm.advance(10u, 100u, world);
    assert(first.instructions == 3u && vm.activeThreadCount() == 1u);
    assert(vm.advance(11u, 100u, world).instructions == 0u);
    PapyrusVmSnapshot suspended = vm.snapshot();
    assert(suspended.threads[0].locals.at("counter").integer == 5);
    PapyrusAdvanceResult resumed = vm.advance(12u, 100u, world);
    assert(resumed.completedThreads == 1u && vm.activeThreadCount() == 0u);
    assert(vm.restore(suspended, error));
    assert(vm.activeThreadCount() == 1u);

    int latentCalls = 0;
    vm.registerNative("Utility.LatentFixture",
        [&](std::span<const PapyrusValue>, std::uint64_t tick, BethesdaWorld&) {
            ++latentCalls;
            NativeCallResult result;
            result.completed = false;
            result.resumeTick = tick + 2u;
            result.value = PapyrusValue::fromInteger(9);
            return result;
        });
    PapyrusFunction latent;
    latent.name = "Fixture.Latent";
    PapyrusInstruction latentCall;
    latentCall.opcode = PapyrusOpcode::CallNative;
    latentCall.name = "Utility.LatentFixture";
    latentCall.destination = "result";
    latent.instructions = {latentCall, finish};
    assert(vm.registerFunction(latent, error));
    assert(vm.startFunction("Fixture.Latent", {}, error) != 0u);
    (void)vm.advance(30u, 100u, world);
    assert(latentCalls == 1);
    (void)vm.advance(31u, 100u, world);
    assert(latentCalls == 1);
    (void)vm.advance(32u, 100u, world);
    assert(latentCalls == 1);

    PapyrusFunction unsupported;
    unsupported.name = "Fixture.Unsupported";
    PapyrusInstruction call;
    call.opcode = PapyrusOpcode::CallNative;
    call.name = "SKSE.ExecuteNativeCode";
    unsupported.instructions = {call};
    assert(vm.registerFunction(unsupported, error));
    assert(vm.startFunction("Fixture.Unsupported", {}, error) != 0u);
    const PapyrusAdvanceResult strict = vm.advance(20u, 100u, world);
    assert(!strict.diagnostics.empty());

    PapyrusFunction compatibilityRoot;
    compatibilityRoot.name = "Fixture.CompatibilityRoot";
    compatibilityRoot.instructions = {finish};
    assert(vm.registerFunction(std::move(compatibilityRoot), error));
    PapyrusFunction unreachableCompatibility;
    unreachableCompatibility.name = "Fixture.UnreachableCompatibility";
    PapyrusInstruction unreachableCall;
    unreachableCall.opcode = PapyrusOpcode::CallStatic;
    unreachableCall.name = "Missing.ScriptFunction";
    unreachableCompatibility.instructions = {unreachableCall, finish};
    assert(vm.registerFunction(std::move(unreachableCompatibility), error));
    const std::vector<std::string> compatibilityRoots{"Fixture.CompatibilityRoot"};
    assert(vm.unresolvedCallBindings(compatibilityRoots).empty());
    assert(!vm.unresolvedCallBindings().empty());

    // Synchronous user calls carry an explicit object context and a saveable
    // call stack. Instance properties are separate from static world state.
    PapyrusVm objectVm;
    std::vector<std::int64_t> captured;
    objectVm.registerNative("Fixture.Capture",
        [&](std::span<const PapyrusValue> values, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (values.size() != 1u || values[0].type != PapyrusValueType::Integer) {
                result.error = "expected one integer";
            } else {
                captured.push_back(values[0].integer);
            }
            return result;
        });

    PapyrusFunction doubleValue;
    doubleValue.name = "Fixture.Double";
    doubleValue.scriptClass = "Fixture";
    doubleValue.parameters = {"value"};
    PapyrusInstruction multiply;
    multiply.opcode = PapyrusOpcode::IntegerMultiply;
    multiply.destination = "result";
    multiply.operands = {PapyrusOperand::fromLocal("value"),
                         PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(2))};
    PapyrusInstruction returnResult = finish;
    returnResult.operands = {PapyrusOperand::fromLocal("result")};
    doubleValue.instructions = {multiply, returnResult};
    assert(objectVm.registerFunction(std::move(doubleValue), error));

    PapyrusFunction objectEvent;
    objectEvent.name = "Fixture.OnEvent";
    objectEvent.scriptClass = "Fixture";
    PapyrusInstruction getCounter;
    getCounter.opcode = PapyrusOpcode::PropertyGet;
    getCounter.name = "Counter";
    getCounter.targetType = "Fixture";
    getCounter.destination = "value";
    getCounter.operands = {PapyrusOperand::fromLocal("self")};
    PapyrusInstruction invokeDouble;
    invokeDouble.opcode = PapyrusOpcode::CallMethod;
    invokeDouble.name = "Double";
    invokeDouble.targetType = "Fixture";
    invokeDouble.destination = "doubled";
    invokeDouble.operands = {PapyrusOperand::fromLocal("self"),
                             PapyrusOperand::fromLocal("value")};
    PapyrusInstruction setCounter;
    setCounter.opcode = PapyrusOpcode::PropertySet;
    setCounter.name = "Counter";
    setCounter.targetType = "Fixture";
    setCounter.operands = {PapyrusOperand::fromLocal("self"),
                           PapyrusOperand::fromLocal("doubled")};
    PapyrusInstruction capture;
    capture.opcode = PapyrusOpcode::CallNative;
    capture.name = "Fixture.Capture";
    capture.operands = {PapyrusOperand::fromLocal("doubled")};
    objectEvent.instructions = {getCounter, invokeDouble, setCounter, capture, finish};
    assert(objectVm.registerFunction(std::move(objectEvent), error));

    const ObjectId scriptedObject = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x1234u));
    std::unordered_map<std::string, PapyrusValue> properties;
    properties.emplace("Counter", PapyrusValue::fromInteger(5));
    assert(objectVm.attachScript(scriptedObject, "Fixture", std::move(properties), error));
    assert(objectVm.startFunctionOnObject(scriptedObject, "Fixture", "OnEvent", {}, error) != 0u);
    const PapyrusAdvanceResult enteredCall = objectVm.advance(40u, 2u, world);
    assert(enteredCall.instructions == 2u);
    const PapyrusVmSnapshot nested = objectVm.snapshot();
    assert(nested.threads.size() == 1u && nested.threads[0].callStack.size() == 1u);
    assert(nested.threads[0].function == "fixture.double");
    assert(objectVm.restore(nested, error));
    const PapyrusAdvanceResult completedCall = objectVm.advance(40u, 100u, world);
    assert(completedCall.completedThreads == 1u && completedCall.diagnostics.empty());
    const PapyrusValue* counter = objectVm.findProperty(scriptedObject, "fixture", "counter");
    assert(counter != nullptr && counter->type == PapyrusValueType::Integer && counter->integer == 10);
    assert(captured == std::vector<std::int64_t>{10});

    PapyrusFunction arrays;
    arrays.name = "Fixture.Arrays";
    PapyrusInstruction create;
    create.opcode = PapyrusOpcode::ArrayCreate;
    create.destination = "items";
    create.operands = {PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(3))};
    PapyrusInstruction set;
    set.opcode = PapyrusOpcode::ArraySetElement;
    set.operands = {PapyrusOperand::fromLocal("items"),
                    PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(1)),
                    PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(7))};
    PapyrusInstruction get;
    get.opcode = PapyrusOpcode::ArrayGetElement;
    get.destination = "item";
    get.operands = {PapyrusOperand::fromLocal("items"),
                    PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(1))};
    PapyrusInstruction greater;
    greater.opcode = PapyrusOpcode::CompareGreater;
    greater.destination = "valid";
    greater.operands = {PapyrusOperand::fromLocal("item"),
                        PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(5))};
    PapyrusInstruction branch;
    branch.opcode = PapyrusOpcode::JumpIfFalse;
    branch.jumpOffset = 2;
    branch.operands = {PapyrusOperand::fromLocal("valid")};
    PapyrusInstruction captureItem = capture;
    captureItem.operands = {PapyrusOperand::fromLocal("item")};
    arrays.instructions = {create, set, get, greater, branch, captureItem, finish};
    assert(objectVm.registerFunction(std::move(arrays), error));
    assert(objectVm.startFunction("Fixture.Arrays", {}, error) != 0u);
    assert(objectVm.advance(41u, 100u, world).diagnostics.empty());
    assert(captured == std::vector<std::int64_t>({10, 7}));

    PapyrusFunction onUpdate;
    onUpdate.name = "Fixture.OnUpdate";
    onUpdate.scriptClass = "Fixture";
    PapyrusInstruction captureUpdate = capture;
    captureUpdate.operands = {
        PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(99))};
    onUpdate.instructions = {captureUpdate, finish};
    assert(objectVm.registerFunction(std::move(onUpdate), error));
    assert(objectVm.registerForUpdate(
        scriptedObject, "Fixture", 2.0 / 60.0, 100u, false, error));
    assert(objectVm.snapshot().updates.size() == 1u);
    assert(objectVm.advance(101u, 100u, world).instructions == 0u);
    assert(objectVm.advance(102u, 100u, world).completedThreads == 1u);
    assert(objectVm.snapshot().updates.empty());
    assert(captured == std::vector<std::int64_t>({10, 7, 99}));

    PapyrusFunction onUpdateGameTime;
    onUpdateGameTime.name = "Fixture.OnUpdateGameTime";
    onUpdateGameTime.scriptClass = "Fixture";
    PapyrusInstruction captureGameTime = capture;
    captureGameTime.operands = {
        PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(100))};
    onUpdateGameTime.instructions = {captureGameTime, finish};
    assert(objectVm.registerFunction(std::move(onUpdateGameTime), error));
    assert(objectVm.registerForUpdate(
        scriptedObject, "Fixture", 1.0 / 60.0, 200u, false, error,
        "OnUpdateGameTime"));
    const PapyrusVmSnapshot gameTimeSnapshot = objectVm.snapshot();
    assert(gameTimeSnapshot.updates.size() == 1u);
    assert(gameTimeSnapshot.updates.front().eventFunction == "onupdategametime");
    assert(objectVm.advance(201u, 100u, world).completedThreads == 1u);
    assert(objectVm.snapshot().updates.empty());
    assert(captured == std::vector<std::int64_t>({10, 7, 99, 100}));

    std::vector<std::uint8_t> pex;
    put32(pex, 0xfa57c0deu);
    pex.push_back(3u); pex.push_back(2u); put16(pex, 1u); put64(pex, 1234u);
    putString(pex, "fixture.psc"); putString(pex, "tester"); putString(pex, "machine");
    const std::vector<std::string> strings = {
        "", "Fixture", "Parent", "OnInit", "None", "value", "Int"};
    put16(pex, static_cast<std::uint16_t>(strings.size()));
    for (const std::string& value : strings) putString(pex, value);
    PexModuleInfo info;
    assert(readPexModuleInfo(pex, info, error));
    assert(info.majorVersion == 3u && info.strings.size() == strings.size() && info.bigEndian);

    // Complete Skyrim PEX object/state/function fixture: OnInit(int value)
    // performs value = value + 1 and returns.
    pex.push_back(0u);                       // no debug info
    put16(pex, 0u);                          // user flags
    put16(pex, 1u);                          // objects
    put16(pex, 1u); put32(pex, 0u);          // Fixture, object size
    put16(pex, 2u); put16(pex, 0u);          // Parent, doc
    put32(pex, 0u); put16(pex, 0u);          // user flags, auto state
    put16(pex, 0u); put16(pex, 0u);          // variables, properties
    put16(pex, 1u);                          // states
    put16(pex, 0u); put16(pex, 1u);          // empty state, one function
    put16(pex, 3u);                          // OnInit
    put16(pex, 4u); put16(pex, 0u);          // return type, doc
    put32(pex, 0u); pex.push_back(0u);       // flags
    put16(pex, 1u); put16(pex, 5u); put16(pex, 6u);  // parameter
    put16(pex, 0u); put16(pex, 2u);          // locals, instructions
    pex.push_back(1u);                       // iadd
    putIdentifier(pex, 5u); putIdentifier(pex, 5u); putInteger(pex, 1);
    pex.push_back(26u);                      // ret
    pex.push_back(static_cast<std::uint8_t>(PexValueKind::None));
    PexScript decoded;
    assert(readPexScript(pex, decoded, error));
    assert(decoded.objects == std::vector<std::string>{"Fixture"});
    assert(decoded.objectInfo.size() == 1u && decoded.objectInfo[0].parentClass == "Parent");
    assert(decoded.functions.size() == 1u && decoded.functions[0].instructions.size() == 2u);
    assert(decoded.functions[0].parameterTypes == std::vector<std::string>{"Int"});
    const PexCompatibilityReport report = inspectPexCompatibility(decoded);
    assert(report.compatibilityErrors.empty());
    PapyrusVm decodedVm;
    PexCompatibilityReport loadReport;
    assert(decodedVm.loadPexScript(decoded, true, loadReport, error));
    assert(decodedVm.startFunction("fixture.oninit", arguments, error) != 0u);
    assert(decodedVm.advance(1u, 10u, world).completedThreads == 1u);

    // PEX auto-properties are accessed through their hidden backing variable,
    // not a propget opcode. VMAD attaches the canonical property name; the VM
    // must bridge ::Quest_var back to that value for retail fragment code.
    PexScript autoPropertyScript;
    PexObjectInfo autoObject;
    autoObject.name = "AutoFixture";
    autoObject.parentClass = "ReferenceAlias";
    PexVariableInfo backingVariable;
    backingVariable.name = "::Quest_var";
    backingVariable.type = "Quest";
    autoObject.variables.push_back(backingVariable);
    PexPropertyInfo questProperty;
    questProperty.name = "Quest";
    questProperty.type = "Quest";
    questProperty.autoVariable = "::Quest_var";
    questProperty.flags = 0x4u;
    autoObject.properties.push_back(questProperty);
    autoPropertyScript.objects.push_back(autoObject.name);
    autoPropertyScript.objectInfo.push_back(autoObject);
    PexFunctionInfo autoEvent;
    autoEvent.objectName = "AutoFixture";
    autoEvent.name = "OnEvent";
    autoEvent.returnType = "None";
    autoEvent.locals = {"::temp0"};
    autoEvent.localTypes = {"None"};
    PexInstructionInfo callAutoProperty;
    callAutoProperty.opcode = 23u;
    callAutoProperty.arguments = {
        PexValue{PexValueKind::Identifier, "Mark"},
        PexValue{PexValueKind::Identifier, "::Quest_var"},
        PexValue{PexValueKind::Identifier, "::temp0"},
        PexValue{PexValueKind::Integer, {}, 0}};
    PexInstructionInfo returnNone;
    returnNone.opcode = 26u;
    returnNone.arguments = {PexValue{}};
    autoEvent.instructions = {callAutoProperty, returnNone};
    autoPropertyScript.functions.push_back(autoEvent);
    PapyrusVm autoPropertyVm;
    ObjectId capturedQuest;
    autoPropertyVm.registerNative("Quest.Mark",
        [&](std::span<const PapyrusValue> values, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (values.size() != 1u || values[0].type != PapyrusValueType::Object) {
                result.error = "Quest.Mark expected its method receiver";
            } else {
                capturedQuest = values[0].object;
            }
            return result;
        });
    assert(autoPropertyVm.loadPexScript(
        autoPropertyScript, true, loadReport, error));
    const ObjectId aliasObject = ObjectId::runtime(77u);
    const ObjectId questObject = ObjectId::persistent(
        makeRecordKey("Skyrim.esm", 0x39645u));
    assert(autoPropertyVm.attachScript(aliasObject, "AutoFixture",
        {{"Quest", PapyrusValue::fromObject(questObject)}}, error));
    assert(autoPropertyVm.startFunctionOnObject(
        aliasObject, "AutoFixture", "OnEvent", {}, error) != 0u);
    assert(autoPropertyVm.advance(2u, 10u, world).diagnostics.empty());
    assert(capturedQuest == questObject);

    std::vector<std::uint8_t> truncated = pex;
    truncated.pop_back();
    assert(!readPexScript(truncated, decoded, error));

    std::cout << "papyrus VM tests passed\n";
    return 0;
}
