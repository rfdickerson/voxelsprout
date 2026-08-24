#include "bethesda/tes3_script.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

using namespace odai::bethesda;

int failures = 0;

void check(bool value, const std::string& message) {
    if (!value) {
        std::cerr << "[tes3 script test] FAIL: " << message << '\n';
        ++failures;
    }
}

void testCompilerAndVm() {
    const std::string source = R"(begin TR_RuntimeTest
short count
short disabled
set count to 0
while ( count < 3 )
  set count to count + 1
endwhile
if ( GetJournalIndex "TR_TestQuest" = 0 )
  Journal "TR_TestQuest" 10
elseif ( count = 99 )
  Journal "TR_TestQuest" 99
else
  Journal "TR_TestQuest" 20
endif
"TR_TestDoor"->Enable
set disabled to "TR_TestDoor"->GetDisabled
Choice "Accept" 1 "Decline" 2
end)";
    Tes3ScriptCompiler compiler;
    const Tes3CompileResult compiled = compiler.compile(source);
    check(compiled.success(), "structured MWScript source compiles");
    check(compiled.program.id == "tr_runtimetest", "begin supplies normalized script id");
    check(compiled.program.locals.at("count") == Tes3LocalType::Short,
          "typed local declaration is retained");
    check(compiled.program.commands.contains("journal") &&
              compiled.program.commands.contains("enable") &&
              compiled.program.commands.contains("choice"),
          "command closure includes direct and target-qualified calls");

    Tes3ScriptVm vm;
    std::string error;
    check(vm.registerProgram(compiled.program, error), error);
    const std::uint64_t threadId = vm.start("TR_RuntimeTest", {}, error);
    check(threadId != 0u, error);
    std::map<std::string, int> journals;
    std::vector<Tes3NativeCall> calls;
    const auto execute = [&](const Tes3NativeCall& call) {
        calls.push_back(call);
        Tes3NativeResult result;
        if (call.command == "getjournalindex") {
            result.value = Tes3Value::fromNumber(journals[call.arguments.at(0).string]);
        } else if (call.command == "journal") {
            journals[call.arguments.at(0).string] =
                static_cast<int>(call.arguments.at(1).number);
        } else if (call.command == "getdisabled") {
            result.value = Tes3Value::fromNumber(1.0);
        } else if (call.command == "choice") {
            result.suspend = true;
            result.suspensionReason = "dialogue-choice";
        } else if (call.command != "enable") {
            result.error = "unexpected native " + call.command;
        }
        return result;
    };
    const Tes3VmStepResult first = vm.step(7u, 100u, execute);
    check(first.diagnostics.empty(), "deterministic VM executes without diagnostics");
    const Tes3ScriptThread& suspended = vm.threads().at(threadId);
    check(suspended.state == Tes3ThreadState::Suspended &&
              suspended.suspensionReason == "dialogue-choice",
          "Choice suspends a saveable script thread");
    check(suspended.locals.at("count").number == 3.0,
          "while loop and numeric assignment execute deterministically");
    check(journals["TR_TestQuest"] == 10,
          "native expression result selects the correct if branch");
    const auto enable = std::find_if(calls.begin(), calls.end(), [](const Tes3NativeCall& call) {
        return call.command == "enable";
    });
    check(enable != calls.end() && enable->target == "\"TR_TestDoor\"",
          "target-qualified call retains its authored target");
    check(suspended.locals.at("disabled").number == 1.0 &&
              std::any_of(calls.begin(), calls.end(), [](const Tes3NativeCall& call) {
                  return call.command == "getdisabled" &&
                      call.target == "\"TR_TestDoor\"";
              }),
          "target-qualified native expressions retain target and return value");

    check(vm.resume(threadId, error), error);
    const Tes3VmStepResult second = vm.step(8u, 10u, execute);
    check(second.completedThreads == 1u &&
              vm.threads().at(threadId).state == Tes3ThreadState::Completed,
          "resumed dialogue result completes at the saved instruction cursor");
}

void testDiagnosticsAndRegistry() {
    Tes3ScriptCompiler compiler;
    const Tes3CompileResult malformed = compiler.compile(
        "begin broken\nif ( 1 )\nJournal q 10\nend", "broken");
    check(!malformed.success(), "unterminated control flow is a compile error");
    const Tes3NativeRegistry registry = Tes3NativeRegistry::coreRuntimeRegistry();
    check(registry.find("JoUrNaL") != nullptr &&
              registry.find("journal")->disposition == Tes3NativeDisposition::Implemented,
          "native registry lookup is case-insensitive");
    check(registry.find("PlaySound3D") != nullptr &&
              !registry.find("PlaySound3D")->gameplayAffecting,
          "presentation-only no-op eligibility is explicit");
    check(registry.find("UnknownTRCommand") == nullptr,
          "unknown commands remain visible to strict closure checks");
}

}  // namespace

int main() {
    testCompilerAndVm();
    testDiagnosticsAndRegistry();
    if (failures == 0) std::cout << "tes3 script tests passed\n";
    return failures == 0 ? 0 : 1;
}
