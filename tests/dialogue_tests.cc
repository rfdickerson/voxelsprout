// Correctness tests for the dialogue-tree system (Dragon Age: Origins
// touchstone, see docs/ROADMAP.md's Party RPG / Narrative section): JSON
// loading, the runtime's condition filtering and effect application, the
// auto-continue narration chain, and defensive handling of malformed/cyclic
// authored data. No Vulkan, no GTest -- same lightweight harness style as
// job_system_tests.cc / economy_tests.cc.

#include "dialogue/dialogue_context.h"
#include "dialogue/dialogue_io.h"
#include "dialogue/dialogue_runtime.h"
#include "dialogue/dialogue_state_io.h"
#include "dialogue/dialogue_types.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

using namespace odai::dialogue;

int g_failures = 0;

void expectTrue(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[dialogue test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectEqStr(const std::string& actual, const std::string& expected, const std::string& message) {
    if (actual != expected) {
        std::cerr << "[dialogue test] FAIL: " << message << " (expected '" << expected << "', got '"
                   << actual << "')\n";
        ++g_failures;
    }
}

// A small two-branch conversation used by most of the runtime tests below:
// an opening line with three choices (one always available, one gated on a
// flag, one gated on companion approval), each leading to its own reply node,
// plus a narration-only node that auto-continues into a terminal goodbye.
constexpr const char* kSampleTree = R"json({
  "id": "sample",
  "start": "greet",
  "nodes": {
    "greet": {
      "speaker": "Alistair",
      "text": "Well met, traveler.",
      "choices": [
        { "text": "Hello.", "next": "hello_reply" },
        { "text": "Have we met before?", "next": "recall_reply",
          "condition": { "flag": "met_alistair" } },
        { "text": "You look troubled.", "next": "confide_reply",
          "condition": { "companion": "alistair", "minApproval": 10 } },
        { "text": "[Leave]", "next": "" }
      ]
    },
    "hello_reply": {
      "speaker": "Alistair",
      "text": "Good day to you too.",
      "choices": [
        { "text": "Continue.", "next": "",
          "effects": [
            { "setFlag": "met_alistair", "value": true },
            { "companion": "alistair", "approvalDelta": 5 }
          ] }
      ]
    },
    "recall_reply": { "speaker": "Alistair", "text": "Aye, that we have.", "autoNext": "goodbye" },
    "confide_reply": { "speaker": "Alistair", "text": "...perhaps I am.", "autoNext": "goodbye" },
    "goodbye": { "speaker": "Alistair", "text": "Safe travels." }
  }
})json";

void testLoadsSimpleTree() {
    const DialogueLoadResult result = loadDialogueTreeFromJson(kSampleTree, "kSampleTree");
    expectTrue(result.ok(), "sample tree loads without errors");
    expectEqStr(result.tree.id, "sample", "tree id parsed");
    expectEqStr(result.tree.startNode, "greet", "start node parsed");
    expectTrue(result.tree.nodes.size() == 5, "all five nodes parsed");
    const auto it = result.tree.nodes.find("greet");
    expectTrue(it != result.tree.nodes.end(), "greet node present");
    if (it != result.tree.nodes.end()) {
        expectTrue(it->second.choices.size() == 4, "greet has four authored choices");
        expectEqStr(it->second.choices[0].text, "Hello.", "first choice text parsed");
        expectEqStr(it->second.choices[0].targetNode, "hello_reply", "first choice target parsed");
    }
}

void testMalformedJsonRecordsError() {
    const DialogueLoadResult result = loadDialogueTreeFromJson("{ not valid json", "bad");
    expectTrue(!result.ok(), "malformed JSON is not ok()");
    expectTrue(!result.errors.empty(), "malformed JSON records an error");
}

void testMissingNodesObjectRecordsError() {
    const DialogueLoadResult result = loadDialogueTreeFromJson(R"({"id":"x","start":"a"})", "no-nodes");
    expectTrue(!result.ok(), "a tree with no 'nodes' object is not ok()");
}

void testUnknownReferencesRecordError() {
    constexpr const char* kBroken = R"json({
      "id": "broken", "start": "a",
      "nodes": { "a": { "text": "hi", "choices": [ { "text": "go", "next": "nowhere" } ] } }
    })json";
    const DialogueLoadResult result = loadDialogueTreeFromJson(kBroken, "kBroken");
    expectTrue(!result.ok(), "a choice targeting an unknown node records an error");
    expectTrue(result.tree.nodes.size() == 1, "the rest of the tree still loads (non-fatal)");
}

void testRuntimeWalksBasicBranchAndApplyEffects() {
    const DialogueLoadResult result = loadDialogueTreeFromJson(kSampleTree, "kSampleTree");
    expectTrue(result.ok(), "sample tree loads for runtime test");

    MapDialogueContext ctx;
    DialogueRuntime runtime;
    runtime.begin(result.tree, ctx);

    expectTrue(!runtime.isFinished(), "runtime is not finished after begin()");
    const DialogueNode* node = runtime.currentNode();
    expectTrue(node != nullptr, "currentNode is non-null at greet");
    if (node != nullptr) expectEqStr(node->speaker, "Alistair", "speaker on greet node");

    // Only the always-available choice and the [Leave] choice should show —
    // the other two are gated on state nothing has set yet.
    const std::vector<const DialogueChoice*> choices = runtime.availableChoices();
    expectTrue(choices.size() == 2, "gated choices are hidden until their condition passes");

    runtime.choose(*choices[0]);  // "Hello." -> hello_reply
    node = runtime.currentNode();
    expectTrue(node != nullptr, "advanced to hello_reply");
    if (node != nullptr) expectEqStr(node->text, "Good day to you too.", "hello_reply text");

    const std::vector<const DialogueChoice*> replyChoices = runtime.availableChoices();
    expectTrue(replyChoices.size() == 1, "hello_reply has exactly one choice");
    runtime.choose(*replyChoices[0]);  // applies effects, then "" ends the dialogue

    expectTrue(ctx.flag("met_alistair"), "choosing the reply set the met_alistair flag");
    expectTrue(ctx.approval("alistair") == 5, "choosing the reply raised alistair's approval by 5");
    expectTrue(runtime.isFinished(), "an empty targetNode ends the dialogue");
    expectTrue(runtime.currentNode() == nullptr, "currentNode is null once finished");
}

void testFlagGatedChoiceUnlocksOnSecondVisit() {
    const DialogueLoadResult result = loadDialogueTreeFromJson(kSampleTree, "kSampleTree");
    MapDialogueContext ctx;
    ctx.setFlag("met_alistair", true);

    DialogueRuntime runtime;
    runtime.begin(result.tree, ctx);
    const std::vector<const DialogueChoice*> choices = runtime.availableChoices();
    expectTrue(choices.size() == 3, "met_alistair flag unlocks the recall choice");
}

void testApprovalGatedChoice() {
    const DialogueLoadResult result = loadDialogueTreeFromJson(kSampleTree, "kSampleTree");
    MapDialogueContext ctx;
    ctx.adjustApproval("alistair", 10);

    DialogueRuntime runtime;
    runtime.begin(result.tree, ctx);
    const std::vector<const DialogueChoice*> choices = runtime.availableChoices();
    expectTrue(choices.size() == 3, "approval >= minApproval unlocks the confide choice");
}

void testAutoContinueChainSkipsNarrationOnlyNodes() {
    const DialogueLoadResult result = loadDialogueTreeFromJson(kSampleTree, "kSampleTree");
    MapDialogueContext ctx;
    ctx.setFlag("met_alistair", true);

    DialogueRuntime runtime;
    runtime.begin(result.tree, ctx);
    const std::vector<const DialogueChoice*> choices = runtime.availableChoices();
    // choices[0]=Hello, choices[1]=recall (unlocked), choices[2]=[Leave]
    runtime.choose(*choices[1]);

    // recall_reply has no choices and an autoNext to goodbye, which has no
    // choices and no autoNext -- the runtime should land directly on goodbye.
    const DialogueNode* node = runtime.currentNode();
    expectTrue(!runtime.isFinished(), "auto-continue lands on a real node, not finished");
    expectTrue(node != nullptr, "currentNode is non-null after auto-continue");
    if (node != nullptr) expectEqStr(node->id, "goodbye", "auto-continue skipped the narration-only node");
    expectTrue(runtime.availableChoices().empty(), "the terminal node has no choices");
}

void testDialogueStateJsonRoundTrip() {
    MapDialogueContext original;
    original.setFlag("met_alistair", true);
    original.setFlag("cleared_tower", false);
    original.adjustApproval("alistair", 15);
    original.adjustApproval("morrigan", -5);

    const std::string json = saveDialogueStateToJson(original);

    MapDialogueContext loaded;
    const bool ok = loadDialogueStateFromJson(json, loaded);
    expectTrue(ok, "loadDialogueStateFromJson reports success on well-formed input");
    expectTrue(loaded.flag("met_alistair"), "round-tripped flag 'met_alistair' stayed true");
    expectTrue(!loaded.flag("cleared_tower"), "round-tripped flag 'cleared_tower' stayed false");
    expectTrue(loaded.approval("alistair") == 15, "round-tripped approval for 'alistair' survives");
    expectTrue(loaded.approval("morrigan") == -5, "round-tripped approval for 'morrigan' survives");
    expectTrue(loaded.approval("wynne") == 0, "a companion never touched defaults to zero approval");
}

void testDialogueStateFileRoundTrip() {
    MapDialogueContext original;
    original.setFlag("recruited_sten", true);
    original.adjustApproval("sten", 8);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "odai_dialogue_state_test.json";
    const bool saved = saveDialogueState(original, path);
    expectTrue(saved, "saveDialogueState writes the file successfully");

    MapDialogueContext loaded;
    const bool loadedOk = loadDialogueState(path, loaded);
    expectTrue(loadedOk, "loadDialogueState reads back a file it just wrote");
    expectTrue(loaded.flag("recruited_sten"), "file round-trip preserves flags");
    expectTrue(loaded.approval("sten") == 8, "file round-trip preserves approval");

    std::filesystem::remove(path);
}

void testDialogueStateMalformedJsonRecordsError() {
    MapDialogueContext ctx;
    const bool ok = loadDialogueStateFromJson("{ not valid json", ctx);
    expectTrue(!ok, "loadDialogueStateFromJson rejects malformed JSON");
    expectTrue(!getDialogueStateLastError().empty(), "a parse failure leaves a non-empty last-error message");
}

void testDialogueStateMissingFileRecordsError() {
    MapDialogueContext ctx;
    const bool ok = loadDialogueState("/nonexistent/odai_dialogue_state_missing.json", ctx);
    expectTrue(!ok, "loadDialogueState reports failure for a path that doesn't exist");
}

void testCyclicAutoNextDoesNotHang() {
    // Built by hand (not via JSON) so the loader's reference validation
    // doesn't get in the way of testing the runtime's own hop-cap guard.
    DialogueTree tree;
    tree.id = "cyclic";
    tree.startNode = "a";
    DialogueNode a;
    a.id = "a";
    a.autoNext = "b";
    DialogueNode b;
    b.id = "b";
    b.autoNext = "a";
    tree.nodes.emplace("a", a);
    tree.nodes.emplace("b", b);

    MapDialogueContext ctx;
    DialogueRuntime runtime;
    runtime.begin(tree, ctx);  // must return, not hang
    expectTrue(runtime.isFinished(), "a cyclic autoNext chain is treated as a dead end, not a hang");
}

}  // namespace

int main() {
    testLoadsSimpleTree();
    testMalformedJsonRecordsError();
    testMissingNodesObjectRecordsError();
    testUnknownReferencesRecordError();
    testRuntimeWalksBasicBranchAndApplyEffects();
    testFlagGatedChoiceUnlocksOnSecondVisit();
    testApprovalGatedChoice();
    testAutoContinueChainSkipsNarrationOnlyNodes();
    testDialogueStateJsonRoundTrip();
    testDialogueStateFileRoundTrip();
    testDialogueStateMalformedJsonRecordsError();
    testDialogueStateMissingFileRecordsError();
    testCyclicAutoNextDoesNotHang();

    if (g_failures != 0) {
        std::cerr << "[dialogue test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[dialogue test] all checks passed\n";
    return 0;
}
