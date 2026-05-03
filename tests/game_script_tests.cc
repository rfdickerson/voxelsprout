#include "game/game_state.h"
#include "game/lua_script.h"

#include <filesystem>
#include <iostream>

namespace {

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[game script test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void testLuaApiAndRollback() {
    odai::game::GameState state;
    odai::game::LuaScriptRuntime runtime;
    expectTrue(runtime.init(state), "Lua runtime initializes");
    expectTrue(runtime.loadScriptString(
                   "function on_activate(ref)\n"
                   "  game.add_item('tax', 1)\n"
                   "  game.add_gold(5)\n"
                   "  game.set_journal('q', 10)\n"
                   "  return { handled = true, message = ref }\n"
                   "end\n",
                   "api_test"),
               "Lua script string loads");
    const odai::game::ScriptCallResult result = runtime.onActivate("target");
    expectTrue(result.handled, "Lua activation handled");
    expectTrue(result.message == "target", "Lua activation returns message");
    expectTrue(state.itemCount("tax") == 1, "Lua can add item");
    expectTrue(state.gold() == 5, "Lua can add gold");
    expectTrue(state.journalStage("q") == 10, "Lua can set journal");

    expectTrue(runtime.loadScriptString(
                   "function on_activate(ref)\n"
                   "  game.add_gold(100)\n"
                   "  error('boom')\n"
                   "end\n",
                   "rollback_test"),
               "Lua rollback script loads");
    const int beforeGold = state.gold();
    const odai::game::ScriptCallResult failed = runtime.onActivate("target");
    expectTrue(!failed.handled, "Failed Lua activation is not handled");
    expectTrue(state.gold() == beforeGold, "Lua call failure rolls back state");
    expectTrue(!runtime.lastError().empty(), "Lua call records error");
}

void testDeathTaxmanQuestScript() {
    odai::game::GameState state;
    odai::game::LuaScriptRuntime runtime;
    expectTrue(runtime.init(state), "Death Taxman runtime initializes");
    const std::filesystem::path scriptPath =
        std::filesystem::path(ODAI_PROJECT_SOURCE_DIR) / "assets" / "scripts" / "mv_deadtaxman.lua";
    expectTrue(runtime.loadScriptFile(scriptPath), "Death Taxman script loads");

    const odai::game::ScriptCallResult corpse = runtime.onActivate("processus vitellius");
    expectTrue(corpse.handled, "Processus activation handled");
    expectTrue(state.journalStage("MV_DeadTaxman") == 10, "Corpse activation starts journal");
    expectTrue(state.itemCount("bk_seydaneentaxrecord") == 1, "Corpse activation adds tax record");
    expectTrue(state.gold() == 200, "Corpse activation adds tax gold");

    const odai::game::DialogueResult socucius =
        runtime.getDialogue("socucius ergalla", "murder of processus vitellius");
    expectTrue(socucius.handled, "Socucius dialogue handled");
    expectTrue(socucius.choices.size() == 2u, "Socucius offers truth/lie choices");

    const odai::game::ScriptCallResult truth = runtime.chooseDialogue("taxman_truth");
    expectTrue(truth.handled, "Truth dialogue choice handled");
    expectTrue(state.journalStage("MV_DeadTaxman") == 30, "Truth advances journal");
    expectTrue(state.gold() == 0, "Truth spends recovered tax gold");

    const odai::game::DialogueResult foryn = runtime.getDialogue("foryn gilnith", "");
    expectTrue(foryn.handled, "Foryn dialogue handled");
    expectTrue(!foryn.choices.empty(), "Foryn dialogue offers resolution");

    const odai::game::ScriptCallResult kill = runtime.chooseDialogue("taxman_kill_foryn");
    expectTrue(kill.handled, "Foryn kill choice handled");
    expectTrue(state.isRefDead("foryn gilnith"), "Foryn marked dead");
    expectTrue(state.journalStage("MV_DeadTaxman") == 70, "Foryn death advances journal");
    expectTrue(state.gold() == 500, "Reward is granted once");

    const odai::game::ScriptCallResult duplicate = runtime.chooseDialogue("taxman_kill_foryn");
    expectTrue(!duplicate.handled, "Completed reward choice is ignored");
    expectTrue(state.gold() == 500, "Reward is not duplicated");
}

void testSeydaNeenDialogueTopicsAndChoices() {
    odai::game::GameState state;
    odai::game::LuaScriptRuntime runtime;
    expectTrue(runtime.init(state), "Seyda Neen runtime initializes");
    const std::filesystem::path scriptPath =
        std::filesystem::path(ODAI_PROJECT_SOURCE_DIR) / "assets" / "scripts" / "mv_deadtaxman.lua";
    expectTrue(runtime.loadScriptFile(scriptPath), "Seyda Neen script loads");

    state.addItem("engraved_ring_of_healing", 1);
    const odai::game::DialogueResult fargothGreeting = runtime.getDialogue("fargoth", "");
    expectTrue(fargothGreeting.handled, "Fargoth greeting handled");
    expectTrue(!fargothGreeting.topics.empty(), "Fargoth greeting exposes topics");

    const odai::game::DialogueResult ringTopic = runtime.getDialogue("fargoth", "missing ring");
    expectTrue(ringTopic.handled, "Fargoth missing ring topic handled");
    expectTrue(ringTopic.choices.size() == 1u, "Fargoth ring topic offers return choice");

    const odai::game::ScriptCallResult returnedRing = runtime.chooseDialogue("fargoth_return_ring");
    expectTrue(returnedRing.handled, "Fargoth ring return handled");
    expectTrue(state.itemCount("engraved_ring_of_healing") == 0, "Fargoth ring removed");
    expectTrue(state.journalStage("MV_FargothRing") == 30, "Fargoth ring quest advances");

    const odai::game::ScriptCallResult duplicateRing = runtime.chooseDialogue("fargoth_return_ring");
    expectTrue(!duplicateRing.handled, "Fargoth duplicate ring return ignored");
    expectTrue(state.journalStage("MV_FargothRing") == 30, "Fargoth ring quest does not repeat");

    const odai::game::DialogueResult hrisskarGreeting = runtime.getDialogue("hrisskar flat-foot", "");
    bool hasHidingTopic = false;
    for (const odai::game::DialogueTopic& topic : hrisskarGreeting.topics) {
        hasHidingTopic = hasHidingTopic || topic.id == "fargoth's hiding place";
    }
    expectTrue(hasHidingTopic, "Hrisskar exposes Fargoth hiding topic after ring return");

    const odai::game::DialogueResult hidingTopic =
        runtime.getDialogue("hrisskar flat-foot", "fargoth's hiding place");
    expectTrue(hidingTopic.handled, "Hrisskar hiding topic handled");
    expectTrue(!hidingTopic.choices.empty(), "Hrisskar hiding topic offers quest choice");

    const int beforeGold = state.gold();
    const odai::game::ScriptCallResult invalid = runtime.chooseDialogue("not_a_real_choice");
    expectTrue(!invalid.handled, "Invalid dialogue choice is ignored");
    expectTrue(state.gold() == beforeGold, "Invalid dialogue choice does not mutate gold");
    expectTrue(state.journalStage("MV_FargothHiding") == 0, "Invalid dialogue choice does not mutate journal");
}

void testFargothNightRoutineAndStashQuest() {
    odai::game::GameState state;
    odai::game::LuaScriptRuntime runtime;
    expectTrue(runtime.init(state), "Fargoth night routine runtime initializes");
    const std::filesystem::path scriptPath =
        std::filesystem::path(ODAI_PROJECT_SOURCE_DIR) / "assets" / "scripts" / "mv_deadtaxman.lua";
    expectTrue(runtime.loadScriptFile(scriptPath), "Fargoth night routine script loads");

    odai::game::LuaScriptRuntime::NpcUpdateCommand beforeQuest =
        runtime.updateNpc("fargoth", -12369.92f, 0.0f, -69672.32f, 22.0f);
    expectTrue(beforeQuest.handled, "Fargoth update handles prequest state");
    expectTrue(beforeQuest.route.empty(), "Fargoth has no hiding route before Hrisskar quest");

    state.setJournalStage("MV_FargothRing", 30);
    state.setJournalStage("MV_FargothHiding", 10);
    odai::game::LuaScriptRuntime::NpcUpdateCommand daytime =
        runtime.updateNpc("fargoth", -12369.92f, 0.0f, -69672.32f, 14.0f);
    expectTrue(daytime.handled, "Fargoth daytime update handled");
    expectTrue(daytime.route.empty(), "Fargoth has no hiding route during daytime");
    expectTrue(state.journalStage("MV_FargothHiding") == 10, "Daytime update does not advance hiding quest");

    odai::game::LuaScriptRuntime::NpcUpdateCommand night =
        runtime.updateNpc("fargoth", -12369.92f, 0.0f, -69672.32f, 22.0f);
    expectTrue(night.handled, "Fargoth night update handled");
    expectTrue(night.route.size() >= 2u, "Fargoth gets hiding route at night");
    expectTrue(state.journalStage("MV_FargothHiding") == 12, "Night route marks Fargoth en route");

    const odai::game::ScriptCallResult earlyStash = runtime.onActivate("fargoth_stash");
    expectTrue(!earlyStash.handled, "Stash cannot be taken before Fargoth reveals it");
    expectTrue(state.itemCount("fargoth_stash") == 0, "Early stash activation gives no item");

    odai::game::LuaScriptRuntime::NpcUpdateCommand reveal =
        runtime.updateNpc("fargoth", -13352.96f, 0.0f, -68649.04f, 22.5f);
    expectTrue(reveal.handled, "Fargoth reveal update handled");
    expectTrue(reveal.stop, "Fargoth stops after revealing stash");
    expectTrue(state.journalStage("MV_FargothHiding") == 15, "Fargoth reveal advances hiding quest");

    const odai::game::ScriptCallResult stash = runtime.onActivate("fargoth_stash");
    expectTrue(stash.handled, "Revealed stash activation handled");
    expectTrue(state.itemCount("fargoth_stash") == 1, "Revealed stash gives stash item");
    const odai::game::ScriptCallResult duplicateStash = runtime.onActivate("fargoth_stash");
    expectTrue(duplicateStash.handled, "Duplicate stash activation gives feedback");
    expectTrue(state.itemCount("fargoth_stash") == 1, "Stash item is not duplicated");

    const int beforeTurnInGold = state.gold();
    const odai::game::ScriptCallResult turnIn = runtime.chooseDialogue("hrisskar_turn_in_stash");
    expectTrue(turnIn.handled, "Hrisskar stash turn-in handled");
    expectTrue(state.itemCount("fargoth_stash") == 0, "Hrisskar turn-in removes stash item");
    expectTrue(state.journalStage("MV_FargothHiding") == 20, "Hrisskar turn-in completes hiding quest");
    expectTrue(state.gold() == beforeTurnInGold + 100, "Hrisskar turn-in pays reward");

    const odai::game::ScriptCallResult duplicateTurnIn = runtime.chooseDialogue("hrisskar_turn_in_stash");
    expectTrue(!duplicateTurnIn.handled, "Hrisskar turn-in cannot repeat");
    expectTrue(state.gold() == beforeTurnInGold + 100, "Hrisskar reward is not duplicated");
}

}  // namespace

int main() {
    testLuaApiAndRollback();
    testDeathTaxmanQuestScript();
    testSeydaNeenDialogueTopicsAndChoices();
    testFargothNightRoutineAndStashQuest();

    if (g_failures != 0) {
        std::cerr << "[game script test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[game script test] all checks passed\n";
    return 0;
}
