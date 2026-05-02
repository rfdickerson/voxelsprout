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

}  // namespace

int main() {
    testLuaApiAndRollback();
    testDeathTaxmanQuestScript();

    if (g_failures != 0) {
        std::cerr << "[game script test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[game script test] all checks passed\n";
    return 0;
}
