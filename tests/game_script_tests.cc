#include "game/game_state.h"
#include "game/lua_script.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

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

void testPlayerStatsInventoryAndQuestViews() {
    odai::game::GameState state;
    odai::game::LuaScriptRuntime runtime;
    expectTrue(runtime.init(state), "Player stats runtime initializes");
    expectTrue(runtime.loadScriptString(
                   "function on_activate(ref)\n"
                   "  game.set_player_max_stats(120, 80, 110)\n"
                   "  game.set_player_stats(90, 50, 70)\n"
                   "  game.damage_player(25)\n"
                   "  game.restore_player(5, 10, 15)\n"
                   "  game.set_item_name('sample_item', 'Sample Item')\n"
                   "  game.add_item('sample_item', 2)\n"
                   "  game.define_quest('sample_quest', 'Sample Quest', 'Start it.')\n"
                   "  game.set_journal('sample_quest', 10)\n"
                   "  game.set_quest_objective('sample_quest', 'Finish it.')\n"
                   "  game.complete_quest('sample_quest')\n"
                   "  return { handled = true, message = 'ok' }\n"
                   "end\n",
                   "player_stats_test"),
               "Player stats script loads");
    const odai::game::ScriptCallResult result = runtime.onActivate("target");
    expectTrue(result.handled, "Player stats activation handled");
    expectTrue(state.playerStats().maxHealth == 120, "Player max health set");
    expectTrue(state.playerStats().health == 70, "Player health damage/restore applied");
    expectTrue(state.playerStats().magicka == 60, "Player magicka restore applied");
    expectTrue(state.playerStats().fatigue == 85, "Player fatigue restore applied");
    const std::vector<odai::game::InventoryEntry> inventory = state.inventoryEntries();
    const std::vector<odai::game::QuestEntry> quests = state.questEntries();
    expectTrue(inventory.size() == 1u, "Inventory view exposes item");
    expectTrue(!inventory.empty() && inventory[0].name == "Sample Item", "Inventory view exposes display name");
    expectTrue(quests.size() == 1u, "Quest view exposes quest");
    expectTrue(!quests.empty() && quests[0].completed, "Quest view exposes completion");
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

void testSeydaNeenScheduleCommands() {
    odai::game::GameState state;
    odai::game::LuaScriptRuntime runtime;
    expectTrue(runtime.init(state), "Seyda Neen schedule runtime initializes");
    const std::filesystem::path scriptPath =
        std::filesystem::path(ODAI_PROJECT_SOURCE_DIR) / "assets" / "scripts" / "mv_deadtaxman.lua";
    expectTrue(runtime.loadScriptFile(scriptPath), "Seyda Neen schedule script loads");

    const odai::game::LuaScriptRuntime::NpcUpdateCommand day =
        runtime.updateNpc("darvame hleran", -11000.0f, 0.0f, -69000.0f, 12.0f);
    expectTrue(day.handled, "Outdoor Seyda Neen NPC schedule is handled");
    expectTrue(day.state == "wander", "Day schedule wanders");
    expectTrue(day.anchor == "darvame_silt_strider", "Darvame day anchor is silt strider");
    expectTrue(day.speed > 0.0f, "Schedule provides movement speed");
    expectTrue(day.wanderRadius > 0.0f, "Wander schedule provides radius");

    const odai::game::LuaScriptRuntime::NpcUpdateCommand night =
        runtime.updateNpc("vodunius nuccius", -11000.0f, 0.0f, -69000.0f, 23.0f);
    expectTrue(night.handled, "Night schedule is handled");
    expectTrue(night.state == "travel", "Night schedule travels to rest anchor");
    expectTrue(night.anchor == "vodunius_home", "Vodunius night anchor is home");
    expectTrue(night.waitSeconds > 0.0f, "Night schedule includes wait time");

    const odai::game::LuaScriptRuntime::NpcUpdateCommand guard =
        runtime.updateNpc("seyda_neen_guard_1", -11000.0f, 0.0f, -69000.0f, 10.0f);
    expectTrue(guard.handled, "Generated Seyda Neen guard schedule is handled");
    expectTrue(guard.state == "wander", "Guard patrol uses wander state");
    expectTrue(guard.anchor == "dock", "Day guard anchor patrols dock");
    expectTrue(guard.priority == 10, "Guard schedule has higher priority");

    const odai::game::LuaScriptRuntime::NpcUpdateCommand unknown =
        runtime.updateNpc("not a seyda neen actor", 0.0f, 0.0f, 0.0f, 10.0f);
    expectTrue(!unknown.handled, "Unknown actor schedule is ignored");
}

void testSeydaNeenActorStatsAndServices() {
    odai::game::GameState state;
    odai::game::LuaScriptRuntime runtime;
    expectTrue(runtime.init(state), "Seyda Neen actor stats runtime initializes");
    const std::filesystem::path scriptPath =
        std::filesystem::path(ODAI_PROJECT_SOURCE_DIR) / "assets" / "scripts" / "mv_deadtaxman.lua";
    expectTrue(runtime.loadScriptFile(scriptPath), "Seyda Neen actor stats script loads");

    const odai::game::DialogueResult fargothStats = runtime.getDialogue("fargoth", "statistics");
    expectTrue(fargothStats.handled, "Fargoth stats topic handled");
    expectTrue(fargothStats.text.find("Level 2") != std::string::npos, "Fargoth level is exposed");
    expectTrue(fargothStats.text.find("Health 41") != std::string::npos, "Fargoth health is exposed");
    expectTrue(fargothStats.text.find("Magicka 82") != std::string::npos, "Fargoth magicka is exposed");

    const odai::game::DialogueResult socuciusServices = runtime.getDialogue("socucius ergalla", "services");
    expectTrue(socuciusServices.handled, "Socucius services topic handled");
    expectTrue(!socuciusServices.choices.empty(), "Socucius exposes service choices");
    bool hasTrainingChoice = false;
    for (const odai::game::DialogueChoice& choice : socuciusServices.choices) {
        hasTrainingChoice = hasTrainingChoice || choice.id == "train:socucius ergalla";
    }
    expectTrue(hasTrainingChoice, "Socucius exposes training choice");

    const odai::game::ScriptCallResult training = runtime.chooseDialogue("train:socucius ergalla");
    expectTrue(training.handled, "Training choice is handled");
    expectTrue(training.message.find("Agent") != std::string::npos, "Training response includes class");

    const odai::game::DialogueResult darvameServices = runtime.getDialogue("darvame hleran", "services");
    expectTrue(darvameServices.handled, "Darvame services topic handled");
    bool hasTravelChoice = false;
    for (const odai::game::DialogueChoice& choice : darvameServices.choices) {
        hasTravelChoice = hasTravelChoice || choice.id == "travel:darvame hleran";
    }
    expectTrue(hasTravelChoice, "Darvame exposes travel choice");
    expectTrue(runtime.chooseDialogue("travel:darvame hleran").handled, "Travel choice is handled");
}

void testCuratedMerchantBuying() {
    odai::game::GameState state;
    odai::game::LuaScriptRuntime runtime;
    expectTrue(runtime.init(state), "Curated merchant runtime initializes");
    const std::filesystem::path scriptPath =
        std::filesystem::path(ODAI_PROJECT_SOURCE_DIR) / "assets" / "scripts" / "mv_deadtaxman.lua";
    expectTrue(runtime.loadScriptFile(scriptPath), "Curated merchant script loads");

    const odai::game::DialogueResult arrilleBarter = runtime.getDialogue("arrille", "barter");
    expectTrue(arrilleBarter.handled, "Arrille barter dialogue handled");
    bool hasDagger = false;
    for (const odai::game::DialogueChoice& choice : arrilleBarter.choices) {
        hasDagger = hasDagger || choice.id == "buy:arrille:iron_dagger";
    }
    expectTrue(hasDagger, "Arrille barter exposes buy choices");

    state.addGold(20);
    const odai::game::ScriptCallResult boughtDagger =
        runtime.chooseDialogue("buy:arrille:iron_dagger");
    expectTrue(boughtDagger.handled, "Arrille purchase is handled");
    expectTrue(state.itemCount("iron_dagger") == 1, "Arrille purchase adds item");
    expectTrue(state.gold() == 2, "Arrille purchase spends gold");

    const int beforeFailedGold = state.gold();
    const int beforeFailedCount = state.itemCount("chitin_armor");
    const odai::game::ScriptCallResult expensivePurchase =
        runtime.chooseDialogue("buy:arrille:chitin_armor");
    expectTrue(expensivePurchase.handled, "Insufficient gold purchase gives feedback");
    expectTrue(state.gold() == beforeFailedGold, "Insufficient gold purchase keeps gold");
    expectTrue(state.itemCount("chitin_armor") == beforeFailedCount, "Insufficient gold purchase adds no item");

    const odai::game::DialogueResult ajiraBarter = runtime.getDialogue("ajira", "barter");
    expectTrue(ajiraBarter.handled, "Ajira barter dialogue handled");
    bool hasMagickaPotion = false;
    for (const odai::game::DialogueChoice& choice : ajiraBarter.choices) {
        hasMagickaPotion = hasMagickaPotion || choice.id == "buy:ajira:standard_restore_magicka_potion";
    }
    expectTrue(hasMagickaPotion, "Ajira barter exposes buy choices");

    expectTrue(!runtime.chooseDialogue("barter:fargoth").handled, "Non-merchant barter is ignored");
    expectTrue(!runtime.chooseDialogue("buy:fargoth:iron_dagger").handled, "Non-merchant buy is ignored");
}

void testBalmoraPlayableQuestSlice() {
    odai::game::GameState state;
    odai::game::LuaScriptRuntime runtime;
    expectTrue(runtime.init(state), "Balmora runtime initializes");
    const std::filesystem::path scriptPath =
        std::filesystem::path(ODAI_PROJECT_SOURCE_DIR) / "assets" / "scripts" / "mv_deadtaxman.lua";
    expectTrue(runtime.loadScriptFile(scriptPath), "Regional gameplay script loads");

    state.addItem("package_for_caius", 1);
    state.setJournalStage("MV_ReportToCaius", 30);
    const odai::game::DialogueResult caius = runtime.getDialogue("caius cosades", "orders");
    expectTrue(caius.handled, "Caius orders dialogue handled");
    expectTrue(!caius.choices.empty(), "Caius offers package delivery");
    expectTrue(runtime.chooseDialogue("caius_deliver_package").handled, "Caius package delivery handled");
    expectTrue(state.journalStage("MV_ReportToCaius") == 100, "Caius quest completes");
    expectTrue(state.itemCount("package_for_caius") == 0, "Caius package removed");

    const odai::game::DialogueResult eydis = runtime.getDialogue("eydis fire-eye", "balmora work");
    expectTrue(eydis.handled, "Eydis Balmora work dialogue handled");
    expectTrue(runtime.chooseDialogue("fg_start_rats").handled, "Fighters Guild rat job starts");
    expectTrue(state.journalStage("MV_FG_Rats") == 10, "Rat job journal starts");
    expectTrue(runtime.chooseDialogue("combat:fire-eye rat").handled, "Rat combat first attack handled");
    expectTrue(runtime.chooseDialogue("combat:fire-eye rat").handled, "Rat combat second attack handled");
    expectTrue(state.isRefDead("fire-eye rat"), "Rat marked dead");
    expectTrue(state.journalStage("MV_FG_Rats") == 100, "Rat job completes");

    expectTrue(runtime.chooseDialogue("mg_start_mushrooms").handled, "Mages Guild mushrooms completes");
    expectTrue(state.itemCount("luminous_russula") == 1, "Mushroom reward item exists");
    expectTrue(runtime.chooseDialogue("tg_start_diamond").handled, "Thieves Guild diamond completes");
    expectTrue(state.itemCount("diamond") == 1, "Diamond acquired");
    expectTrue(runtime.chooseDialogue("hh_start_records").handled, "House Hlaalu records completes");
    expectTrue(state.itemCount("hlaalu_records") >= 1, "Hlaalu records acquired");
}

}  // namespace

int main() {
    testLuaApiAndRollback();
    testPlayerStatsInventoryAndQuestViews();
    testDeathTaxmanQuestScript();
    testSeydaNeenDialogueTopicsAndChoices();
    testFargothNightRoutineAndStashQuest();
    testSeydaNeenScheduleCommands();
    testSeydaNeenActorStatsAndServices();
    testCuratedMerchantBuying();
    testBalmoraPlayableQuestSlice();

    if (g_failures != 0) {
        std::cerr << "[game script test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[game script test] all checks passed\n";
    return 0;
}
