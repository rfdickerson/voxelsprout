#include "bethesda/bethesda_session.h"
#include "bethesda/save_game.h"
#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>
using namespace odai::bethesda;
int main() {
    BethesdaSession session;
    std::string error;
    assert(session.configure({odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition,
                              "inventory-fixture",
                              "skyrim-bleak-falls",
                              1u,
                              {}},
                             error));
    const auto sword = makeRecordKey("Fixture.esm", 0x100);
    const auto axe = makeRecordKey("Fixture.esm", 0x101);
    const auto potion = makeRecordKey("Fixture.esm", 0x102);
    session.setSkyrimItems({{sword, {sword, "Sword", {}, {}, 12, 0}},
                            {axe, {axe, "Axe", {}, {}, 20, 0}},
                            {potion, {potion, "Healing", {}, {}, 0, 30}}});
    RuntimeObject player;
    player.id = session.playerObject();
    player.base = makeRecordKey("Skyrim.esm", 7);
    player.kind = RuntimeObjectKind::Actor;
    player.actorValues.emplace();
    player.actorValues->health = 50;
    player.inventory = {{sword, 1, true}, {axe, 1, false}, {potion, 2, false}};
    assert(session.world().addInitialObject(player, error));
    assert(session.useInventoryItem(player.id, axe, error));
    assert(session.useInventoryItem(player.id, potion, error));
    auto step = session.advance(1. / 60.);
    assert(step.diagnostics.empty());
    auto *current = session.world().find(player.id);
    assert(current->actorValues->health == 80);
    assert(!current->inventory[0].equipped && current->inventory[1].equipped);
    assert(current->inventory[2].count == 1);
    // Two same-frame attempts cannot heal twice using the same remaining potion.
    assert(session.useInventoryItem(player.id, potion, error));
    assert(session.useInventoryItem(player.id, potion, error));
    const auto path =
        std::filesystem::temp_directory_path() /
        ("odai-inventory-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    assert(!saveOdaiGameAtomic(path, session, error));
    assert(!std::filesystem::exists(path));
    step = session.advance(1. / 60.);
    assert(current->actorValues->health == 100);
    assert(current->inventory.size() == 2);
    assert(!session.useInventoryItem(player.id, potion, error));
    // Journal metadata stays immutable while saved stage state chooses the visible entry.
    SkyrimQuestDefinition journal;
    journal.record = makeRecordKey("Fixture.esm", 0x400);
    journal.editorId = "JournalFixture";
    journal.title = "A Journey";
    SkyrimQuestStageDefinition early, future;
    early.index = 10;
    early.logEntries.push_back({0, {}, 0, "Recover the <Alias=Treasure>."});
    future.index = 20;
    future.logEntries.push_back({0, {}, 0, "The journey is complete."});
    journal.stages = {early, future};
    assert(session.registerQuestDefinition(journal, {}, error));
    auto& journalState = session.quest("JournalFixture");
    RuntimeObject treasure;
    treasure.id = ObjectId::persistent(sword);
    treasure.base = sword;
    treasure.kind = RuntimeObjectKind::Item;
    assert(session.world().addInitialObject(treasure, error));
    QuestAliasRuntimeState alias;
    alias.name = "Treasure";
    alias.id = 0;
    alias.handle = session.world().allocateRuntimeId();
    alias.target = ObjectId::persistent(sword);
    journalState.aliases.push_back(alias);
    assert(session.questJournalSummary(journalState).empty());
    session.setQuestStage("JournalFixture", 10);
    assert(session.questJournalTitle(journalState) == "A Journey");
    assert(session.questJournalSummary(journalState) == "Recover the Sword.");
    assert(saveOdaiGameAtomic(path, session, error));
    SaveLoadReport report;
    if (!loadOdaiGame(path, session, {}, report, error)) { std::cerr << error << "\n"; return 1; }
    assert(session.questJournalSummary(session.quest("JournalFixture")) == "Recover the Sword.");
    assert(session.questJournalTitle(session.quest("JournalFixture")) == "A Journey");
    current = session.world().find(player.id);
    assert(current->inventory[1].equipped);
    assert(current->actorValues->health == 100 && current->inventory.size() == 2);
    RuntimeObject corpse;
    corpse.id = session.world().allocateRuntimeId();
    corpse.base = makeRecordKey("Fixture.esm", 0x900);
    corpse.kind = RuntimeObjectKind::Actor;
    corpse.actorValues.emplace();
    corpse.inventory = {{axe, 5, false}};
    assert(session.world().addInitialObject(corpse, error));
    assert(!session.lootObject(player.id, corpse.id, axe, 1).accepted);
    session.world().find(corpse.id)->actorValues->dead = true;
    assert(session.lootObject(player.id, corpse.id, axe, 4).accepted);
    assert(session.lootObject(player.id, corpse.id, axe, 4).accepted);
    step = session.advance(1. / 60.);
    assert(!step.diagnostics.empty()); // The second stale request was rejected atomically.
    assert(session.world().find(corpse.id)->inventory[0].count == 1);
    current = session.world().find(player.id);
    assert(current->inventory[1].count == 5);
    assert(!session.lootObject(player.id, corpse.id, axe, 2).accepted);
    std::filesystem::remove(path);
    std::cout << "Skyrim inventory tests passed\n";
}
