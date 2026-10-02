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
    assert(session.configure({odai::importer::bethesda::BethesdaGame::SkyrimSpecialEdition,
                              "inventory-fixture",
                              "skyrim-bleak-falls",
                              1u,
                              {}},
                             error));
    const auto sword = makeRecordKey("Fixture.esm", 0x100);
    const auto axe = makeRecordKey("Fixture.esm", 0x101);
    const auto potion = makeRecordKey("Fixture.esm", 0x102);
    session.setSkyrimItems({{sword, {sword, "Sword", {}, {}, 12, 0, "WEAP", 1}},
                            {axe, {axe, "Axe", {}, {}, 20, 0, "WEAP", 3}},
                            {potion, {potion, "Healing", {}, {}, 0, 30}}});
    RuntimeObject player;
    player.id = session.playerObject();
    player.base = makeRecordKey("Skyrim.esm", 7);
    player.kind = RuntimeObjectKind::Actor;
    player.actorValues.emplace();
    player.actorValues->health = 50;
    player.inventory = {{sword, 1, true, kEquipmentRightHand}, {axe, 1, false}, {potion, 2, false}};
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
    // Slot changes preserve clothing, displace only conflicting hands, and
    // persist both biped and hand ownership through save/load.
    const auto armor = makeRecordKey("Fixture.esm", 0x103);
    const auto shield = makeRecordKey("Fixture.esm", 0x104);
    const auto greatsword = makeRecordKey("Fixture.esm", 0x105);
    session.setSkyrimItems({{axe, {axe, "Axe", {}, {}, 20, 0, "WEAP", 3}},
        {armor, {armor, "Armor", {}, {}, 0, 0, "ARMO", 0, 1u << 2}},
        {shield, {shield, "Shield", {}, {}, 0, 0, "ARMO", 0, 1u << 9}},
        {greatsword, {greatsword, "Greatsword", {}, {}, 30, 0, "WEAP", 5}}});
    current->inventory = {{armor, 1, true, 1u << 2}, {shield, 1, false},
        {axe, 1, false}, {greatsword, 1, false}};
    assert(session.equipActorItem(player.id, shield, true, false, error));
    assert(session.equipActorItem(player.id, axe, true, false, error));
    session.advance(1. / 60.);
    assert(current->inventory[0].equipped && current->inventory[1].equipped && current->inventory[2].equipped);
    assert(session.equipActorItem(player.id, greatsword, true, false, error));
    session.advance(1. / 60.);
    assert(current->inventory[0].equipped && !current->inventory[1].equipped && !current->inventory[2].equipped);
    assert(current->inventory[3].equipped);
    assert(saveOdaiGameAtomic(path, session, error));
    assert(loadOdaiGame(path, session, {}, report, error));
    current = session.world().find(player.id);
    assert(current->inventory[3].equipmentSlots == (kEquipmentLeftHand | kEquipmentRightHand));
    assert(current->inventory[0].equipped);
    assert(session.equipActorItem(player.id, armor, true, false, error, true, true));
    session.advance(1. / 60.);
    assert(session.equipActorItem(player.id, armor, false, false, error));
    assert(!session.advance(1. / 60.).diagnostics.empty());
    assert(current->inventory[0].equipped && current->inventory[0].preventUnequip);
    assert(saveOdaiGameAtomic(path, session, error));
    assert(loadOdaiGame(path, session, {}, report, error));
    current = session.world().find(player.id);
    assert(current->inventory[0].preventUnequip);
    assert(session.equipActorItem(player.id, armor, false, false, error, false, true));
    session.advance(1. / 60.);
    assert(!current->inventory[0].equipped && !current->inventory[0].preventUnequip);
    std::filesystem::remove(path);
    std::cout << "Skyrim inventory tests passed\n";
}
