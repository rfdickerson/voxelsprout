#include "bethesda/route_checkpoint.h"
#include <iostream>
#include <stdexcept>
using namespace odai::bethesda;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
    BethesdaSession session;
    std::string error;
    require(session.configure({odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition,
        "synthetic", "skyrim-bleak-falls", 1u}, error), "configure");
    for (const auto& record : skyrimBleakFallsScenario().questRecords) {
        SkyrimQuestDefinition definition{};
        definition.editorId = record.editorId;
        definition.record = makeRecordKey(record.plugin,record.localFormId);
        require(session.registerQuestDefinition(definition,{},error), "register quest");
    }
    auto* mq102 = session.findQuest(ObjectId::persistent(makeRecordKey("Skyrim.esm",0x4e50du)));
    require(mq102 != nullptr,"MQ102 missing");
    mq102->running = true;
    mq102->objectives.push_back({10,true,false,false,{}});
    require(assessBleakFallsStart(session)["ok"], "valid bootstrap rejected");
    require(!assessBleakFallsStart(session)["player_inventory_verified"], "absent player verified");
    const auto before = session.deterministicHash();
    const auto snapshot = routeCheckpoint(session);
    require(snapshot == routeCheckpoint(session), "unstable snapshot");
    require(before == session.deterministicHash(), "report mutated session");
    RuntimeObject player;
    player.id = session.playerObject();
    player.base = makeRecordKey("Skyrim.esm",0x7u);
    player.kind = RuntimeObjectKind::Actor;
    player.actorValues.emplace();
    require(session.world().addInitialObject(player,error), "add player");
    require(assessBleakFallsStart(session)["player_inventory_verified"], "player inventory not verified");
    auto* livePlayer = session.world().find(player.id);
    livePlayer->inventory.push_back({makeRecordKey("Skyrim.esm",0xad06fu),1,false});
    require(!assessBleakFallsStart(session)["ok"], "early Dragonstone accepted as fresh");
    livePlayer->inventory.clear();
    session.setQuestStage("MS13",50);
    require(!assessBleakFallsStart(session)["ok"], "progress accepted as fresh");
    require(snapshot != routeCheckpoint(session), "progress not observed");
    odai::importer::fnv::ResolvedContentProfile profile;
    profile.sourcePath = "/private/person/profile.json";
    profile.dataRoot = "/private/person/Data";
    profile.layers.push_back({"mod", "mod", "/private/person/mod", true, 1, "1.7", "private source"});
    profile.archives.push_back({"/private/person/Data/Skyrim.bsa","base",true,0});
    const auto metadata = routeProfileMetadata(profile).dump();
    require(metadata.find("/private") == std::string::npos, "personal path leaked");
    require(metadata.find("Skyrim.bsa") != std::string::npos, "archive identity lost");
    std::cout << "Route checkpoint tests passed\n";
}
