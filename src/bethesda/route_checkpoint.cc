#include "bethesda/route_checkpoint.h"

namespace odai::bethesda {
using nlohmann::json;
json bleakFallsRouteContract() {
    return {{"version", 1}, {"scenario", "skyrim-bleak-falls"},
        {"start", "Riverwood / post-Helgen / Hadvar"},
        {"bootstrap", json::array({{"MQ101",900,true}, {"MQ102",10,false},
            {"MQ102A",0,false}, {"MQ102A",1,false}, {"MQ102A",5,false},
            {"MQ102A",10,false}, {"MQ102A",20,false}})},
        {"route", {"Riverwood conversations", "Bleak Falls entrance puzzle",
            "Spider and Arvel; recover Golden Claw", "Authored dungeon triggers",
            "Claw door", "Word wall and naturally resident boss",
            "Loot Dragonstone", "Return Golden Claw to Lucan",
            "Whiterun conversations and Dragonstone hand-in"}},
        {"policy", "After bootstrap: ordinary player actions only; no stage, item or residency injection"},
        {"release_gate_passed", false}};
}
json scenarioStartContract(const std::string& id) {
    if (id == "skyrim-bleak-falls") return bleakFallsRouteContract();
    const auto* scenario = findScenario(id);
    if (!scenario) return nullptr;
    json seeds = json::array();
    for (const auto& seed : scenario->prerequisiteQuests)
        seeds.push_back({seed.editorId, seed.stage, seed.completed});
    return {{"version", 1}, {"scenario", id}, {"bootstrap", seeds},
        {"start", id == "skyrim-helgen-ralof" ? "Helgen cave exit / Ralof" : scenario->startMarker}, {"release_gate_passed", false}};
}
json assessScenarioStart(const BethesdaSession& session) {
    if (session.config().scenarioId == "skyrim-bleak-falls") return assessBleakFallsStart(session);
    const auto* scenario = findScenario(session.config().scenarioId);
    json errors = json::array();
    if (!scenario) errors.push_back("Unknown scenario");
    else for (const auto& record : scenario->questRecords) {
        int stage = 0;
        bool completed = false;
        for (const auto& seed : scenario->prerequisiteQuests) {
            if (seed.editorId == record.editorId) { stage = seed.stage; completed = seed.completed; }
        }
        const auto* quest = session.findQuest(record.editorId);
        if (!quest || quest->stage != stage || quest->completed != completed || quest->failed)
            errors.push_back("Unexpected quest state: " + record.editorId);
    }
    return {{"ok", errors.empty()}, {"errors", errors}, {"release_gate_passed", false}};
}
json routeProfileMetadata(const importer::fnv::ResolvedContentProfile& profile) {
    json layers = json::array(), archives = json::array(), plugins = json::array();
    for (const auto& layer : profile.layers)
        layers.push_back({{"id",layer.id}, {"version",layer.version},
            {"enabled",layer.enabled}, {"priority",layer.priority}});
    for (const auto& archive : profile.archives)
        archives.push_back({{"file",archive.path.filename().string()},
            {"layer",archive.layerId}, {"required",archive.required}});
    for (const auto& plugin : profile.plugins)
        plugins.push_back(std::filesystem::path(plugin).filename().string());
    return {{"game",importer::fnv::bethesdaGameName(profile.game)},
        {"fingerprint",profile.fingerprint}, {"plugins",plugins},
        {"layers",layers}, {"archives",archives}};
}
json routeCheckpoint(const BethesdaSession& session) {
    json quests = json::object(), activators = json::array();
    for (const auto& record : skyrimBleakFallsScenario().questRecords) {
        const auto* quest = session.findQuest(record.editorId);
        if (!quest) { quests[record.editorId] = nullptr; continue; }
        json objectives = json::array(), aliases = json::array();
        for (const auto& obj : quest->objectives)
            objectives.push_back({{"index",obj.index}, {"displayed",obj.displayed},
                {"completed",obj.completed}, {"failed",obj.failed}});
        for (const auto& alias : quest->aliases) {
            const auto* target = session.world().find(alias.target);
            aliases.push_back({{"id",alias.id}, {"target",alias.target.toString()},
                {"created_object",alias.createdObject.toString()},
                {"materialized",alias.createdObjectMaterialized},
                {"target_registered",target != nullptr}});
            if (target) {
                aliases.back()["position"] = target->transform.position;
                aliases.back()["enabled"] = target->enabled;
                aliases.back()["cell"] = target->currentSpace.cell.toString();
                aliases.back()["escort_waiting"] = session.sceneActorWaiting(target->id);
                if (target->aiState) {
                    aliases.back()["walking"] = target->aiState->walking;
                    aliases.back()["pause_seconds"] = target->aiState->pauseSeconds;
                }
                if (target->navigationRequest) {
                    aliases.back()["navigation"] = {
                        {"destination", target->navigationRequest->destination.toString()},
                        {"revision", target->navigationRequest->revision},
                        {"status", static_cast<int>(target->navigationRequest->status)}};
                }
            }
        }
        quests[record.editorId] = {{"stage",quest->stage}, {"running",quest->running},
            {"completed",quest->completed}, {"failed",quest->failed},
            {"completed_stages",quest->completedStages},
            {"objectives",objectives}, {"aliases",aliases}};
    }
    for (const auto& object : session.world().orderedObjects()) {
        if (!object.activatorState) continue;
        const auto& state = *object.activatorState;
        activators.push_back({{"object",object.id.toString()},
            {"activation_count",state.activationCount}, {"opened",state.opened},
            {"puzzle_states",state.puzzleStates}});
    }
    json inventory = json::array();
    const auto* player = session.world().find(session.playerObject());
    if (player) for (const auto& item : player->inventory)
        inventory.push_back({{"item",item.item.toString()}, {"count",item.count},
            {"equipped",item.equipped}});
    json playerState = {{"registered",player != nullptr}, {"inventory",inventory}};
    if (player) {
        playerState["cell"] = player->currentSpace.cell.toString();
        playerState["worldspace"] = player->currentSpace.worldspace.toString();
        playerState["in_dialogue"] = player->inDialogueWithPlayer;
        if (player->actorValues) playerState["dead"] = player->actorValues->dead;
    }
    if (const auto physical = session.physics().characterState(session.playerObject())) {
        playerState["feet"] = {physical->position.x, physical->position.y, physical->position.z};
        playerState["velocity"] = {physical->velocity.x, physical->velocity.y, physical->velocity.z};
        playerState["grounded"] = physical->grounded;
    }
    json scenes = json::array();
    for (const auto& [record, playing] : session.scenes()) {
        scenes.push_back({{"scene",record.toString()}, {"playing_flag",playing}});
        const auto progress = session.sceneProgress().find(record);
        if (progress != session.sceneProgress().end()) {
            scenes.back()["phase"] = progress->second.phase;
            scenes.back()["completed_actions"] = progress->second.completed;
        }
        for (const auto& line : session.sceneSpeech()) if (line.scene == record)
            scenes.back()["speech"] = {{"speaker", line.speaker.toString()}, {"voice_key", line.voiceKey}, {"presented", line.presented}};
    }
    return {{"scenes",scenes}, {"quests",quests}, {"player",playerState}, {"activators",activators},
        {"next_story_event_sequence",session.nextStoryEventSequence()},
        {"trigger_coverage", "Activation counters and story sequence only; not a complete trigger trace"}};
}
json assessBleakFallsStart(const BethesdaSession& session) {
    json errors = json::array(), seeds = json::array();
    for (const auto& seed : skyrimBleakFallsScenario().prerequisiteQuests)
        seeds.push_back({seed.editorId,seed.stage,seed.completed});
    if (seeds != bleakFallsRouteContract()["bootstrap"])
        errors.push_back("Bootstrap definition differs from route contract v1");
    if (session.config().scenarioId != "skyrim-bleak-falls") errors.push_back("Wrong scenario");
    for (const auto& [id, stage] : std::vector<std::pair<std::string,int>>{
            {"MQ101",900},{"MQ102",10},{"MQ102A",20},{"MQ102B",0},{"MS13",0},{"MQ103",0}}) {
        const auto* quest = session.findQuest(id);
        if (!quest || quest->stage != stage || quest->failed ||
            quest->completed != (id == "MQ101")) errors.push_back("Unexpected quest state: " + id);
    }
    const auto* quest = session.findQuest("MQ102");
    bool displayed = false;
    if (quest) for (const auto& objective : quest->objectives)
        if (objective.index == 10 && objective.displayed && !objective.completed && !objective.failed)
            displayed = true;
    if (!quest || !quest->running || !displayed) errors.push_back("MQ102 objective 10 is not active");
    const auto* player = session.world().find(session.playerObject());
    if (player) for (const auto& item : player->inventory)
        if (item.count > 0 && (item.item == makeRecordKey("Skyrim.esm",0x999e7u) ||
                              item.item == makeRecordKey("Skyrim.esm",0xad06fu)))
            errors.push_back("Player already holds route quest loot");
    return {{"ok",errors.empty()}, {"errors",errors},
        {"player_inventory_verified",player != nullptr}, {"release_gate_passed",false}};
}
}
