#include "bethesda/bethesda_session.h"
#include "bethesda/save_game.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>

#include <nlohmann/json.hpp>

using namespace odai::bethesda;

namespace {

std::string fixtureChecksum(const std::string& bytes) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

void registerSaveFixture(BethesdaSession& session) {
    std::string error;
    QuestRuntimeState& ms13 = session.quest("MS13");
    if (ms13.aliases.empty()) {
        QuestAliasRuntimeState arvel;
        arvel.id = 1;
        arvel.name = "Arvel";
        arvel.handle = ObjectId::runtime(0x71000001u);
        arvel.target = ObjectId::persistent(
            makeRecordKey("Skyrim.esm", 0x39646u));
        ms13.aliases.push_back(std::move(arvel));
        QuestAliasRuntimeState claw;
        claw.id = 11;
        claw.name = "GoldenClaw";
        claw.handle = ObjectId::runtime(0x7100000bu);
        claw.createdObject = makeRecordKey("Skyrim.esm", 0x39647u);
        claw.createdInAliasId = 1;
        const std::vector<RuntimeObject> objects = session.world().orderedObjects();
        claw.createdObjectMaterialized = std::any_of(
            objects.begin(), objects.end(),
            [&](const RuntimeObject& object) {
                return std::any_of(object.inventory.begin(), object.inventory.end(),
                    [&](const InventoryEntry& entry) {
                        return entry.item == claw.createdObject && entry.count > 0;
                    });
            });
        if (claw.createdObjectMaterialized) {
            RuntimeObject itemInstance;
            itemInstance.id = session.world().allocateRuntimeId();
            itemInstance.base = claw.createdObject;
            itemInstance.kind = RuntimeObjectKind::Item;
            itemInstance.enabled = false;
            itemInstance.persistent = true;
            assert(session.world().addInitialObject(itemInstance, error));
            claw.target = itemInstance.id;
        }
        ms13.aliases.push_back(std::move(claw));
    }
    PapyrusInstruction finish;
    finish.opcode = PapyrusOpcode::Return;
    PapyrusFunction callee;
    callee.name = "SaveFixture.Callee";
    PapyrusInstruction wait;
    wait.opcode = PapyrusOpcode::WaitTicks;
    wait.operands = {PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(1000))};
    callee.instructions = {wait, finish};
    assert(session.papyrus().registerFunction(std::move(callee), error));

    PapyrusFunction caller;
    caller.name = "SaveFixture.Caller";
    PapyrusInstruction call;
    call.opcode = PapyrusOpcode::CallStatic;
    call.name = "SaveFixture.Callee";
    caller.instructions = {call, finish};
    assert(session.papyrus().registerFunction(std::move(caller), error));

    PapyrusFunction onUpdate;
    onUpdate.name = "SaveFixture.OnUpdate";
    onUpdate.scriptClass = "SaveFixture";
    onUpdate.instructions = {finish};
    assert(session.papyrus().registerFunction(std::move(onUpdate), error));
    PapyrusFunction onUpdateGameTime;
    onUpdateGameTime.name = "SaveFixture.OnUpdateGameTime";
    onUpdateGameTime.scriptClass = "SaveFixture";
    onUpdateGameTime.instructions = {finish};
    assert(session.papyrus().registerFunction(std::move(onUpdateGameTime), error));
    assert(session.papyrus().attachScript(
        ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x14u)),
        "SaveFixture", {}, error));
}

std::shared_ptr<const odai::anim::AnimationView> saveAnimationView() {
    static const auto skeleton = [] {
        odai::anim::Skeleton value;
        value.bones.push_back({"NPC Root [Root]", -1});
        return std::make_shared<odai::anim::Skeleton>(std::move(value));
    }();
    auto view = std::make_shared<odai::anim::AnimationView>();
    view->skeleton = skeleton;
    odai::anim::AnimationClip idle;
    idle.name = "idle";
    idle.duration = 1.0f;
    view->clips.push_back(std::move(idle));
    view->stateClips.emplace("idle", "idle");
    view->supportedBehaviorGraph = true;
    return view;
}

void registerPhysicalFixture(BethesdaSession& session, const ObjectId& actor) {
    PhysicsCharacterConfig config;
    config.position = {10.0f, 100.0f, 20.0f};
    std::string error;
    assert(session.registerActorAnimation(
        actor, saveAnimationView(), nullptr, config, error));
}

}  // namespace

int main() {
    BethesdaSession original;
    std::string error;
    const BethesdaSessionConfig config{
        odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition,
        "profile-a", "skyrim-bleak-falls", 77u};
    assert(original.configure(config, error));
    RuntimeObject player;
    player.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x14u));
    player.base = makeRecordKey("Skyrim.esm", 0x7u);
    player.kind = RuntimeObjectKind::Actor;
    player.actorValues = ActorValues{83.0f, 42.0f, 10.0f, false};
    player.outfit = makeRecordKey("Skyrim.esm", 0x10e2d2u);
    player.factions = {makeRecordKey("Skyrim.esm", 0x5c84du)};
    player.referenceTypes = {makeRecordKey("Skyrim.esm", 0x130f7u)};
    player.originSpace = RuntimeSpaceState{
        RuntimeSpaceKind::Interior,
        makeRecordKey("Skyrim.esm", 0x133c8u), {}, 0, 0};
    player.currentSpace = RuntimeSpaceState{
        RuntimeSpaceKind::Exterior,
        makeRecordKey("Skyrim.esm", 0xd74u),
        makeRecordKey("Skyrim.esm", 0x3cu), -7, -15};
    player.navigationRequest = RuntimeNavigationRequest{
        ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x12345u)), 4u,
        NavigationRequestStatus::Moving};
    RuntimeCombatState savedCombat;
    savedCombat.nextMeleeAttackTick = 48u;
    savedCombat.attacksStarted = 3u;
    savedCombat.hitsLanded = 2u;
    savedCombat.combatTarget =
        ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x111111u));
    savedCombat.lastTarget =
        ObjectId::persistent(makeRecordKey("Skyrim.esm", 0xabcdefu));
    player.combatState = savedCombat;
    RuntimeAiState ai;
    ai.walking = true;
    ai.projectedToNavigation = true;
    ai.wanderOrigin = {10.0f, 20.0f, 30.0f};
    ai.wanderTarget = {40.0f, 50.0f, 60.0f};
    ai.path = {
        RuntimePathStep{RuntimePathStepKind::ActivateDoor,
            {25.0f, 30.0f, 35.0f}, {35.0f, 40.0f, 45.0f},
            makeRecordKey("Skyrim.esm", 0x1234u)},
        RuntimePathStep{RuntimePathStepKind::Walk, {40.0f, 50.0f, 60.0f}, {}, {}}};
    ai.pathIndex = 1u;
    ai.pauseSeconds = 0.75f;
    ai.randomState = 12345u;
    ai.scriptedMoveActive = true;
    ai.scriptedMoveRevision = 4u;
    player.aiState = ai;
    player.inventory.push_back({makeRecordKey("Skyrim.esm", 0x39647u), 1, false});
    assert(original.world().addInitialObject(player, error));
    RuntimeObject puzzleDoor;
    puzzleDoor.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x4e2b9u));
    puzzleDoor.base = makeRecordKey("Skyrim.esm", 0x4e2b2u);
    puzzleDoor.kind = RuntimeObjectKind::Activator;
    puzzleDoor.persistent = true;
    puzzleDoor.activatorState = RuntimeActivatorState{
        {3, 1, 2}, {2, 3, 1}, 3, 2u, false};
    assert(original.world().addInitialObject(puzzleDoor, error));
    RuntimeObject giftActor;
    giftActor.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x13482u));
    giftActor.base = makeRecordKey("Skyrim.esm", 0x13475u);
    giftActor.kind = RuntimeObjectKind::Actor;
    assert(original.world().addInitialObject(giftActor, error));
    registerPhysicalFixture(original, player.id);
    std::unordered_map<std::string, PapyrusValue> properties;
    properties.emplace("GoldenClawTaken", PapyrusValue::fromBoolean(true));
    assert(original.papyrus().attachScript(
        player.id, "MS13QuestScript", std::move(properties), error));
    original.setQuestStage("MS13", 50);
    const RecordKey location = makeRecordKey("Skyrim.esm", 0x7819bu);
    const RecordKey keyword = makeRecordKey("Skyrim.esm", 0x10f63cu);
    const RecordKey global = makeRecordKey("Skyrim.esm", 0x3f0d0u);
    assert(original.registerLocation(location, {}, {keyword}, error));
    original.locationsForRestore().at(location).keywordData.at(keyword) = 6.5f;
    original.locationsForRestore().at(location).loaded = true;
    assert(original.registerGlobalVariable(global, 2.0f, error));
    original.globalVariablesForRestore().at(global) = 8.0f;
    original.forcedWeatherForRestore() =
        makeRecordKey("Skyrim.esm", 0x10e1f2u);
    original.storyEventsForRestore().push_back(StoryEventRuntimeState{
        1u, keyword, {PapyrusValue::fromObject(ObjectId::persistent(location)),
                     PapyrusValue::fromInteger(4)}});
    original.setNextStoryEventSequence(2u);
    original.scriptDebugLogsForRestore().push_back("save-fixture");
    original.advance(5.0 / 60.0);
    assert(original.queueActorAnimationEvent(player.id, {"weaponSwing", "right"}));
    registerSaveFixture(original);
    assert(original.papyrus().registerForUpdate(
        player.id, "SaveFixture", 120.0, original.clock().tick(), false,
        error, "OnUpdateGameTime"));
    original.giftMenuRequestsForRestore().push_back(
        GiftMenuRequestState{1u, giftActor.id, player.id, {}, false, false, false});
    original.setNextGiftMenuSequence(2u);
    assert(original.papyrus().startFunction("SaveFixture.Caller", {}, error) != 0u);
    assert(original.papyrus().advance(
        original.clock().tick(), 1u, original.world()).instructions == 1u);
    assert(original.papyrus().snapshot().threads[0].callStack.size() == 1u);
    const std::uint64_t expectedHash = original.deterministicHash();

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "odai-save-v8-test.json";
    assert(saveOdaiGameAtomic(path, original, error));

    BethesdaSession loaded;
    assert(loaded.configure(config, error));
    registerPhysicalFixture(loaded, player.id);
    registerSaveFixture(loaded);
    SaveLoadReport report;
    assert(loadOdaiGame(path, loaded, {}, report, error));
    assert(!report.contentReconciled);
    assert(loaded.deterministicHash() == expectedHash);
    const PapyrusValue* restoredProperty = loaded.papyrus().findProperty(
        player.id, "MS13QuestScript", "GoldenClawTaken");
    assert(restoredProperty != nullptr && restoredProperty->boolean);
    assert(loaded.papyrus().snapshot().threads[0].callStack.size() == 1u);
    const RuntimeObject* restoredPlayer = loaded.world().find(player.id);
    assert(restoredPlayer != nullptr && restoredPlayer->outfit == player.outfit &&
           restoredPlayer->factions == player.factions &&
           restoredPlayer->referenceTypes == player.referenceTypes &&
           restoredPlayer->originSpace == player.originSpace &&
           restoredPlayer->currentSpace == player.currentSpace &&
           restoredPlayer->navigationRequest == player.navigationRequest &&
           restoredPlayer->aiState == player.aiState);
    const QuestRuntimeState* restoredMs13 = loaded.findQuest("MS13");
    assert(restoredMs13 != nullptr);
    const auto restoredClawAlias = std::find_if(restoredMs13->aliases.begin(),
        restoredMs13->aliases.end(),
        [](const QuestAliasRuntimeState& alias) { return alias.id == 11; });
    assert(restoredClawAlias != restoredMs13->aliases.end() &&
           restoredClawAlias->target.kind == ObjectIdKind::Spawned);
    const RuntimeObject* restoredClawInstance =
        loaded.world().find(restoredClawAlias->target);
    assert(restoredClawInstance != nullptr &&
           restoredClawInstance->kind == RuntimeObjectKind::Item &&
           restoredClawInstance->base == makeRecordKey("Skyrim.esm", 0x39647u));
    assert(loaded.locations() == original.locations());
    assert(loaded.globalVariables() == original.globalVariables());
    assert(loaded.forcedWeather() == original.forcedWeather());
    assert(loaded.storyEvents() == original.storyEvents());
    assert(loaded.nextStoryEventSequence() == original.nextStoryEventSequence());
    assert(loaded.scriptDebugLogs() == original.scriptDebugLogs());
    assert(loaded.giftMenuRequests() == original.giftMenuRequests());
    assert(loaded.nextGiftMenuSequence() == original.nextGiftMenuSequence());
    assert(loaded.physics().snapshot() == original.physics().snapshot());
    assert(loaded.animationSnapshots().size() == 1u);
    assert(loaded.animationSnapshots().front().thirdPerson ==
        original.animationSnapshots().front().thirdPerson);

    // Stream residency may not have instantiated saved actors yet. Loading
    // stages their exact physical/graph state, and later registration consumes it.
    BethesdaSession deferred;
    assert(deferred.configure(config, error));
    registerSaveFixture(deferred);
    assert(loadOdaiGame(path, deferred, {}, report, error));
    assert(deferred.deterministicHash() == expectedHash);
    assert(deferred.physicsSnapshots() == original.physicsSnapshots());
    assert(deferred.animationSnapshots() == original.animationSnapshots());
    registerPhysicalFixture(deferred, player.id);
    assert(deferred.deterministicHash() == expectedHash);

    const std::uint64_t residentHash = loaded.deterministicHash();
    assert(loaded.unregisterActorAnimation(player.id));
    assert(loaded.deterministicHash() == residentHash);
    registerPhysicalFixture(loaded, player.id);
    assert(loaded.deterministicHash() == residentHash);

    // V7 did not persist actor faction membership or distinguish game-time
    // update registrations from real-time OnUpdate timers.
    const std::filesystem::path versionSeven = path.string() + ".v7";
    {
        nlohmann::json root;
        std::ifstream input(path, std::ios::binary);
        input >> root;
        root["version"] = 7u;
        root["payload"].erase("gift_menus");
        root["payload"].erase("next_gift_menu_sequence");
        for (auto& object : root["payload"]["world"]["objects"]) {
            object.erase("factions");
            object.erase("origin_space");
            object.erase("current_space");
        }
        for (auto& update : root["payload"]["papyrus"]["updates"]) {
            update.erase("event");
        }
        root["checksum"] = fixtureChecksum(root["payload"].dump());
        std::ofstream output(versionSeven, std::ios::binary | std::ios::trunc);
        output << root.dump(2) << '\n';
    }
    BethesdaSession migratedSeven;
    assert(migratedSeven.configure(config, error));
    registerPhysicalFixture(migratedSeven, player.id);
    registerSaveFixture(migratedSeven);
    assert(loadOdaiGame(versionSeven, migratedSeven, {}, report, error));
    assert(migratedSeven.world().find(player.id)->factions.empty());
    assert(migratedSeven.world().find(player.id)->originSpace.kind ==
           RuntimeSpaceKind::Unknown);
    assert(migratedSeven.world().find(player.id)->currentSpace.kind ==
           RuntimeSpaceKind::Unknown);
    assert(migratedSeven.papyrus().snapshot().updates.size() == 1u);
    assert(migratedSeven.papyrus().snapshot().updates.front().eventFunction == "onupdate");
    assert(migratedSeven.giftMenuRequests().empty());
    assert(std::any_of(report.diagnostics.begin(), report.diagnostics.end(),
        [](const std::string& diagnostic) {
            return diagnostic.find("pre-version-8") != std::string::npos;
        }));

    // V6 had VMAD puzzle state but not quest-created inventory provenance.
    const std::filesystem::path versionSix = path.string() + ".v6";
    {
        nlohmann::json root;
        std::ifstream input(path, std::ios::binary);
        input >> root;
        root["version"] = 6u;
        for (auto& quest : root["payload"]["quests"]) {
            for (auto& alias : quest["aliases"]) {
                alias.erase("created_object");
                alias.erase("created_in_alias_id");
                alias.erase("created_level");
                alias.erase("created_object_materialized");
            }
        }
        root["checksum"] = fixtureChecksum(root["payload"].dump());
        std::ofstream output(versionSix, std::ios::binary | std::ios::trunc);
        output << root.dump(2) << '\n';
    }
    BethesdaSession migratedSix;
    assert(migratedSix.configure(config, error));
    registerPhysicalFixture(migratedSix, player.id);
    registerSaveFixture(migratedSix);
    assert(loadOdaiGame(versionSix, migratedSix, {}, report, error));
    const QuestRuntimeState* migratedSixMs13 = migratedSix.findQuest("MS13");
    assert(migratedSixMs13 != nullptr && migratedSixMs13->aliases.size() == 2u &&
           migratedSixMs13->aliases[1].createdObjectMaterialized);

    // V5 had deterministic combat but no VMAD-backed puzzle state.
    const std::filesystem::path versionFive = path.string() + ".v5";
    {
        nlohmann::json root;
        std::ifstream input(path, std::ios::binary);
        input >> root;
        root["version"] = 5u;
        for (auto& object : root["payload"]["world"]["objects"]) {
            object.erase("activator_state");
        }
        root["checksum"] = fixtureChecksum(root["payload"].dump());
        std::ofstream output(versionFive, std::ios::binary | std::ios::trunc);
        output << root.dump(2) << '\n';
    }
    BethesdaSession migratedFive;
    assert(migratedFive.configure(config, error));
    registerPhysicalFixture(migratedFive, player.id);
    registerSaveFixture(migratedFive);
    assert(loadOdaiGame(versionFive, migratedFive, {}, report, error));
    assert(migratedFive.world().find(puzzleDoor.id) != nullptr &&
           !migratedFive.world().find(puzzleDoor.id)->activatorState.has_value());

    // V4 had typed navigation but no deterministic melee cooldown/counters.
    const std::filesystem::path versionFour = path.string() + ".v4";
    {
        nlohmann::json root;
        std::ifstream input(path, std::ios::binary);
        input >> root;
        root["version"] = 4u;
        for (auto& object : root["payload"]["world"]["objects"]) {
            object.erase("combat_state");
            object.erase("activator_state");
        }
        root["checksum"] = fixtureChecksum(root["payload"].dump());
        std::ofstream output(versionFour, std::ios::binary | std::ios::trunc);
        output << root.dump(2) << '\n';
    }
    BethesdaSession migratedFour;
    assert(migratedFour.configure(config, error));
    registerPhysicalFixture(migratedFour, player.id);
    registerSaveFixture(migratedFour);
    assert(loadOdaiGame(versionFour, migratedFour, {}, report, error));
    assert(migratedFour.world().find(player.id) != nullptr &&
           !migratedFour.world().find(player.id)->combatState.has_value());

    // V3 persisted only walk points. Promote those arrays into typed V4 steps.
    const std::filesystem::path versionThree = path.string() + ".v3";
    {
        nlohmann::json root;
        std::ifstream input(path, std::ios::binary);
        input >> root;
        root["version"] = 3u;
        for (auto& object : root["payload"]["world"]["objects"]) {
            object.erase("combat_state");
            object.erase("activator_state");
            if (!object["ai_state"].is_null()) {
                nlohmann::json oldPath = nlohmann::json::array();
                for (const auto& step : object["ai_state"]["path"]) {
                    oldPath.push_back(step["position"]);
                }
                object["ai_state"]["path"] = std::move(oldPath);
            }
        }
        root["checksum"] = fixtureChecksum(root["payload"].dump());
        std::ofstream output(versionThree, std::ios::binary | std::ios::trunc);
        output << root.dump(2) << '\n';
    }
    BethesdaSession migratedThree;
    assert(migratedThree.configure(config, error));
    registerPhysicalFixture(migratedThree, player.id);
    registerSaveFixture(migratedThree);
    assert(loadOdaiGame(versionThree, migratedThree, {}, report, error));
    const RuntimeObject* migratedThreePlayer = migratedThree.world().find(player.id);
    assert(migratedThreePlayer != nullptr && migratedThreePlayer->aiState.has_value());
    assert(migratedThreePlayer->aiState->path.front().kind == RuntimePathStepKind::Walk);

    // V2 had physical/graph state but no deterministic AI path cursor.
    const std::filesystem::path versionTwo = path.string() + ".v2";
    {
        nlohmann::json root;
        std::ifstream input(path, std::ios::binary);
        input >> root;
        root["version"] = 2u;
        for (auto& object : root["payload"]["world"]["objects"]) {
            object.erase("ai_state");
            object.erase("combat_state");
            object.erase("activator_state");
        }
        root["checksum"] = fixtureChecksum(root["payload"].dump());
        std::ofstream output(versionTwo, std::ios::binary | std::ios::trunc);
        output << root.dump(2) << '\n';
    }
    BethesdaSession migratedTwo;
    assert(migratedTwo.configure(config, error));
    registerPhysicalFixture(migratedTwo, player.id);
    registerSaveFixture(migratedTwo);
    assert(loadOdaiGame(versionTwo, migratedTwo, {}, report, error));
    assert(!report.diagnostics.empty());
    assert(migratedTwo.world().find(player.id) != nullptr &&
           !migratedTwo.world().find(player.id)->aiState.has_value());

    // V1 also omitted physical and behavior-graph state, so migration seeds
    // Jolt from the saved actor transform.
    const std::filesystem::path versionOne = path.string() + ".v1";
    {
        nlohmann::json root;
        std::ifstream input(path, std::ios::binary);
        input >> root;
        root["version"] = 1u;
        root["payload"].erase("physics");
        root["payload"].erase("animations");
        for (auto& object : root["payload"]["world"]["objects"]) {
            object.erase("ai_state");
            object.erase("combat_state");
            object.erase("activator_state");
        }
        root["checksum"] = fixtureChecksum(root["payload"].dump());
        std::ofstream output(versionOne, std::ios::binary | std::ios::trunc);
        output << root.dump(2) << '\n';
    }
    BethesdaSession migrated;
    assert(migrated.configure(config, error));
    registerPhysicalFixture(migrated, player.id);
    registerSaveFixture(migrated);
    assert(loadOdaiGame(versionOne, migrated, {}, report, error));
    assert(!report.diagnostics.empty());
    const auto migratedPhysics = migrated.physics().characterState(player.id);
    const RuntimeObject* migratedPlayer = migrated.world().find(player.id);
    assert(migratedPhysics.has_value() && migratedPlayer != nullptr);
    assert(std::fabs(migratedPhysics->position.x -
        static_cast<float>(migratedPlayer->transform.position[0])) < 1.0e-4f);

    // Simulate interruption after the old destination was staged but before
    // the new temporary generation was renamed into place.
    const std::filesystem::path previous = path.string() + ".previous";
    std::error_code filesystemError;
    std::filesystem::rename(path, previous, filesystemError);
    assert(!filesystemError);
    BethesdaSession recovered;
    assert(recovered.configure(config, error));
    registerPhysicalFixture(recovered, player.id);
    registerSaveFixture(recovered);
    assert(loadOdaiGame(path, recovered, {}, report, error));
    assert(report.recoveredPrevious && recovered.deterministicHash() == expectedHash);
    std::filesystem::rename(previous, path, filesystemError);
    assert(!filesystemError);

    BethesdaSession changed;
    BethesdaSessionConfig changedConfig = config;
    changedConfig.contentFingerprint = "profile-b";
    assert(changed.configure(changedConfig, error));
    registerPhysicalFixture(changed, player.id);
    registerSaveFixture(changed);
    assert(!loadOdaiGame(path, changed, {}, report, error));
    SaveLoadOptions reconcile;
    reconcile.recordAvailable = [](const RecordKey& key) { return key.plugin == "skyrim.esm"; };
    assert(loadOdaiGame(path, changed, reconcile, report, error));
    assert(report.contentReconciled);

    // AI cursor corruption is structural, not a best-effort replan: reject it
    // before touching the live session just like malformed VM/physics state.
    const std::filesystem::path invalidAi = path.string() + ".invalid-ai";
    {
        nlohmann::json root;
        std::ifstream input(path, std::ios::binary);
        input >> root;
        root["payload"]["world"]["objects"][0]["ai_state"]["path_index"] = 999u;
        root["checksum"] = fixtureChecksum(root["payload"].dump());
        std::ofstream output(invalidAi, std::ios::binary | std::ios::trunc);
        output << root.dump(2) << '\n';
    }
    BethesdaSession invalidAiTarget;
    assert(invalidAiTarget.configure(config, error));
    registerPhysicalFixture(invalidAiTarget, player.id);
    registerSaveFixture(invalidAiTarget);
    const std::uint64_t beforeInvalidAi = invalidAiTarget.deterministicHash();
    assert(!loadOdaiGame(invalidAi, invalidAiTarget, {}, report, error));
    assert(invalidAiTarget.deterministicHash() == beforeInvalidAi);

    const std::filesystem::path invalidCombat = path.string() + ".invalid-combat";
    {
        nlohmann::json root;
        std::ifstream input(path, std::ios::binary);
        input >> root;
        root["payload"]["world"]["objects"][0]["combat_state"]["hits_landed"] = 99u;
        root["checksum"] = fixtureChecksum(root["payload"].dump());
        std::ofstream output(invalidCombat, std::ios::binary | std::ios::trunc);
        output << root.dump(2) << '\n';
    }
    BethesdaSession invalidCombatTarget;
    assert(invalidCombatTarget.configure(config, error));
    registerPhysicalFixture(invalidCombatTarget, player.id);
    registerSaveFixture(invalidCombatTarget);
    const std::uint64_t beforeInvalidCombat = invalidCombatTarget.deterministicHash();
    assert(!loadOdaiGame(invalidCombat, invalidCombatTarget, {}, report, error));
    assert(invalidCombatTarget.deterministicHash() == beforeInvalidCombat);

    const std::filesystem::path invalidActivator = path.string() + ".invalid-activator";
    {
        nlohmann::json root;
        std::ifstream input(path, std::ios::binary);
        input >> root;
        for (auto& object : root["payload"]["world"]["objects"]) {
            if (!object["activator_state"].is_null()) {
                object["activator_state"]["puzzle_solution"] = {1, 2};
            }
        }
        root["checksum"] = fixtureChecksum(root["payload"].dump());
        std::ofstream output(invalidActivator, std::ios::binary | std::ios::trunc);
        output << root.dump(2) << '\n';
    }
    BethesdaSession invalidActivatorTarget;
    assert(invalidActivatorTarget.configure(config, error));
    registerPhysicalFixture(invalidActivatorTarget, player.id);
    registerSaveFixture(invalidActivatorTarget);
    const std::uint64_t beforeInvalidActivator = invalidActivatorTarget.deterministicHash();
    assert(!loadOdaiGame(
        invalidActivator, invalidActivatorTarget, {}, report, error));
    assert(invalidActivatorTarget.deterministicHash() == beforeInvalidActivator);

    // Corruption must be rejected before the live session is mutated.
    {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(16);
        file.put('X');
    }
    BethesdaSession corruptTarget;
    assert(corruptTarget.configure(config, error));
    registerSaveFixture(corruptTarget);
    assert(!loadOdaiGame(path, corruptTarget, {}, report, error));

    std::error_code removeError;
    std::filesystem::remove(path, removeError);
    std::filesystem::remove(previous, removeError);
    std::filesystem::remove(versionSeven, removeError);
    std::filesystem::remove(versionSix, removeError);
    std::filesystem::remove(versionOne, removeError);
    std::filesystem::remove(versionTwo, removeError);
    std::filesystem::remove(versionThree, removeError);
    std::filesystem::remove(versionFour, removeError);
    std::filesystem::remove(versionFive, removeError);
    std::filesystem::remove(invalidAi, removeError);
    std::filesystem::remove(invalidCombat, removeError);
    std::filesystem::remove(invalidActivator, removeError);
    std::cout << "ODAI save tests passed\n";
    return 0;
}
