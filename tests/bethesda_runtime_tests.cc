#include "bethesda/bethesda_session.h"
#include "bethesda/condition.h"
#include "bethesda/runtime_ids.h"
#include "bethesda/runtime_world.h"
#include "bethesda/scenario.h"
#include "bethesda/skyrim_quest.h"
#include "bethesda/skyrim_dialogue.h"
#include "bethesda/skyrim_runtime_records.h"
#include "bethesda/vmad_reader.h"
#include "bethesda/whiterun_presentation.h"
#include "bethesda/oblivion_presentation.h"

#include <bit>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector>

using namespace odai::bethesda;

namespace {

void put16(std::vector<std::uint8_t>& bytes, std::uint16_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8u));
}

void put32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (int byte = 0; byte < 4; ++byte)
        bytes.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
}

void putString(std::vector<std::uint8_t>& bytes, const std::string& value) {
    put16(bytes, static_cast<std::uint16_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

std::vector<std::uint8_t> ctda(
    std::uint8_t comparison, bool orWithNext, float wanted, std::uint16_t function) {
    std::vector<std::uint8_t> bytes(28u, 0u);
    bytes[0] = static_cast<std::uint8_t>((comparison << 5u) | (orWithNext ? 1u : 0u));
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(wanted);
    for (int byte = 0; byte < 4; ++byte) bytes[4u + byte] = bits >> (byte * 8);
    bytes[8] = static_cast<std::uint8_t>(function);
    bytes[9] = static_cast<std::uint8_t>(function >> 8u);
    return bytes;
}

}  // namespace

int main() {
    {
        constexpr OblivionReferenceCamera camera =
            imperialMarketReferenceCamera();
        assert(std::abs(camera.horizontalFovDegrees - 70.0f) < 0.001f);
        assert(camera.position[0] >= 32768.0f && camera.position[0] < 36864.0f);
        assert(camera.position[2] <= -65536.0f && camera.position[2] > -69632.0f);
        assert(camera.position[1] > 3600.0f && camera.position[1] < 4200.0f);
        assert(camera.pitchDegrees < 0.0f);
        const float yaw = camera.yawDegrees * 3.14159265358979323846f / 180.0f;
        assert(std::cos(yaw) < -0.4f && std::sin(yaw) < -0.85f);
    }
    {
        constexpr OblivionReferenceCamera camera = anvilHarborReferenceCamera();
        assert(std::abs(camera.horizontalFovDegrees - 72.0f) < 0.001f);
        assert(static_cast<int>(std::floor(camera.position[0] / 4096.0f)) == -48);
        // Streamer engine Z is the negative of the plugin's exterior-cell Y.
        assert(static_cast<int>(std::floor(-camera.position[2] / 4096.0f)) == -8);
        assert(camera.position[1] > 380.0f && camera.position[1] < 460.0f);
        assert(camera.yawDegrees > 40.0f && camera.yawDegrees < 50.0f);
        assert(camera.pitchDegrees < -6.0f && camera.pitchDegrees > -11.0f);
    }
    {
        constexpr float marker[3] = {10000.0f, 800.0f, -20000.0f};
        constexpr OblivionReferenceCamera camera =
            greatForestReferenceCamera(marker);
        assert(std::abs(camera.horizontalFovDegrees - 70.0f) < 0.001f);
        assert(camera.position[0] == marker[0] - 2200.0f);
        assert(camera.position[1] == marker[1] + 230.0f);
        assert(camera.position[2] == marker[2] - 1700.0f);
        const float yaw = camera.yawDegrees * 3.14159265358979323846f / 180.0f;
        const float toMarkerX = marker[0] - camera.position[0];
        const float toMarkerZ = marker[2] - camera.position[2];
        assert(toMarkerX * std::cos(yaw) + toMarkerZ * std::sin(yaw) > 2700.0f);
    }
    {
        const float gate[3] = {100.0f, 200.0f, 300.0f};
        const WhiterunReferenceCamera camera =
            whiterunReferenceCamera(gate, -90.0f);
        assert(std::abs(camera.horizontalFovDegrees - 75.0f) < 0.001f);
        assert(camera.position[0] > gate[0] + 2300.0f);
        assert(camera.position[2] > gate[2]);
        assert(camera.pitchDegrees > 0.0f);
        const float yaw = camera.yawDegrees * 3.14159265358979323846f / 180.0f;
        const float toGateX = gate[0] - camera.position[0];
        const float toGateZ = gate[2] - camera.position[2];
        assert(toGateX * std::cos(yaw) + toGateZ * std::sin(yaw) > 0.0f);
        // The look target is intentionally left of the gate centre, placing
        // the gatehouse/banner in the right half of a 16:9 reference frame.
        assert(toGateX * -std::sin(yaw) + toGateZ * std::cos(yaw) > 0.0f);

        const float movedGate[3] = {-400.0f, 225.0f, 900.0f};
        const WhiterunReferenceCamera moved =
            whiterunReferenceCamera(movedGate, -90.0f);
        assert(std::abs((moved.position[0] - camera.position[0]) + 500.0f) < 0.001f);
        assert(std::abs((moved.position[1] - camera.position[1]) - 25.0f) < 0.001f);
        assert(std::abs((moved.position[2] - camera.position[2]) - 600.0f) < 0.001f);

        const WhiterunReferenceCamera market =
            whiterunMarketReferenceCamera(gate, -90.0f);
        assert(std::abs(market.horizontalFovDegrees - 75.0f) < 0.001f);
        assert(market.position[2] > gate[2]);
        assert(market.position[0] > gate[0] + 400.0f);
        assert(market.pitchDegrees > 0.0f && market.pitchDegrees < 10.0f);
        const float marketYaw =
            market.yawDegrees * 3.14159265358979323846f / 180.0f;
        assert(std::cos(marketYaw) > 0.9f);
        assert(std::abs(std::sin(marketYaw)) < 0.1f);
    }

    const ScenarioDefinition* whiterunScenario =
        findScenario("skyrim-whiterun-showcase");
    assert(whiterunScenario != nullptr);
    assert(whiterunScenario->basePlugin == "Skyrim.esm");
    assert(whiterunScenario->worldspace == "WhiterunWorld");
    assert(whiterunScenario->startMarker.empty());
    assert(whiterunScenario->questRecords.empty());
    assert(whiterunScenario->prerequisiteQuests.size() == 5u);
    BethesdaSession whiterunSession;
    std::string whiterunError;
    assert(whiterunSession.configure({
        odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition,
        "whiterun-fixture", whiterunScenario->id, 17u}, whiterunError));
    const QuestRuntimeState* dragonRising = whiterunSession.findQuest("MQ104");
    const QuestRuntimeState* wayOfTheVoice = whiterunSession.findQuest("MQ105");
    assert(dragonRising != nullptr && dragonRising->stage == 160 &&
           dragonRising->completed);
    assert(wayOfTheVoice != nullptr && wayOfTheVoice->stage == 10 &&
           wayOfTheVoice->running && !wayOfTheVoice->completed);

    const ScenarioDefinition* riftenScenario =
        findScenario("skyrim-riften-showcase");
    assert(riftenScenario != nullptr);
    assert(riftenScenario->basePlugin == "Skyrim.esm");
    assert(riftenScenario->worldspace == "RiftenWorld");
    assert(riftenScenario->startMarker.empty());
    assert(riftenScenario->questRecords.empty());
    assert(riftenScenario->prerequisiteQuests.empty());

    const RecordKey key = makeRecordKey("Data/Skyrim.ESM", 0x1234u);
    assert(key.plugin == "skyrim.esm");
    RecordKey parsed;
    assert(parseRecordKey(key.toString(), parsed));
    assert(parsed == key);

    const RecordKey tes3Quest = makeTes3RecordKey("dial", "TR_m3_TT_Lloris5");
    assert(tes3Quest.valid());
    assert(tes3Quest == makeTes3RecordKey("DIAL", "tr_M3_tt_lloris5"));
    assert(tes3Quest.toString() == "tes3:DIAL:tr_m3_tt_lloris5");
    assert(parseRecordKey(tes3Quest.toString(), parsed));
    assert(parsed == tes3Quest);

    const RecordKey colonId = makeTes3RecordKey("CELL", "Test:Sanctuary%Interior");
    assert(parseRecordKey(colonId.toString(), parsed));
    assert(parsed == colonId);

    const RecordKey frmr = makeTes3ReferenceKey("TR_Mainland.esm", 0x1234u);
    assert(frmr.toString() == "frmr:tr_mainland.esm:0x00001234");
    assert(parseRecordKey(frmr.toString(), parsed));
    assert(parsed == frmr);

    FixedStepClock clock;
    std::uint32_t calls = 0u;
    const FixedStepResult clockResult = clock.advance(3.0 / 60.0,
        [&](std::uint64_t tick, double step) {
            assert(tick == calls);
            assert(step == 1.0 / 60.0);
            ++calls;
        });
    assert(clockResult.steps == 3u && calls == 3u && clock.tick() == 3u);

    // Deterministic simulation iteration is a cached identity index, not an
    // owning copy/sort of every RuntimeObject. Decorative/non-actor residency
    // must not enter the actor loop, and spawn/destroy must invalidate it.
    {
        BethesdaWorld indexed;
        std::string indexedError;
        RuntimeObject decoration;
        decoration.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x100u));
        decoration.base = makeRecordKey("Skyrim.esm", 0x101u);
        decoration.kind = RuntimeObjectKind::Item;
        assert(indexed.addInitialObject(decoration, indexedError));
        RuntimeObject indexedActor;
        indexedActor.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x20u));
        indexedActor.base = makeRecordKey("Skyrim.esm", 0x21u);
        indexedActor.kind = RuntimeObjectKind::Actor;
        assert(indexed.addInitialObject(indexedActor, indexedError));
        assert(indexed.orderedObjectIds().size() == 2u);
        assert(indexed.orderedActorIds().size() == 1u &&
               indexed.orderedActorIds().front() == indexedActor.id);
        WorldCommand destroy;
        destroy.type = WorldCommandType::Destroy;
        destroy.target = indexedActor.id;
        (void)indexed.queue(std::move(destroy));
        assert(indexed.applyQueuedCommands().applied == 1u);
        assert(indexed.orderedActorIds().empty());
    }

    BethesdaSession session;
    std::string error;
    assert(session.configure({
        odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition,
        "fixture-content", "skyrim-bleak-falls", 0x12345678u}, error));
    const QuestRuntimeState* unbound = session.findQuest("mq101");
    assert(unbound != nullptr && unbound->stage == 900 && unbound->completed);
    assert(unbound->record == makeRecordKey("Skyrim.esm", 0x3372bu));
    const QuestRuntimeState* beforeTheStorm = session.findQuest("MQ102");
    assert(beforeTheStorm != nullptr && beforeTheStorm->stage == 10 &&
           beforeTheStorm->running &&
           beforeTheStorm->record == makeRecordKey("Skyrim.esm", 0x4e50du));
    const QuestRuntimeState* goldenClawQuest = session.findQuest("MS13");
    assert(goldenClawQuest != nullptr && goldenClawQuest->stage == 0 &&
           goldenClawQuest->record == makeRecordKey("Skyrim.esm", 0x39645u));

    RuntimeObject player;
    player.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x14u));
    player.base = makeRecordKey("Skyrim.esm", 0x7u);
    player.kind = RuntimeObjectKind::Actor;
    player.actorValues.emplace();
    assert(session.world().addInitialObject(player, error));

    const RecordKey claw = makeRecordKey("Skyrim.esm", 0x39647u);
    WorldCommand add;
    add.type = WorldCommandType::AddItem;
    add.target = player.id;
    add.item = claw;
    add.itemCount = 1;
    (void)session.world().queue(add);
    WorldCommand damage;
    damage.type = WorldCommandType::AdjustActorValue;
    damage.target = player.id;
    damage.actorValue = ActorValue::Health;
    damage.actorValueDelta = -25.0f;
    (void)session.world().queue(damage);
    const RecordKey riverwoodFriendFaction =
        makeRecordKey("Skyrim.esm", 0x5c84du);
    WorldCommand addFaction;
    addFaction.type = WorldCommandType::AddToFaction;
    addFaction.target = player.id;
    addFaction.faction = riverwoodFriendFaction;
    (void)session.world().queue(addFaction);
    const BethesdaSessionStep step = session.advance(1.0 / 60.0);
    assert(step.worldCommands == 3u && step.diagnostics.empty());
    const RuntimeObject* updated = session.world().find(player.id);
    assert(updated != nullptr && updated->inventory.size() == 1u);
    assert(updated->inventory[0].item == claw && updated->inventory[0].count == 1);
    assert(updated->actorValues->health == 75.0f);
    assert(updated->factions == std::vector<RecordKey>{riverwoodFriendFaction});
    WorldCommand removeFaction = addFaction;
    removeFaction.type = WorldCommandType::RemoveFromFaction;
    (void)session.world().queue(removeFaction);
    assert(session.world().applyQueuedCommands().applied == 1u);
    assert(session.world().find(player.id)->factions.empty());

    // ACHR ownership remains immutable provenance while current space follows
    // packages, doors, and teleports. Replaying the same ordered mutations must
    // produce the same hash regardless of presentation-cell churn.
    {
        const RuntimeSpaceState house{
            RuntimeSpaceKind::Interior,
            makeRecordKey("Skyrim.esm", 0x133c8u), {}, 0, 0};
        const RuntimeSpaceState riverwood{
            RuntimeSpaceKind::Exterior,
            makeRecordKey("Skyrim.esm", 0xd74u),
            makeRecordKey("Skyrim.esm", 0x3c), -7, -15};
        auto exerciseSpaceMove = [&](BethesdaWorld& world) {
            RuntimeObject actor;
            actor.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x13482u));
            actor.base = makeRecordKey("Skyrim.esm", 0x13475u);
            actor.kind = RuntimeObjectKind::Actor;
            assert(world.addInitialObject(actor, error));
            WorldCommand origin;
            origin.type = WorldCommandType::SetOriginSpace;
            origin.target = actor.id;
            origin.originSpace = house;
            (void)world.queue(origin);
            WorldCommand current;
            current.type = WorldCommandType::SetCurrentSpace;
            current.target = actor.id;
            current.currentSpace = riverwood;
            (void)world.queue(current);
            const CommandApplyResult applied = world.applyQueuedCommands();
            assert(applied.applied == 2u && applied.diagnostics.empty());
            const RuntimeObject* moved = world.find(actor.id);
            assert(moved != nullptr && moved->originSpace == house &&
                   moved->currentSpace == riverwood && !moved->interior);

            WorldCommand overwrite = origin;
            overwrite.originSpace = riverwood;
            (void)world.queue(overwrite);
            const CommandApplyResult rejected = world.applyQueuedCommands();
            assert(rejected.applied == 0u && rejected.diagnostics.size() == 1u);
        };
        BethesdaWorld first;
        BethesdaWorld replay;
        exerciseSpaceMove(first);
        exerciseSpaceMove(replay);
        assert(first.deterministicHash() == replay.deterministicHash());

        RuntimeObject marker;
        marker.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x2bfa8u));
        marker.base = makeRecordKey("Skyrim.esm", 0x1u);
        marker.kind = RuntimeObjectKind::Activator;
        marker.currentSpace = riverwood;
        marker.transform.position = {123.0, 45.0, 678.0};
        assert(first.addInitialObject(marker, error));
        WorldCommand teleport;
        teleport.type = WorldCommandType::TeleportToReference;
        teleport.target = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x13482u));
        teleport.destination = marker.id;
        (void)first.queue(teleport);
        assert(first.applyQueuedCommands().applied == 1u);
        const RuntimeObject* teleported = first.find(teleport.target);
        assert(teleported != nullptr && teleported->currentSpace == riverwood &&
               teleported->transform == marker.transform);
    }

    {
        BethesdaSession giftSession;
        assert(giftSession.configure({
            odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition,
            "gift-fixture", "skyrim-bleak-falls", 15u}, error));
        assert(giftSession.world().addInitialObject(player, error));
        RuntimeObject giver;
        giver.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x13482u));
        giver.base = makeRecordKey("Skyrim.esm", 0x13475u);
        giver.kind = RuntimeObjectKind::Actor;
        const RecordKey supplies = makeRecordKey("Skyrim.esm", 0x1397eu);
        giver.inventory.push_back({supplies, 2, false});
        assert(giftSession.world().addInitialObject(giver, error));
        giftSession.giftMenuRequestsForRestore().push_back(
            GiftMenuRequestState{1u, giver.id, player.id, {}, false, false, true});
        giftSession.setNextGiftMenuSequence(2u);
        const GiftTransferResult gift =
            giftSession.transferGiftMenuItem(1u, supplies, 1);
        assert(gift.accepted && gift.diagnostic.empty());
        assert(giftSession.world().applyQueuedCommands().applied == 2u);
        assert(giftSession.world().find(giver.id)->inventory.front().count == 1);
        const auto received = std::find_if(
            giftSession.world().find(player.id)->inventory.begin(),
            giftSession.world().find(player.id)->inventory.end(),
            [&](const InventoryEntry& entry) { return entry.item == supplies; });
        assert(received != giftSession.world().find(player.id)->inventory.end() &&
               received->count == 1);
        assert(giftSession.closeGiftMenu(1u, error));
        assert(giftSession.giftMenuRequests().empty());
    }

    BethesdaSession puzzle;
    assert(puzzle.configure({
        odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition,
        "puzzle-fixture", "skyrim-bleak-falls", 29u}, error));
    RuntimeObject puzzlePlayer = player;
    puzzlePlayer.inventory.push_back({claw, 1, false});
    assert(puzzle.world().addInitialObject(puzzlePlayer, error));
    RuntimeObject puzzleDoor;
    puzzleDoor.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x4e2b9u));
    puzzleDoor.base = makeRecordKey("Skyrim.esm", 0x4e2b2u);
    puzzleDoor.kind = RuntimeObjectKind::Activator;
    puzzleDoor.persistent = true;
    puzzleDoor.activatorState = RuntimeActivatorState{
        {3, 1, 2}, {2, 3, 1}, 3, 0u, false};
    assert(puzzle.world().addInitialObject(puzzleDoor, error));
    RuntimeObject emptyActor;
    emptyActor.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x600u));
    emptyActor.base = makeRecordKey("Skyrim.esm", 0x7u);
    emptyActor.kind = RuntimeObjectKind::Actor;
    assert(puzzle.world().addInitialObject(emptyActor, error));
    const PuzzleDoorActivationResult missing = puzzle.activatePuzzleDoor(
        emptyActor.id, puzzleDoor.id, claw,
        makeRecordKey("Skyrim.esm", 0x39645u), 50);
    assert(missing.accepted && missing.missingRequiredItem && !missing.opened);
    assert(puzzle.world().applyQueuedCommands().applied == 1u);
    const PuzzleDoorActivationResult wrong = puzzle.activatePuzzleDoor(
        player.id, puzzleDoor.id, claw,
        makeRecordKey("Skyrim.esm", 0x39645u), 50);
    assert(wrong.accepted && wrong.incorrectCombination && !wrong.opened);
    assert(puzzle.world().applyQueuedCommands().applied == 1u);
    for (const std::size_t ring : {0u, 0u, 1u, 1u, 2u, 2u}) {
        assert(puzzle.rotatePuzzleRing(puzzleDoor.id, ring, error));
        assert(puzzle.world().applyQueuedCommands().applied == 1u);
    }
    const PuzzleDoorActivationResult opened = puzzle.activatePuzzleDoor(
        player.id, puzzleDoor.id, claw,
        makeRecordKey("Skyrim.esm", 0x39645u), 50);
    assert(opened.accepted && opened.opened);
    assert(puzzle.world().applyQueuedCommands().applied == 1u);
    assert(puzzle.world().find(puzzleDoor.id)->activatorState->opened);
    assert(puzzle.findQuest("MS13")->stage == 50);

    const ObjectId destination = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x12345u));
    const RecordKey outfit = makeRecordKey("Skyrim.esm", 0x10e2d2u);
    WorldCommand move;
    move.type = WorldCommandType::RequestMoveTo;
    move.target = player.id;
    move.destination = destination;
    (void)session.world().queue(move);
    WorldCommand setOutfit;
    setOutfit.type = WorldCommandType::SetOutfit;
    setOutfit.target = player.id;
    setOutfit.outfit = outfit;
    (void)session.world().queue(setOutfit);
    const CommandApplyResult physicalCommands = session.world().applyQueuedCommands();
    assert(physicalCommands.applied == 2u && physicalCommands.diagnostics.empty());
    updated = session.world().find(player.id);
    assert(updated != nullptr && updated->outfit == outfit);
    assert(updated->navigationRequest.has_value() &&
           updated->navigationRequest->destination == destination &&
           updated->navigationRequest->status == NavigationRequestStatus::Pending);
    WorldCommand moving;
    moving.type = WorldCommandType::SetNavigationStatus;
    moving.target = player.id;
    moving.navigationRevision = updated->navigationRequest->revision;
    moving.navigationStatus = NavigationRequestStatus::Moving;
    (void)session.world().queue(moving);
    assert(session.world().applyQueuedCommands().applied == 1u);
    assert(session.world().find(player.id)->navigationRequest->status ==
           NavigationRequestStatus::Moving);

    // Melee is sampled and resolved on the fixed tick from Jolt-owned
    // character positions. The nearest living actor in the facing cone takes
    // damage; cooldown and stamina are persistent runtime state.
    {
        BethesdaSession combat;
        assert(combat.configure({
            odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition,
            "combat-fixture", "skyrim-bleak-falls", 31u}, error));
        RuntimeObject attacker = player;
        attacker.transform.position = {0.0, 0.0, 0.0};
        RuntimeObject target = player;
        target.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x200u));
        target.transform.position = {100.0, 0.0, 0.0};
        target.actorValues = ActorValues{20.0f, 100.0f, 100.0f, false};
        RuntimeObject behind = player;
        behind.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x300u));
        behind.transform.position = {-60.0, 0.0, 0.0};
        assert(combat.world().addInitialObject(attacker, error));
        assert(combat.world().addInitialObject(target, error));
        assert(combat.world().addInitialObject(behind, error));
        PhysicsCharacterConfig controller;
        assert(combat.registerActorController(attacker.id, controller, error));
        controller.position = {100.0f, 0.0f, 0.0f};
        assert(combat.registerActorController(target.id, controller, error));
        controller.position = {-60.0f, 0.0f, 0.0f};
        assert(combat.registerActorController(behind.id, controller, error));

        MeleeAttackResult first;
        (void)combat.advance(1.0 / 60.0,
            [&](std::uint64_t tick, double) {
                assert(tick == 0u);
                first = combat.performMeleeAttack(attacker.id, {1.0f, 0.0f, 0.0f});
            });
        assert(first.accepted && first.hit && first.killed && first.target == target.id);
        const RuntimeObject* defeated = combat.world().find(target.id);
        const RuntimeObject* spent = combat.world().find(attacker.id);
        assert(defeated != nullptr && defeated->actorValues->dead &&
               defeated->actorValues->health == 0.0f);
        assert(spent != nullptr && spent->actorValues->stamina == 90.0f &&
               spent->combatState.has_value() &&
               spent->combatState->nextMeleeAttackTick == 24u &&
               spent->combatState->lastTarget == target.id);
        const auto knockedBack = combat.physics().characterState(target.id);
        assert(knockedBack.has_value() &&
               (knockedBack->position.x > 100.0f || knockedBack->position.y > 0.0f));
        MeleeAttackResult cooldown;
        (void)combat.advance(1.0 / 60.0,
            [&](std::uint64_t, double) {
                cooldown = combat.performMeleeAttack(
                    attacker.id, {1.0f, 0.0f, 0.0f});
            });
        assert(!cooldown.accepted && cooldown.diagnostic == "melee attack is on cooldown");
        const MeleeAttackResult malformed =
            combat.performMeleeAttack(attacker.id, {});
        assert(!malformed.accepted && !malformed.diagnostic.empty());
    }

    // Authored static collision blocks the hit query even though both virtual
    // characters are within nominal range.
    {
        BethesdaSession occluded;
        assert(occluded.configure({
            odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition,
            "combat-wall-fixture", "skyrim-bleak-falls", 32u}, error));
        RuntimeObject attacker = player;
        RuntimeObject target = player;
        target.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x400u));
        assert(occluded.world().addInitialObject(attacker, error));
        assert(occluded.world().addInitialObject(target, error));
        PhysicsCharacterConfig controller;
        assert(occluded.registerActorController(attacker.id, controller, error));
        controller.position = {100.0f, 0.0f, 0.0f};
        assert(occluded.registerActorController(target.id, controller, error));
        const std::vector<odai::math::Vector3> wall{
            {50.0f, -100.0f, -100.0f}, {50.0f, 200.0f, -100.0f},
            {50.0f, 200.0f, 100.0f}, {50.0f, -100.0f, 100.0f}};
        const std::vector<std::uint32_t> triangles{0u, 1u, 2u, 0u, 2u, 3u};
        assert(occluded.physics().addStaticCollision(
            ObjectId::runtime(99u), wall, triangles, error));
        MeleeAttackResult blocked;
        (void)occluded.advance(1.0 / 60.0,
            [&](std::uint64_t, double) {
                blocked = occluded.performMeleeAttack(
                    attacker.id, {1.0f, 0.0f, 0.0f});
            });
        assert(blocked.accepted && !blocked.hit);
    }

    // A combat package target drives the same melee resolver without any
    // renderer or frame-loop participation.
    {
        BethesdaSession hostile;
        assert(hostile.configure({
            odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition,
            "hostile-fixture", "skyrim-bleak-falls", 33u}, error));
        RuntimeObject victim = player;
        victim.transform.position = {100.0, 0.0, 0.0};
        RuntimeObject enemy = player;
        enemy.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x500u));
        RuntimeCombatState aggression;
        aggression.combatTarget = victim.id;
        enemy.combatState = aggression;
        assert(hostile.world().addInitialObject(victim, error));
        assert(hostile.world().addInitialObject(enemy, error));
        PhysicsCharacterConfig controller;
        controller.position = {100.0f, 0.0f, 0.0f};
        assert(hostile.registerActorController(victim.id, controller, error));
        controller.position = {0.0f, 0.0f, 0.0f};
        assert(hostile.registerActorController(enemy.id, controller, error));
        (void)hostile.advance(1.0 / 60.0);
        const RuntimeObject* struck = hostile.world().find(victim.id);
        const RuntimeObject* aggressor = hostile.world().find(enemy.id);
        assert(struck != nullptr && struck->actorValues->health == 90.0f);
        assert(aggressor != nullptr && aggressor->combatState.has_value() &&
               aggressor->combatState->combatTarget == victim.id &&
               aggressor->combatState->hitsLanded == 1u);
        (void)hostile.advance(1.0 / 60.0);
        assert(hostile.world().find(victim.id)->actorValues->health == 90.0f);
    }

    BethesdaSession replay;
    assert(replay.configure({
        odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition,
        "fixture-content", "skyrim-bleak-falls", 0x12345678u}, error));
    assert(replay.world().addInitialObject(player, error));
    (void)replay.world().queue(add);
    (void)replay.world().queue(damage);
    (void)replay.world().queue(addFaction);
    (void)replay.advance(1.0 / 60.0);
    (void)replay.world().queue(removeFaction);
    (void)replay.world().applyQueuedCommands();
    (void)replay.world().queue(move);
    (void)replay.world().queue(setOutfit);
    (void)replay.world().applyQueuedCommands();
    (void)replay.world().queue(moving);
    (void)replay.world().applyQueuedCommands();
    assert(replay.deterministicHash() == session.deterministicHash());

    // Controller-only actors use the same fixed-step authority as animated
    // actors. Presentation supplies velocity intent; Jolt publishes the sole
    // world transform, and streaming residency may remove/re-add the controller
    // without changing replay state.
    {
        BethesdaSession physicalSession;
        assert(physicalSession.configure({
            odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition,
            "physical-fixture", "skyrim-bleak-falls", 9u}, error));
        RuntimeObject physicalPlayer = player;
        physicalPlayer.transform.position = {0.0, 100.0, 0.0};
        assert(physicalSession.world().addInitialObject(physicalPlayer, error));
        PhysicsCharacterConfig controller;
        controller.position = {0.0f, 100.0f, 0.0f};
        assert(physicalSession.registerActorController(
            physicalPlayer.id, controller, error));
        PhysicsCharacterInput intent;
        intent.desiredVelocity = {60.0f, 0.0f, 0.0f};
        assert(physicalSession.setActorControllerInput(physicalPlayer.id, intent));
        std::uint32_t beforeTicks = 0u;
        (void)physicalSession.advance(1.0 / 60.0,
            [&](std::uint64_t tick, double fixedStep) {
                assert(tick == 0u && fixedStep == 1.0 / 60.0);
                WorldCommand facing;
                facing.type = WorldCommandType::SetTransform;
                facing.target = physicalPlayer.id;
                facing.transform = physicalPlayer.transform;
                facing.transform.rotationRadians[1] = 1.25f;
                (void)physicalSession.world().queue(std::move(facing));
                ++beforeTicks;
            });
        assert(beforeTicks == 1u);
        const auto physical = physicalSession.physics().characterState(physicalPlayer.id);
        const RuntimeObject* movedPlayer = physicalSession.world().find(physicalPlayer.id);
        assert(physical.has_value() && physical->position.x > 0.0f);
        assert(movedPlayer != nullptr &&
               movedPlayer->transform.position[0] == physical->position.x &&
               movedPlayer->transform.position[1] == physical->position.y &&
               movedPlayer->transform.position[2] == physical->position.z);
        assert(movedPlayer->transform.rotationRadians[1] == 1.25f);
        const std::uint64_t residentHash = physicalSession.deterministicHash();
        assert(physicalSession.unregisterActorController(physicalPlayer.id));
        assert(physicalSession.deterministicHash() == residentHash);
        assert(physicalSession.registerActorController(
            physicalPlayer.id, controller, error));
        assert(physicalSession.deterministicHash() == residentHash);
    }

    std::vector<Condition> conditions;
    for (const auto& fixture : {ctda(0u, true, 1.0f, 1u), ctda(0u, false, 1.0f, 2u),
                                ctda(3u, false, 2.0f, 3u)}) {
        Condition condition;
        assert(readCondition(fixture, condition, error));
        conditions.push_back(condition);
    }
    const ConditionEvaluation evaluated = evaluateConditions(
        conditions,
        [](const Condition& condition) -> std::optional<float> {
            if (condition.function == 1u) return 0.0f;
            if (condition.function == 2u) return 1.0f;
            if (condition.function == 3u) return 2.0f;
            return std::nullopt;
        }, true);
    assert(evaluated.matched && evaluated.diagnostics.empty());
    conditions[2].function = 999u;
    assert(!evaluateConditions(conditions, {}, true).matched);
    assert(evaluateConditions(conditions, {}, false).matched);

    {
        std::vector<std::uint8_t> editorId{'T', 'e', 's', 't', 'L', 'o', 'c', '\0'};
        std::vector<std::uint8_t> parentBytes;
        put32(parentBytes, 0x100u);
        std::vector<std::uint8_t> keywordBytes;
        put32(keywordBytes, 0x200u);
        put32(keywordBytes, 0x201u);
        odai::importer::fnv::EsmRecordView locationRecord;
        locationRecord.type = "LCTN";
        locationRecord.subrecords = {
            {"EDID", editorId.data(), static_cast<std::uint32_t>(editorId.size())},
            {"PNAM", parentBytes.data(), static_cast<std::uint32_t>(parentBytes.size())},
            {"KWDA", keywordBytes.data(), static_cast<std::uint32_t>(keywordBytes.size())}};
        SkyrimLocationDefinition locationDefinition;
        assert(readSkyrimLocation(locationRecord,
            makeRecordKey("Skyrim.esm", 0x300u), locationDefinition, error));
        assert(locationDefinition.editorId == "TestLoc" &&
               locationDefinition.parentFormId == 0x100u &&
               locationDefinition.keywordFormIds ==
                   std::vector<std::uint32_t>({0x200u, 0x201u}));
        keywordBytes.pop_back();
        locationRecord.subrecords.back().size =
            static_cast<std::uint32_t>(keywordBytes.size());
        assert(!readSkyrimLocation(locationRecord,
            makeRecordKey("Skyrim.esm", 0x300u), locationDefinition, error));

        std::vector<std::uint8_t> globalEditorId{'T', 'e', 's', 't', 'G', 'l', 'o', 'b', '\0'};
        std::vector<std::uint8_t> globalValue;
        put32(globalValue, std::bit_cast<std::uint32_t>(3.5f));
        odai::importer::fnv::EsmRecordView globalRecord;
        globalRecord.type = "GLOB";
        globalRecord.subrecords = {
            {"EDID", globalEditorId.data(), static_cast<std::uint32_t>(globalEditorId.size())},
            {"FLTV", globalValue.data(), static_cast<std::uint32_t>(globalValue.size())}};
        SkyrimGlobalVariableDefinition globalDefinition;
        assert(readSkyrimGlobalVariable(globalRecord,
            makeRecordKey("Skyrim.esm", 0x301u), globalDefinition, error));
        assert(globalDefinition.editorId == "TestGlob" && globalDefinition.initialValue == 3.5f);
    }

    std::vector<std::uint8_t> vmad;
    put16(vmad, 5u); put16(vmad, 2u); put16(vmad, 1u);
    putString(vmad, "QF_MQ101_Fixture"); vmad.push_back(0u); put16(vmad, 2u);
    putString(vmad, "Stage"); vmad.push_back(static_cast<std::uint8_t>(VmadValueType::Integer));
    vmad.push_back(0u); put32(vmad, 900u);
    putString(vmad, "Target"); vmad.push_back(static_cast<std::uint8_t>(VmadValueType::Object));
    vmad.push_back(0u); put16(vmad, 0u); put16(vmad, 0xffffu); put32(vmad, 0x1234u);
    VmadAttachments attachments;
    assert(readVmadAttachments(vmad, attachments, error));
    assert(attachments.scripts.size() == 1u &&
           attachments.scripts[0].properties[0].value.integer == 900 &&
           attachments.scripts[0].properties[1].value.object.formId == 0x1234u &&
           attachments.trailingOffset == vmad.size());
    vmad.pop_back();
    assert(!readVmadAttachments(vmad, attachments, error));

    // INFO-specific VMAD tails use bit-selected begin/end fragments after the
    // common attachment prefix and reject malformed trailing data.
    std::vector<std::uint8_t> infoVmad;
    put16(infoVmad, 5u); put16(infoVmad, 2u); put16(infoVmad, 1u);
    putString(infoVmad, "TIF_Fixture"); infoVmad.push_back(0u); put16(infoVmad, 0u);
    infoVmad.push_back(0u);       // unknown
    infoVmad.push_back(2u);       // authored end fragment
    putString(infoVmad, "TIF_Fixture");
    infoVmad.push_back(0u);
    putString(infoVmad, "TIF_Fixture");
    putString(infoVmad, "Fragment_0");
    VmadInfoAttachments infoAttachments;
    assert(readVmadInfoAttachments(infoVmad, infoAttachments, error));
    assert(infoAttachments.flags == 2u && infoAttachments.fragments.size() == 1u &&
           infoAttachments.fragments[0].scriptClass == "TIF_Fixture" &&
           infoAttachments.fragments[0].function == "Fragment_0");
    infoVmad.push_back(0xffu);
    assert(!readVmadInfoAttachments(infoVmad, infoAttachments, error));
    infoVmad.pop_back();

    // Typed DIAL/INFO extraction retains stable identity, localized string
    // IDs, CTDA gates, DNAM response links, and the full INFO fragment.
    std::vector<std::uint8_t> dialEdid{'F','i','x','t','u','r','e','T','o','p','i','c',0u};
    std::vector<std::uint8_t> dialFull; put32(dialFull, 0x1001u);
    std::vector<std::uint8_t> dialQuest; put32(dialQuest, 0x39645u);
    std::vector<std::uint8_t> dialBranch; put32(dialBranch, 0x705u);
    std::vector<std::uint8_t> dialData; put32(dialData, 0u);
    odai::importer::fnv::EsmRecordView dialRecord;
    dialRecord.type = "DIAL";
    dialRecord.subrecords = {
        {"EDID", dialEdid.data(), static_cast<std::uint32_t>(dialEdid.size())},
        {"FULL", dialFull.data(), static_cast<std::uint32_t>(dialFull.size())},
        {"QNAM", dialQuest.data(), static_cast<std::uint32_t>(dialQuest.size())},
        {"BNAM", dialBranch.data(), static_cast<std::uint32_t>(dialBranch.size())},
        {"DATA", dialData.data(), static_cast<std::uint32_t>(dialData.size())}};
    SkyrimDialogueTopicDefinition dialogueTopic;
    assert(readSkyrimDialogueTopic(dialRecord,
        makeRecordKey("Skyrim.esm", 0x700u), dialogueTopic, error));
    assert(dialogueTopic.editorId == "FixtureTopic" &&
           dialogueTopic.promptStringId == 0x1001u &&
           dialogueTopic.rawQuestFormId == 0x39645u &&
           dialogueTopic.rawBranchFormId == 0x705u);

    std::vector<std::uint8_t> branchEdid{'F','i','x','t','u','r','e','B','r','a','n','c','h',0u};
    std::vector<std::uint8_t> branchQuest; put32(branchQuest, 0x39645u);
    std::vector<std::uint8_t> branchStart; put32(branchStart, 0x700u);
    std::vector<std::uint8_t> branchFlags; put32(branchFlags, 1u);
    odai::importer::fnv::EsmRecordView branchRecord;
    branchRecord.type = "DLBR";
    branchRecord.subrecords = {
        {"EDID", branchEdid.data(), static_cast<std::uint32_t>(branchEdid.size())},
        {"QNAM", branchQuest.data(), static_cast<std::uint32_t>(branchQuest.size())},
        {"SNAM", branchStart.data(), static_cast<std::uint32_t>(branchStart.size())},
        {"DNAM", branchFlags.data(), static_cast<std::uint32_t>(branchFlags.size())}};
    SkyrimDialogueBranchDefinition dialogueBranch;
    assert(readSkyrimDialogueBranch(branchRecord,
        makeRecordKey("Skyrim.esm", 0x705u), dialogueBranch, error));
    assert(dialogueBranch.editorId == "FixtureBranch" &&
           dialogueBranch.rawQuestFormId == 0x39645u &&
           dialogueBranch.rawStartingTopicFormId == 0x700u &&
           dialogueBranch.flags == 1u);

    std::vector<std::uint8_t> infoFlags; put32(infoFlags, 1u);
    std::vector<std::uint8_t> responseLink; put32(responseLink, 0x702u);
    std::vector<std::uint8_t> responseData(24u, 0u);
    responseData[12] = 1u;
    std::vector<std::uint8_t> responseText; put32(responseText, 0x2001u);
    std::vector<std::uint8_t> infoPrompt; put32(infoPrompt, 0x2002u);
    std::vector<std::uint8_t> speakerCondition = ctda(0u, false, 1.0f, 72u);
    speakerCondition[12] = 0x88u; speakerCondition[13] = 0x08u;
    std::vector<std::uint8_t> questVariableCondition = ctda(0u, false, 1.0f, 629u);
    questVariableCondition[12] = 0x45u; questVariableCondition[13] = 0x96u;
    questVariableCondition[14] = 0x03u;
    std::vector<std::uint8_t> questVariableName{
        ':', ':', 'r', 'o', 'u', 't', 'e', 'R', 'e', 'a', 'd', 'y', '_', 'v', 'a', 'r', 0u};
    odai::importer::fnv::EsmRecordView infoRecord;
    infoRecord.type = "INFO";
    infoRecord.subrecords = {
        {"VMAD", infoVmad.data(), static_cast<std::uint32_t>(infoVmad.size())},
        {"ENAM", infoFlags.data(), static_cast<std::uint32_t>(infoFlags.size())},
        {"RNAM", infoPrompt.data(), static_cast<std::uint32_t>(infoPrompt.size())},
        {"DNAM", responseLink.data(), static_cast<std::uint32_t>(responseLink.size())},
        {"TRDT", responseData.data(), static_cast<std::uint32_t>(responseData.size())},
        {"NAM1", responseText.data(), static_cast<std::uint32_t>(responseText.size())},
        {"CTDA", speakerCondition.data(), static_cast<std::uint32_t>(speakerCondition.size())},
        {"CTDA", questVariableCondition.data(),
            static_cast<std::uint32_t>(questVariableCondition.size())},
        {"CIS2", questVariableName.data(),
            static_cast<std::uint32_t>(questVariableName.size())}};
    SkyrimDialogueInfoDefinition dialogueInfo;
    assert(readSkyrimDialogueInfo(infoRecord,
        makeRecordKey("Skyrim.esm", 0x701u), dialogueTopic.record,
        makeRecordKey("Skyrim.esm", 0x39645u), dialogueInfo, error));
    assert(dialogueInfo.rawResponseInfoFormId == 0x702u &&
           dialogueInfo.promptStringId == 0x2002u &&
           dialogueInfo.responses.size() == 1u &&
           dialogueInfo.responses[0].textStringId == 0x2001u &&
           dialogueInfo.conditions.size() == 2u &&
           dialogueInfo.conditions[1].stringParameter2 == "::routeReady_var" &&
           dialogueInfo.scripts.flags == 2u);

    // A QUST definition retains stable identity while exposing stage,
    // objective, alias, condition, VMAD, and transitive FormID closure data.
    std::vector<std::pair<std::string, std::vector<std::uint8_t>>> ownedSubrecords;
    odai::importer::fnv::EsmRecordView questRecord;
    questRecord.type = "QUST";
    questRecord.formId = 0x39645u;
    const auto addSubrecord = [&](std::string type, std::vector<std::uint8_t> bytes) {
        ownedSubrecords.emplace_back(std::move(type), std::move(bytes));
        const auto& owned = ownedSubrecords.back();
        questRecord.subrecords.push_back({owned.first, owned.second.data(),
            static_cast<std::uint32_t>(owned.second.size())});
    };
    addSubrecord("EDID", {'M', 'S', '1', '3', 0u});
    std::vector<std::uint8_t> dnam(12u, 0u);
    dnam[0] = 3u; dnam[2] = 60u;
    addSubrecord("DNAM", std::move(dnam));
    addSubrecord("INDX", {30u, 0u});
    addSubrecord("QSDT", {1u});
    std::vector<std::uint8_t> stageCondition = ctda(0u, false, 1.0f, 42u);
    stageCondition[12] = 0x34u; stageCondition[13] = 0x12u;
    addSubrecord("CTDA", std::move(stageCondition));
    addSubrecord("QSDT", {0u});
    addSubrecord("QOBJ", {10u, 0u});
    addSubrecord("NNAM", {0x78u, 0x56u, 0x34u, 0x12u});
    addSubrecord("ALST", {1u, 0u, 0u, 0u});
    addSubrecord("ALID", {'C', 'l', 'a', 'w', 0u});
    addSubrecord("ALFR", {0xefu, 0xcdu, 0xabu, 0u});
    addSubrecord("ALCO", {0x47u, 0x96u, 0x03u, 0u});
    addSubrecord("ALCA", {1u, 0u, 0u, 0x80u});
    addSubrecord("ALCL", {0u, 0u, 0u, 0u});
    addSubrecord("ALED", {});
    vmad.push_back(0u);  // Restore the byte removed by the malformed-input check.
    // Complete TES5 QUST-specific VMAD tail: header, one stage fragment, and
    // one alias-script attachment.
    vmad.push_back(0u);
    put16(vmad, 1u);
    putString(vmad, "QF_MQ101_Fixture");
    put16(vmad, 30u);
    put16(vmad, 0u);
    put32(vmad, 1u);
    vmad.push_back(0u);
    putString(vmad, "QF_MQ101_Fixture");
    putString(vmad, "Fragment_0");
    put16(vmad, 1u);
    put16(vmad, 0u); put16(vmad, 1u); put32(vmad, 0x39645u);
    put16(vmad, 5u); put16(vmad, 2u); put16(vmad, 1u);
    putString(vmad, "MS13GoldenClawScript");
    vmad.push_back(0u); put16(vmad, 0u);
    addSubrecord("VMAD", vmad);

    SkyrimQuestDefinition questDefinition;
    assert(readSkyrimQuest(
        questRecord, makeRecordKey("Skyrim.esm", 0x39645u), questDefinition, error));
    assert(questDefinition.editorId == "MS13" && questDefinition.questFlags == 3u &&
           questDefinition.priority == 60u);
    assert(questDefinition.stages.size() == 1u && questDefinition.stages[0].index == 30u &&
           questDefinition.stages[0].conditions.size() == 1u &&
           questDefinition.stages[0].logEntries.size() == 2u &&
           questDefinition.stages[0].logEntries[0].conditions.size() == 1u &&
           questDefinition.stages[0].logEntries[1].conditions.empty());
    assert(questDefinition.objectives.size() == 1u &&
           questDefinition.objectives[0].displayTextId == 0x12345678u);
    assert(questDefinition.stageFragments.size() == 1u &&
           questDefinition.stageFragments[0].stage == 30u &&
           questDefinition.stageFragments[0].scriptClass == "QF_MQ101_Fixture" &&
           questDefinition.stageFragments[0].function == "Fragment_0");
    assert(questDefinition.aliasScripts.size() == 1u &&
           questDefinition.aliasScripts[0].object.alias == 1u &&
           questDefinition.aliasScripts[0].scripts.size() == 1u &&
           questDefinition.aliasScripts[0].scripts[0].className ==
               "MS13GoldenClawScript");
    questDefinition.objectives[0].displayText = "Retrieve the Golden Claw";
    assert(questDefinition.aliases.size() == 1u && questDefinition.aliases[0].name == "Claw" &&
           questDefinition.aliases[0].forcedReferenceFormId == 0xabcdefu &&
           questDefinition.aliases[0].createdObjectFormId == 0x39647u &&
           questDefinition.aliases[0].createdInAliasId == 1);
    assert(std::binary_search(
        questDefinition.referencedFormIds.begin(), questDefinition.referencedFormIds.end(), 0x1234u));
    assert(session.registerQuestDefinition(
        questDefinition,
        [](std::uint32_t formId) -> std::optional<ObjectId> {
            if (formId == 0xabcdefu) {
                return ObjectId::persistent(makeRecordKey("Skyrim.esm", 0xabcdefu));
            }
            if (formId == 0x39647u) {
                return ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x39647u));
            }
            return std::nullopt;
        }, error));
    goldenClawQuest = session.findQuest("MS13");
    assert(goldenClawQuest != nullptr && goldenClawQuest->objectives.size() == 1u &&
           goldenClawQuest->objectives[0].displayText == "Retrieve the Golden Claw" &&
           goldenClawQuest->aliases.size() == 1u && goldenClawQuest->aliases[0].target.valid() &&
           goldenClawQuest->aliases[0].createdObject ==
               makeRecordKey("Skyrim.esm", 0x39647u));

    // Dynamic TES5 aliases retain the ALFL/ALFA/ALRT topology used by
    // MQ103: a boss reference is found by reference type inside a forced
    // location, then a quest item is created in that boss alias.
    std::vector<std::pair<std::string, std::vector<std::uint8_t>>> dynamicOwned;
    odai::importer::fnv::EsmRecordView dynamicQuestRecord;
    dynamicQuestRecord.type = "QUST";
    dynamicQuestRecord.formId = 0xd0800u;
    const auto addDynamicSubrecord =
        [&](std::string type, std::vector<std::uint8_t> bytes) {
            dynamicOwned.emplace_back(std::move(type), std::move(bytes));
            const auto& owned = dynamicOwned.back();
            dynamicQuestRecord.subrecords.push_back({owned.first, owned.second.data(),
                static_cast<std::uint32_t>(owned.second.size())});
        };
    addDynamicSubrecord("EDID", {'M', 'Q', '1', '0', '3', 'F', 'i', 'x', 't', 'u', 'r', 'e', 0u});
    addDynamicSubrecord("ALLS", {53u, 0u, 0u, 0u});
    addDynamicSubrecord("ALID", {'B', 'l', 'e', 'a', 'k', 'F', 'a', 'l', 'l', 's', 'L', 'o', 'c', 0u});
    addDynamicSubrecord("ALFL", {0xe9u, 0x8eu, 0x01u, 0u});
    addDynamicSubrecord("ALED", {});
    addDynamicSubrecord("ALST", {52u, 0u, 0u, 0u});
    addDynamicSubrecord("ALID", {'B', 'l', 'e', 'a', 'k', 'F', 'a', 'l', 'l', 's', 'B', 'o', 's', 's', 0u});
    addDynamicSubrecord("FNAM", {0x90u, 0x02u, 0u, 0u});
    addDynamicSubrecord("ALFA", {53u, 0u, 0u, 0u});
    addDynamicSubrecord("ALRT", {0xf7u, 0x30u, 0x01u, 0u});
    addDynamicSubrecord("ALED", {});
    addDynamicSubrecord("ALST", {18u, 0u, 0u, 0u});
    addDynamicSubrecord("ALID", {'D', 'r', 'a', 'g', 'o', 'n', 's', 't', 'o', 'n', 'e', 0u});
    addDynamicSubrecord("ALCO", {0x02u, 0xf2u, 0x0du, 0u});
    addDynamicSubrecord("ALCA", {52u, 0u, 0u, 0u});
    addDynamicSubrecord("ALCL", {0u, 0u, 0u, 0u});
    addDynamicSubrecord("ALED", {});
    SkyrimQuestDefinition dynamicQuestDefinition;
    assert(readSkyrimQuest(dynamicQuestRecord,
        makeRecordKey("Skyrim.esm", 0xd0800u), dynamicQuestDefinition, error));
    assert(dynamicQuestDefinition.aliases.size() == 3u);
    const auto parsedBossAlias = std::find_if(dynamicQuestDefinition.aliases.begin(),
        dynamicQuestDefinition.aliases.end(),
        [](const SkyrimQuestAliasDefinition& alias) { return alias.id == 52; });
    assert(parsedBossAlias != dynamicQuestDefinition.aliases.end() &&
           parsedBossAlias->findMatchingReferenceInAliasId == 53 &&
           parsedBossAlias->referenceTypeFormId == 0x130f7u);
    assert(session.registerQuestDefinition(dynamicQuestDefinition,
        [](std::uint32_t formId) -> std::optional<ObjectId> {
            if (formId == 0x18ee9u || formId == 0x130f7u || formId == 0xdf202u) {
                return ObjectId::persistent(makeRecordKey("Skyrim.esm", formId));
            }
            return std::nullopt;
        }, error));
    RuntimeObject fixtureBoss;
    fixtureBoss.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x9bcd6u));
    fixtureBoss.base = makeRecordKey("Skyrim.esm", 0xb7989u);
    fixtureBoss.kind = RuntimeObjectKind::Actor;
    fixtureBoss.location = makeRecordKey("Skyrim.esm", 0x18ee9u);
    fixtureBoss.referenceTypes = {makeRecordKey("Skyrim.esm", 0x130f7u)};
    fixtureBoss.actorValues.emplace();
    assert(session.world().addInitialObject(fixtureBoss, error));
    assert(session.bindQuestInventoryForActor(
        fixtureBoss.id, fixtureBoss.base, error) == 1u);
    assert(session.advance(1.0 / 60.0).diagnostics.empty());
    const QuestRuntimeState* dynamicQuest = session.findQuest("MQ103Fixture");
    assert(dynamicQuest != nullptr);
    const auto runtimeBossAlias = std::find_if(dynamicQuest->aliases.begin(),
        dynamicQuest->aliases.end(),
        [](const QuestAliasRuntimeState& alias) { return alias.id == 52; });
    const auto runtimeDragonstoneAlias = std::find_if(dynamicQuest->aliases.begin(),
        dynamicQuest->aliases.end(),
        [](const QuestAliasRuntimeState& alias) { return alias.id == 18; });
    assert(runtimeBossAlias != dynamicQuest->aliases.end() &&
           runtimeBossAlias->target == fixtureBoss.id);
    assert(runtimeDragonstoneAlias != dynamicQuest->aliases.end() &&
           runtimeDragonstoneAlias->target.kind == ObjectIdKind::Spawned);
    const RuntimeObject* dragonstoneInstance =
        session.world().find(runtimeDragonstoneAlias->target);
    const RuntimeObject* fixtureBossWithLoot = session.world().find(fixtureBoss.id);
    assert(dragonstoneInstance != nullptr &&
           dragonstoneInstance->kind == RuntimeObjectKind::Item &&
           dragonstoneInstance->base == makeRecordKey("Skyrim.esm", 0xdf202u) &&
           fixtureBossWithLoot != nullptr && fixtureBossWithLoot->inventory.size() == 1u &&
           fixtureBossWithLoot->inventory[0].item == makeRecordKey("Skyrim.esm", 0xdf202u));
    PapyrusFunction removeSpawnedItem;
    removeSpawnedItem.name = "Fixture.RemoveSpawnedItem";
    PapyrusInstruction removeItem;
    removeItem.opcode = PapyrusOpcode::CallMethod;
    removeItem.name = "RemoveItem";
    removeItem.targetType = "ObjectReference";
    removeItem.operands = {
        PapyrusOperand::fromLiteral(PapyrusValue::fromObject(fixtureBoss.id)),
        PapyrusOperand::fromLiteral(
            PapyrusValue::fromObject(runtimeDragonstoneAlias->target)),
        PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(1)),
        PapyrusOperand::fromLiteral(PapyrusValue::fromBoolean(false)),
        PapyrusOperand::fromLiteral(PapyrusValue{})};
    PapyrusInstruction returnAfterRemove;
    returnAfterRemove.opcode = PapyrusOpcode::Return;
    removeSpawnedItem.instructions = {removeItem, returnAfterRemove};
    assert(session.papyrus().registerFunction(std::move(removeSpawnedItem), error));
    assert(session.papyrus().startFunction("Fixture.RemoveSpawnedItem", {}, error) != 0u);
    assert(session.advance(1.0 / 60.0).diagnostics.empty());
    fixtureBossWithLoot = session.world().find(fixtureBoss.id);
    assert(fixtureBossWithLoot != nullptr && fixtureBossWithLoot->inventory.empty());
    RuntimeObject ambiguousBoss = fixtureBoss;
    ambiguousBoss.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x9bcd7u));
    assert(session.world().addInitialObject(ambiguousBoss, error));
    assert(session.bindDynamicQuestAliasesForObject(ambiguousBoss.id, error) == 0u);
    assert(error.find("ambiguous dynamic quest alias MQ103Fixture:52") !=
           std::string::npos);
    dynamicQuest = session.findQuest("MQ103Fixture");
    const auto unchangedBossAlias = std::find_if(dynamicQuest->aliases.begin(),
        dynamicQuest->aliases.end(),
        [](const QuestAliasRuntimeState& alias) { return alias.id == 52; });
    assert(unchangedBossAlias != dynamicQuest->aliases.end() &&
           unchangedBossAlias->target == fixtureBoss.id);
    error.clear();

    dialogueBranch.quest = makeRecordKey("Skyrim.esm", 0x39645u);
    dialogueBranch.startingTopic = dialogueTopic.record;
    dialogueTopic.quest = dialogueBranch.quest;
    dialogueTopic.branch = dialogueBranch.record;
    dialogueTopic.prompt = "I could help with the fixture item.";
    dialogueInfo.prompt = "Wait, you mean this fixture item?";
    dialogueInfo.responses[0].text = "Thank you for the fixture item.";
    dialogueInfo.responseInfo = makeRecordKey("Skyrim.esm", 0x702u);
    dialogueInfo.linkedTopics = {makeRecordKey("Skyrim.esm", 0x703u)};
    assert(session.registerDialogueBranch(dialogueBranch, error));
    assert(session.registerDialogueTopic(dialogueTopic, error));
    assert(session.registerDialogueInfo(dialogueInfo, error));
    SkyrimDialogueTopicDefinition linkedTopic;
    linkedTopic.record = makeRecordKey("Skyrim.esm", 0x703u);
    linkedTopic.quest = dialogueBranch.quest;
    linkedTopic.branch = dialogueBranch.record;
    linkedTopic.prompt = "A linked follow-up.";
    assert(session.registerDialogueTopic(linkedTopic, error));
    SkyrimDialogueInfoDefinition linkedInfo;
    linkedInfo.record = makeRecordKey("Skyrim.esm", 0x704u);
    linkedInfo.topic = linkedTopic.record;
    linkedInfo.quest = dialogueBranch.quest;
    linkedInfo.responses.push_back({0u, "Linked response.", 0u, 0u});
    assert(session.registerDialogueInfo(linkedInfo, error));
    RuntimeObject dialogueSpeaker;
    dialogueSpeaker.id = session.world().allocateRuntimeId();
    dialogueSpeaker.base = makeRecordKey("Skyrim.esm", 0x888u);
    dialogueSpeaker.kind = RuntimeObjectKind::Actor;
    dialogueSpeaker.actorValues.emplace();
    assert(session.world().addInitialObject(dialogueSpeaker, error));
    session.setResolvedFormResolver([](std::uint32_t formId) -> std::optional<ObjectId> {
        if (formId == 0x888u) {
            return ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x888u));
        }
        if (formId == 0x39645u) {
            return ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x39645u));
        }
        return std::nullopt;
    });
    PapyrusFunction infoFragment;
    infoFragment.name = "TIF_Fixture.Fragment_0";
    infoFragment.scriptClass = "TIF_Fixture";
    infoFragment.parentClass = "TopicInfo";
    infoFragment.parameters = {"akSpeakerRef"};
    PapyrusInstruction getOwningQuest;
    getOwningQuest.opcode = PapyrusOpcode::CallMethod;
    getOwningQuest.name = "GetOwningQuest";
    getOwningQuest.targetType = "TopicInfo";
    getOwningQuest.destination = "owner";
    getOwningQuest.operands = {PapyrusOperand::fromLocal("self")};
    PapyrusInstruction dialogueSetStage;
    dialogueSetStage.opcode = PapyrusOpcode::CallMethod;
    dialogueSetStage.name = "SetStage";
    dialogueSetStage.targetType = "Quest";
    dialogueSetStage.operands = {PapyrusOperand::fromLocal("owner"),
        PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(20))};
    PapyrusInstruction dialogueReturn;
    dialogueReturn.opcode = PapyrusOpcode::Return;
    infoFragment.instructions = {getOwningQuest, dialogueSetStage, dialogueReturn};
    session.papyrus().registerClassParent("TIF_Fixture", "TopicInfo");
    assert(session.papyrus().registerFunction(std::move(infoFragment), error));
    assert(session.papyrus().attachScript(
        ObjectId::persistent(dialogueInfo.record), "TIF_Fixture", {}, error));
    assert(session.papyrus().attachScript(
        ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x39645u)), "RouteState",
        {{"::routeReady_var", PapyrusValue::fromBoolean(true)}}, error));
    const std::vector<SkyrimDialogueChoice> choices =
        session.availableDialogueChoices(dialogueSpeaker.id, player.id, true);
    assert(choices.size() == 1u &&
           choices[0].prompt == "Wait, you mean this fixture item?");
    const std::vector<RecordKey> linkedEligibility{linkedTopic.record};
    const std::vector<SkyrimDialogueChoice> linkedChoices =
        session.availableDialogueChoices(
            dialogueSpeaker.id, player.id, true, linkedEligibility);
    assert(linkedChoices.size() == 1u && linkedChoices[0].info == linkedInfo.record);
    // Bit 1 is absent: selecting the begin phase must not run an authored end
    // fragment. Finishing the response (bit 2) dispatches it on the next tick.
    assert(session.selectDialogueInfo(
        dialogueInfo.record, dialogueSpeaker.id, player.id, 1u).accepted);
    assert(session.findQuest("MS13")->stage == 0);
    const SkyrimDialogueSelectionResult selectedDialogue =
        session.selectDialogueInfo(dialogueInfo.record, dialogueSpeaker.id, player.id, 2u);
    assert(selectedDialogue.accepted &&
           selectedDialogue.nextTopics == dialogueInfo.linkedTopics &&
           selectedDialogue.responses ==
               std::vector<std::string>{"Thank you for the fixture item."});
    assert(session.advance(1.0 / 60.0).diagnostics.empty());
    assert(session.findQuest("MS13")->stage == 20);

    PapyrusFunction containerChanged;
    containerChanged.name = "MS13GoldenClawScript.OnContainerChanged";
    containerChanged.parameters = {"new_container", "old_container"};
    PapyrusInstruction returnFromContainerChanged;
    returnFromContainerChanged.opcode = PapyrusOpcode::Return;
    containerChanged.instructions = {returnFromContainerChanged};
    assert(session.papyrus().registerFunction(std::move(containerChanged), error));
    assert(session.papyrus().attachScript(
        goldenClawQuest->aliases[0].handle, "MS13GoldenClawScript", {}, error));

    RuntimeObject arvel;
    arvel.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x3a1e2u));
    arvel.base = makeRecordKey("Skyrim.esm", 0xabcdefu);
    arvel.kind = RuntimeObjectKind::Actor;
    arvel.actorValues.emplace();
    arvel.actorValues->health = 0.0f;
    arvel.actorValues->dead = true;
    assert(session.world().addInitialObject(arvel, error));
    assert(session.bindQuestInventoryForActor(arvel.id, arvel.base, error) == 1u);
    assert(session.bindQuestInventoryForActor(arvel.id, arvel.base, error) == 0u);
    const BethesdaSessionStep materialized = session.advance(1.0 / 60.0);
    assert(materialized.diagnostics.empty());
    assert(session.papyrus().activeThreadCount() == 1u);
    assert(session.advance(1.0 / 60.0).diagnostics.empty());
    assert(session.papyrus().activeThreadCount() == 0u);
    const RuntimeObject* arvelWithClaw = session.world().find(arvel.id);
    assert(arvelWithClaw != nullptr && arvelWithClaw->inventory.size() == 1u &&
           arvelWithClaw->inventory[0].item == claw);
    const LootTransferResult looted = session.lootObject(player.id, arvel.id);
    assert(looted.accepted && looted.transferred.size() == 1u &&
           looted.transferred[0].item == claw);
    const BethesdaSessionStep lootStep = session.advance(1.0 / 60.0);
    assert(lootStep.diagnostics.empty());
    assert(session.papyrus().activeThreadCount() == 1u);
    assert(session.advance(1.0 / 60.0).diagnostics.empty());
    assert(session.papyrus().activeThreadCount() == 0u);
    updated = session.world().find(player.id);
    arvelWithClaw = session.world().find(arvel.id);
    assert(updated != nullptr && updated->inventory[0].count == 2 &&
           arvelWithClaw != nullptr && arvelWithClaw->inventory.empty());

    PapyrusFunction stageFragment;
    stageFragment.name = "QF_MQ101_Fixture.Fragment_0";
    PapyrusInstruction displayObjective;
    displayObjective.opcode = PapyrusOpcode::CallMethod;
    displayObjective.name = "SetObjectiveDisplayed";
    displayObjective.targetType = "Quest";
    displayObjective.operands = {
        PapyrusOperand::fromLiteral(PapyrusValue::fromObject(
            ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x39645u)))),
        PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(10)),
        PapyrusOperand::fromLiteral(PapyrusValue::fromBoolean(true))};
    PapyrusInstruction returnFromFragment;
    returnFromFragment.opcode = PapyrusOpcode::Return;
    stageFragment.instructions = {displayObjective, returnFromFragment};
    assert(session.papyrus().registerFunction(std::move(stageFragment), error));
    assert(session.papyrus().attachScript(
        ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x39645u)),
        "QF_MQ101_Fixture", {}, error));

    PapyrusFunction advanceQuest;
    advanceQuest.name = "Fixture.AdvanceQuest";
    PapyrusInstruction setStage;
    setStage.opcode = PapyrusOpcode::CallMethod;
    setStage.name = "SetStage";
    setStage.targetType = "Quest";
    setStage.operands = {
        PapyrusOperand::fromLiteral(PapyrusValue::fromObject(
            ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x39645u)))),
        PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(30))};
    PapyrusInstruction achievement;
    achievement.opcode = PapyrusOpcode::CallStatic;
    achievement.name = "Game.AddAchievement";
    achievement.operands = {
        PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(8))};
    const ObjectId mapMarker = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x10fef8u));
    PapyrusInstruction addToMap;
    addToMap.opcode = PapyrusOpcode::CallMethod;
    addToMap.name = "AddToMap";
    addToMap.targetType = "ObjectReference";
    addToMap.operands = {
        PapyrusOperand::fromLiteral(PapyrusValue::fromObject(mapMarker))};
    PapyrusInstruction sideQuestStatistic;
    sideQuestStatistic.opcode = PapyrusOpcode::CallMethod;
    sideQuestStatistic.name = "IncSideQuests";
    sideQuestStatistic.targetType = "AchievementsScript";
    sideQuestStatistic.operands = {
        PapyrusOperand::fromLiteral(PapyrusValue::fromObject(player.id))};
    PapyrusInstruction finishQuest;
    finishQuest.opcode = PapyrusOpcode::Return;
    advanceQuest.instructions = {
        setStage, achievement, addToMap, sideQuestStatistic, finishQuest};
    assert(session.papyrus().registerFunction(std::move(advanceQuest), error));
    assert(session.papyrus().startFunction("Fixture.AdvanceQuest", {}, error) != 0u);
    const BethesdaSessionStep questStep = session.advance(1.0 / 60.0);
    assert(questStep.diagnostics.empty());
    goldenClawQuest = session.findQuest("MS13");
    assert(goldenClawQuest != nullptr && goldenClawQuest->stage == 30 &&
           std::binary_search(goldenClawQuest->completedStages.begin(),
                              goldenClawQuest->completedStages.end(), 30));
    // A native SetStage posts the fragment without mutating the VM thread
    // vector under iteration. The fragment then executes deterministically on
    // the next fixed tick.
    assert(session.papyrus().activeThreadCount() == 1u);
    const BethesdaSessionStep fragmentStep = session.advance(1.0 / 60.0);
    assert(fragmentStep.diagnostics.empty());
    goldenClawQuest = session.findQuest("MS13");
    assert(goldenClawQuest != nullptr && goldenClawQuest->objectives[0].displayed);
    assert(session.statistics().at("achievement:8") == 1 &&
           session.statistics().at("side_quests_completed") == 1);
    assert(session.discoveries() == std::vector<RecordKey>{mapMarker.reference});

    const RecordKey runtimeLocation = makeRecordKey("Skyrim.esm", 0x7819bu);
    const RecordKey runtimeKeyword = makeRecordKey("Skyrim.esm", 0x10f63cu);
    const RecordKey runtimeGlobal = makeRecordKey("Skyrim.esm", 0x3f0d0u);
    assert(session.registerLocation(runtimeLocation, {}, {runtimeKeyword}, error));
    assert(session.registerGlobalVariable(runtimeGlobal, 2.0f, error));
    session.setLocationLoaded(runtimeLocation, true);
    session.setResolvedFormResolver([&](std::uint32_t formId) -> std::optional<ObjectId> {
        return formId == 0x10fef8u ? std::optional<ObjectId>(mapMarker) : std::nullopt;
    });
    std::vector<PapyrusValue> capturedNativeValues;
    session.papyrus().registerNative("Fixture.CaptureNativeValue",
        [&](std::span<const PapyrusValue> values, std::uint64_t, BethesdaWorld&) {
            NativeCallResult result;
            if (values.size() != 1u) result.error = "capture expects one value";
            else capturedNativeValues.push_back(values[0]);
            return result;
        });
    PapyrusFunction nativeState;
    nativeState.name = "Fixture.NativeState";
    const PapyrusOperand locationObject = PapyrusOperand::fromLiteral(
        PapyrusValue::fromObject(ObjectId::persistent(runtimeLocation)));
    const PapyrusOperand keywordObject = PapyrusOperand::fromLiteral(
        PapyrusValue::fromObject(ObjectId::persistent(runtimeKeyword)));
    const PapyrusOperand globalObject = PapyrusOperand::fromLiteral(
        PapyrusValue::fromObject(ObjectId::persistent(runtimeGlobal)));
    PapyrusInstruction setKeywordData;
    setKeywordData.opcode = PapyrusOpcode::CallMethod;
    setKeywordData.targetType = "Location";
    setKeywordData.name = "SetKeywordData";
    setKeywordData.operands = {locationObject, keywordObject,
        PapyrusOperand::fromLiteral(PapyrusValue::fromFloat(4.25))};
    PapyrusInstruction hasKeyword;
    hasKeyword.opcode = PapyrusOpcode::CallMethod;
    hasKeyword.targetType = "Location";
    hasKeyword.name = "HasKeyword";
    hasKeyword.destination = "has_keyword";
    hasKeyword.operands = {locationObject, keywordObject};
    PapyrusInstruction isLoaded = hasKeyword;
    isLoaded.name = "IsLoaded";
    isLoaded.destination = "is_loaded";
    isLoaded.operands = {locationObject};
    PapyrusInstruction setGlobal;
    setGlobal.opcode = PapyrusOpcode::CallMethod;
    setGlobal.targetType = "GlobalVariable";
    setGlobal.name = "SetValue";
    setGlobal.operands = {globalObject,
        PapyrusOperand::fromLiteral(PapyrusValue::fromFloat(7.5))};
    PapyrusInstruction getGlobal = setGlobal;
    getGlobal.name = "GetValue";
    getGlobal.destination = "global_value";
    getGlobal.operands = {globalObject};
    PapyrusInstruction storyEvent;
    storyEvent.opcode = PapyrusOpcode::CallMethod;
    storyEvent.targetType = "Keyword";
    storyEvent.name = "SendStoryEvent";
    storyEvent.operands = {keywordObject, locationObject,
        PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(9))};
    PapyrusInstruction openLog;
    openLog.opcode = PapyrusOpcode::CallStatic;
    openLog.name = "Debug.OpenUserLog";
    openLog.operands = {
        PapyrusOperand::fromLiteral(PapyrusValue::fromString("scenario-fixture"))};
    PapyrusInstruction getForm;
    getForm.opcode = PapyrusOpcode::CallStatic;
    getForm.name = "Game.GetForm";
    getForm.destination = "resolved_form";
    getForm.operands = {
        PapyrusOperand::fromLiteral(PapyrusValue::fromInteger(0x10fef8u))};
    const auto captureLocal = [&](const std::string& local) {
        PapyrusInstruction instruction;
        instruction.opcode = PapyrusOpcode::CallNative;
        instruction.name = "Fixture.CaptureNativeValue";
        instruction.operands = {PapyrusOperand::fromLocal(local)};
        return instruction;
    };
    nativeState.instructions = {setKeywordData, hasKeyword, captureLocal("has_keyword"),
        isLoaded, captureLocal("is_loaded"), setGlobal, getGlobal,
        captureLocal("global_value"), storyEvent, openLog, getForm,
        captureLocal("resolved_form"), finishQuest};
    assert(session.papyrus().registerFunction(std::move(nativeState), error));
    assert(session.papyrus().startFunction("Fixture.NativeState", {}, error) != 0u);
    const BethesdaSessionStep nativeStep = session.advance(1.0 / 60.0);
    assert(nativeStep.diagnostics.empty());
    assert(capturedNativeValues.size() == 4u && capturedNativeValues[0].boolean &&
           capturedNativeValues[1].boolean && capturedNativeValues[2].real == 7.5 &&
           capturedNativeValues[3].object == mapMarker);
    assert(session.locations().at(runtimeLocation).keywordData.at(runtimeKeyword) == 4.25f);
    assert(session.globalVariables().at(runtimeGlobal) == 7.5f);
    assert(session.storyEvents().size() == 1u &&
           session.storyEvents()[0].keyword == runtimeKeyword);
    assert(session.scriptDebugLogs() == std::vector<std::string>{"scenario-fixture"});

    std::cout << "bethesda runtime tests passed\n";
    return 0;
}
