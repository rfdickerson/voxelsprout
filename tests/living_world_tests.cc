#include "bethesda/bethesda_session.h"
#include "bethesda/gameplay_catalog.h"
#include "bethesda/living_world.h"
#include "bethesda/save_game.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#ifdef NDEBUG
#undef assert
#define assert(expression)                                                        \
    do {                                                                          \
        if (!(expression)) {                                                      \
            std::cerr << "living-world check failed: " #expression << " at "    \
                      << __FILE__ << ':' << __LINE__ << '\n';                     \
            std::abort();                                                         \
        }                                                                         \
    } while (false)
#endif

namespace {

using namespace odai::bethesda;

ObjectId reference(std::uint32_t id) {
    return ObjectId::persistent(makeTes3ReferenceKey("morrowind.esm", id));
}

RuntimeSpaceState balmoraSpace() {
    RuntimeSpaceState space;
    space.kind = RuntimeSpaceKind::Exterior;
    space.worldspace = makeTes3RecordKey("WRLD", "vardenfell");
    space.gridX = -3;
    space.gridZ = -2;
    return space;
}

RuntimeObject object(
    ObjectId id, std::string type, std::string base,
    RuntimeObjectKind kind, std::array<double, 3> position) {
    RuntimeObject value;
    value.id = std::move(id);
    value.base = makeTes3RecordKey(std::move(type), std::move(base));
    value.kind = kind;
    value.transform.position = position;
    value.originSpace = balmoraSpace();
    value.currentSpace = balmoraSpace();
    if (kind == RuntimeObjectKind::Actor) value.actorValues.emplace();
    return value;
}

ActivityAnchor anchor(
    ObjectId id, ActivityAnchorKind kind, std::array<double, 3> position,
    std::uint32_t capacity = 1u) {
    ActivityAnchor value;
    value.object = std::move(id);
    value.kind = kind;
    value.space = balmoraSpace();
    value.position = position;
    value.capacity = capacity;
    return value;
}

GameplayCellPayload balmoraPayload() {
    GameplayCellPayload payload;
    payload.contentFingerprint = "balmora-living-fixture";
    payload.space = balmoraSpace();
    payload.anchors = {
        anchor(reference(100u), ActivityAnchorKind::Bed, {0.0, 0.0, 0.0}, 4u),
        anchor(reference(101u), ActivityAnchorKind::Meal, {10.0, 0.0, 0.0}, 4u),
        anchor(reference(102u), ActivityAnchorKind::ShopCounter, {20.0, 0.0, 0.0}, 2u),
        anchor(reference(103u), ActivityAnchorKind::Tavern, {30.0, 0.0, 0.0}, 4u),
        anchor(reference(104u), ActivityAnchorKind::Patrol, {40.0, 0.0, 0.0}, 2u),
        anchor(reference(105u), ActivityAnchorKind::Worship, {50.0, 0.0, 0.0}, 1u),
    };
    ActorArchetype merchant;
    merchant.actor = reference(1u);
    merchant.base = makeTes3RecordKey("NPC_", "balmora_merchant");
    merchant.homeSpace = balmoraSpace();
    merchant.roles = ActorRole::Citizen | ActorRole::Merchant;
    merchant.authoredPackages.push_back(BehaviorPackage{
        "tes3:temple-duty", BehaviorPackageSource::AuthoredTes3,
        RuntimeActivityKind::Worship, ActivityAnchorKind::Worship,
        reference(105u), 540u, 600u, 100, true, "authored temple duty"});
    ActorArchetype citizen;
    citizen.actor = reference(2u);
    citizen.base = makeTes3RecordKey("NPC_", "balmora_citizen");
    citizen.homeSpace = balmoraSpace();
    citizen.roles = ActorRole::Citizen | ActorRole::Merchant;
    citizen.authoredPosition = {5.0, 0.0, 0.0};
    ActorArchetype guard;
    guard.actor = reference(3u);
    guard.base = makeTes3RecordKey("NPC_", "balmora_guard");
    guard.homeSpace = balmoraSpace();
    guard.roles = ActorRole::Citizen | ActorRole::Guard;
    guard.authoredPosition = {40.0, 0.0, 0.0};
    payload.actors = {merchant, citizen, guard};
    payload.physicsPolicies.push_back(PhysicsPolicy{
        reference(200u), PhysicsClassification::Breakable, 2.0f,
        false, true, false, false});
    payload.physicsPolicies.push_back(PhysicsPolicy{
        reference(201u), PhysicsClassification::Breakable, 1.0f,
        true, true, false, true});
    return payload;
}

void addFixtureObjects(BethesdaWorld& world) {
    std::string error;
    assert(world.addInitialObject(object(reference(1u), "NPC_", "balmora_merchant",
        RuntimeObjectKind::Actor, {0.0, 0.0, 0.0}), error));
    assert(world.addInitialObject(object(reference(2u), "NPC_", "balmora_citizen",
        RuntimeObjectKind::Actor, {5.0, 0.0, 0.0}), error));
    assert(world.addInitialObject(object(reference(3u), "NPC_", "balmora_guard",
        RuntimeObjectKind::Actor, {40.0, 0.0, 0.0}), error));
    for (std::uint32_t id = 100u; id <= 105u; ++id) {
        assert(world.addInitialObject(object(reference(id), "ACTI", "anchor_" +
            std::to_string(id), RuntimeObjectKind::Activator,
            {static_cast<double>((id - 100u) * 10u), 0.0, 0.0}), error));
    }
    assert(world.addInitialObject(object(reference(200u), "MISC", "breakable_crate",
        RuntimeObjectKind::Item, {70.0, 5.0, 0.0}), error));
    assert(world.addInitialObject(object(reference(201u), "MISC", "quest_vase",
        RuntimeObjectKind::Item, {75.0, 5.0, 0.0}), error));
}

}  // namespace

int main() {
    using namespace odai::bethesda;
    std::string error;
    GameplayCellPayload payload = balmoraPayload();

    const std::filesystem::path sidecar =
        std::filesystem::temp_directory_path() / "odai-living-world-cell.json";
    assert(saveGameplayCellPayloadAtomic(sidecar, payload, error));
    GameplayCellPayload loaded;
    assert(loadGameplayCellPayload(
        sidecar, payload.contentFingerprint, loaded, error));
    assert(loaded == payload);
    assert(!loadGameplayCellPayload(sidecar, "different-profile", loaded, error));

    const ScheduleCompileResult compiled =
        SystemicScheduleCompiler{}.compile({payload});
    assert(compiled.actors.size() == 3u);
    const ActorSchedule& merchantSchedule = compiled.actors.at(reference(1u));
    const ScheduleEntry* atNine =
        SystemicScheduleCompiler::entryAt(merchantSchedule, 9u * 60u);
    assert(atNine != nullptr);
    assert(atNine->source == BehaviorPackageSource::AuthoredTes3);
    assert(atNine->activity == RuntimeActivityKind::Worship);
    assert(atNine->anchor == reference(105u));

    BethesdaSession session;
    assert(session.configure({odai::importer::fnv::BethesdaGame::Morrowind,
        payload.contentFingerprint, "", 11u}, error));
    addFixtureObjects(session.world());
    assert(session.installGameplayCells({payload}, error));

    // No resident spaces means analytical simulation. At 09:00 the authored
    // TES3 package beats the generated shop routine and reconciles directly.
    LivingWorldStep offscreen =
        session.livingWorld().advanceGameMinutes(60u, session.world());
    assert(offscreen.offscreenReconciliations >= 2u);
    assert(session.world().applyQueuedCommands().diagnostics.empty());
    const RuntimeObject* merchant = session.world().find(reference(1u));
    assert(merchant != nullptr && merchant->livingState.has_value());
    assert(merchant->livingState->activity == RuntimeActivityKind::Worship);
    assert(merchant->transform.position[0] == 50.0);

    session.setGameplayResidentSpaces({balmoraSpace()});
    (void)session.livingWorld().advanceGameMinutes(1u, session.world());
    assert(session.world().applyQueuedCommands().diagnostics.empty());

    LivingWorldStimulus theft;
    theft.kind = StimulusKind::Theft;
    theft.source = reference(1u);
    theft.subject = reference(200u);
    theft.position = {40.0, 0.0, 0.0};
    theft.sightRadius = 512.0f;
    theft.hearingRadius = 512.0f;
    theft.crime = RuntimeCrimeKind::Theft;
    theft.bounty = 25;
    const std::uint64_t theftSequence = session.postLivingWorldStimulus(theft);
    LivingWorldStep reaction = session.livingWorld().advance(
        0.0, session.world(), [](const auto&, const auto&) { return true; });
    assert(!reaction.witnesses.empty());
    assert(reaction.witnesses.front().stimulusSequence == theftSequence);
    assert(session.world().applyQueuedCommands().diagnostics.empty());
    merchant = session.world().find(reference(1u));
    const RuntimeObject* guard = session.world().find(reference(3u));
    assert(merchant->livingState->bounty == 25);
    assert(merchant->livingState->crimesCommitted == 1u);
    assert(guard != nullptr && guard->livingState.has_value());
    assert(guard->livingState->source == RuntimeBehaviorSource::CrimeOrEmergency);

    // The sidecar policy initializes a transient physical delta. It resets to
    // the authored transform after 72 unloaded game-hours.
    session.setGameplayResidentSpaces({});
    (void)session.livingWorld().advanceGameMinutes(0u, session.world());
    assert(session.world().applyQueuedCommands().diagnostics.empty());
    RuntimeObject* crate = session.world().find(reference(200u));
    assert(crate != nullptr && crate->physicalState.has_value());
    WorldCommand breakQuestObject;
    breakQuestObject.type = WorldCommandType::BreakObject;
    breakQuestObject.target = reference(201u);
    (void)session.world().queue(std::move(breakQuestObject));
    assert(session.world().applyQueuedCommands().diagnostics.empty());
    const RuntimeObject* questVase = session.world().find(reference(201u));
    assert(questVase != nullptr && questVase->physicalState->broken &&
        questVase->physicalState->protectedFromDestruction &&
        questVase->physicalState->meaningful);
    WorldCommand move;
    move.type = WorldCommandType::SetPosition;
    move.target = crate->id;
    move.transform.position = {700.0, 50.0, 0.0};
    (void)session.world().queue(std::move(move));
    assert(session.world().applyQueuedCommands().diagnostics.empty());
    (void)session.livingWorld().advanceGameMinutes(1u, session.world());
    assert(session.world().applyQueuedCommands().diagnostics.empty());
    (void)session.livingWorld().advanceGameMinutes(72u * 60u, session.world());
    assert(session.world().applyQueuedCommands().diagnostics.empty());
    crate = session.world().find(reference(200u));
    assert(crate->transform.position[0] == 70.0);
    assert(crate->transform.position[1] == 5.0);

    // Jolt dynamic bodies are stable runtime identities and round-trip their
    // physical snapshot independently of ImportedScene serialization.
    BethesdaPhysicsWorld physics;
    assert(physics.initialize(error));
    PhysicsDynamicBodyConfig body;
    body.position = {0.0f, 100.0f, 0.0f};
    body.boundsHalfExtents = {8.0f, 8.0f, 8.0f};
    body.massKilograms = 2.0f;
    body.buoyant = true;
    assert(physics.addDynamicBody(reference(300u), body, error));
    PhysicsHingeConfig hinge;
    hinge.worldAnchor = {0.0f, 100.0f, 0.0f};
    hinge.minimumAngleRadians = -1.5707963f;
    hinge.maximumAngleRadians = 1.5707963f;
    hinge.frictionTorqueNewtonMetres = 0.1f;
    assert(physics.addWorldHingeConstraint(reference(300u), hinge, error));
    assert(physics.hasConstraint(reference(300u)));
    assert(physics.removeConstraint(reference(300u)));
    assert(!physics.hasConstraint(reference(300u)));
    assert(physics.addDynamicBodyImpulse(reference(300u), {10.0f, 0.0f, 0.0f}));
    (void)physics.step(1.0f / 60.0f);
    std::vector<PhysicsDynamicBodySnapshot> dynamic = physics.dynamicBodySnapshots();
    assert(dynamic.size() == 1u);
    assert(dynamic[0].linearVelocity.x > 0.0f);
    const PhysicsDynamicBodySnapshot savedDynamic = dynamic[0];
    assert(physics.setDynamicBodyTransform(reference(300u), {1000.0f, 0.0f, 0.0f}, {}));
    assert(physics.restoreDynamicBody(savedDynamic, error));
    assert(std::fabs(physics.dynamicBodySnapshots()[0].position.x -
        savedDynamic.position.x) < 0.01f);

    // Save v9 includes living-world time, crimes, schedules, and meaningful
    // physical persistence state without changing ImportedScene.
    const std::filesystem::path save =
        std::filesystem::temp_directory_path() / "odai-living-world-save.json";
    assert(saveOdaiGameAtomic(save, session, error));
    BethesdaSession restored;
    assert(restored.configure({odai::importer::fnv::BethesdaGame::Morrowind,
        payload.contentFingerprint, "", 11u}, error));
    assert(restored.installGameplayCells({payload}, error));
    SaveLoadReport report;
    assert(loadOdaiGame(save, restored, {}, report, error));
    assert(restored.livingWorld().absoluteGameMinute() ==
        session.livingWorld().absoluteGameMinute());
    const RuntimeObject* restoredMerchant = restored.world().find(reference(1u));
    assert(restoredMerchant != nullptr && restoredMerchant->livingState.has_value());
    assert(restoredMerchant->livingState->bounty == 25);
    assert(restored.world().find(reference(200u))->physicalState.has_value());

    std::error_code ignored;
    std::filesystem::remove(sidecar, ignored);
    std::filesystem::remove(save, ignored);
    std::cout << "living world tests passed\n";
    return 0;
}
