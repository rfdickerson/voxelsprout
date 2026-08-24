#include "games/newvegas/newvegas_actors.h"
#include "bethesda/bethesda_session.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace odai::games::newvegas;

namespace {

SkinnedActor walkingActor(bool controllerOwned) {
    SkinnedActor actor;
    actor.name = "MovementFixture";
    actor.placed = true;
    actor.renderVisible = true;
    actor.wanders = true;
    actor.walkSpeedUnitsPerSecond = 120.0f;
    actor.wanderTarget[0] = 1000.0f;
    actor.runtimeControllerOwned = controllerOwned;
    return actor;
}

std::uint64_t replayHash(const std::vector<double>& frameDeltas) {
    using namespace odai::bethesda;
    BethesdaSession session;
    std::string error;
    assert(session.configure({
        odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition,
        "actor-movement-fixture", "skyrim-bleak-falls", 41u}, error));
    RuntimeObject object;
    object.id = ObjectId::persistent(makeRecordKey("Skyrim.esm", 0x1234u));
    object.base = makeRecordKey("Skyrim.esm", 0x7u);
    object.kind = RuntimeObjectKind::Actor;
    object.transform.position = {0.0, 100.0, 0.0};
    object.actorValues.emplace();
    assert(session.world().addInitialObject(object, error));
    PhysicsCharacterConfig config;
    config.position = {0.0f, 100.0f, 0.0f};
    assert(session.registerActorController(object.id, config, error));

    std::vector<SkinnedActor> actors{walkingActor(true)};
    actors[0].position[1] = 100.0f;
    const auto noGround = [](float, float, float, float&) { return false; };
    const auto noWalls = [](float&, float&, float, float, float) {};
    for (const double frameDelta : frameDeltas) {
        (void)session.advance(frameDelta,
            [&](std::uint64_t, double fixedStep) {
                const auto physical = session.physics().characterState(object.id);
                assert(physical.has_value());
                actors[0].position[0] = physical->position.x;
                actors[0].position[1] = physical->position.y;
                actors[0].position[2] = physical->position.z;
                actors[0].runtimeControllerBlocked = physical->blocked;
                updateActorWandering(actors, static_cast<float>(fixedStep), nullptr,
                    noGround, noWalls, -1);
                PhysicsCharacterInput input;
                input.desiredVelocity = actors[0].runtimeRequestedVelocity;
                assert(session.setActorControllerInput(object.id, input));
            });
    }
    return session.deterministicHash();
}

}  // namespace

int main() {
    const auto noGround = [](float, float, float, float&) { return false; };
    const auto noWalls = [](float&, float&, float, float, float) {};

    // Dialogue follows a live head anchor when it remains anatomically close
    // to the imported bind-pose head. A broken TES3 controller used to return
    // a perfectly finite pelvis-height point, which passed the old > 1 check
    // and made the conversation zoom centre on the actor's crotch.
    SkinnedActor dialogueActor;
    dialogueActor.standingHeightUnits = 180.0f;
    dialogueActor.headHeightUnits = 158.0f;
    dialogueActor.headAnchorBone = 0;
    dialogueActor.poseScratch.push_back(odai::math::Matrix4::identity());
    dialogueActor.headAnchorLocal[1] = 164.0f;
    assert(std::fabs(conversationFaceHeight(dialogueActor) - 164.0f) < 1.0e-4f);
    dialogueActor.headAnchorLocal[1] = 72.0f;
    assert(std::fabs(conversationFaceHeight(dialogueActor) - 158.0f) < 1.0e-4f);
    dialogueActor.headAnchorBone = -1;
    dialogueActor.headHeightUnits = 0.0f;
    dialogueActor.humanoid = true;
    assert(std::fabs(conversationFaceHeight(dialogueActor) - 158.4f) < 1.0e-4f);
    dialogueActor.humanoid = false;
    assert(std::fabs(conversationFaceHeight(dialogueActor) - 117.0f) < 1.0e-4f);

    std::vector<SkinnedActor> controlled{walkingActor(true)};
    updateActorWandering(controlled, 1.0f / 60.0f, nullptr, noGround, noWalls, -1);
    assert(controlled[0].position[0] == 0.0f);
    assert(controlled[0].position[1] == 0.0f);
    assert(controlled[0].position[2] == 0.0f);
    assert(controlled[0].walking);
    assert(std::fabs(controlled[0].runtimeRequestedVelocity.x - 120.0f) < 1.0e-4f);
    assert(controlled[0].runtimeRequestedVelocity.y == 0.0f);
    assert(std::fabs(controlled[0].runtimeRequestedVelocity.z) < 1.0e-4f);

    std::vector<SkinnedActor> legacy{walkingActor(false)};
    updateActorWandering(legacy, 1.0f / 60.0f, nullptr, noGround, noWalls, -1);
    assert(std::fabs(legacy[0].position[0] - 2.0f) < 1.0e-4f);
    assert(legacy[0].runtimeRequestedVelocity.x == 0.0f);

    controlled[0].runtimeControllerBlocked = true;
    controlled[0].scriptedMoveActive = true;
    controlled[0].scriptedMoveRevision = 17u;
    updateActorWandering(controlled, 1.0f / 60.0f, nullptr, noGround, noWalls, -1);
    assert(!controlled[0].walking);
    assert(controlled[0].runtimeRequestedVelocity.x == 0.0f);
    assert(!controlled[0].scriptedMoveActive);
    assert(controlled[0].scriptedMoveRevision == 0u);

    std::vector<SkinnedActor> deathReload{walkingActor(true)};
    deathReload[0].runtimeDead = true;
    updateActorWandering(deathReload, 1.0f / 60.0f, nullptr, noGround, noWalls, -1);
    assert(!deathReload[0].walking &&
           deathReload[0].runtimeRequestedVelocity.x == 0.0f);
    deathReload[0].runtimeDead = false;  // loading an earlier live save
    updateActorWandering(deathReload, 1.0f / 60.0f, nullptr, noGround, noWalls, -1);
    assert(deathReload[0].walking &&
           std::fabs(deathReload[0].runtimeRequestedVelocity.x - 120.0f) < 1.0e-4f);

    odai::importer::fnv::FalloutNavMeshRecord floor;
    floor.formId = 1u;
    floor.vertices = {0.0f, 0.0f, 0.0f, 100.0f, 0.0f, 0.0f,
        0.0f, 100.0f, 0.0f};
    odai::importer::fnv::FalloutNavMeshTriangle triangle;
    triangle.vertex[0] = 0u;
    triangle.vertex[1] = 1u;
    triangle.vertex[2] = 2u;
    floor.triangles.push_back(triangle);
    ActorNavigationWorld navigation;
    navigation.addCell({0, 0}, {floor});
    std::vector<SkinnedActor> doorActors{walkingActor(true)};
    doorActors[0].projectedToNavigation = true;
    doorActors[0].wanderTarget[0] = 0.0f;
    doorActors[0].wanderPath.push_back(ActorNavigationStep{
        ActorNavigationStepKind::ActivateDoor, {0.0f, 0.0f, 0.0f},
        {900.0f, 25.0f, -300.0f}, 0x1234u});
    doorActors[0].wanderPathIndex = 1u;
    updateActorWandering(
        doorActors, 1.0f / 60.0f, &navigation, noGround, noWalls, -1);
    assert(doorActors[0].position[0] == 900.0f);
    assert(doorActors[0].position[1] == 25.0f);
    assert(doorActors[0].position[2] == -300.0f);
    assert(doorActors[0].runtimeControllerNeedsRelocation);

    assert(replayHash({1.0 / 30.0}) == replayHash({1.0 / 60.0, 1.0 / 60.0}));

    std::cout << "new vegas actor movement tests passed\n";
    return 0;
}
