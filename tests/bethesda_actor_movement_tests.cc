#include "games/bethesda/bethesda_actors.h"
#include "bethesda/bethesda_session.h"
#include "anim/hkx_packfile.h"
#include "import/bethesda/skyrim_animation_assets.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace odai::games::bethesda;

namespace {

SkinnedActor walkingActor(bool controllerOwned) {
    SkinnedActor actor;
    actor.name = "MovementFixture";
    actor.placed = true;
    actor.renderVisible = true;
    actor.wanders = true;
    actor.walkSpeedUnitsPerSecond = 120.0f;
    actor.wanderTarget[0] = 1000.0f;
    actor.yawRadians = actorYawForDirection(1.0f, 0.0f);
    actor.runtimeControllerOwned = controllerOwned;
    return actor;
}

std::uint64_t replayHash(const std::vector<double>& frameDeltas) {
    using namespace odai::bethesda;
    BethesdaSession session;
    std::string error;
    assert(session.configure({
        odai::importer::bethesda::BethesdaGame::SkyrimSpecialEdition,
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
    {
        std::vector<SkinnedActor> actors{walkingActor(false)};
        const float destination[3] = {90.0f, 0.0f, 90.0f};
        applyActorScriptedMovement(actors[0], destination, 1u, true);
        const auto noGround = [](float, float, float, float&) { return false; };
        const auto noWalls = [](float&, float&, float, float, float) {};
        for (int tick = 0; tick < 300; ++tick) {
            applyActorScriptedMovement(actors[0], destination, 1u, true);
            updateActorWandering(actors, 1.0f / 60.0f, nullptr, noGround, noWalls, -1);
        }
        assert(actors[0].scriptedMoveArrived);
        assert(!actors[0].scriptedMoveActive && !actors[0].wanders);
        const float arrivedX = actors[0].position[0];
        updateActorWandering(actors, 1.0f, nullptr, noGround, noWalls, -1);
        assert(actors[0].position[0] == arrivedX);
        const float escortDestination[3] = {195.0f, 170.0f, -100.0f};
        applyActorScriptedMovement(actors[0], escortDestination, 2u, true);
        assert(!actors[0].scriptedMoveArrived && actors[0].scriptedMoveActive);
        assert(actors[0].wanderTarget[2] == -100.0f);
        applyActorScriptedMovement(actors[0], escortDestination, 3u, false);
        assert(!actors[0].wanders && !actors[0].scriptedMoveActive);
    }
    const auto noGround = [](float, float, float, float&) { return false; };
    const auto noWalls = [](float&, float&, float, float, float) {};

    // Presentation actors stand in front of the camera and face back toward
    // it. The logical facing must also equal the direction produced when the
    // renderer rotates the model's +X forward axis.
    const float diagonal = std::sqrt(0.5f);
    const float paradeYaw = actorYawForDirection(-diagonal, -diagonal);
    assert(std::fabs(paradeYaw - (3.14159265358979323846f / 4.0f)) < 1.0e-5f);
    const odai::math::Vector3 paradeFacing = actorFacing(paradeYaw);
    const odai::math::Vector3 renderedFacing = odai::math::transformDirection(
        odai::math::Matrix4::rotationY(paradeYaw), {0.0f, 0.0f, -1.0f});
    assert(paradeFacing.x < 0.0f && paradeFacing.z < 0.0f);
    assert(std::fabs(paradeFacing.x - renderedFacing.x) < 1.0e-5f);
    assert(std::fabs(paradeFacing.z - renderedFacing.z) < 1.0e-5f);

    // Pin every cardinal direction, not just a convenient diagonal. A wrong
    // local-forward assumption can pass one heading while becoming backward
    // or sideways after a quarter turn.
    for (const odai::math::Vector3 target : {
             odai::math::Vector3{1.0f, 0.0f, 0.0f},
             odai::math::Vector3{-1.0f, 0.0f, 0.0f},
             odai::math::Vector3{0.0f, 0.0f, 1.0f},
             odai::math::Vector3{0.0f, 0.0f, -1.0f}}) {
        const float yaw = actorYawForDirection(target.x, target.z);
        const odai::math::Vector3 logical = actorFacing(yaw);
        const odai::math::Vector3 rendered = odai::math::transformDirection(
            odai::math::Matrix4::rotationY(yaw), {0.0f, 0.0f, -1.0f});
        assert((logical.x * target.x) + (logical.z * target.z) > 0.9999f);
        assert((rendered.x * target.x) + (rendered.z * target.z) > 0.9999f);
    }

    // Skyrim's child races use the shared humanoid skeleton. Their fallback
    // idle must drive visible torso/head/arm bones, not merely a sub-unit COM
    // translation that makes the rendered child appear frozen.
    odai::anim::Skeleton childRig;
    for (const char* name : {"NPC COM [COM ]", "NPC L Thigh [LThg]",
             "NPC R Thigh [RThg]", "NPC Spine1 [Spn1]", "NPC Spine2 [Spn2]",
             "NPC Head [Head]", "NPC L UpperArm [LUar]", "NPC R UpperArm [RUar]"}) {
        odai::anim::Bone bone;
        bone.name = name;
        bone.parentIndex = childRig.bones.empty() ? -1 : 0;
        bone.localRotation.w = 1.0f;
        childRig.bones.push_back(std::move(bone));
    }
    odai::importer::bethesda::FalloutAssetSource emptyAssets;
    odai::anim::AnimationClip childIdle;
    std::string childIdleWhy;
    assert(loadActorIdleClip(emptyAssets,
        "Actors\\Character\\Character Assets\\skeleton.nif", childRig, 0u,
        childIdle, childIdleWhy));
    assert(childIdle.tracks.size() >= 4u);

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

    // A sharp turn accelerates continuously instead of jumping from a fixed
    // shuffle speed to a full stride on one threshold-crossing frame.
    std::vector<SkinnedActor> turning{walkingActor(true)};
    turning[0].yawRadians = actorYawForDirection(-1.0f, 0.0f);
    float previousSpeed = 0.0f;
    float maximumSpeedStep = 0.0f;
    for (int tick = 0; tick < 100; ++tick) {
        updateActorWandering(
            turning, 1.0f / 60.0f, nullptr, noGround, noWalls, -1);
        const float speed = std::sqrt(
            (turning[0].runtimeRequestedVelocity.x *
                turning[0].runtimeRequestedVelocity.x) +
            (turning[0].runtimeRequestedVelocity.z *
                turning[0].runtimeRequestedVelocity.z));
        assert(speed + 1.0e-4f >= previousSpeed);
        maximumSpeedStep = std::max(maximumSpeedStep, speed - previousSpeed);
        previousSpeed = speed;
    }
    assert(previousSpeed > 119.0f);
    assert(maximumSpeedStep < 12.0f);

    // Cross-fade from the exact displayed idle pose into walk. A one-bone
    // fixture makes the discontinuity measurable without renderer involvement.
    SkinnedActor blended;
    blended.placed = true;
    blended.renderVisible = true;
    odai::anim::Bone blendBone;
    blendBone.name = "root";
    blendBone.parentIndex = -1;
    blendBone.localRotation.w = 1.0f;
    blendBone.localScale = {1.0f, 1.0f, 1.0f};
    blended.character.skeleton.bones.push_back(blendBone);
    blended.character.inverseBindMatrices.push_back(odai::math::Matrix4::identity());
    blended.sampler.bindSkeleton(
        blended.character.skeleton, blended.character.inverseBindMatrices);
    odai::anim::BoneTrack idleTrack;
    idleTrack.boneIndex = 0;
    idleTrack.translationKeys.push_back({0.0f, {0.0f, 0.0f, 0.0f}});
    blended.idleClip.duration = 1.0f;
    blended.idleClip.tracks.push_back(idleTrack);
    odai::anim::BoneTrack walkTrack;
    walkTrack.boneIndex = 0;
    walkTrack.translationKeys.push_back({0.0f, {10.0f, 0.0f, 0.0f}});
    blended.walkClip.duration = 1.0f;
    blended.walkClip.tracks.push_back(walkTrack);
    std::vector<SkinnedActor> blendedActors{std::move(blended)};
    updateActorPoses(blendedActors, 1.0f / 60.0f);
    assert(std::fabs(blendedActors[0].poseScratch[0](0, 3)) < 1.0e-4f);
    blendedActors[0].walking = true;
    updateActorPoses(blendedActors, 1.0f / 60.0f);
    const float firstWalkPose = blendedActors[0].poseScratch[0](0, 3);
    assert(firstWalkPose > 0.0f && firstWalkPose < 1.0f);
    for (int frame = 0; frame < 12; ++frame) {
        updateActorPoses(blendedActors, 1.0f / 60.0f);
    }
    assert(blendedActors[0].poseScratch[0](0, 3) > 9.9f);

    std::vector<SkinnedActor> deathReload{walkingActor(true)};
    deathReload[0].runtimeDead = true;
    updateActorWandering(deathReload, 1.0f / 60.0f, nullptr, noGround, noWalls, -1);
    assert(!deathReload[0].walking &&
           deathReload[0].runtimeRequestedVelocity.x == 0.0f);
    deathReload[0].runtimeDead = false;  // loading an earlier live save
    updateActorWandering(deathReload, 1.0f / 60.0f, nullptr, noGround, noWalls, -1);
    assert(deathReload[0].walking &&
           std::fabs(deathReload[0].runtimeRequestedVelocity.x - 120.0f) < 1.0e-4f);

    // TES3 has generated collision-derived navigation but no authored NAVM
    // meshes. It must still project actors and constrain their movement.
    odai::importer::ImportedScene generatedScene;
    odai::importer::ImportedSceneCollisionTriangle generatedFloorA;
    const float floorA[9] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 128.0f, 128.0f, 0.0f, 0.0f};
    std::copy_n(floorA, 9u, generatedFloorA.vertices);
    odai::importer::ImportedSceneCollisionTriangle generatedFloorB;
    const float floorB[9] = {
        128.0f, 0.0f, 0.0f, 0.0f, 0.0f, 128.0f, 128.0f, 0.0f, 128.0f};
    std::copy_n(floorB, 9u, generatedFloorB.vertices);
    generatedScene.collisionTriangles = {generatedFloorA, generatedFloorB};
    ActorNavigationWorld generatedNavigation;
    generatedNavigation.addGeneratedCell({0, 0}, generatedScene);
    assert(generatedNavigation.meshCount() == 0u);
    assert(generatedNavigation.generatedNodeCount() != 0u);
    assert(generatedNavigation.hasNavigation());
    std::vector<SkinnedActor> tes3Actors{walkingActor(true)};
    tes3Actors[0].position[0] = -10.0f;
    tes3Actors[0].position[1] = 80.0f;
    tes3Actors[0].position[2] = -10.0f;
    updateActorWandering(
        tes3Actors, 1.0f / 60.0f, &generatedNavigation, noGround, noWalls, -1);
    assert(tes3Actors[0].projectedToNavigation);
    assert(std::fabs(tes3Actors[0].position[0] - 32.0f) < 1.0e-4f);
    assert(std::fabs(tes3Actors[0].position[1]) < 1.0e-4f);
    assert(std::fabs(tes3Actors[0].position[2] - 32.0f) < 1.0e-4f);
    assert(tes3Actors[0].runtimeControllerNeedsRelocation);

    odai::importer::bethesda::FalloutNavMeshRecord floor;
    floor.formId = 1u;
    floor.vertices = {0.0f, 0.0f, 0.0f, 100.0f, 0.0f, 0.0f,
        0.0f, 100.0f, 0.0f};
    odai::importer::bethesda::FalloutNavMeshTriangle triangle;
    triangle.vertex[0] = 0u;
    triangle.vertex[1] = 1u;
    triangle.vertex[2] = 2u;
    floor.triangles.push_back(triangle);
    ActorNavigationWorld navigation;
    navigation.addCell({0, 0}, {floor});
    auto followFloor = floor;
    for (auto& coordinate : followFloor.vertices) coordinate *= 100.f;
    ActorNavigationWorld followNavigation;
    followNavigation.addCell({0, 0}, {followFloor});
    auto follower = walkingActor(true);
    follower.followsPlayer = true;
    follower.renderVisible = false; // followers must move behind the camera
    follower.projectedToNavigation = true;
    updateActorFollowTarget(follower, &followNavigation, {800.f, 0.f, -20.f}, true, 1.f / 60.f);
    assert(follower.scriptedMoveActive && follower.wanders && !follower.wanderPath.empty());
    assert(follower.followSpeedMultiplier == 2.5f);
    updateActorFollowTarget(follower, &followNavigation, {590.f, 0.f, -20.f}, true, 1.f / 60.f);
    assert(follower.followSpeedMultiplier == 2.5f);
    updateActorFollowTarget(follower, &followNavigation, {440.f, 0.f, -20.f}, true, 1.f / 60.f);
    assert(follower.followSpeedMultiplier == 1.f);
    updateActorFollowTarget(follower, &followNavigation, {610.f, 0.f, -20.f}, true, 1.f / 60.f);
    assert(follower.followSpeedMultiplier == 1.f);

    std::vector<SkinnedActor> followers{follower};
    updateActorWandering(followers, 1.f / 60.f, &followNavigation, noGround, noWalls, -1);
    followers[0].wanderPath = {
        {ActorNavigationStepKind::Walk, {0.f, 0.f, 0.f}, {}, 0},
        {ActorNavigationStepKind::Walk, {800.f, 0.f, -20.f}, {}, 0}};
    followers[0].wanderPathIndex = 1;
    followers[0].wanderTarget[0] = followers[0].wanderTarget[1] = followers[0].wanderTarget[2] = 0.f;
    updateActorWandering(followers, 1.f / 60.f, &followNavigation, noGround, noWalls, -1);
    assert(followers[0].wanderPathIndex == 2);
    assert(followers[0].walking);
    assert(odai::math::length(followers[0].runtimeRequestedVelocity) > 0.f);
    updateActorFollowTarget(follower, &followNavigation, {100.f, 0.f, 0.f}, true, .5f);
    assert(!follower.wanders && !follower.scriptedMoveActive && follower.wanderPath.empty());
    updateActorFollowTarget(follower, &followNavigation, {200.f, 0.f, 0.f}, true, .5f);
    assert(!follower.wanders); // stopping-distance hysteresis
    updateActorFollowTarget(follower, &followNavigation, {300.f, 0.f, -20.f}, true, .5f);
    assert(follower.wanders && follower.followSpeedMultiplier == 1.f);
    updateActorFollowTarget(follower, nullptr, {800.f, 0.f, 0.f}, true, .5f);
    assert(!follower.wanders && follower.wanderPath.empty());
    follower.runtimeDead = true;
    updateActorFollowTarget(follower, &followNavigation, {800.f, 0.f, -20.f}, true, .5f);
    assert(!follower.wanders);

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

    // Optional installed-data fixture: no retail bytes enter the repository,
    // but a developer with Skyrim installed proves the Player/Nord/OTFT and
    // ARMO -> ARMA assembly path against the actual virtual Data source.
    if (const char* skyrimData = std::getenv("ODAI_SKYRIM_DATA")) {
        SkinnedActor avatar;
        std::string detail;
        assert(loadSkyrimPlayerAvatar(
            std::filesystem::path(skyrimData),
            "ArmorIronBandedNoHelmetOutfit", 47u, avatar, detail));
        std::cerr << "retail avatar: " << detail << ", equipped="
                  << avatar.inventoryFormIds.size() << "\n";
        assert(avatar.name == "SkyrimPlayerAvatar");
        assert(avatar.instanceSlot == 47u);
        assert(!avatar.runtimeObjectId.valid() && avatar.referenceFormId == 0u);
        assert(avatar.inventoryFormIds.size() >= 3u);
        assert(avatar.character.parts.size() >= 3u);
        for (const std::string_view requiredFace : {
                 "actors\\character\\character assets\\malehead.nif",
                 "actors\\character\\character assets\\eyesmale.nif",
                 "actors\\character\\character assets\\mouth\\mouthhuman.nif",
                 "actors\\character\\character assets\\faceparts\\malebrows.nif",
                 "actors\\character\\character assets\\hair\\male\\hairline01.nif",
                 "actors\\character\\character assets\\hair\\male\\hair01.nif"}) {
            assert(std::any_of(
                avatar.character.parts.begin(), avatar.character.parts.end(),
                [&](const odai::importer::bethesda::FalloutCharacterPart& part) {
                    return part.sourcePath == requiredFace;
                }));
        }
        assert(!avatar.idleClip.tracks.empty() && !avatar.walkClip.tracks.empty());
        assert(!avatar.idleClip.name.starts_with("procedural"));
        assert(!avatar.walkClip.name.starts_with("procedural"));
        assert(avatar.authoredLocomotionClips.size() == 7u);
        odai::importer::bethesda::FalloutAssetSource retailAssets;
        assert(retailAssets.open(skyrimData));
        std::string graphFingerprint, graphError;
        auto graphProgram = odai::importer::bethesda::loadSkyrimBehaviorProgram(retailAssets,
            "meshes\\actors\\character\\behaviors\\0_master.hkx", graphFingerprint, graphError);
        assert(graphProgram && !graphProgram->graph.eventNames.empty());
        for (const bool female : {false, true}) {
            auto view = odai::importer::bethesda::loadSkyrimNpcAnimationView(retailAssets,
                avatar.character.skeleton, avatar.character.inverseBindMatrices, female,
                avatar.idleClip, avatar.walkClip, graphProgram, graphFingerprint);
            std::cerr << "retail NPC " << (female ? "female" : "male") << ": " << view->clips.size()
                      << " clips, graph executable=" << (view->behavior && view->behavior->executable()) << "\n";
            assert(view->stateClips.contains("run_forward"));
            assert(view->stateClips.contains("walk_left"));
            assert(view->stateClips.contains("sneak"));
            assert(view->stateClips.contains("swim"));
            assert(view->stateClips.contains("attack_1hm"));
            odai::anim::BehaviorGraphInstance runtime;
            assert(runtime.bind(*view, graphError));
            odai::anim::AnimationInputState input;
            input.attacking = true; input.weaponStyle = "1hm";
            const auto attackPose = runtime.step(input, 1.0f / 60.0f);
            assert(attackPose.activeState == "attack_1hm" && attackPose.actionActive);
        }
        std::vector<std::uint8_t> retailHkx;
        std::string decodeError;
        std::vector<std::uint8_t> retailMasterGraph;
        assert(retailAssets.resolveAsset(
            "meshes\\actors\\character\\behaviors\\0_master.hkx",
            retailMasterGraph, decodeError));
        odai::anim::HkxDecodedBehaviorGraph behaviorGraph;
        assert(odai::anim::decodeHkxBehaviorGraph(
            retailMasterGraph, behaviorGraph, decodeError));
        assert(behaviorGraph.name == "0_Master.hkb");
        assert(behaviorGraph.nodes.size() > 1000u);
        assert(behaviorGraph.clipGeneratorCount > 250u);
        assert(behaviorGraph.behaviorReferenceCount > 10u);
        assert(behaviorGraph.stateMachineCount > 100u);
        assert(behaviorGraph.transitionEffectCount > 30u);
        assert(retailAssets.resolveAsset(
            "meshes\\actors\\character\\animations\\male\\mt_walkforward.hkx",
            retailHkx, decodeError));
        odai::anim::AnimationClip decoded;
        odai::anim::HkxDecodedClipMetadata metadata;
        std::vector<std::uint8_t> retailSkeletonHkx;
        assert(retailAssets.resolveAsset(
            "meshes\\actors\\character\\character assets\\skeleton.hkx",
            retailSkeletonHkx, decodeError));
        odai::anim::HkxDecodedSkeleton sourceSkeleton;
        assert(odai::anim::decodeHkxAnimationSkeleton(
            retailSkeletonHkx, sourceSkeleton, decodeError));
        assert(sourceSkeleton.boneNames.size() == 99u);
        assert(std::count(sourceSkeleton.translationLocked.begin(),
            sourceSkeleton.translationLocked.end(), true) == 94);
        if (!odai::anim::decodeHkxAnimationClip(retailHkx, avatar.character.skeleton,
                "retail walk", decoded, metadata, decodeError, &sourceSkeleton)) {
            std::cerr << "retail HKX decode failed: " << decodeError << "\n";
        }
        assert(!decoded.tracks.empty());
        assert(metadata.frameCount > 2u && metadata.transformTrackCount == 99u);
        assert(metadata.boundTracks >= 90u && metadata.missingTracks <= 9u);
        for (const odai::anim::BoneTrack& track : decoded.tracks) {
            assert(track.translationKeys.size() == metadata.frameCount);
            assert(track.rotationKeys.size() == metadata.frameCount);
            assert(track.scaleKeys.size() == metadata.frameCount);
        }
    }

    std::cout << "bethesda actor movement tests passed\n";
    return 0;
}
