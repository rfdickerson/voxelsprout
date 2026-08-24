#include "anim/hkx_packfile.h"
#include "anim/skyrim_animation.h"
#include "bethesda/bethesda_physics_world.h"
#include "bethesda/bethesda_session.h"
#include "bethesda/runtime_ids.h"
#include "bethesda/save_game.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <Jolt/Jolt.h>

namespace {

void writeU32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

std::vector<std::uint8_t> syntheticPackfile() {
    constexpr std::size_t dataStart = 112u;
    const std::string classes = std::string("hkaSkeleton\0hkbBehaviorGraph\0", 29u);
    std::vector<std::uint8_t> bytes(dataStart + classes.size(), 0u);
    const std::uint8_t magic[] = {0x57u, 0xe0u, 0xe0u, 0x57u, 0x10u, 0xc0u, 0xc0u, 0x10u};
    std::memcpy(bytes.data(), magic, sizeof(magic));
    writeU32(bytes, 12u, 8u);
    bytes[16] = 8u;
    bytes[17] = 1u;
    writeU32(bytes, 20u, 1u);
    const char version[] = "hk_2010.2.0-r1";
    std::memcpy(bytes.data() + 40u, version, sizeof(version));
    const char section[] = "__classnames__";
    std::memcpy(bytes.data() + 64u, section, sizeof(section));
    bytes[83] = 0xffu;
    for (std::size_t field = 0; field < 7u; ++field) {
        writeU32(bytes, 84u + field * 4u,
            field == 0u ? static_cast<std::uint32_t>(dataStart) :
                static_cast<std::uint32_t>(bytes.size() - dataStart));
    }
    std::memcpy(bytes.data() + dataStart, classes.data(), classes.size());
    return bytes;
}

void testHkxInspection() {
    auto bytes = syntheticPackfile();
    odai::anim::HkxPackfileSummary summary;
    std::string error;
    assert(odai::anim::inspectHkxPackfile(bytes, summary, error));
    assert(summary.pointerSize == 8u && summary.littleEndian);
    assert(summary.containsSkeleton && summary.containsBehaviorGraph);
    bytes[111] = 0xffu;
    assert(!odai::anim::inspectHkxPackfile(bytes, summary, error));
}

odai::anim::Skeleton makeRig() {
    odai::anim::Skeleton rig;
    rig.bones.push_back({"NPC Root [Root]", -1});
    rig.bones.push_back({"WeaponSword", 0, {0.0f, 0.0f, 10.0f}});
    rig.bones.push_back({"QUIVER", 0, {0.0f, -5.0f, 20.0f}});
    return rig;
}

void testRigBindingAndGraphSnapshot() {
    const odai::anim::Skeleton rig = makeRig();
    const std::vector<std::string> names{"NPC Root [Root]", "weaponsword", "missing"};
    const auto binding = odai::anim::bindTracksByName(names, rig);
    assert(binding.exactMatches == 1u && binding.caseInsensitiveMatches == 1u);
    assert(binding.missingTracks.size() == 1u && binding.coverage() > 0.66f);

    odai::anim::AnimationClip idle;
    idle.name = "idle";
    idle.duration = 1.0f;
    odai::anim::AnimationClip walk = idle;
    walk.name = "walk";
    odai::anim::BoneTrack root;
    root.boneIndex = 0;
    root.translationKeys = {{0.0f, {}}, {1.0f, {100.0f, 0.0f, 0.0f}}};
    walk.tracks.push_back(root);
    odai::anim::AnimationView view;
    view.skeleton = &rig;
    view.clips = {idle, walk};
    view.stateClips = {{"idle", "idle"}, {"locomotion", "walk"}};
    view.socketBoneNames = {"WeaponSword", "QUIVER"};
    view.supportedBehaviorGraph = true;

    odai::anim::BehaviorGraphInstance first;
    std::string error;
    assert(first.bind(view, error));
    odai::anim::AnimationInputState input;
    input.movementSpeed = 120.0f;
    input.animationDriven = true;
    auto output = first.step(input, 1.0f / 60.0f);
    assert(output.activeState == "locomotion" && output.desiredRootMotion.x > 1.0f);
    assert(output.socketTransforms.size() == 2u);
    const auto saved = first.snapshot();
    output = first.step(input, 1.0f / 60.0f);

    odai::anim::BehaviorGraphInstance restored;
    assert(restored.bind(view, error));
    assert(restored.restore(saved, error));
    const auto replay = restored.step(input, 1.0f / 60.0f);
    assert(replay.activeState == output.activeState);
    assert(std::fabs(replay.desiredRootMotion.x - output.desiredRootMotion.x) < 1.0e-4f);
}

void testJoltCharacterGroundingAndSnapshot() {
    using namespace odai::bethesda;
    BethesdaPhysicsWorld world;
    std::string error;
    assert(world.initialize(error));
    const std::vector<odai::math::Vector3> vertices{
        {-500.0f, 0.0f, -500.0f}, {500.0f, 0.0f, -500.0f},
        {500.0f, 0.0f, 500.0f}, {-500.0f, 0.0f, 500.0f}};
    const std::vector<std::uint32_t> indices{0u, 1u, 2u, 0u, 2u, 3u};
    assert(world.addStreamedStaticCollision(17u, vertices, indices, error));
    const auto floorHit = world.castDown({0.0f, 100.0f, 0.0f}, 200.0f);
    assert(floorHit.has_value());
    assert(std::fabs(floorHit->position.y) < 1.0e-3f);
    assert(std::fabs(floorHit->distance - 100.0f) < 1.0e-3f);
    assert(floorHit->normal.y > 0.99f);
    const auto footHit = world.castDown({0.0f, 40.0f, 0.0f}, 80.0f);
    assert(footHit.has_value() && std::fabs(footHit->position.y) < 1.0e-3f);
    assert(!footHit->object.has_value());
    const ObjectId actor = ObjectId::runtime(7u);
    PhysicsCharacterConfig config;
    config.position = {0.0f, 100.0f, 0.0f};
    assert(world.addCharacter(actor, config, error));
    for (int tick = 0; tick < 180; ++tick) world.step(1.0f / 60.0f);
    const auto state = world.characterState(actor);
    assert(state.has_value());
    assert(state->grounded);
    assert(state->position.y > -1.0f && state->position.y < 1.0f);
    const auto saved = world.snapshot();
    assert(saved.size() == 1u && saved.front().object == actor);
    PhysicsCharacterInput input;
    input.desiredVelocity = {200.0f, 0.0f, 0.0f};
    assert(world.setCharacterInput(actor, input));
    world.step(1.0f / 60.0f);
    assert(world.restore(saved, error));
    assert(std::fabs(world.characterState(actor)->position.x - saved.front().position.x) < 1.0e-4f);
    assert(world.removeStreamedStaticCollision(17u));
    for (int tick = 0; tick < 60; ++tick) world.step(1.0f / 60.0f);
    assert(!world.characterState(actor)->grounded);
    assert(world.characterState(actor)->position.y < 40.0f);
}

void testJoltOverlappingRetailCollisionWarningIsRecoverable() {
    using namespace odai::bethesda;
    BethesdaPhysicsWorld world;
    std::string error;
    assert(world.initialize(error));

    // The host callback itself is the regression boundary: Jolt's debug-build
    // default intentionally breaks here, while ODAI must treat Trace as a
    // recoverable compatibility diagnostic.
    JPH::Trace("ODAI recoverable trace fixture");

    // More coincident triangles than Jolt permits in one leaf force its AABB
    // builder down the documented random-split warning path. That warning must
    // remain diagnostic: the default Jolt callback traps if the host forgets
    // to install one, which used to crash exterior streaming on retail meshes.
    const std::vector<odai::math::Vector3> vertices{
        {-100.0f, 0.0f, -100.0f}, {100.0f, 0.0f, -100.0f},
        {0.0f, 0.0f, 100.0f}};
    std::vector<std::uint32_t> indices;
    for (int triangle = 0; triangle < 16; ++triangle) {
        indices.insert(indices.end(), {0u, 1u, 2u});
    }
    assert(world.addStreamedStaticCollision(23u, vertices, indices, error));
    const auto hit = world.castDown({0.0f, 100.0f, 0.0f}, 200.0f);
    assert(hit.has_value());
    assert(std::fabs(hit->position.y) < 1.0e-3f);
}

void testSessionFixedTickAndSaveContinuation() {
    using namespace odai::bethesda;
    const auto rig = std::make_shared<odai::anim::Skeleton>(makeRig());
    auto third = std::make_shared<odai::anim::AnimationView>();
    third->skeleton = rig.get();
    odai::anim::AnimationClip idle;
    idle.name = "idle";
    idle.duration = 1.0f;
    third->clips.push_back(idle);
    third->supportedBehaviorGraph = true;
    auto first = std::make_shared<odai::anim::AnimationView>(*third);

    BethesdaSessionConfig config;
    config.game = odai::importer::fnv::BethesdaGame::SkyrimSpecialEdition;
    config.contentFingerprint = "animation-save-fixture";
    std::string error;
    BethesdaSession session;
    assert(session.configure(config, error));
    RuntimeObject actor;
    actor.id = ObjectId::runtime(99u);
    actor.base = makeRecordKey("skyrim.esm", 0x7u);
    actor.kind = RuntimeObjectKind::Actor;
    actor.actorValues = ActorValues{};
    actor.transform.position = {0.0, 0.0, 120.0};
    assert(session.world().addInitialObject(actor, error));
    PhysicsCharacterConfig physical;
    physical.position = {0.0f, 0.0f, 120.0f};
    assert(session.registerActorAnimation(actor.id, third, first, physical, error));
    odai::anim::AnimationInputState input;
    input.weaponDrawn = true;
    assert(session.setActorAnimationInput(actor.id, input));
    const auto advanced = session.advance(1.0 / 30.0);
    assert(advanced.clock.steps == 2u);
    const auto snapshots = session.animationSnapshots();
    assert(snapshots.size() == 1u && snapshots.front().firstPerson.has_value());
    assert(snapshots.front().thirdPerson.fixedTick == 2u &&
        snapshots.front().firstPerson->fixedTick == 2u);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "odai-animation-save-v2.odai";
    assert(saveOdaiGameAtomic(path, session, error));
    const std::uint64_t expectedHash = session.deterministicHash();
    BethesdaSession restored;
    assert(restored.configure(config, error));
    assert(restored.world().addInitialObject(actor, error));
    assert(restored.registerActorAnimation(actor.id, third, first, physical, error));
    SaveLoadReport report;
    assert(loadOdaiGame(path, restored, {}, report, error));
    assert(restored.deterministicHash() == expectedHash);
    std::error_code removeError;
    std::filesystem::remove(path, removeError);
}

}  // namespace

int main() {
    testHkxInspection();
    testRigBindingAndGraphSnapshot();
    testJoltCharacterGroundingAndSnapshot();
    testJoltOverlappingRetailCollisionWarningIsRecoverable();
    testSessionFixedTickAndSaveContinuation();
    std::cout << "Skyrim animation/Jolt tests passed\n";
    return 0;
}
