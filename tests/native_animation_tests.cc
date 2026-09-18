#include "anim/native_program.h"
#include "bethesda/character_movement.h"
#include "anim/skyrim_animation.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <chrono>
#include "import/fnv/skyrim_animation_assets.h"

using namespace odai::anim;

int main(int argc, char** argv) {
    const std::string json = R"({"version":1,"rules":[
      {"id":"walk","state":"locomotion","priority":50,"blend_time":0.2,
       "conditions":{"sex":"female","fatigue":{"min":0.3},"tags":["guard"]},
       "variants":[{"clip":"a","weight":1},{"clip":"b","weight":3}]}
    ]})";
    NativeAnimationProgram program;
    std::string error;
    assert(compileNativeAnimationPack(json, "pack", 2, program, error));
    AnimationSelectorContext context;
    context.values = {{"sex", std::string("female")}, {"fatigue", 0.5}};
    assert(!program.select("locomotion", context));
    context.tags.insert("guard");
    assert(program.select("locomotion", context));
    context.values["fatigue"] = 0.1;
    assert(!program.select("locomotion", context));
    context.values["fatigue"] = 0.5;
    auto replacement = program.rules.front();
    replacement.id = "later"; replacement.layer = 3;
    program.rules.push_back(replacement);
    assert(program.select("locomotion", context)->id == "later");
    replacement.id = "alpha";
    program.rules.push_back(replacement);
    assert(program.select("locomotion", context)->id == "alpha");
    program.rules.front().priority = 51;
    assert(program.select("locomotion", context)->id == "walk");
    const auto count = program.rules.size();
    for (const auto* malformed : {"{}", R"({"version":3,"rules":[]})",
        R"({"version":1,"rules":[{"id":"x","state":"idle","variants":[]}]})",
        R"({"version":1,"rules":[{"id":"x","state":"idle","variants":[{"clip":"a","weight":0}]}]})"}) {
        assert(!compileNativeAnimationPack(malformed, "bad", 0, program, error));
        assert(program.rules.size() == count);
    }
    std::uint64_t first = 123, second = 123;
    bool sawA = false, sawB = false;
    for (int i = 0; i < 100; ++i) {
        const auto chosen = chooseNativeVariant(program.rules.front(), first);
        assert(chosen == chooseNativeVariant(program.rules.front(), second));
        sawA |= chosen == 0; sawB |= chosen == 1;
    }
    assert(sawA && sawB);

    AnimationView view;
    {
        AnimationView jumpView;
        auto jumpSkeleton = std::make_shared<Skeleton>();
        jumpSkeleton->bones.push_back({"root", -1});
        jumpView.skeleton = jumpSkeleton;
        AnimationClip takeoff; takeoff.name = "dova-jump"; takeoff.duration = 1; takeoff.loop = false;
        AnimationClip landing = takeoff; landing.name = "landing";
        jumpView.clips = {takeoff, landing};
        jumpView.stateClips = {{"landing", "landing"}};
        auto jumpProgram = std::make_shared<NativeAnimationProgram>();
        assert(compileNativeAnimationPack(R"({"version":1,"rules":[
            {"id":"takeoff","state":"jump","hold_until_landing":true,
             "conditions":{"speed":{"min":1}},
             "variants":[{"clip":"dova-jump","loop":false}]}]})",
            "dova", 1, *jumpProgram, error));
        jumpView.nativeProgram = jumpProgram;
        BehaviorGraphInstance jumping;
        assert(jumping.bind(jumpView, error));
        AnimationInputState airborne;
        airborne.grounded = false; airborne.verticalVelocity = 200; airborne.movementSpeed = 230;
        assert(jumping.step(airborne, .1f).activeClip == "dova-jump");
        const auto action = jumping.snapshot();
        airborne.verticalVelocity = -200; airborne.movementSpeed = 0;
        assert(jumping.step(airborne, .1f).activeClip == "dova-jump");
        assert(jumping.snapshot().stateTime > action.stateTime);
        assert(jumping.snapshot().randomState == action.randomState);
        airborne.grounded = true; airborne.landed = true;
        assert(jumping.step(airborne, .1f).activeClip == "landing");
    }
    auto rig = std::make_shared<Skeleton>();
    rig->bones.push_back({"root", -1});
    view.skeleton = rig;
    for (const auto* name : {"idle", "walk", "a", "b"}) {
        AnimationClip clip; clip.name = name; clip.duration = 1; clip.loop = true;
        view.clips.push_back(clip);
    }
    view.stateClips = {{"idle", "idle"}, {"locomotion", "walk"}};
    view.nativeProgram = std::make_shared<NativeAnimationProgram>(program);
    view.sourceFingerprint = "native-fixture";
    BehaviorGraphInstance instance, replay;
    assert(instance.bind(view, error));
    AnimationInputState input; input.movementSpeed = 20; input.selectorContext = context;
    const auto started = instance.step(input, 1.f / 60);
    assert(!started.proceduralFallback && !started.authoredGraphExecuted);
    assert(started.activeRule == "walk" && started.activeProvider == "pack");
    const auto saved = instance.snapshot();
    assert(instance.step(input, 1.f / 60).activeClip == started.activeClip);
    assert(instance.snapshot().randomState == saved.randomState);
    assert(replay.bind(view, error) && replay.restore(saved, error));
    const auto continued = replay.step(input, 1.f / 60);
    assert(continued.activeClip == started.activeClip);
    assert(replay.snapshot() == instance.snapshot());
    input.selectorContext.tags.clear();
    const auto builtin = instance.step(input, 1.f / 60);
    assert(builtin.activeClip == "walk" && builtin.activeRule.empty());
    assert(instance.snapshot().previousClip == started.activeClip);
    auto missing = std::make_shared<NativeAnimationProgram>(program);
    for (auto& rule : missing->rules) for (auto& variant : rule.variants) variant.clip = "missing";
    view.nativeProgram = missing;
    assert(instance.bind(view, error));
    input.selectorContext = context;
    const auto fallback = instance.step(input, 1.f / 60);
    assert(fallback.activeClip == "walk" && !fallback.fallbackReason.empty());
    {
        const auto translated = [](const char* name, float x) {
            AnimationClip result; result.name = name; result.duration = 1;
            BoneTrack track; track.boneIndex = 0; track.translationKeys = {{0, {x,0,0}}};
            result.tracks.push_back(track); return result;
        };
        const auto zero = translated("zero", 0), ten = translated("ten", 10), reference = translated("reference", 4);
        AnimationSampler sampler; sampler.bindSkeleton(*rig);
        std::vector<odai::math::Matrix4> pose;
        const WeightedAnimationPose samples[]{{&zero, 0, .5f}, {&ten, 0, .5f}};
        const AnimationPoseLayer layers[]{{&ten, &reference, 0, .5f, true, {1}}};
        sampler.sampleComposed(*rig, samples, layers, pose);
        assert(std::abs(odai::math::transformPoint(pose[0], {}).x - 8.f) < .0001f);
        const AnimationPoseLayer masked[]{{&ten, nullptr, 0, 1, false, {0}}};
        sampler.sampleComposed(*rig, samples, masked, pose);
        assert(std::abs(odai::math::transformPoint(pose[0], {}).x - 5.f) < .0001f);
        assert(!compileNativeAnimationPack(R"({"version":1,"rules":[{"id":"x","state":"idle",
            "variants":[{"clip":"a"}],"layers":[{"id":"x","clip":"a","additive":true,"bones":["root"]}]}]})",
            "bad", 0, program, error));
    }
    // A short first-person action cannot finish the shared gameplay action.
    AnimationView authority = view, presentation = view;
    authority.nativeProgram.reset(); presentation.nativeProgram.reset();
    AnimationClip attack; attack.name = "attack"; attack.duration = .8f; attack.loop = false;
    attack.annotations = {{.1f, "HitFrame"}};
    authority.clips.push_back(attack);
    attack.duration = .2f;
    presentation.clips.push_back(attack);
    BehaviorGraphInstance primary, secondary;
    assert(primary.bind(authority, error) && secondary.bind(presentation, error));
    AnimationInputState action; action.attacking = true;
    for (int tick = 0; tick < 5; ++tick) {
        const auto main = primary.step(action, .1f);
        const auto clock = primary.snapshot();
        auto follower = action;
        follower.sharedState = clock.state;
        follower.sharedStateTime = clock.stateTime;
        follower.sharedActionIdentity = clock.actionIdentity;
        follower.ownsGameplayEvents = false;
        const auto shown = secondary.step(follower, .1f);
        assert(shown.activeState == main.activeState && shown.clipEvents.empty());
        assert(secondary.snapshot().actionIdentity == clock.actionIdentity);
        assert(std::abs(secondary.snapshot().stateTime - clock.stateTime) < .0001f);
        action.attacking = false;
    }
    {
        using namespace odai::bethesda;
        CharacterMovementSettings settings; settings.enabled = true;
        CharacterMovementState movement;
        float x = 4, z = 0;
        assert(!advanceCharacterMovement(movement, settings, true, false, false, .016f, x, z));
        assert(advanceCharacterMovement(movement, settings, false, false, true, .016f, x, z)); // coyote
        assert(!advanceCharacterMovement(movement, settings, false, false, true, .016f, x, z));
        assert(!advanceCharacterMovement(movement, settings, true, true, true, .016f, x, z)); // held on landing
        movement = {};
        assert(!advanceCharacterMovement(movement, settings, false, false, true, .016f, x, z));
        const auto buffered = movement;
        assert(advanceCharacterMovement(movement, settings, true, true, false, .016f, x, z));
        auto restoredMovement = buffered;
        assert(advanceCharacterMovement(restoredMovement, settings, true, true, false, .016f, x, z));
        assert(restoredMovement == movement);
        movement = {}; x = 6; z = 0;
        advanceCharacterMovement(movement, settings, true, false, false, .016f, x, z);
        x = -6;
        advanceCharacterMovement(movement, settings, false, false, false, .016f, x, z);
        assert(x > 5.8f); // sprint momentum cannot reverse instantly in air
        assert(classifyLanding(1.9f, settings) == LandingSeverity::Light);
        assert(classifyLanding(2.f, settings) == LandingSeverity::Hard);
        assert(classifyLanding(5.f, settings) == LandingSeverity::Stagger);
        assert(classifyLanding(8.f, settings) == LandingSeverity::Severe);
        settings.bufferSeconds = -1;
        assert(!validCharacterMovementSettings(settings));
    }
    {
        NativeAnimationProgram sprintProgram;
        std::string sprintError;
        assert(compileNativeAnimationPack(R"({"version":1,"rules":[
            {"id":"sprint-jump","state":"jump","priority":2,
             "conditions":{"sprinting":true},
             "variants":[{"clip":"sprint_jump","loop":false}]},
            {"id":"jump","state":"jump","priority":1,
             "variants":[{"clip":"jump","loop":false}]}
        ]})", "test", 1, sprintProgram, sprintError));
        AnimationSelectorContext sprintContext;
        sprintContext.values["sprinting"] = true;
        assert(sprintProgram.select("jump", sprintContext)->id == "sprint-jump");
    }
    // Packs from distinct virtual paths retain profile precedence; shared paths
    // resolve only the winning bytes through the same asset source as meshes.
    const auto root = std::filesystem::temp_directory_path() / ("odai-native-packs-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto low = root / "low", high = root / "high";
    std::filesystem::create_directories(low / "odai/animations");
    std::filesystem::create_directories(high / "odai/animations");
    std::ofstream(low / "odai/animations/a.json") << json;
    std::ofstream(high / "odai/animations/b.json") << json;
    std::ofstream(high / "odai/animations/broken.json") << "{}";
    odai::importer::fnv::FalloutAssetSource assets;
    assert(assets.addModDirectory(low) && assets.addModDirectory(high));
    odai::importer::fnv::loadSkyrimNativeAnimationPacks(assets, view);
    assert(view.nativeProgram->rules.size() == 2);
    const auto* winning = view.nativeProgram->select("locomotion", context);
    assert(winning && winning->id.ends_with("b.json#walk"));
    assert(!view.diagnostics.empty());
    assert(view.sourceFingerprint.find("a.json@") != std::string::npos);
    // Native replacement jumps must not add their authored pelvis lift to the
    // capsule's jump arc; landing compression remains authored.
    std::filesystem::create_directories(high / "meshes");
    std::ofstream(high / "meshes/jump.json") << R"({"duration":1,"tracks":[
        {"bone":"NPC COM [COM ]","translationKeys":[
            {"time":0,"value":[0,10,0]},{"time":1,"value":[2,90,3]}]}]})";
    std::ofstream(high / "odai/animations/jump.json") << R"({"version":1,"rules":[
        {"id":"jump","state":"jump","variants":[{"clip":"meshes/jump.json","loop":false}]}]})";
    odai::importer::fnv::FalloutAssetSource jumpAssets;
    assert(jumpAssets.addModDirectory(high));
    AnimationView jumpView;
    auto jumpRig = std::make_shared<Skeleton>();
    jumpRig->bones.push_back({"NPC COM [COM ]", -1});
    jumpView.skeleton = jumpRig;
    odai::importer::fnv::loadSkyrimNativeAnimationPacks(jumpAssets, jumpView);
    const auto jumpClip = std::find_if(jumpView.clips.begin(), jumpView.clips.end(),
        [](const auto& clip) { return clip.name == "meshes\\jump.json"; });
    assert(jumpClip != jumpView.clips.end());
    const auto& jumpKeys = jumpClip->tracks.front().translationKeys;
    assert(jumpKeys.back().value.y == jumpKeys.front().value.y);
    assert(jumpKeys.back().value.x == 2 && jumpKeys.back().value.z == 3);
    // Same source used as controller jump, authored action and looping layer:
    // none may inherit another use's root adjustment or looping policy.
    std::ofstream(high / "odai/animations/jump.json") << R"({"version":1,"rules":[
      {"id":"jump","state":"jump","variants":[{"clip":"meshes/jump.json","loop":false}]},
      {"id":"authored","state":"idle","variants":[{"clip":"meshes/jump.json","loop":false}]},
      {"id":"loop","state":"walk","variants":[{"clip":"meshes/jump.json","loop":true}]}]})";
    AnimationView sharedView; sharedView.skeleton = jumpRig;
    odai::importer::fnv::loadSkyrimNativeAnimationPacks(jumpAssets, sharedView);
    const auto findRuleClip = [&](const std::string& state) -> const AnimationClip& {
        const auto* rule = sharedView.nativeProgram->select(state, {});
        assert(rule);
        const auto found = std::find_if(sharedView.clips.begin(), sharedView.clips.end(),
            [&](const auto& clip) { return clip.name == rule->variants.front().clip; });
        assert(found != sharedView.clips.end());
        return *found;
    };
    const auto& controlled = findRuleClip("jump");
    const auto& authored = findRuleClip("idle");
    const auto& looping = findRuleClip("walk");
    assert(controlled.tracks.front().translationKeys.back().value.y == 10);
    assert(authored.tracks.front().translationKeys.back().value.y == 90);
    assert(looping.tracks.front().translationKeys.back().value.y == 90);
    assert(!controlled.loop && !authored.loop && looping.loop);

    std::filesystem::remove_all(root);
    if (argc > 1 && std::string_view(argv[1]) == "--benchmark") {
        auto benchmarkRig = std::make_shared<Skeleton>();
        for (int i = 0; i < 64; ++i) benchmarkRig->bones.push_back({"bone" + std::to_string(i), i - 1});
        AnimationView benchmarkView;
        benchmarkView.skeleton = benchmarkRig;
        AnimationClip idle; idle.name = "idle"; idle.duration = 1;
        benchmarkView.clips.push_back(idle);
        benchmarkView.stateClips = {{"idle", "idle"}, {"locomotion", "idle"}};
        auto benchmarkProgram = std::make_shared<NativeAnimationProgram>();
        for (int i = 0; i < 128; ++i) {
            NativeAnimationRule rule; rule.id = std::to_string(i); rule.state = "locomotion"; rule.priority = i;
            rule.variants = {{"idle", 1, true}};
            benchmarkProgram->rules.push_back(std::move(rule));
        }
        benchmarkView.nativeProgram = benchmarkProgram;
        std::vector<BehaviorGraphInstance> actors(200);
        for (auto& actor : actors) assert(actor.bind(benchmarkView, error));
        AnimationInputState movement; movement.movementSpeed = 100;
        const auto begin = std::chrono::steady_clock::now();
        std::size_t matrices = 0;
        for (int tick = 0; tick < 300; ++tick) for (auto& actor : actors)
            matrices += actor.step(movement, 1.f / 60).pose.size();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        std::cout << "Synthetic native animation benchmark: 200 actors, 64 bones, 128 rules, 300 ticks; "
            << seconds * 1.e6 / 60000 << " us/actor/tick; " << matrices << " sampled matrices\n";
    }
    std::cout << "Native animation selector tests passed\n";
}
