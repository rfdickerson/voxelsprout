#include "anim/native_program.h"
#include "anim/pose_graph.h"
#include "anim/pose_modifiers.h"
#include "anim/skyrim_animation.h"
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
using namespace odai::anim;
using namespace odai::math;
namespace {
bool near(float a, float b, float tolerance = 1.e-4f) {
  return std::abs(a - b) < tolerance;
}
AnimationClip clip(std::string name, float offset) {
  AnimationClip c;
  c.name = std::move(name);
  c.duration = 1;
  c.loop = true;
  BoneTrack t;
  t.boneIndex = 0;
  t.translationKeys = {{0, {offset, 0, 0}}, {1, {offset + 1, 0, 0}}};
  c.tracks.push_back(t);
  return c;
}
const char *graphText =
    R"({"root":"fsm","parameters":{"speed":0,"attack":false},"nodes":[
{"id":"idle","type":"clip","clip":"idle"},
{"id":"walk","type":"clip","clip":"walk"},
{"id":"cached","type":"cache","inputs":["walk"]},
{"id":"move","type":"blend1d","parameter":"speed","samples":[{"node":"idle","x":0},{"node":"cached","x":1}]},
{"id":"hit","type":"clip","clip":"hit"},
{"id":"fsm","type":"state_machine","inputs":["move","hit"],"initial":"move","transitions":[
{"from":"move","to":"hit","parameter":"attack","equals":true,"duration":0},
{"from":"hit","to":"move","parameter":"attack","equals":false,"duration":0}]}]})";
} // namespace
int main() {
  {
    const std::array<PoseGraphSample, 4> square{
        {{"a", 0, 0}, {"b", 1, 0}, {"c", 1, 1}, {"d", 0, 1}}};
    const auto low = blendSpaceWeights(square, .2f, .19999f, true);
    const auto high = blendSpaceWeights(square, .2f, .20001f, true);
    for (int i = 0; i < 4; ++i)
      assert(near(low[i], high[i], .0001f));
  }
  Skeleton skeleton;
  skeleton.bones.push_back({"root", -1});
  std::vector<AnimationClip> clips{clip("idle", 0), clip("walk", 10),
                                   clip("hit", 20)};
  std::string error;
  PoseGraphProgram graph;
  assert(compilePoseGraph(graphText, graph, error));
  auto invalid = nlohmann::json::parse(graphText);
  invalid["nodes"][2]["inputs"] = {"fsm"};
  const auto root = graph.root;
  assert(!compilePoseGraph(invalid.dump(), graph, error));
  assert(graph.root == root);
  invalid = nlohmann::json::parse(graphText);
  invalid["parameters"]["speed"] = false;
  assert(!compilePoseGraph(invalid.dump(), graph, error));
  invalid = nlohmann::json::parse(graphText);
  invalid["nodes"][0]["id"] = "walk";
  assert(!compilePoseGraph(invalid.dump(), graph, error));
  PoseGraphInstance instance;
  PoseEvaluationPacket packet;
  LocalPose pose;
  assert(instance.advance(graph, skeleton, nullptr, {{"speed", .25}}, clips,
                          .1f, packet, error));
  assert(evaluatePosePacket(packet, skeleton, clips, pose, error));
  assert(near(pose[0].translation.x, 2.6f));
  {
    PoseGraphProgram procedural;
    assert(compilePoseGraph(
        R"({"root":"offset","parameters":{"x":2,"y":0,"z":0},"nodes":[
      {"id":"base","type":"clip","clip":"idle"},
      {"id":"offset","type":"translate","inputs":["base"],"role":"root","target_parameters":["x","y","z"]}]})",
        procedural, error));
    HumanoidRigMapping rig;
    rig.roles["root"] = 0;
    PoseGraphInstance evaluator;
    PoseEvaluationPacket evaluated;
    LocalPose local;
    assert(evaluator.advance(procedural, skeleton, &rig, {}, clips, .1f,
                             evaluated, error));
    assert(evaluatePosePacket(evaluated, skeleton, clips, local, error));
    assert(near(local[0].translation.x, 2.1f));
  }
  const auto saved = instance.save();
  PoseGraphInstance restored;
  assert(restored.restore(saved, error));
  assert(!restored.restore(
      R"({"event":"x","clocks":{"a":{"time":-1,"state":"","previous":"","elapsed":0,"duration":0}}})",
      error));
  assert(restored.save() == saved);
  PoseEvaluationPacket second;
  assert(instance.advance(graph, skeleton, nullptr, {{"attack", true}}, clips,
                          .1f, packet, error));
  assert(restored.advance(graph, skeleton, nullptr, {{"attack", true}}, clips,
                          .1f, second, error));
  assert(instance.save() == restored.save());
  assert(packet.eventClip == "hit");
  assert(evaluatePosePacket(packet, skeleton, clips, pose, error));
  assert(near(pose[0].translation.x, 20.1f));
  auto badPacket = packet;
  badPacket.instructions[0].inputs = {999};
  const auto before = pose;
  assert(!evaluatePosePacket(badPacket, skeleton, clips, pose, error));
  assert(near(before[0].translation.x, pose[0].translation.x));
  const auto beforeMissing = instance.save();
  auto missing = clips;
  missing.pop_back();
  assert(!instance.advance(graph, skeleton, nullptr, {{"attack", true}},
                           missing, .1f, packet, error));
  assert(instance.save() == beforeMissing);
  std::vector<PoseGraphSample> triangle{{"a", 0, 0}, {"b", 1, 0}, {"c", 0, 1}};
  auto weights = blendSpaceWeights(triangle, .25f, .25f, true);
  assert(near(weights[0], .5f) && near(weights[1], .25f) &&
         near(weights[2], .25f));
  weights = blendSpaceWeights(triangle, 1, 1, true);
  assert(near(weights[0], 0) && near(weights[1], .5f) && near(weights[2], .5f));
  std::vector<PoseGraphSample> line{{"a", 0, 0}, {"b", 1, 0}, {"c", 2, 0}};
  weights = blendSpaceWeights(line, .5f, 0, false);
  assert(near(weights[0], .5f) && near(weights[1], .5f) && near(weights[2], 0));
  auto leader = clip("lead", 0), follower = clip("follow", 0);
  leader.annotations = {{.2f, "sync:left"}, {.6f, "sync:right"}};
  follower.duration = 2;
  follower.annotations = {{.2f, "sync:left"}, {1.4f, "sync:right"}};
  assert(near(synchronizedClipTime(leader, .4f, follower), .8f));
  assert(near(synchronizedClipTime(leader, 1.4f, follower), 2.8f));
  follower.annotations.clear();
  assert(near(synchronizedClipTime(leader, .4f, follower), .8f));
  Skeleton limb;
  limb.bones = {{"hip", -1, {0, 0, 0}},
                {"knee", 0, {0, -1, 0}},
                {"foot", 1, {0, -1, 0}},
                {"toe", 2, {0, 0, .2f}}};
  LocalPose rest;
  for (const auto &b : limb.bones)
    rest.push_back({b.localTranslation, b.localRotation, b.localScale});
  auto solved = rest;
  LimbTarget target;
  target.position = {.5f, -1.5f, .2f};
  assert(solveTwoBoneIk(limb, {0, 1, 2}, target, solved));
  auto world = composePoseWorld(limb, solved);
  assert(length(transformPoint(world[2], {}) - target.position) < .001f);
  assert(near(
      length(transformPoint(world[1], {}) - transformPoint(world[0], {})), 1));
  assert(near(
      length(transformPoint(world[2], {}) - transformPoint(world[1], {})), 1));
  target.position = {10, 0, 0};
  solved = rest;
  assert(solveTwoBoneIk(limb, {0, 1, 2}, target, solved));
  world = composePoseWorld(limb, solved);
  assert(length(transformPoint(world[2], {})) <= 2.001f);
  assert(!solveTwoBoneIk(limb, {2, 1, 0}, target, solved));
  target.position = {.5f, -1.5f, .2f};
  target.normal = normalize(Vector3{.2f, 1, .1f});
  target.alignNormal = true;
  solved = rest;
  assert(solveTwoBoneIk(limb, {0, 1, 2}, target, solved));
  world = composePoseWorld(limb, solved);
  const auto up = normalize(transformPoint(world[2], {0, 1, 0}) -
                            transformPoint(world[2], {}));
  assert(dot(up, target.normal) > .999f);
  PoseInertializer inertial;
  auto base = rest;
  auto result = inertial.evaluate(base, false, .2f, .01f);
  base[0].translation.x = 10;
  result = inertial.evaluate(base, true, .2f, .01f);
  assert(near(result[0].translation.x, 0));
  for (int i = 0; i < 8; ++i)
    result = inertial.evaluate(base, false, .2f, .01f);
  const auto previous = result[0].translation.x;
  base[0].translation.x = -5;
  result = inertial.evaluate(base, true, .2f, .01f);
  assert(near(result[0].translation.x, previous));
  PoseInertializer resumed;
  assert(resumed.restore(inertial.save(), error));
  for (int i = 0; i < 25; ++i) {
    result = inertial.evaluate(base, false, .2f, .01f);
    const auto other = resumed.evaluate(base, false, .2f, .01f);
    assert(near(result[0].translation.x, other[0].translation.x));
  }
  assert(near(result[0].translation.x, -5));
  NativeAnimationProgram native;
  nlohmann::json pack = {
      {"version", 2},
      {"rules", nlohmann::json::array(
                    {{{"id", "modern"},
                      {"state", "idle"},
                      {"variants", nlohmann::json::array({{{"clip", "idle"}}})},
                      {"graph", nlohmann::json::parse(graphText)}}})}};
  assert(compileNativeAnimationPack(pack.dump(), "test", 0, native, error));
  assert(native.rules[0].graph);
  AnimationView view;
  view.skeleton = std::make_shared<Skeleton>(skeleton);
  view.clips = clips;
  view.stateClips = {{"idle", "idle"}};
  view.nativeProgram = std::make_shared<NativeAnimationProgram>(native);
  BehaviorGraphInstance runtime;
  assert(runtime.bind(view, error));
  AnimationInputState input;
  auto output = runtime.step(input, .1f);
  assert(output.evaluationPacket);
  assert(output.localPose.size() == 1);
  const auto snapshot = runtime.snapshot();
  BehaviorGraphInstance continuation;
  assert(continuation.bind(view, error));
  assert(continuation.restore(snapshot, error));
  const auto a = runtime.step(input, .1f), b = continuation.step(input, .1f);
  assert(near(a.pose[0].m[3], b.pose[0].m[3]));
  {
    AnimationView feetView = view;
    feetView.skeleton = std::make_shared<Skeleton>(limb);
    auto rig = std::make_shared<HumanoidRigMapping>();
    rig->roles["pelvis"] = 0;
    rig->limbs["left_leg"] = {0, 1, 2};
    feetView.humanoidRig = rig;
    BehaviorGraphInstance feet;
    assert(feet.bind(feetView, error));
    AnimationInputState contact;
    contact.grounded = true;
    contact.footIkEnabled = true;
    contact.footContacts[0].valid = true;
    contact.footContacts[0].position = {0, -2, 0};
    contact.footContacts[0].normal = normalize(Vector3{.2f, 1, .1f});
    assert(!feet.step(contact, .1f).localPose.empty());
    assert(feet.snapshot().footPlants.contains("left_leg"));
    assert(feet.snapshot().footPlants.at("left_leg").size() == 6);
    contact.actorPosition.x = .1f;
    contact.footContacts[0].position.x = .1f;
    const auto planted = feet.step(contact, .1f);
    // The ground probe moved under the translated actor, but the stance foot
    // remains at the original world-space contact until the authored lift.
    assert(near(feet.snapshot().footPlants.at("left_leg")[0], 0.f));
    const auto plantedWorld = composePoseWorld(limb, planted.localPose);
    assert(std::abs(transformPoint(plantedWorld[2], {}).x + .1f) < .01f);
    contact.footContacts[0].valid = false;
    (void)feet.step(contact, .1f);
    assert(feet.snapshot().footPlants.empty());
    contact.footContacts[0].valid = true;
    (void)feet.step(contact, .1f);
    assert(feet.snapshot().footPlants.contains("left_leg"));
    contact.ragdollActive = true;
    (void)feet.step(contact, .1f);
    assert(feet.snapshot().footPlants.empty());
    contact.ragdollActive = false;
    contact.teleported = true;
    contact.grounded = false;
    const auto airborne = feet.step(contact, .1f);
    assert(airborne.resetHistory && feet.snapshot().footPlants.empty());
  }
  std::cout << "Pose graph and procedural motion tests passed\n";
}
