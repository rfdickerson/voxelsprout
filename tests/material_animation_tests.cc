#include "import/fnv/nif_material_animation.h"
#include "import/imported_lighting_material.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
using namespace odai::importer;
using namespace odai::importer::fnv;
namespace {
int failures = 0;
void check(bool ok, const char *message) {
  if (!ok) {
    std::cerr << message << '\n';
    ++failures;
  }
}
void near(float a, float b, const char *message) {
  check(std::abs(a - b) < 1e-5f, message);
}
template <class T> void put(std::vector<std::uint8_t> &b, T v) {
  auto p = reinterpret_cast<const std::uint8_t *>(&v);
  b.insert(b.end(), p, p + sizeof(v));
}
struct Fixture {
  std::vector<std::vector<std::uint8_t>> bytes{4};
  std::vector<MaterialAnimationBlock> blocks;
  Fixture(bool color = false, std::uint32_t variable = 6) {
    auto &p = bytes[0];
    put(p, int(-1));
    put(p, 0u);
    put(p, 1);
    auto &c = bytes[1];
    put(c, int(-1));
    put(c, std::uint16_t(8));
    for (float v : {1.f, 0.f, 0.f, 2.f})
      put(c, v);
    put(c, 0);
    put(c, 2);
    put(c, variable);
    auto &i = bytes[2];
    for (int n = 0; n < (color ? 3 : 1); ++n)
      put(i, 0.f);
    put(i, 3);
    auto &d = bytes[3];
    put(d, 2u);
    put(d, 1u);
    for (int key = 0; key < 2; ++key) {
      put(d, float(key * 2));
      for (int n = 0; n < (color ? 3 : 1); ++n)
        put(d, float(key + n));
    }
    blocks = {
        {"BSEffectShaderProperty", bytes[0]},
        {color ? "BSEffectShaderPropertyColorController"
               : "BSEffectShaderPropertyFloatController",
         bytes[1]},
        {color ? "NiPoint3Interpolator" : "NiFloatInterpolator", bytes[2]},
        {color ? "NiPosData" : "NiFloatData", bytes[3]}};
  }
  auto parse(std::uint32_t &unsupported) {
    return readNifMaterialAnimation(blocks, 0, {}, unsupported);
  }
};
} // namespace
int main() {
  for (const std::uint32_t slot : {0u, 4u, 6u}) {
    Fixture flip(false, slot);
    put(flip.bytes[1], 2u); put(flip.bytes[1], 4); put(flip.bytes[1], 5);
    flip.blocks[1] = {"NiFlipController", flip.bytes[1]};
    flip.blocks.push_back({"NiSourceTexture", {}});
    flip.blocks.push_back({"NiSourceTexture", {}});
    std::vector<std::string> paths(6);
    paths[4] = "synthetic/frame_a.dds"; paths[5] = "synthetic/frame_b.dds";
    std::uint32_t unsupportedFlip = 0;
    const auto parsed = readNifMaterialAnimation(flip.blocks, 0, paths, unsupportedFlip);
    const auto expected = slot == 4u ? MaterialAnimatedValue::GlowFrame :
        slot == 6u ? MaterialAnimatedValue::NormalFrame : MaterialAnimatedValue::DiffuseFrame;
    check(parsed.size() == 1u && parsed[0].target == expected && parsed[0].texturePaths == std::vector<std::string>{paths[4], paths[5]} && unsupportedFlip == 0,
          "NiFlip TexType maps base/glow/normal to independent material roles");
    paths[5].clear();
    check(readNifMaterialAnimation(flip.blocks, 0, paths, unsupportedFlip).empty() && unsupportedFlip > 0,
          "Missing flip source is reported instead of inventing a frame");
  }
  Fixture fixture;
  std::uint32_t unsupported = 0;
  auto tracks = fixture.parse(unsupported);
  check(tracks.size() == 1 && unsupported == 0,
        "linked scalar controller resolves");
  if (tracks.empty())
    return 1;
  auto t = tracks.front();
  near(sampleMaterialAnimation(t, 1), .5f, "linear midpoint");
  near(sampleMaterialAnimation(t, 3), .5f, "loop wraps");
  near(sampleMaterialAnimation(t, -1), .5f, "negative loop time wraps");
  t.cycle = 1;
  near(sampleMaterialAnimation(t, 2.5f), .75f, "reverse cycle reflects");
  t.cycle = 2;
  near(sampleMaterialAnimation(t, 3), 1, "clamp holds last key");
  t.backwards = true;
  near(sampleMaterialAnimation(t, 0), 1, "backwards begins at end");
  t.backwards = false;
  t.frequency = 2;
  t.phase = .5f;
  near(sampleMaterialAnimation(t, .25f), .5f,
       "frequency and phase apply before clamp");
  t = tracks.front();
  t.interpolation = 5;
  near(sampleMaterialAnimation(t, 1), 0, "constant key holds");
  t.interpolation = 2;
  t.keys[0].forward = 2;
  near(sampleMaterialAnimation(t, 1), .75f,
       "quadratic uses authored tangent deltas");
  t = tracks.front();
  t.interpolation = 3;
  near(sampleMaterialAnimation(t, .5f), .25f,
       "neutral TCB reproduces a linear ramp");
  t.keys[0].tension = t.keys[1].tension = 1;
  near(sampleMaterialAnimation(t, .5f), .15625f,
       "TCB tension changes the curve");
  Fixture color(true, 0);
  unsupported = 0;
  auto colors = color.parse(unsupported);
  check(colors.size() == 3 && unsupported == 0,
        "point3 controller yields RGB channels");
  if (colors.size() == 3)
    near(sampleMaterialAnimation(colors[2], 1), 2.5f,
         "color values retain independent channels");
  // Missing references, detached targets, cycles and malformed key payloads.
  auto original = fixture.bytes[1];
  int missing = 99;
  std::memcpy(fixture.bytes[1].data() + 26, &missing, 4);
  unsupported = 0;
  check(fixture.parse(unsupported).empty() && unsupported == 1,
        "missing interpolator rejected");
  std::memcpy(fixture.bytes[1].data(), original.data(), original.size());
  int detached = 3;
  std::memcpy(fixture.bytes[1].data() + 22, &detached, 4);
  unsupported = 0;
  check(fixture.parse(unsupported).empty() && unsupported == 1,
        "detached target rejected");
  std::memcpy(fixture.bytes[1].data(), original.data(), original.size());
  int cycle = 1;
  std::memcpy(fixture.bytes[1].data(), &cycle, 4);
  unsupported = 0;
  check(fixture.parse(unsupported).size() == 1 && unsupported == 1,
        "cyclic chain terminates");
  std::memcpy(fixture.bytes[1].data(), original.data(), original.size());
  fixture.blocks[3].data = fixture.blocks[3].data.first(9);
  unsupported = 0;
  check(fixture.parse(unsupported).empty() && unsupported == 1,
        "truncated keys rejected atomically");
  t = tracks[0];
  t.keys[1].time = t.keys[0].time;
  check(!validMaterialAnimationTrack(t), "duplicate key times rejected");
  t = tracks[0];
  t.keys[0].value = std::numeric_limits<float>::infinity();
  check(!validMaterialAnimationTrack(t), "nonfinite values rejected");
  Fixture flip(false, 0);
  flip.blocks[1].type = "NiFlipController";
  put(flip.bytes[1], 2u);
  put(flip.bytes[1], 2);
  put(flip.bytes[1], 3);
  flip.blocks[1].data = flip.bytes[1];
  unsupported = 0;
  auto flipped = readNifMaterialAnimation(
      flip.blocks, 0, {"", "", "a.dds", "b.dds"}, unsupported);
  check(flipped.size() == 1 && flipped[0].texturePaths.size() == 2 &&
            unsupported == 0,
        "linked diffuse flip controller resolves ordered source frames");
  unsupported = 0;
  check(readNifMaterialAnimation(flip.blocks, 0, {"", "", "a.dds", ""},
                                 unsupported)
                .empty() &&
            unsupported == 1,
        "missing flip source is reported without partial animation");
  ImportedScene scene;
  ImportedNifLightingMaterial material;
  material.valid = 1;
  material.shaderType = 0xffffffffu;
  material.emissiveMultiplier = 1;
  material.emissive[0] = material.emissive[1] = material.emissive[2] = 1;
  material.animations = tracks;
  t = tracks[0];
  t.target = MaterialAnimatedValue::Alpha;
  material.animations.push_back(t);
  t.target = MaterialAnimatedValue::DiffuseFrame;
  t.keys[1].value = 2;
  t.textures = {0, 0xffffffffu, 1};
  t.texturePaths = {"synthetic/a.dds", "synthetic/missing.dds",
                    "synthetic/b.dds"};
  material.animations.push_back(t);
  scene.lightingMaterials.push_back(material);
  scene.textures.resize(2);
  const auto path = std::filesystem::temp_directory_path() /
                    "odai-material-animation-test.bin";
  check(saveImportedScene(scene, path), "animated scene saves");
  for (bool runtime : {false, true}) {
    ImportedScene restored;
    check(runtime ? loadImportedSceneRuntime(path, restored)
                  : loadImportedScene(path, restored),
          "animated scene reloads");
    check(restored.lightingMaterials.size() == 1,
          "material ownership survives reload");
    if (restored.lightingMaterials.empty())
      continue;
    const auto &m = restored.lightingMaterials[0];
    check(m.animations.size() == 3, "tracks survive both loaders");
    auto gpu = sampleImportedMaterial(m, makeImportedNifGpuMaterial(m, {}), 1);
    near(gpu.animationUv[0], .5f, "UV controller reaches GPU material");
    near(gpu.animationColor[3], .5f, "alpha controller reaches GPU material");
    check(gpu.animationState[0] == 0xffffffffu,
          "missing flip frame retains static texture fallback");
    gpu = sampleImportedMaterial(m, makeImportedNifGpuMaterial(m, {}), 1.9f);
    check(gpu.animationState[0] == 0xffffffffu,
          "flip frame selection uses floor");
    gpu = sampleImportedMaterial(m, makeImportedNifGpuMaterial(m, {}), 0);
    near(gpu.animationUv[0], 0, "reload restarts at local time zero");
  }
  for (const auto target : {MaterialAnimatedValue::NormalFrame, MaterialAnimatedValue::GlowFrame}) {
    ImportedNifLightingMaterial animated;
    auto flipTrack = t;
    flipTrack.target = target;
    flipTrack.textures = {123u, 0xffffffffu, 456u};
    animated.animations = {flipTrack};
    GpuImportedMaterial baseGpu;
    baseGpu.textures[0] = 91u; baseGpu.textures[1] = 92u;
    const int role = target == MaterialAnimatedValue::NormalFrame ? 0 : 1;
    auto sampled = sampleImportedMaterial(animated, baseGpu, 0);
    check(sampled.textures[role] == 123u, "Non-diffuse flip reaches its typed GPU texture role");
    sampled = sampleImportedMaterial(animated, baseGpu, 1);
    check(sampled.textures[role] == baseGpu.textures[role], "Missing non-diffuse frame retains the authored static map");
    scene.lightingMaterials[0].animations = {flipTrack};
    scene.lightingMaterials[0].animations[0].textures = {0u, 0xffffffffu, 1u};
    check(saveImportedScene(scene, path), "Non-diffuse flip scene saves");
    ImportedScene reloaded;
    check(loadImportedSceneRuntime(path, reloaded) &&
          reloaded.lightingMaterials[0].animations[0].target == target,
          "Non-diffuse flip target survives runtime cooking");
  }
  // A v38 scene has no animation appendix and supplies neutral defaults.
  ImportedScene old;
  old.lightingMaterials.resize(1);
  check(saveImportedScene(old, path), "neutral scene saves");
  {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(4);
    const std::uint32_t v = 38;
    f.write(reinterpret_cast<const char *>(&v), 4);
  }
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 8);
  ImportedScene restored;
  check(loadImportedSceneRuntime(path, restored) &&
            restored.lightingMaterials[0].animations.empty(),
        "v38 has neutral animation defaults");
  check(saveImportedScene(scene, path), "resave animation fixture");
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
  check(!loadImportedSceneRuntime(path, restored),
        "truncated animation appendix rejected");
  std::filesystem::remove(path);
  return failures ? 1 : 0;
}
