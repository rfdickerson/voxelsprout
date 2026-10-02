#include "import/bethesda/nif_particles.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
using namespace odai::importer::bethesda;
using Bytes = std::vector<std::uint8_t>;
template <class T> void put(Bytes &b, T v) {
  auto *p = (std::uint8_t *)&v;
  b.insert(b.end(), p, p + sizeof(v));
}
void str(Bytes &b, const std::string &s) {
  put(b, std::uint32_t(s.size()));
  b.insert(b.end(), s.begin(), s.end());
}
int failures = 0;
void check(bool value, const char *name) {
  if (!value) {
    std::cerr << name << "\n";
    ++failures;
  }
}
Bytes fixture(bool missing = false, NifEmitterShape shape = NifEmitterShape::Box, bool withLod = false, bool bare = false, bool unsupportedColor = false) {
  std::vector<std::string> types = {"NiNode",
                                    "NiParticleSystem",
                                    "NiPSysBoxEmitter",
                                    "BSPSysSimpleColorModifier",
                                    "BSPSysScaleModifier",
                                    "NiPSysGravityModifier",
                                    "NiPSysRotationModifier",
                                    "NiPSysEmitterCtlr",
                                    "NiFloatInterpolator",
                                    "NiPSysDragModifier",
                                    "NiPSysDragModifier",
                                    "NiPSysDragModifier",
                                    "BSEffectShaderProperty",
                                    "NiAlphaProperty"};
  types[2] = shape == NifEmitterShape::Cylinder ? "NiPSysCylinderEmitter" :
      shape == NifEmitterShape::Sphere ? "NiPSysSphereEmitter" : "NiPSysBoxEmitter";
  std::vector<Bytes> b(types.size());
  auto av = [&](int i) {
    put(b[i], -1);
    put(b[i], 0u);
    put(b[i], -1);
    put(b[i], 0u);
    for (float f :
         {10.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f})
      put(b[i], f);
    put(b[i], -1);
  };
  av(0);
  put(b[0], 1u);
  put(b[0], 1);
  put(b[0], 0u);
  av(1);
  for (int i = 0; i < 4; ++i)
    put(b[1], 0.f);
  put(b[1], -1);
  put(b[1], 12);
  put(b[1], 13);
  auto mod = [&](int i) {
    put(b[i], -1);
    put(b[i], 0u);
    put(b[i], missing ? 99 : 1);
    put(b[i], std::uint8_t(1));
  };
  mod(2);
  for (float f :
       {20.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 3.f, 0.f, 2.f, .2f})
    put(b[2], f);
  put(b[2], 0);
  for (int i = 0; i < (shape == NifEmitterShape::Box ? 3 : shape == NifEmitterShape::Cylinder ? 2 : 1); ++i)
    put(b[2], 4.f);
  mod(3);
  for (float f : {.1f, .9f, .1f, .3f, .7f, 1.f, 1.f, 1.f, 1.f, 0.f, 1.f, 1.f,
                  1.f, .5f, 1.f, 1.f, 1.f, 0.f})
    put(b[3], f);
  mod(4);
  put(b[4], 2u);
  put(b[4], 1.f);
  put(b[4], 2.f);
  mod(5);
  put(b[5], 0);
  for (float f : {0.f, 0.f, 1.f, 0.f, -2.f})
    put(b[5], f);
  put(b[5], 0u);
  mod(6);
  put(b[6], .1f);
  put(b[6], 0.f);
  put(b[7], -1);
  put(b[7], std::uint16_t(72));
  for (float f : {1.f, 0.f, 0.f, 3.f})
    put(b[7], f);
  put(b[7], 1);
  put(b[7], 8);
  put(b[8], 6.f);
  put(b[8], -1);
  for (int i = 9; i < 12; ++i) {
    mod(i);
    put(b[i], 0);
    for (int j = 0; j < 3; ++j)
      put(b[i], float(j == i - 9));
    put(b[i], .08f);
  }
  put(b[12], -1);
  put(b[12], 0u);
  put(b[12], -1);
  for (int i = 0; i < 6; ++i)
    put(b[12], 0u);
  str(b[12], "textures/effects/synthetic.dds");
  for (int i = 0; i < 10; ++i)
    put(b[12], 0.f);
  put(b[12], 40.f);
  put(b[13], -1);
  put(b[13], 0u);
  put(b[13], -1);
  put(b[13], std::uint16_t(0x10ed));
  if (withLod) {
    types.push_back("BSPSysLODModifier");
    b.emplace_back();
    mod(14);
    for (int i = 0; i < 4; ++i) put(b[14], 0.f);
  }
  if (bare) {
    const float initial[]{0.25f, 0.5f, 0.75f, 0.4f};
    std::memcpy(b[2].data() + 13 + 6 * sizeof(float), initial, sizeof(initial));
    // Unreferenced plain NiObject blocks stand in for absent optional modifiers.
    for (int i : {3, 4, 5, 6, 9, 10, 11}) {
      types[i] = "NiObject";
      b[i].clear();
    }
  }
  if (unsupportedColor) types[3] = "NiPSysColorModifier";
  Bytes out;
  std::string h = "Gamebryo File Format, Version 20.2.0.7\n";
  out.insert(out.end(), h.begin(), h.end());
  put(out, 0x14020007u);
  put(out, std::uint8_t(1));
  put(out, 12u);
  put(out, std::uint32_t(b.size()));
  put(out, 83u);
  for (int i = 0; i < 3; ++i)
    put(out, std::uint8_t(0));
  put(out, std::uint16_t(types.size()));
  for (auto &t : types)
    str(out, t);
  for (unsigned i = 0; i < b.size(); ++i)
    put(out, std::uint16_t(i));
  for (auto &v : b)
    put(out, std::uint32_t(v.size()));
  put(out, 0u);
  put(out, 0u);
  put(out, 0u);
  for (auto &v : b)
    out.insert(out.end(), v.begin(), v.end());
  put(out, 1u);
  put(out, 0);
  return out;
}
int main() {
  {
    NifMist bare, restored;
    std::string error;
    check(parseNifMist(fixture(false, NifEmitterShape::Box, false, true), bare, error),
          "Emitter without optional modifiers parses");
    check(bare.gravity == 0 && bare.drag == 0 && bare.scales == std::vector<float>({1, 1}),
          "Absent forces and scale preserve neutral defaults");
    check(bare.colors[0] == 0.25f && bare.colors[3] == 0.4f, "Initial emitter color retained");
    NifMist rejected;
    check(!parseNifMist(fixture(false, NifEmitterShape::Box, false, true, true),
                       rejected, error),
          "Unsupported color modifier cannot masquerade as absent");
    auto particles = sampleNifMist(bare, 0, 42);
    check(particles.size() == 1 && particles[0].color[3] == 0.4f,
          "Absent color modifier does not invent a birth fade");
    check(decodeNifMist(encodeNifMist(bare), restored), "Bare emitter cooked roundtrip");
    check(restored.colors == bare.colors && restored.gravity == 0 && restored.drag == 0,
          "Optional defaults survive streaming serialization");
  }

  NifMist m;
  std::string error;
  NifMist lodFixture;
  check(parseNifMist(fixture(false, NifEmitterShape::Box, true), lodFixture, error),
        "unconsumed LOD must not remove the supported mill-particle subset");
  auto bytes = fixture();
  check(parseNifMist(bytes, m, error), error.c_str());
  check(m.rate == 6 && m.life == 2 && m.scales.size() == 2,
        "source emission survives");
  auto p = sampleNifMist(m, 1, 42), again = sampleNifMist(m, 1, 42);
  check(!p.empty() && p.size() == again.size(), "live particles");
  for (size_t i = 0; i < p.size(); ++i)
    check(p[i].position == again[i].position,
          "deterministic independent evaluation");
  check(sampleNifMist(m, -1, 42).empty(), "negative time empty");
  check(sampleNifMist(m, 0, 42).size() == 1, "reload starts fresh");
  auto cooked = encodeNifMist(m);
  NifMist loaded;
  check(decodeNifMist(cooked, loaded), "typed cooked roundtrip");
  auto q = sampleNifMist(loaded, 1, 42);
  check(q.size() == p.size() && q.front().position == p.front().position,
        "cooked runtime identical");
  cooked.pop_back();
  check(!decodeNifMist(cooked, loaded), "truncated cooked data rejected");
  check(!parseNifMist(fixture(true), loaded, error),
        "missing modifier target rejected");
  bytes.resize(bytes.size() / 2);
  check(!parseNifMist(bytes, loaded, error), "truncated NIF rejected");
  m.gravity = -100;
  m.drag = 0;
  auto down = sampleNifMist(m, 1, 42);
  check(down.front().position[2] < p.front().position[2],
        "signed gravity moves particles downward");
  for (const auto shape : {NifEmitterShape::Cylinder, NifEmitterShape::Sphere}) {
    NifMist volume;
    check(parseNifMist(fixture(false, shape), volume, error), "Authored volume emitter parses");
    check(volume.emitterShape == shape && volume.volumeRadius == 4, "Emitter shape and dimensions survive");
    volume.nodes = {MistNode{}}; volume.emitterNode = volume.gravityNode = 0;
    volume.gravity = volume.drag = volume.speed = volume.speedVariation = 0;
    volume.rate = 200;
    const auto emitted = sampleNifMist(volume, 1, 77);
    check(emitted.size() > 100, "Volume fixture has enough samples for distribution bounds");
    float meanRadiusSquared = 0;
    for (const auto& particle : emitted) {
      const auto& v = particle.position;
      const float radiusSquared = v[0]*v[0]+v[1]*v[1]+(shape == NifEmitterShape::Sphere ? v[2]*v[2] : 0);
      check(radiusSquared <= 16.001f && std::abs(v[2]) <= (shape == NifEmitterShape::Sphere ? 4.001f : 2.001f),
            "Volume roots stay inside authored cylinder/sphere");
      meanRadiusSquared += radiusSquared / emitted.size();
    }
    check(meanRadiusSquared > 5 && meanRadiusSquared < 12, "Volume sampling fills the volume rather than its shell");
    NifMist roundtrip;
    check(decodeNifMist(encodeNifMist(volume), roundtrip), "Volume metadata survives cooking");
    const auto restored = sampleNifMist(roundtrip, 1, 77);
    check(restored.size() == emitted.size() && restored.front().position == emitted.front().position,
          "Volume cooking preserves deterministic birth positions");
    volume.volumeRadius = volume.volumeHeight = 0;
    volume.declination = 1.57079632679f; volume.planarAngle = 0; volume.speed = 20;
    const auto horizontal = sampleNifMist(volume, 1, 77);
    check(horizontal.front().position[0] > 19 && std::abs(horizontal.front().position[2]) < 0.001f,
          "Authored declination emits horizontally rather than always upward");
  }
  auto legacy = encodeNifMist(m);
  legacy.resize(legacy.size() - 29u);
  check(decodeNifMist(legacy, loaded) && loaded.emitterShape == NifEmitterShape::Box,
        "Legacy cooked mist retains neutral box shape");
  if (!failures)
    std::cout << "particle checks passed\n";
  return failures ? 1 : 0;
}
