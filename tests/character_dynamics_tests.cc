#include "anim/character_dynamics.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

using namespace odai;

int main(int argc, char** argv) {
    anim::BodyMorphProgram program;
    std::string error;
    const std::string pack = R"({
      "version": 1,
      "vertex_count": 3,
      "topology_fingerprint": "fixture-v1",
      "targets": [
        {"id":"muscular","deltas":[{"vertex":1,"position":[2,0,0]}]},
        {"id":"weight","deltas":[{"vertex":2,"position":[0,4,0]}]}
      ],
      "outfit_mappings":{"iron":["weight"]}
    })";
    assert(anim::compileBodyMorphProgram(pack, program, error));
    const std::vector<math::Vector3> base(3);
    anim::BodyMorphSnapshot sliders{"fixture-v1", {{"muscular", 1.0f}, {"weight", 0.5f}}};
    std::vector<math::Vector3> result;
    assert(anim::applyBodyMorphs(program, sliders, base, "iron", result, error));
    assert(result[1].x == 0.0f && result[2].y == 2.0f);
    assert(!anim::applyBodyMorphs(program, sliders, base, "unmapped", result, error));
    sliders.topologyFingerprint = "wrong";
    assert(!anim::applyBodyMorphs(program, sliders, base, "iron", result, error));

    anim::HairChainSimulator hair;
    anim::HairChainConfig hairConfig;
    hairConfig.bones = {0, 1, 2};
    hairConfig.damping = 0.1f;
    hairConfig.collisions.push_back({0, 0.25f});
    assert(hair.configure(hairConfig, error));
    std::vector<math::Matrix4> authored{
        math::Matrix4::translation({0, 2, 0}),
        math::Matrix4::translation({0, 1, 0}),
        math::Matrix4::translation({0, 0, 0})};
    std::vector<math::Matrix4> simulated;
    assert(hair.update(authored, 1.0f / 60.0f, true, simulated));
    assert(simulated.size() == authored.size());
    assert(hair.snapshot().positions.front().y == 2.0f);
    authored[0] = math::Matrix4::translation({100, 2, 0});
    authored[1] = math::Matrix4::translation({100, 1, 0});
    authored[2] = math::Matrix4::translation({100, 0, 0});
    assert(hair.update(authored, 1.0f / 60.0f, true, simulated));
    assert(std::fabs(hair.snapshot().positions.back().x - 100.0f) < 0.001f);
    assert(hair.update(authored, 1.0f / 60.0f, false, simulated));
    assert(simulated[2](1, 3) == authored[2](1, 3));

    if (argc > 1 && std::string_view(argv[1]) == "--benchmark") {
        constexpr int actors = 200;
        constexpr int ticks = 300;
        const auto start = std::chrono::steady_clock::now();
        for (int actor = 0; actor < actors; ++actor) {
            anim::HairChainSimulator measured;
            assert(measured.configure(hairConfig, error));
            for (int tick = 0; tick < ticks; ++tick) {
                authored[0](0, 3) = static_cast<float>(tick) * 0.01f;
                assert(measured.update(authored, 1.0f / 60.0f, true, simulated));
            }
        }
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "hair benchmark: " << actors << " actors x " << ticks
                  << " ticks = " << elapsed << " ms ("
                  << (elapsed * 1000.0 / (actors * ticks)) << " us/actor-tick)\n";
    }

    std::cout << "character dynamics tests passed\n";
}
