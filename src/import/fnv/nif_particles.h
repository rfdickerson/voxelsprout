#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
namespace odai::importer::fnv {
struct MistKey {
  float time = 0, value = 0, forward = 0, backward = 0;
};
struct MistNode {
  int parent = -1;
  std::array<float, 12> transform{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  std::array<std::vector<MistKey>, 3> angles;
  float period = 0, frequency = 1, phase = 0;
};
enum class NifEmitterShape : std::uint8_t { Box, Cylinder, Sphere };
// Supported Skyrim alpha-particle subset, in source (Z-up) coordinates.
struct NifMist {
  std::string texture;
  NifEmitterShape emitterShape = NifEmitterShape::Box;
  float volumeRadius = 0, volumeHeight = 0;
  float declination = 0, declinationVariation = 0, planarAngle = 0, planarVariation = 0;
  std::vector<MistNode> nodes;
  int emitterNode = -1, gravityNode = -1;
  float speed = 0, speedVariation = 0, radius = 1, radiusVariation = 0,
        life = 1, lifeVariation = 0;
  float rate = 0, period = 0, frequency = 1, phase = 0;
  std::array<float, 3> box{}, gravityAxis{0, 0, 1};
  float gravity = 0, drag = 0, rotationSpeed = 0, rotationVariation = 0;
  float softDepth = 0;
  std::array<float, 6> colorTimes{};
  std::array<float, 12> colors{};
  std::vector<float> scales;
};
struct MistParticle {
  std::array<float, 3> position{};
  std::array<float, 4> color{};
  float radius = 0, angle = 0;
};
bool parseNifMist(const std::vector<std::uint8_t> &, NifMist &, std::string &);
std::vector<MistParticle> sampleNifMist(const NifMist &, float elapsed,
                                        std::uint32_t seed);
std::vector<std::uint8_t> encodeNifMist(const NifMist &);
bool decodeNifMist(const std::vector<std::uint8_t> &, NifMist &);
} // namespace odai::importer::fnv
