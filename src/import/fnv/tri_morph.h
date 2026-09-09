#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace odai::importer::fnv {
using TriPosition = std::array<float, 3>;
struct TriMorph {
  std::string name;
  float scale = 0;
  std::vector<std::array<std::int16_t, 3>> deltas;
};
struct TriModifier {
  std::string name;
  std::vector<std::uint32_t> vertices;
  std::vector<TriPosition>
      positions; // absolute replacements, not expression deltas
};
// FRTRI003 data stays in source coordinates and original vertex order. A NIF
// vertex count alone is not proof that this topology matches an actor head.
struct TriMorphSet {
  std::vector<TriPosition> positions;
  std::vector<std::array<std::uint32_t, 3>> triangles, uvTriangles;
  std::vector<std::array<std::uint32_t, 4>> quads, uvQuads;
  std::vector<std::array<float, 2>> uvs;
  std::array<std::uint32_t, 6> reserved{};
  bool hasUvFaces = false;
  // Retail files can repeat names. Keep ordinal identity; a name-only binding
  // must diagnose ambiguity instead of silently choosing a different target.
  bool ambiguousMorphNames = false, ambiguousModifierNames = false;
  std::vector<TriMorph> morphs;
  std::vector<TriModifier> modifiers;
};
enum class TriReadStatus {
  Ok,
  UnsupportedFormat,
  MalformedData,
  ResourceLimit
};
// Transactional: failure clears output. Bounds and allocation limits apply even
// when the file originates from a locally installed mod.
TriReadStatus readTriMorphs(std::span<const std::uint8_t> bytes,
                            TriMorphSet &output, std::string &error);
struct TriMorphWeight {
  std::uint32_t morph = 0;
  float weight = 0;
};
// Evaluates additive relative expressions against a neutral source-order head.
// Does not apply absolute chargen modifiers, convert coordinate bases or update
// normals. No cumulative drift: every call starts from neutralPositions.
// Weights are explicit 0..1; invalid requests leave output empty.
bool evaluateTriMorphs(const TriMorphSet &set,
                       std::span<const TriPosition> neutralPositions,
                       std::span<const TriMorphWeight> weights,
                       std::vector<TriPosition> &output, std::string &error);
} // namespace odai::importer::fnv
