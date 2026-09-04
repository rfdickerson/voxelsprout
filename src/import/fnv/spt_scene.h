#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "import/fnv/nif_scene.h"

namespace odai::importer::fnv {

// Renderer-independent subset of an Oblivion SpeedTree source. Retail SPT 2
// stores a tagged procedural description rather than triangles; the importer
// validates the stream and turns it into the same NifModel-shaped geometry the
// single imported-scene path already consumes.
struct OblivionSptTree {
    std::string sourcePath;
    std::string barkTexturePath;
    std::string leafTexturePath;
    std::string billboardTexturePath;
    std::uint32_t seed = 0u;
    float billboardWidth = 0.0f;
    float billboardHeight = 0.0f;
    float wind[8] = {};
    std::uint32_t taggedSectionCount = 0u;
    std::uint32_t splineTokenCount = 0u;
};

bool parseOblivionSpt(
    const std::vector<std::uint8_t>& bytes,
    std::string_view sourcePath,
    std::string_view leafTexturePath,
    std::uint32_t seed,
    float billboardWidth,
    float billboardHeight,
    const float wind[8],
    OblivionSptTree& outTree,
    std::string& outError);

// Produces deterministic vegetation LOD geometry in Bethesda model space
// (Z-up). The current safe fallback uses Oblivion's complete retail per-species
// billboard rather than inventing branches from partially decoded splines.
// Collision is intentionally absent: presentation must not invent gameplay
// collision.
bool buildOblivionSptModel(
    const OblivionSptTree& tree, NifModel& outModel, std::string& outError);

} // namespace odai::importer::fnv
