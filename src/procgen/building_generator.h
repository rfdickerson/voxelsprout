#pragma once

#include <cstdint>

#include "procgen/mesh_emit.h"

// Era-styled low-poly building generator. Buildings are assembled from CSG
// primitives and booleans in LOCAL space — the lot spans [0, lotWidth] x
// [0, lotDepth] in XZ with the ground at y = 0 and the street assumed on the
// -Z side — then triangulated for per-tile placement via appendTriMesh.
namespace odai::procgen {

enum class Era : std::uint8_t {
    E1890s,  // brick row-houses, cornices, chimneys, sawtooth mills
    E1930s,  // art-deco stepped setback towers, brick factories
    E1960s,  // modernist glass slabs, curtain walls, tank farms
};

enum class BuildingKind : std::uint8_t { Residential, Commercial, Industrial };

struct BuildingDesc {
    Era era = Era::E1890s;
    BuildingKind kind = BuildingKind::Residential;
    int level = 1;            // 1..3 development level (drives mass/height)
    int wealthTier = 1;       // 0..2 (residential: trailer park .. estates)
    float lotWidth = 0.8f;    // world units, X extent of the buildable pad
    float lotDepth = 0.8f;    // world units, Z extent
    std::uint32_t seed = 0;   // same desc => bit-identical mesh
    // LOD tier: 0 = massing only (far zoom), 1 = adds the era window pass —
    // painted-on facade quads (Victorian sashes, deco ribbons, curtain-wall
    // mullions), ~2 triangles per window. Identical massing at both tiers.
    int detail = 1;
};

TriMesh generateBuilding(const BuildingDesc& desc);

}  // namespace odai::procgen
