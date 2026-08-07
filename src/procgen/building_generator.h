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
    // Optional named-material-library slots for the facade details. Zero — the
    // default — means "no library material", and the generator's built-in
    // metallic/roughness constants are used directly, which is exactly what it
    // did before materials existed.
    //
    // A caller that sets these is promising it has uploaded a material table
    // where those indices are populated: the shader reads the table entry, so
    // pointing at an empty slot would render glazing as a flat rough dielectric
    // rather than glass. See Renderer::setImportedMaterialTable.
    std::uint32_t glassMaterial = 0u;
    std::uint32_t mullionMaterial = 0u;
    // The building mass itself. Walls and roofs carry no distinguishing
    // coefficients — they are all the default rough dielectric — so they are
    // told apart by surface normal: upward-facing goes to roofMaterial, the
    // rest to wallMaterial. That also sweeps up cornices, chimney caps and
    // parapet tops as "roof", which is what you want when tuning them.
    std::uint32_t wallMaterial = 0u;
    std::uint32_t roofMaterial = 0u;
};

TriMesh generateBuilding(const BuildingDesc& desc);

// The facade coefficients the generator uses when no library slot is supplied.
// Exposed so a caller populating a material table can seed those slots with the
// same values and get identical shading before it starts editing them.
inline constexpr float kBuildingGlassMetallic = 0.0f;
inline constexpr float kBuildingGlassRoughness = 0.09f;
inline constexpr float kBuildingMullionMetallic = 0.90f;
inline constexpr float kBuildingMullionRoughness = 0.34f;

}  // namespace odai::procgen
