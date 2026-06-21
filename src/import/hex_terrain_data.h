#pragma once

#include <cstdint>
#include <vector>

// GPU-instanced, tessellated hex-terrain render input.
//
// A StrategyMap is converted (game/strategy_hex_terrain.cc) into this Vulkan-free
// struct, which the renderer uploads once and draws as one instanced, height-
// displaced hex per land tile. It mirrors the role of ImportedScene: pure data on
// the game -> render seam, no Vulkan types.
//
// Geometry model:
//   - One shared base hex fan (center + 6 corners) in LOCAL XZ, reinterpreted as
//     six 3-control-point triangle patches. The GPU tessellator subdivides it and
//     the evaluation shader displaces Y.
//   - One HexTileInstance per land tile supplies the tile's world XZ center, its
//     own elevation height, and its six neighbour heights (one per hex edge).
//
// Height has two layers, both reconstructed in-shader from per-instance data (no
// extra textures or descriptors):
//   - CONTINUOUS base: each hex corner sits at the average of the three tiles that
//     meet there (own + the two edge-neighbours adjacent to that corner), so two
//     tiles sharing a corner compute an identical height -> crack free. The base is
//     planar within each of the six fan sectors.
//   - Per-tile DETAIL: fbm roughness or stepped strip-mine terraces, windowed to the
//     hex interior so it reaches 0 before any shared edge.
namespace odai::importer {

// One base-mesh vertex in local hex space (tile center at the origin). cornerIndex is
// 0..5 for the six corners (corner k at angle -30+60k degrees) and 6 for the center;
// the vertex shader uses it to pick that control point's continuous base height.
struct HexBaseVertex {
    float localXZ[2] = {};
    std::uint32_t cornerIndex = 0u;
};

// Per-tile instance attributes. Field order is also the vertex-attribute layout for a
// VK_VERTEX_INPUT_RATE_INSTANCE binding (all 4-byte, tightly packed, stride 64):
//   centerXZ             RG32F   @0
//   classFlags           R32_UINT@8
//   detailParams         RGBA32F @12   {feature, amplitude, windowEnd(world), noiseFreq}
//   ownAndNear           RGBA32F @28   {ownElevY, neighborElevY[0..2]}
//   farAndSize           RGBA32F @44   {neighborElevY[3..5], hexSize}
//   neighborTerrainPacked R32_UINT@60  bits 4k..4k+3 = TerrainType of neighbor k (k=0..5)
struct HexTileInstance {
    float centerXZ[2] = {};
    // bits 0-7 TerrainType, bits 8-15 TileFlag_*, bits 16-31 terrain texture index
    // (builder writes a scene index; the renderer remaps it to a bindless slot at
    // upload, 0xFFFF == none -> palette fallback in the fragment shader).
    std::uint32_t classFlags = 0u;
    float detailParams[4] = {};
    float ownElevY = 0.0f;               // world Y of this tile's own elevation
    float neighborElevY[6] = {};         // world Y per hex edge (edge k faces dir 60k deg)
    float hexSize = 0.0f;                // circumradius, for tessellation LOD scaling
    // Neighbor terrain types packed 4 bits each: bits (4k)..(4k+3) = TerrainType of
    // the neighbor in edge direction k (k=0..5, same indexing as neighborElevY).
    // Defaults to own terrain type on all 6 edges so map-border tiles blend to themselves.
    std::uint32_t neighborTerrainPacked = 0u;
};

// detailParams[0] feature selectors (the evaluation shader switches on these).
enum HexDetailFeature : std::uint32_t {
    HexDetail_None = 0u,
    HexDetail_Rough = 1u,       // generic fbm roughness (hills / mountains / forest)
    HexDetail_StripMine = 2u,   // concentric stepped terraces
};

// A contiguous run of instances sharing one cullable bound (one map chunk). Drawn
// with a single instanced vkCmdDrawIndexed using firstInstance = firstInstance.
struct HexTerrainPage {
    std::uint32_t firstInstance = 0u;
    std::uint32_t instanceCount = 0u;
    float boundsMin[3] = {};
    float boundsMax[3] = {};
};

struct HexTerrainData {
    std::vector<HexBaseVertex> baseVertices;     // center + 6 corners (7)
    std::vector<std::uint32_t> baseIndices;      // 18 (six triangle patches)
    std::vector<HexTileInstance> instances;      // grouped contiguously by page
    std::vector<HexTerrainPage> pages;           // one per chunk

    float hexSize = 0.0f;                         // circumradius (also per instance)

    [[nodiscard]] bool empty() const { return instances.empty(); }
};

}  // namespace odai::importer
