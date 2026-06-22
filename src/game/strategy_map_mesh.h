#pragma once

#include "game/strategy_map.h"
#include "game/units.h"
#include "import/imported_scene.h"

#include <vector>

// Converts a StrategyMap into an ImportedScene the existing Vulkan renderer can
// upload directly. Geometry is emitted into the packed render-stream
// (packedVertices / packedIndices / packedDraws) with per-vertex terrain color.
// When terrainTextures is populated, top-face vertices receive world-space tiling
// UVs and a bindless texture index; the imported-static shader samples the texture
// and falls back to vertex color for any terrain type with no texture entry.
namespace odai::game {

struct StrategyMapMeshOptions {
    bool drawGridOverlay = true;       // Thin dark edge lines per hex.
    bool drawSettlements = true;       // Box markers at settlement tiles.
    float gridLineWidth = 1.5f;        // World-unit width of overlay edge quads.
    float settlementHeight = 64.0f;    // Base marker height, scaled by tier.

    // 3D relief (true) extrudes hexes into prisms with side skirts so elevation
    // reads as solid land. 2D board mode (false) flattens every tile to the y=0
    // plane and omits skirts. The two modes produce different geometry, so the
    // caller re-meshes when toggling.
    bool extruded = true;

    // When false, land (non-water) tiles emit no top fan or skirts: the GPU-
    // instanced, tessellated hex-terrain path (game/strategy_hex_terrain.cc) owns
    // the land surface instead. Water tiles, the grid overlay, and markers are
    // unaffected. Set false whenever the caller also uploads HexTerrainData, so the
    // land surface is not drawn twice.
    bool emitLandSurface = true;

    // NxN hexes per chunk. The mesher emits one packed draw (and one page range
    // with tight bounds) per chunk so the renderer can frustum-cull per chunk.
    // Clamped to >= 1.
    std::uint32_t chunkSize = 16;

    // Emit ocean/coast tiles as flat water-level quads into scene.waterPatches so
    // the existing animated reflective/refractive water shader renders them.
    bool emitWaterPatches = true;

    // Vertical exaggeration for elevation-based land height. 0 = flat plateau;
    // 0.35 = subtle relief so terrain types read as raised/sunken without steep
    // faceting. Matches HexTerrainOptions::heightExaggeration so the two builders
    // can be swapped without a camera-framing jump. Ignored in 2D board mode.
    float heightExaggeration = 0.35f;

    // When true, tile visibility (Hidden/Explored/Visible) controls top-face color:
    // Hidden tiles are rendered near-black; Explored tiles are desaturated blue-gray;
    // Visible tiles keep their full terrain color. Skirt colors follow the top face.
    // The caller must set MapTile::visibility before each mesh build; the option simply
    // reads that field and applies it, so toggling re-meshes cleanly.
    bool fogOfWar = false;

    // When fogOfWar is true and playerOwner != 0, enemy units whose tile has
    // visibility != Visible are hidden (the player can't see what they can't see).
    // 0 disables the filter so all units are always drawn (preserves current behavior
    // when fog is off or playerOwner is unknown).
    std::uint8_t playerOwner = 0;

    // Per-terrain-type textures indexed by TerrainType cast to int. Entries with
    // width==0 fall back to vertex color. Multiple terrain types may share a file
    // (same sourcePath); the mesher deduplicates so each unique path uploads once.
    // The vector may be shorter than TerrainType::Count; missing entries fall back.
    std::vector<odai::importer::ImportedSceneTexture> terrainTextures;
};

// Build a renderable scene for the map. The result has sourceTag "strategy_map"
// and valid bounds for camera framing. Options taken by value so the caller can
// std::move the (potentially large) terrainTextures into the call.
odai::importer::ImportedScene buildStrategyMapScene(
    const StrategyMap& map,
    StrategyMapMeshOptions options = {});

// Same, but also emits a short token box per live unit (colored by owner and
// reddened by missing health). Unit markers join the whole-map overlay page so
// they are covered by a page range and never culled.
odai::importer::ImportedScene buildStrategyMapScene(
    const StrategyMap& map,
    const std::vector<Unit>& units,
    StrategyMapMeshOptions options = {});

// Class -> scene-texture-index map for a terrain-texture set, deduplicated by source
// path (entries with no pixels map to 0xFFFFFFFF). The mesher uploads textures in this
// exact index order, so the hex-terrain builder uses the same mapping to reference the
// same bindless slots. Result length == TerrainType::Count.
[[nodiscard]] std::vector<std::uint32_t> terrainTextureSceneIndices(
    const std::vector<odai::importer::ImportedSceneTexture>& terrainTextures);

}  // namespace odai::game
