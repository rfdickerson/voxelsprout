#pragma once

#include "game/strategy_map.h"
#include "import/hex_terrain_data.h"

// Converts a StrategyMap into HexTerrainData: one shared subdivided hex base mesh
// plus one instance per land tile, a continuous elevation field texture, and the
// constants the displacement shader needs. Pure CPU; mirrors strategy_map_mesh.cc.
//
// Land surfaces move to this GPU-instanced, tessellated, displaced path. Water,
// the grid overlay, and markers continue to come from buildStrategyMapScene (call
// it with StrategyMapMeshOptions::emitLandSurface = false to avoid double-drawing
// the land tops).
namespace odai::game {

struct HexTerrainOptions {
    // NxN tiles per page; one instanced draw + one cullable bound per page. Matches
    // the strategy-map mesher's chunkSize so land and overlays page identically.
    std::uint32_t chunkSize = 16;

    // Vertical exaggeration applied to the elevation relief (1.0 == literal).
    float heightExaggeration = 1.0f;

    // When true, mark a sparse deterministic set of Hills tiles as strip mines so the
    // stepped-terrace detail feature is visible in the demo map. Off by default.
    bool demoStripMines = false;
};

odai::importer::HexTerrainData buildHexTerrain(const StrategyMap& map,
                                               HexTerrainOptions options = {});

}  // namespace odai::game
