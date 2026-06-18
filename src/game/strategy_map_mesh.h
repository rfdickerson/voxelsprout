#pragma once

#include "game/strategy_map.h"
#include "import/imported_scene.h"

// Converts a StrategyMap into an ImportedScene the existing Vulkan renderer can
// upload directly. Geometry is emitted into the packed render-stream
// (packedVertices / packedIndices / packedDraws) with per-vertex terrain color
// and textureIndex == invalid, so the imported-static shader shades it with the
// vertex color. No textures or new shaders are required.
namespace odai::game {

struct StrategyMapMeshOptions {
    bool drawGridOverlay = true;       // Thin dark edge lines per hex.
    bool drawSettlements = true;       // Box markers at settlement tiles.
    float gridLineWidth = 4.0f;        // World-unit width of overlay edge quads.
    float settlementHeight = 64.0f;    // Base marker height, scaled by tier.
};

// Build a renderable scene for the map. The result has sourceTag "strategy_map"
// and valid bounds for camera framing.
odai::importer::ImportedScene buildStrategyMapScene(
    const StrategyMap& map,
    const StrategyMapMeshOptions& options = {});

}  // namespace odai::game
