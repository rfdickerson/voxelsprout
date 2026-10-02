#pragma once

#include "import/bethesda/fallout_records.h"
#include "math/math.h"

#include <cmath>
#include <span>

namespace odai::importer::bethesda {

// A non-owning view of the decoded LAND height posts. Renderer geometry and
// physics collision use these same coordinates and the same source heights.
// The owner keeps the LAND record alive until either consumer finishes building.
struct MorrowindTerrainSurface {
    int cellX = 0;
    int cellZ = 0;
    std::span<const float> heights;

    [[nodiscard]] bool valid() const {
        if (heights.size() != static_cast<std::size_t>(kMorrowindLandGridSize *
                                                      kMorrowindLandGridSize)) return false;
        for (float height : heights) if (!std::isfinite(height)) return false;
        return true;
    }
    [[nodiscard]] odai::math::Vector3 post(int row, int col) const {
        const float size = kLandPostSpacing * static_cast<float>(kMorrowindLandGridSize - 1);
        return {
            static_cast<float>(cellX) * size + static_cast<float>(col) * kLandPostSpacing,
            heights[static_cast<std::size_t>(row * kMorrowindLandGridSize + col)],
            -(static_cast<float>(cellZ) * size + static_cast<float>(row) * kLandPostSpacing)};
    }
};

[[nodiscard]] inline MorrowindTerrainSurface morrowindTerrainSurface(
    const FalloutCellRecord& cell) {
    if (!cell.hasGridCoords || cell.land == nullptr ||
        cell.land->gridSize != kMorrowindLandGridSize || !cell.land->hasHeights)
        return {};
    return {cell.gridX, cell.gridZ, cell.land->heights};
}

}  // namespace odai::importer::bethesda
