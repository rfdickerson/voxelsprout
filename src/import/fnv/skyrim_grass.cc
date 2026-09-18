#include "import/fnv/cell_builder.h"

#include <algorithm>
#include <cmath>

namespace odai::importer::fnv {

bool grassWaterAllowed(const FalloutGrassRecord& grass, float height) {
    const float distance = float(grass.waterDistance);
    switch (grass.waterMode) {
    case 0: return height >= distance;
    case 1: return height >= 0 && height <= distance;
    case 2: return height <= -distance;
    case 3: return height <= 0 && height >= -distance;
    case 4: return std::abs(height) >= distance;
    case 5: return std::abs(height) <= distance;
    case 6: return height <= distance;
    case 7: return height >= -distance;
    default: return false;
    }
}

namespace {
std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16; value *= 0x7feb352du;
    value ^= value >> 15; value *= 0x846ca68bu;
    return value ^ (value >> 16);
}
float unit(std::uint32_t value) { return float(mix(value) >> 8) / 16777216.0f; }
}

std::vector<FalloutPlacedReference> scatterSkyrimGrass(
    const FalloutCellRecord& cell, const FalloutWorldTables& tables) {
    std::vector<FalloutPlacedReference> result;
    const auto* world = tables.findWorldspace(cell.worldspaceFormId);
    if (!tables.skyrim || cell.isInterior || !cell.hasGridCoords || !cell.land ||
        (world && world->noGrass) || cell.land->gridSize != kLandGridSize ||
        !cell.land->hasHeights || cell.land->heights.size() != kLandVertexCount) return result;
    const auto& land = *cell.land;
    const bool water = cell.hasWater || (world && world->hasDefaultHeights);
    const float waterHeight = cell.hasWater ? cell.waterHeight :
        (world ? world->defaultWaterHeight : 0.0f);
    // One jittered candidate per 64-unit square; GRAS density is a percentage.
    // Half-open ownership prevents duplicate roots at cell boundaries. The seed
    // uses world/cell/sample coordinates, never traversal order or a global RNG.
    constexpr int side = 64;
    const std::uint32_t cellSeed = mix(cell.worldspaceFormId) ^
        mix(std::uint32_t(cell.gridX)) ^ mix(std::uint32_t(cell.gridZ) + 0x9e3779b9u);
    for (int y = 0; y < side; ++y) for (int x = 0; x < side; ++x) {
        const auto seed = mix(cellSeed ^ std::uint32_t(y * side + x));
        const float col = (float(x) + 0.1f + 0.8f * unit(seed)) * 0.5f;
        const float row = (float(y) + 0.1f + 0.8f * unit(seed + 1)) * 0.5f;
        const int c = int(col), r = int(row);
        const float fx = col - c, fy = row - r;
        const auto h = [&](int yy, int xx) { return land.heights[yy * kLandGridSize + xx]; };
        const float h00 = h(r,c), h10 = h(r,c+1), h01 = h(r+1,c), h11 = h(r+1,c+1);
        const float height = fy >= fx ? h00 + fx*(h11-h00) + (fy-fx)*(h01-h00) :
            h00 + fy*(h11-h00) + (fx-fy)*(h10-h00);
        const float dx = (fy >= fx ? h11-h01 : h10-h00) / kLandPostSpacing;
        const float dy = (fy >= fx ? h01-h00 : h11-h10) / kLandPostSpacing;
        const float slope = std::atan(std::sqrt(dx*dx+dy*dy)) * (180.0f / 3.14159265f);
        if (!std::isfinite(height) || !std::isfinite(slope)) continue;
        const int quadrant = (c >= 16 ? 1 : 0) | (r >= 16 ? 2 : 0);
        // Stochastic selection follows the complete ordered terrain paint stack,
        // including overlays with no grass. Such layers suppress underlying grass.
        std::uint32_t texture = land.quadrantBaseTextureFormId[quadrant];
        for (const auto& layer : land.textureLayers) {
            if (layer.quadrant != quadrant) continue;
            const float opacity = sampleLandLayerOpacity(layer, row - (r >= 16 ? 16 : 0),
                                                        col - (c >= 16 ? 16 : 0));
            if (unit(seed ^ mix(std::uint32_t(layer.layerIndex) + 73u)) < opacity)
                texture = layer.textureFormId;
        }
        const auto association = tables.landGrass.find(texture);
        if (association == tables.landGrass.end() || association->second.empty()) continue;
        const auto& ids = association->second;
        const auto id = ids[mix(seed + 2) % ids.size()];
        const auto found = tables.grasses.find(id);
        if (found == tables.grasses.end()) continue;
        const auto& grass = found->second;
        if (grass.deleted || !grass.valid || grass.modelPath.empty() ||
            unit(seed+3)*100.0f >= grass.density || slope < grass.minSlope ||
            slope > grass.maxSlope || (water && !grassWaterAllowed(grass, height-waterHeight))) continue;
        // In a waterless world only above/either-at-least modes make sense.
        if (!water && grass.waterMode != 0 && grass.waterMode != 4 && grass.waterMode != 7) continue;
        FalloutPlacedReference placement{};
        placement.baseFormId = id;
        placement.position[0] = float(cell.gridX)*kExteriorCellSize + col*kLandPostSpacing;
        placement.position[1] = float(cell.gridZ)*kExteriorCellSize + row*kLandPostSpacing;
        placement.position[2] = height;
        placement.rotationRadians[2] = unit(seed+4)*6.2831853f;
        if ((grass.flags & 4u) != 0u) { // Fit To Slope, clockwise REFR angles
            const float length = std::sqrt(dx*dx + dy*dy + 1.0f);
            // NiMatrix3 XYZ applies local yaw before the slope tilt, so yaw
            // does not change the transformed local Z axis.
            placement.rotationRadians[0] = std::atan2(-dy, 1.0f);
            placement.rotationRadians[1] = std::asin(std::clamp(dx / length, -1.0f, 1.0f));
        }
        placement.scale = std::max(0.1f, 1.0f + (unit(seed+5)*2-1)*
            std::clamp(grass.heightRange, 0.0f, 1.0f));
        result.push_back(placement);
    }
    return result;
}
} // namespace odai::importer::fnv
