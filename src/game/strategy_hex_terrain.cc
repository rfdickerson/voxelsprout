#include "game/strategy_hex_terrain.h"

#include "game/strategy_map_mesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace odai::game {

namespace {

using odai::importer::HexBaseVertex;
using odai::importer::HexDetailFeature;
using odai::importer::HexTerrainData;
using odai::importer::HexTerrainPage;
using odai::importer::HexTileInstance;

constexpr float kInradiusFrac = 0.8660254f;  // sqrt(3)/2: inradius / circumradius

// Per-tile detail amplitude as a fraction of hexSize, by terrain. Kept modest so the
// surface reads as rolling relief rather than spikes (the shader smooths further).
// Water terrains produce no land instance and are never indexed here.
float terrainDetailAmplitudeFrac(TerrainType terrain) {
    switch (terrain) {
        case TerrainType::Mountains: return 0.45f;
        case TerrainType::Hills:     return 0.20f;
        case TerrainType::Forest:    return 0.09f;
        case TerrainType::Jungle:    return 0.09f;
        case TerrainType::Snow:      return 0.06f;
        case TerrainType::Desert:    return 0.05f;
        case TerrainType::Plains:    return 0.035f;
        case TerrainType::Grassland: return 0.035f;
        case TerrainType::Tundra:    return 0.035f;
        default:                     return 0.0f;
    }
}

}  // namespace

HexTerrainData buildHexTerrain(
    const StrategyMap& map,
    const std::vector<odai::importer::ImportedSceneTexture>& terrainTextures,
    HexTerrainOptions options) {
    HexTerrainData data{};
    const float H = map.hexSize;
    data.hexSize = H;

    if (map.width == 0u || map.height == 0u || map.tiles.empty() || H <= 0.0f) {
        return data;
    }

    // Class -> scene-texture index (same dedup the mesher uses). Packed into classFlags
    // bits 16-31; the renderer remaps it to a bindless slot at upload. 0xFFFF == none.
    const std::vector<std::uint32_t> texSceneIdx = terrainTextureSceneIndices(terrainTextures);

    // --- Shared base hex fan in local XZ: center + 6 corners (corner k at -30+60k
    //     deg), six 3-control-point triangle patches. cornerIndex 6 marks the center. ---
    data.baseVertices.push_back(HexBaseVertex{{0.0f, 0.0f}, 6u});  // index 0: center
    for (std::uint32_t corner = 0; corner < 6u; ++corner) {
        const float angle = odai::math::radians((60.0f * static_cast<float>(corner)) - 30.0f);
        data.baseVertices.push_back(HexBaseVertex{{H * std::cos(angle), H * std::sin(angle)}, corner});
    }
    for (std::uint32_t corner = 0; corner < 6u; ++corner) {
        const std::uint32_t next = (corner + 1u) % 6u;
        data.baseIndices.push_back(0u);          // center
        data.baseIndices.push_back(1u + next);   // matches strategy_map_mesh fan winding
        data.baseIndices.push_back(1u + corner);
    }

    // --- Elevation range over LAND tiles only (drives the continuous base height). ---
    int minElev = std::numeric_limits<int>::max();
    int maxElev = std::numeric_limits<int>::min();
    for (const MapTile& tile : map.tiles) {
        if (terrainIsWater(tile.terrain)) {
            continue;
        }
        minElev = std::min(minElev, static_cast<int>(tile.elevation));
        maxElev = std::max(maxElev, static_cast<int>(tile.elevation));
    }
    if (minElev > maxElev) {
        return data;  // no land
    }
    const float exaggeration = std::max(0.0f, options.heightExaggeration);
    const float elevDenom = static_cast<float>(std::max(1, maxElev - minElev));
    const float reliefWorld = static_cast<float>(maxElev - minElev) * map.elevationStep * exaggeration;
    const float waterLevelY = 0.06f * H;   // matches the mesher's water surface
    const float landFloorY = 0.12f * H;    // lowest land sits just above the water

    // World Y for any tile: water surface for sea tiles, elevation-driven for land.
    const auto tileHeightY = [&](std::uint32_t col, std::uint32_t row) -> float {
        const MapTile& t = map.at(col, row);
        if (terrainIsWater(t.terrain)) {
            return waterLevelY;
        }
        const float normalized =
            (static_cast<float>(t.elevation) - static_cast<float>(minElev)) / elevDenom;
        return landFloorY + (std::clamp(normalized, 0.0f, 1.0f) * reliefWorld);
    };

    // Conservative Y span for page culling bounds.
    const float maxDetailAmp = 0.55f * H * exaggeration;
    const float pageMinY = std::min(waterLevelY, landFloorY) - maxDetailAmp;
    const float pageMaxY = landFloorY + reliefWorld + maxDetailAmp;

    // --- One instance per land tile, grouped into contiguous per-chunk pages. ---
    const std::uint32_t chunkSize = std::max<std::uint32_t>(options.chunkSize, 1u);
    const std::uint32_t chunkCols = (map.width + chunkSize - 1u) / chunkSize;
    const std::uint32_t chunkRows = (map.height + chunkSize - 1u) / chunkSize;
    for (std::uint32_t chunkRow = 0; chunkRow < chunkRows; ++chunkRow) {
        for (std::uint32_t chunkCol = 0; chunkCol < chunkCols; ++chunkCol) {
            const std::uint32_t firstInstance = static_cast<std::uint32_t>(data.instances.size());
            float loX = std::numeric_limits<float>::max();
            float loZ = std::numeric_limits<float>::max();
            float hiX = std::numeric_limits<float>::lowest();
            float hiZ = std::numeric_limits<float>::lowest();

            const std::uint32_t rowEnd = std::min(map.height, (chunkRow + 1u) * chunkSize);
            const std::uint32_t colEnd = std::min(map.width, (chunkCol + 1u) * chunkSize);
            for (std::uint32_t row = chunkRow * chunkSize; row < rowEnd; ++row) {
                for (std::uint32_t col = chunkCol * chunkSize; col < colEnd; ++col) {
                    const MapTile& tile = map.at(col, row);
                    if (terrainIsWater(tile.terrain)) {
                        continue;
                    }
                    const odai::math::Vector3 center = tileCenterWorld(map, col, row);
                    const float ownY = tileHeightY(col, row);

                    const std::size_t terrainClass = static_cast<std::size_t>(tile.terrain);
                    std::uint32_t texIndex16 = 0xFFFFu;  // none -> palette fallback
                    if (terrainClass < texSceneIdx.size() &&
                        texSceneIdx[terrainClass] != 0xFFFFFFFFu) {
                        texIndex16 = texSceneIdx[terrainClass] & 0xFFFFu;
                    }

                    HexTileInstance inst{};
                    inst.centerXZ[0] = center.x;
                    inst.centerXZ[1] = center.z;
                    inst.classFlags = static_cast<std::uint32_t>(tile.terrain) |
                                      (static_cast<std::uint32_t>(tile.flags) << 8u) |
                                      (texIndex16 << 16u);
                    inst.ownElevY = ownY;
                    inst.hexSize = H;

                    // Neighbour heights and terrain types indexed by hex edge (edge k
                    // faces physical direction 60k deg). Default to own values so map-
                    // edge tiles flatten outward and blend to themselves.
                    for (int k = 0; k < 6; ++k) {
                        inst.neighborElevY[k] = ownY;
                    }
                    const std::uint32_t ownTerrainNibble = static_cast<std::uint32_t>(tile.terrain) & 0xFu;
                    inst.neighborTerrainPacked =
                        ownTerrainNibble | (ownTerrainNibble << 4u) | (ownTerrainNibble << 8u) |
                        (ownTerrainNibble << 12u) | (ownTerrainNibble << 16u) | (ownTerrainNibble << 20u);

                    for (int dir = 0; dir < 6; ++dir) {
                        int nc = 0;
                        int nr = 0;
                        if (!tileNeighbor(map, static_cast<int>(col), static_cast<int>(row), dir, nc, nr)) {
                            continue;
                        }
                        const odai::math::Vector3 nCenter =
                            tileCenterWorld(map, static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr));
                        float degrees = odai::math::degrees(std::atan2(nCenter.z - center.z, nCenter.x - center.x));
                        if (degrees < 0.0f) {
                            degrees += 360.0f;
                        }
                        const int edge = static_cast<int>(std::lround(degrees / 60.0f)) % 6;
                        inst.neighborElevY[edge] =
                            tileHeightY(static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr));
                        const std::uint32_t nTerrainNibble =
                            static_cast<std::uint32_t>(map.at(static_cast<std::uint32_t>(nc),
                                                               static_cast<std::uint32_t>(nr)).terrain) & 0xFu;
                        const std::uint32_t shift = static_cast<std::uint32_t>(edge) * 4u;
                        inst.neighborTerrainPacked =
                            (inst.neighborTerrainPacked & ~(0xFu << shift)) | (nTerrainNibble << shift);
                    }

                    std::uint32_t feature = HexDetailFeature::HexDetail_Rough;
                    if (options.demoStripMines && tile.terrain == TerrainType::Hills &&
                        (((col * 3u) + (row * 5u)) % 17u) == 0u) {
                        feature = HexDetailFeature::HexDetail_StripMine;
                    }
                    inst.detailParams[0] = static_cast<float>(feature);
                    inst.detailParams[1] = terrainDetailAmplitudeFrac(tile.terrain) * H * exaggeration;
                    inst.detailParams[2] = 0.95f * kInradiusFrac * H;  // window end < inradius -> crack-free
                    inst.detailParams[3] = 1.6f / H;                   // detail noise frequency (broad/smooth)
                    data.instances.push_back(inst);

                    loX = std::min(loX, center.x - H);
                    hiX = std::max(hiX, center.x + H);
                    loZ = std::min(loZ, center.z - H);
                    hiZ = std::max(hiZ, center.z + H);
                }
            }

            const std::uint32_t instanceCount = static_cast<std::uint32_t>(data.instances.size()) - firstInstance;
            if (instanceCount == 0u) {
                continue;
            }
            HexTerrainPage page{};
            page.firstInstance = firstInstance;
            page.instanceCount = instanceCount;
            page.boundsMin[0] = loX; page.boundsMin[1] = pageMinY; page.boundsMin[2] = loZ;
            page.boundsMax[0] = hiX; page.boundsMax[1] = pageMaxY; page.boundsMax[2] = hiZ;
            data.pages.push_back(page);
        }
    }

    return data;
}

}  // namespace odai::game
