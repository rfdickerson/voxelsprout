// Offline generator for a sample strategic hex map. Deterministic (integer hash
// noise, no RNG state) so output is reproducible. Writes the native .smap file
// the runtime loads, and a .bin ImportedScene the existing viewer can render
// directly via ODAI_IMPORTED_SCENE.

#include "game/strategy_map.h"
#include "game/strategy_map_io.h"
#include "game/strategy_map_mesh.h"
#include "import/imported_scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace odai::game;

std::uint32_t hashCoords(std::int32_t x, std::int32_t y, std::uint32_t seed) {
    std::uint32_t h = seed + 0x9E3779B9u;
    h ^= static_cast<std::uint32_t>(x) * 0x85EBCA77u;
    h = (h ^ (h >> 15)) * 0xC2B2AE3Du;
    h ^= static_cast<std::uint32_t>(y) * 0x27D4EB2Fu;
    h = (h ^ (h >> 13)) * 0x165667B1u;
    return h ^ (h >> 16);
}

float hashFloat(std::int32_t x, std::int32_t y, std::uint32_t seed) {
    return static_cast<float>(hashCoords(x, y, seed) & 0xFFFFFFu) / static_cast<float>(0x1000000u);
}

float smoothstep(float t) {
    return t * t * (3.0f - (2.0f * t));
}

// Bilinearly-interpolated value noise at a continuous grid position.
float valueNoise(float x, float y, std::uint32_t seed) {
    const float fx = std::floor(x);
    const float fy = std::floor(y);
    const auto ix = static_cast<std::int32_t>(fx);
    const auto iy = static_cast<std::int32_t>(fy);
    const float tx = smoothstep(x - fx);
    const float ty = smoothstep(y - fy);
    const float v00 = hashFloat(ix, iy, seed);
    const float v10 = hashFloat(ix + 1, iy, seed);
    const float v01 = hashFloat(ix, iy + 1, seed);
    const float v11 = hashFloat(ix + 1, iy + 1, seed);
    const float top = v00 + ((v10 - v00) * tx);
    const float bottom = v01 + ((v11 - v01) * tx);
    return top + ((bottom - top) * ty);
}

float fbm(float x, float y, std::uint32_t seed) {
    float sum = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    for (int octave = 0; octave < 4; ++octave) {
        sum += valueNoise(x * frequency, y * frequency, seed + static_cast<std::uint32_t>(octave) * 101u) * amplitude;
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    return sum;
}

TerrainType classifyTerrain(std::int16_t elevation, float latitude01, float moisture) {
    if (elevation <= 0) {
        return TerrainType::Ocean;
    }
    if (elevation == 1) {
        return TerrainType::Coast;
    }
    const float polar = std::abs((latitude01 * 2.0f) - 1.0f);  // 0 at equator, 1 at poles.
    if (elevation >= 6) {
        return (polar > 0.45f || elevation >= 7) ? TerrainType::Snow : TerrainType::Mountains;
    }
    if (elevation >= 5) {
        return TerrainType::Mountains;
    }
    if (elevation >= 4) {
        return TerrainType::Hills;
    }
    if (polar > 0.78f) {
        return TerrainType::Snow;
    }
    if (polar > 0.62f) {
        return TerrainType::Tundra;
    }
    if (polar < 0.18f && moisture < 0.42f) {
        return TerrainType::Desert;
    }
    // Tropical wet lowlands read as jungle; temperate wet areas as forest.
    if (polar < 0.28f && moisture > 0.55f) {
        return TerrainType::Jungle;
    }
    if (moisture > 0.62f) {
        return TerrainType::Forest;
    }
    return (moisture > 0.45f) ? TerrainType::Grassland : TerrainType::Plains;
}

// Greedy hex walk from one tile toward another by minimizing world distance.
std::vector<std::array<std::uint32_t, 2>> hexWalk(
    const StrategyMap& map,
    std::uint32_t startCol, std::uint32_t startRow,
    std::uint32_t goalCol, std::uint32_t goalRow) {
    std::vector<std::array<std::uint32_t, 2>> path;
    int col = static_cast<int>(startCol);
    int row = static_cast<int>(startRow);
    const odai::math::Vector3 goal = tileCenterWorld(map, goalCol, goalRow);
    for (int step = 0; step < static_cast<int>(map.width + map.height) * 2; ++step) {
        path.push_back({static_cast<std::uint32_t>(col), static_cast<std::uint32_t>(row)});
        if (col == static_cast<int>(goalCol) && row == static_cast<int>(goalRow)) {
            break;
        }
        int bestCol = col;
        int bestRow = row;
        float bestDistance = std::numeric_limits<float>::max();
        for (int direction = 0; direction < 6; ++direction) {
            int nc = 0;
            int nr = 0;
            if (!tileNeighbor(map, col, row, direction, nc, nr)) {
                continue;
            }
            const odai::math::Vector3 center = tileCenterWorld(map, static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr));
            const float dx = center.x - goal.x;
            const float dz = center.z - goal.z;
            const float distance = (dx * dx) + (dz * dz);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestCol = nc;
                bestRow = nr;
            }
        }
        if (bestCol == col && bestRow == row) {
            break;  // Local minimum; stop.
        }
        col = bestCol;
        row = bestRow;
    }
    return path;
}

}  // namespace

int main(int argc, char** argv) {
    std::uint32_t width = 80;
    std::uint32_t height = 60;
    std::uint32_t seed = 1337u;
    std::string smapPath = "strategy_map.smap";
    std::string scenePath = "strategy_map_scene.bin";

    if (argc > 1) {
        smapPath = argv[1];
    }
    if (argc > 2) {
        scenePath = argv[2];
    }
    if (argc > 3) {
        width = static_cast<std::uint32_t>(std::max(2, std::atoi(argv[3])));
    }
    if (argc > 4) {
        height = static_cast<std::uint32_t>(std::max(2, std::atoi(argv[4])));
    }
    if (argc > 5) {
        seed = static_cast<std::uint32_t>(std::strtoul(argv[5], nullptr, 10));
    }

    StrategyMap map{};
    map.resize(width, height);
    map.hexSize = 64.0f;
    map.elevationStep = 28.0f;

    // Elevation + terrain from layered noise, with an island falloff so the map
    // is framed by ocean (readable strategic landmass).
    const float noiseScale = 0.14f;
    for (std::uint32_t row = 0; row < height; ++row) {
        for (std::uint32_t col = 0; col < width; ++col) {
            const float nx = static_cast<float>(col) * noiseScale;
            const float nz = static_cast<float>(row) * noiseScale;
            float base = fbm(nx, nz, seed);

            const float u = (static_cast<float>(col) / static_cast<float>(width - 1)) - 0.5f;
            const float v = (static_cast<float>(row) / static_cast<float>(height - 1)) - 0.5f;
            const float falloff = 1.0f - std::min(1.0f, (std::sqrt((u * u) + (v * v)) * 1.9f));
            base = (base * 0.65f) + (falloff * 0.55f) - 0.25f;

            const auto elevation = static_cast<std::int16_t>(std::lround(base * 9.0f));
            const float latitude01 = static_cast<float>(row) / static_cast<float>(height - 1);
            const float moisture = fbm(nx + 31.7f, nz - 12.3f, seed + 7u);

            MapTile& tile = map.at(col, row);
            tile.elevation = std::clamp<std::int16_t>(elevation, -2, 8);
            tile.terrain = classifyTerrain(tile.elevation, latitude01, moisture);
            tile.visibility = TileVisibility::Hidden;
        }
    }

    // Rivers: from a few high land tiles, walk to the lowest neighbor until water.
    int riversPlaced = 0;
    for (std::uint32_t row = 1; row < height - 1 && riversPlaced < 6; ++row) {
        for (std::uint32_t col = 1; col < width - 1 && riversPlaced < 6; ++col) {
            const MapTile& tile = map.at(col, row);
            if (tile.elevation < 5 || (hashCoords(static_cast<int>(col), static_cast<int>(row), seed + 99u) % 23u) != 0u) {
                continue;
            }
            int currentCol = static_cast<int>(col);
            int currentRow = static_cast<int>(row);
            for (int step = 0; step < static_cast<int>(width + height); ++step) {
                map.at(static_cast<std::uint32_t>(currentCol), static_cast<std::uint32_t>(currentRow)).flags |= TileFlag_River;
                int bestCol = currentCol;
                int bestRow = currentRow;
                std::int16_t bestElevation = map.at(static_cast<std::uint32_t>(currentCol), static_cast<std::uint32_t>(currentRow)).elevation;
                for (int direction = 0; direction < 6; ++direction) {
                    int nc = 0;
                    int nr = 0;
                    if (!tileNeighbor(map, currentCol, currentRow, direction, nc, nr)) {
                        continue;
                    }
                    const std::int16_t elevation = map.at(static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr)).elevation;
                    if (elevation < bestElevation) {
                        bestElevation = elevation;
                        bestCol = nc;
                        bestRow = nr;
                    }
                }
                if (bestCol == currentCol && bestRow == currentRow) {
                    break;
                }
                currentCol = bestCol;
                currentRow = bestRow;
                if (terrainIsWater(map.at(static_cast<std::uint32_t>(currentCol), static_cast<std::uint32_t>(currentRow)).terrain)) {
                    break;
                }
            }
            ++riversPlaced;
        }
    }

    // Settlements on temperate land, spaced apart in tile distance.
    const std::array<const char*, 8> kNames = {
        "Aldenmoor", "Brackton", "Caer Lys", "Duneholt", "Eastmere", "Fenwick", "Grangate", "Highford"};
    const int spacing = static_cast<int>(std::max<std::uint32_t>(4u, std::min(width, height) / 4u));
    for (std::uint32_t row = 0; row < height && map.settlements.size() < kNames.size(); ++row) {
        for (std::uint32_t col = 0; col < width && map.settlements.size() < kNames.size(); ++col) {
            const MapTile& tile = map.at(col, row);
            const bool habitable =
                tile.elevation >= 2 && tile.elevation <= 4 &&
                (tile.terrain == TerrainType::Grassland || tile.terrain == TerrainType::Plains);
            if (!habitable || (hashCoords(static_cast<int>(col), static_cast<int>(row), seed + 5u) % 3u) != 0u) {
                continue;
            }
            bool tooClose = false;
            for (const Settlement& existing : map.settlements) {
                const int dc = static_cast<int>(existing.col) - static_cast<int>(col);
                const int dr = static_cast<int>(existing.row) - static_cast<int>(row);
                if ((dc * dc) + (dr * dr) < spacing * spacing) {
                    tooClose = true;
                    break;
                }
            }
            if (tooClose) {
                continue;
            }
            Settlement settlement{};
            settlement.name = kNames[map.settlements.size()];
            settlement.col = col;
            settlement.row = row;
            settlement.tier = static_cast<std::uint8_t>(1u + (map.settlements.size() % 3u));
            settlement.owner = static_cast<std::uint8_t>(1u + (map.settlements.size() % 4u));
            map.settlements.push_back(settlement);
        }
    }

    // Territory + borders: each land tile is owned by its nearest settlement.
    if (!map.settlements.empty()) {
        for (std::uint32_t row = 0; row < height; ++row) {
            for (std::uint32_t col = 0; col < width; ++col) {
                MapTile& tile = map.at(col, row);
                if (terrainIsWater(tile.terrain)) {
                    continue;
                }
                const odai::math::Vector3 center = tileCenterWorld(map, col, row);
                float bestDistance = std::numeric_limits<float>::max();
                std::uint8_t bestOwner = 0;
                for (const Settlement& settlement : map.settlements) {
                    const odai::math::Vector3 site = tileCenterWorld(map, settlement.col, settlement.row);
                    const float dx = site.x - center.x;
                    const float dz = site.z - center.z;
                    const float distance = (dx * dx) + (dz * dz);
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        bestOwner = settlement.owner;
                    }
                }
                tile.owner = bestOwner;
            }
        }
        for (std::uint32_t row = 0; row < height; ++row) {
            for (std::uint32_t col = 0; col < width; ++col) {
                MapTile& tile = map.at(col, row);
                if (tile.owner == 0) {
                    continue;
                }
                for (int direction = 0; direction < 6; ++direction) {
                    int nc = 0;
                    int nr = 0;
                    if (!tileNeighbor(map, static_cast<int>(col), static_cast<int>(row), direction, nc, nr)) {
                        continue;
                    }
                    if (map.at(static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr)).owner != tile.owner) {
                        tile.flags |= TileFlag_Border;
                        break;
                    }
                }
            }
        }
    }

    // Roads connect each settlement to the next, skipping water tiles.
    for (std::size_t i = 0; i + 1 < map.settlements.size(); ++i) {
        const Settlement& a = map.settlements[i];
        const Settlement& b = map.settlements[i + 1];
        for (const std::array<std::uint32_t, 2>& step : hexWalk(map, a.col, a.row, b.col, b.row)) {
            MapTile& tile = map.at(step[0], step[1]);
            if (!terrainIsWater(tile.terrain)) {
                tile.flags |= TileFlag_Road;
            }
        }
    }

    if (!saveStrategyMap(map, smapPath)) {
        std::cerr << "strategy_map_gen: " << getStrategyMapLastError() << "\n";
        return 1;
    }

    const odai::importer::ImportedScene scene = buildStrategyMapScene(map);
    if (!odai::importer::saveImportedScene(scene, scenePath)) {
        std::cerr << "strategy_map_gen: failed to save scene: " << odai::importer::getImportedSceneLastError() << "\n";
        return 1;
    }

    std::cout << "strategy_map_gen: wrote " << smapPath << " (" << width << "x" << height
              << ", " << map.settlements.size() << " settlements) and " << scenePath
              << " (" << scene.packedVertices.size() << " vertices, "
              << scene.packedDraws.size() << " draws)\n";
    return 0;
}
