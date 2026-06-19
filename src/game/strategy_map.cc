#include "game/strategy_map.h"

#include <cassert>
#include <cmath>

namespace odai::game {

namespace {

constexpr float kSqrt3 = 1.7320508075688772f;

// redblobgames odd-r offset neighbor tables, indexed by row parity then direction.
// Each entry is {dcol, drow}.
constexpr std::array<std::array<std::array<int, 2>, 6>, 2> kOddRNeighbors = {{
    // Even rows.
    {{{{+1, 0}}, {{0, -1}}, {{-1, -1}}, {{-1, 0}}, {{-1, +1}}, {{0, +1}}}},
    // Odd rows.
    {{{{+1, 0}}, {{+1, -1}}, {{0, -1}}, {{-1, 0}}, {{0, +1}}, {{+1, +1}}}},
}};

}  // namespace

TileColor terrainColor(TerrainType terrain) {
    switch (terrain) {
        case TerrainType::Ocean:      return {0.10f, 0.24f, 0.50f};
        case TerrainType::Coast:      return {0.20f, 0.46f, 0.68f};
        case TerrainType::Grassland:  return {0.32f, 0.54f, 0.24f};
        case TerrainType::Plains:     return {0.56f, 0.60f, 0.30f};
        case TerrainType::Forest:     return {0.18f, 0.38f, 0.20f};
        case TerrainType::Jungle:     return {0.13f, 0.45f, 0.16f};
        case TerrainType::Hills:      return {0.46f, 0.42f, 0.26f};
        case TerrainType::Mountains:  return {0.46f, 0.44f, 0.43f};
        case TerrainType::Desert:     return {0.82f, 0.73f, 0.44f};
        case TerrainType::Tundra:     return {0.54f, 0.57f, 0.50f};
        case TerrainType::Snow:       return {0.90f, 0.92f, 0.96f};
        case TerrainType::Count:      break;
    }
    return {1.0f, 0.0f, 1.0f};  // Obvious error color.
}

bool terrainIsWater(TerrainType terrain) {
    return terrain == TerrainType::Ocean || terrain == TerrainType::Coast;
}

void StrategyMap::resize(std::uint32_t newWidth, std::uint32_t newHeight) {
    width = newWidth;
    height = newHeight;
    tiles.assign(static_cast<std::size_t>(newWidth) * static_cast<std::size_t>(newHeight), MapTile{});
}

bool StrategyMap::inBounds(int col, int row) const {
    return col >= 0 && row >= 0 &&
           static_cast<std::uint32_t>(col) < width &&
           static_cast<std::uint32_t>(row) < height;
}

std::size_t StrategyMap::index(std::uint32_t col, std::uint32_t row) const {
    return (static_cast<std::size_t>(row) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(col);
}

const MapTile& StrategyMap::at(std::uint32_t col, std::uint32_t row) const {
    assert(col < width && row < height);
    return tiles[index(col, row)];
}

MapTile& StrategyMap::at(std::uint32_t col, std::uint32_t row) {
    assert(col < width && row < height);
    return tiles[index(col, row)];
}

odai::math::Vector3 tileCenterWorld(const StrategyMap& map, std::uint32_t col, std::uint32_t row) {
    const float rowOffset = (row & 1u) ? 0.5f : 0.0f;
    const float x = map.hexSize * kSqrt3 * (static_cast<float>(col) + rowOffset);
    const float z = map.hexSize * 1.5f * static_cast<float>(row);
    const float y = static_cast<float>(map.at(col, row).elevation) * map.elevationStep;
    return odai::math::Vector3{x, y, z};
}

odai::math::Vector3 tileCornerWorld(const StrategyMap& map, std::uint32_t col, std::uint32_t row, int corner) {
    const odai::math::Vector3 center = tileCenterWorld(map, col, row);
    const float angleDegrees = (60.0f * static_cast<float>(corner)) - 30.0f;
    const float angle = odai::math::radians(angleDegrees);
    return odai::math::Vector3{
        center.x + (map.hexSize * std::cos(angle)),
        center.y,
        center.z + (map.hexSize * std::sin(angle)),
    };
}

std::array<int, 2> hexNeighborOffset(int row, int direction) {
    const int parity = row & 1;
    const int wrapped = ((direction % 6) + 6) % 6;
    return kOddRNeighbors[static_cast<std::size_t>(parity)][static_cast<std::size_t>(wrapped)];
}

bool tileNeighbor(const StrategyMap& map, int col, int row, int direction, int& outCol, int& outRow) {
    const std::array<int, 2> offset = hexNeighborOffset(row, direction);
    outCol = col + offset[0];
    outRow = row + offset[1];
    return map.inBounds(outCol, outRow);
}

int tileNeighborCount(const StrategyMap& map, int col, int row) {
    int count = 0;
    for (int direction = 0; direction < 6; ++direction) {
        int neighborCol = 0;
        int neighborRow = 0;
        if (tileNeighbor(map, col, row, direction, neighborCol, neighborRow)) {
            ++count;
        }
    }
    return count;
}

}  // namespace odai::game
