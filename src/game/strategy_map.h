#pragma once

#include "math/math.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Strategy-map game data model.
// Responsible for: the topology, terrain, and markers of an explorable strategic
// map (Civ / Total War style). Pure CPU data with no Vulkan or renderer types.
// Topology is a pointy-top hex grid stored in odd-r offset coordinates.
namespace odai::game {

enum class TerrainType : std::uint8_t {
    Ocean = 0,
    Coast,
    Grassland,
    Plains,
    Forest,
    Jungle,
    Hills,
    Mountains,
    Desert,
    Tundra,
    Snow,
    Count
};

struct TileColor {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

// Base albedo for a terrain type, used by the mesher for per-vertex color.
TileColor terrainColor(TerrainType terrain);

// True for terrain that should read as water (ocean / shallow coast).
bool terrainIsWater(TerrainType terrain);

// Per-tile overlay flags. Rivers/roads/borders are intentionally simple bits so
// the data model stays topology-agnostic; the mesher decides how to draw them.
enum TileFlagBits : std::uint8_t {
    TileFlag_None = 0u,
    TileFlag_River = 1u << 0,
    TileFlag_Road = 1u << 1,
    TileFlag_Border = 1u << 2,
};

// Fog-of-war state. Stored per tile so a future gameplay layer can drive
// visibility without changing the map format.
enum class TileVisibility : std::uint8_t {
    Hidden = 0,
    Explored = 1,
    Visible = 2,
};

struct MapTile {
    TerrainType terrain = TerrainType::Ocean;
    std::int16_t elevation = 0;  // discrete level; world Y = elevation * elevationStep.
    std::uint8_t flags = 0;      // Bitwise OR of TileFlagBits.
    std::uint8_t owner = 0;      // Territory owner id; 0 == unowned.
    TileVisibility visibility = TileVisibility::Hidden;
};

struct Settlement {
    std::string name;
    std::uint32_t col = 0;
    std::uint32_t row = 0;
    std::uint8_t tier = 1;   // 1 == village ... larger == city.
    std::uint8_t owner = 0;  // Matches MapTile::owner for the controlling faction.
};

struct StrategyMap {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    float hexSize = 64.0f;        // Circumradius (center-to-corner) in world units.
    float elevationStep = 16.0f;  // World units per elevation level.
    std::vector<MapTile> tiles;   // width * height, row-major odd-r offset coordinates.
    std::vector<Settlement> settlements;

    // Allocate width * height default tiles.
    void resize(std::uint32_t newWidth, std::uint32_t newHeight);

    [[nodiscard]] bool inBounds(int col, int row) const;
    [[nodiscard]] std::size_t index(std::uint32_t col, std::uint32_t row) const;
    [[nodiscard]] const MapTile& at(std::uint32_t col, std::uint32_t row) const;
    [[nodiscard]] MapTile& at(std::uint32_t col, std::uint32_t row);
};

// Hex geometry helpers (pointy-top, odd-r). The map lies in the world XZ plane
// with +Y up; elevation contributes to Y.
[[nodiscard]] odai::math::Vector3 tileCenterWorld(const StrategyMap& map, std::uint32_t col, std::uint32_t row);

// World-space position of hex corner [0, 6) for the given tile, at the tile's
// elevation. Corner 0 is the top-right vertex, advancing counter-clockwise.
[[nodiscard]] odai::math::Vector3 tileCornerWorld(const StrategyMap& map, std::uint32_t col, std::uint32_t row, int corner);

// Offset (dcol, drow) to neighbor in direction [0, 6) for a tile in the given row.
[[nodiscard]] std::array<int, 2> hexNeighborOffset(int row, int direction);

// Resolve a neighbor coordinate; returns false if out of bounds.
[[nodiscard]] bool tileNeighbor(const StrategyMap& map, int col, int row, int direction, int& outCol, int& outRow);

// Count of in-bounds neighbors (6 for interior tiles, fewer at edges).
[[nodiscard]] int tileNeighborCount(const StrategyMap& map, int col, int row);

// Odd-r offset <-> axial conversion. Axial {q, r} is the canonical hex coordinate
// the distance formula operates on; exposed for tests and future pathfinding.
[[nodiscard]] std::array<int, 2> offsetToAxial(int col, int row);
[[nodiscard]] std::array<int, 2> axialToOffset(int q, int r);

// Grid distance in hex steps between two odd-r offset tiles (cube distance).
[[nodiscard]] int hexDistance(int colA, int rowA, int colB, int rowB);

}  // namespace odai::game
