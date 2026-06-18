#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "game/strategy_map.h"
#include "game/strategy_map_io.h"
#include "game/strategy_map_mesh.h"
#include "import/imported_scene.h"

namespace {

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[strategy map test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectEqualU32(std::uint32_t actual, std::uint32_t expected, const char* message) {
    if (actual != expected) {
        std::cerr << "[strategy map test] FAIL: " << message
                  << " (expected " << expected << ", got " << actual << ")\n";
        ++g_failures;
    }
}

void expectNear(float actual, float expected, float epsilon, const char* message) {
    if (std::fabs(actual - expected) > epsilon) {
        std::cerr << "[strategy map test] FAIL: " << message
                  << " (expected " << expected << ", got " << actual << ")\n";
        ++g_failures;
    }
}

odai::game::StrategyMap makeSampleMap() {
    using namespace odai::game;
    StrategyMap map{};
    map.resize(5, 4);
    map.hexSize = 10.0f;
    map.elevationStep = 4.0f;
    for (std::uint32_t row = 0; row < map.height; ++row) {
        for (std::uint32_t col = 0; col < map.width; ++col) {
            MapTile& tile = map.at(col, row);
            tile.terrain = static_cast<TerrainType>((col + row) % static_cast<std::uint32_t>(TerrainType::Count));
            tile.elevation = static_cast<std::int16_t>(col - row);
            tile.owner = static_cast<std::uint8_t>(row % 3u);
            tile.visibility = TileVisibility::Explored;
        }
    }
    map.at(1, 1).flags = TileFlag_River | TileFlag_Road;
    map.at(2, 2).flags = TileFlag_Border;
    map.settlements.push_back(Settlement{"Testburg", 2, 1, 2, 1});
    map.settlements.push_back(Settlement{"Hexford", 3, 3, 1, 2});
    return map;
}

void testModelIndexingAndBounds() {
    using namespace odai::game;
    StrategyMap map{};
    map.resize(6, 5);
    expectEqualU32(static_cast<std::uint32_t>(map.tiles.size()), 30u, "resize allocates width*height tiles");
    expectTrue(map.inBounds(0, 0), "origin is in bounds");
    expectTrue(map.inBounds(5, 4), "max corner is in bounds");
    expectTrue(!map.inBounds(6, 4), "column past width is out of bounds");
    expectTrue(!map.inBounds(0, -1), "negative row is out of bounds");
    expectEqualU32(static_cast<std::uint32_t>(map.index(3, 2)), 15u, "row-major index is correct");
}

void testHexNeighbors() {
    using namespace odai::game;
    StrategyMap map{};
    map.resize(5, 5);
    // Interior tiles have all six neighbors regardless of row parity.
    expectEqualU32(static_cast<std::uint32_t>(tileNeighborCount(map, 2, 2)), 6u, "even-row interior tile has 6 neighbors");
    expectEqualU32(static_cast<std::uint32_t>(tileNeighborCount(map, 2, 1)), 6u, "odd-row interior tile has 6 neighbors");
    // Corners have fewer.
    expectTrue(tileNeighborCount(map, 0, 0) < 6, "corner tile has fewer than 6 neighbors");

    // Neighbor relation is symmetric: if B is a neighbor of A, A is a neighbor of B.
    bool symmetric = true;
    for (int direction = 0; direction < 6; ++direction) {
        int nc = 0;
        int nr = 0;
        if (!tileNeighbor(map, 2, 2, direction, nc, nr)) {
            continue;
        }
        bool found = false;
        for (int back = 0; back < 6; ++back) {
            int bc = 0;
            int br = 0;
            if (tileNeighbor(map, nc, nr, back, bc, br) && bc == 2 && br == 2) {
                found = true;
                break;
            }
        }
        symmetric = symmetric && found;
    }
    expectTrue(symmetric, "hex neighbor relation is symmetric");
}

void testHexGeometry() {
    using namespace odai::game;
    StrategyMap map{};
    map.resize(3, 3);
    map.hexSize = 10.0f;
    map.elevationStep = 5.0f;
    map.at(1, 1).elevation = 2;

    const odai::math::Vector3 center = tileCenterWorld(map, 1, 1);
    expectNear(center.y, 10.0f, 1e-4f, "tile world Y follows elevation * step");

    // Every corner sits hexSize away from the center in the XZ plane.
    for (int corner = 0; corner < 6; ++corner) {
        const odai::math::Vector3 c = tileCornerWorld(map, 1, 1, corner);
        const float dx = c.x - center.x;
        const float dz = c.z - center.z;
        expectNear(std::sqrt((dx * dx) + (dz * dz)), map.hexSize, 1e-3f, "hex corner is circumradius from center");
        expectNear(c.y, center.y, 1e-4f, "hex corner shares tile elevation");
    }
}

void testSerializationRoundTrip() {
    using namespace odai::game;
    const StrategyMap original = makeSampleMap();

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "odai_strategy_map_roundtrip.smap";
    expectTrue(saveStrategyMap(original, path), "save strategy map succeeds");

    StrategyMap loaded{};
    expectTrue(loadStrategyMap(path, loaded), "load strategy map succeeds");

    expectEqualU32(loaded.width, original.width, "round-trip preserves width");
    expectEqualU32(loaded.height, original.height, "round-trip preserves height");
    expectNear(loaded.hexSize, original.hexSize, 1e-6f, "round-trip preserves hexSize");
    expectNear(loaded.elevationStep, original.elevationStep, 1e-6f, "round-trip preserves elevationStep");
    expectEqualU32(static_cast<std::uint32_t>(loaded.tiles.size()),
                   static_cast<std::uint32_t>(original.tiles.size()), "round-trip preserves tile count");

    bool tilesMatch = loaded.tiles.size() == original.tiles.size();
    for (std::size_t i = 0; i < loaded.tiles.size() && tilesMatch; ++i) {
        const MapTile& a = original.tiles[i];
        const MapTile& b = loaded.tiles[i];
        tilesMatch = a.terrain == b.terrain && a.elevation == b.elevation &&
                     a.flags == b.flags && a.owner == b.owner && a.visibility == b.visibility;
    }
    expectTrue(tilesMatch, "round-trip preserves every tile field");

    bool settlementsMatch = loaded.settlements.size() == original.settlements.size();
    for (std::size_t i = 0; i < loaded.settlements.size() && settlementsMatch; ++i) {
        const Settlement& a = original.settlements[i];
        const Settlement& b = loaded.settlements[i];
        settlementsMatch = a.name == b.name && a.col == b.col && a.row == b.row &&
                           a.tier == b.tier && a.owner == b.owner;
    }
    expectTrue(settlementsMatch, "round-trip preserves settlements");

    std::error_code removeError;
    std::filesystem::remove(path, removeError);
}

void testLoadRejectsGarbage() {
    using namespace odai::game;
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "odai_strategy_map_garbage.smap";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        const char junk[8] = {'n', 'o', 'p', 'e', 0, 0, 0, 0};
        output.write(junk, sizeof(junk));
    }
    StrategyMap loaded{};
    expectTrue(!loadStrategyMap(path, loaded), "loading a non-strategy-map file fails cleanly");
    expectTrue(!getStrategyMapLastError().empty(), "failed load sets an error message");

    std::error_code removeError;
    std::filesystem::remove(path, removeError);
}

void testMesherProducesRenderableScene() {
    using namespace odai::game;
    const StrategyMap map = makeSampleMap();
    const odai::importer::ImportedScene scene = buildStrategyMapScene(map);

    expectTrue(scene.sourceTag == "strategy_map", "mesher tags the scene as strategy_map");
    expectTrue(!scene.packedVertices.empty(), "mesher emits packed vertices");
    expectTrue(!scene.packedIndices.empty(), "mesher emits packed indices");
    expectTrue(scene.packedDraws.size() >= 2u, "mesher emits terrain, overlay, and settlement draws");
    expectTrue(scene.packedIndices.size() % 3u == 0u, "mesher emits whole triangles");

    // Indices must reference valid vertices.
    bool indicesValid = true;
    for (const std::uint32_t index : scene.packedIndices) {
        if (index >= scene.packedVertices.size()) {
            indicesValid = false;
            break;
        }
    }
    expectTrue(indicesValid, "every packed index references a valid vertex");

    // Bounds must be non-degenerate so the camera can frame the map.
    expectTrue(scene.boundsMax[0] > scene.boundsMin[0], "scene has non-zero X extent");
    expectTrue(scene.boundsMax[2] > scene.boundsMin[2], "scene has non-zero Z extent");

    // Untextured vertex color is what the shader displays.
    bool allUntextured = true;
    bool colorVariety = false;
    const auto& first = scene.packedVertices.front();
    for (const auto& vertex : scene.packedVertices) {
        if (vertex.textureIndex != 0xffffffffu) {
            allUntextured = false;
        }
        if (vertex.color[0] != first.color[0] || vertex.color[1] != first.color[1] || vertex.color[2] != first.color[2]) {
            colorVariety = true;
        }
    }
    expectTrue(allUntextured, "mesher leaves vertices untextured (vertex-color path)");
    expectTrue(colorVariety, "mesher colors tiles differently by terrain");
}

}  // namespace

int main() {
    testModelIndexingAndBounds();
    testHexNeighbors();
    testHexGeometry();
    testSerializationRoundTrip();
    testLoadRejectsGarbage();
    testMesherProducesRenderableScene();

    if (g_failures != 0) {
        std::cerr << "[strategy map test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[strategy map test] all checks passed\n";
    return 0;
}
