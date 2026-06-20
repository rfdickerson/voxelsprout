#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "game/strategy_hex_terrain.h"
#include "game/strategy_map.h"
#include "game/strategy_map_io.h"
#include "game/strategy_map_mesh.h"
#include "game/units.h"
#include "import/hex_terrain_data.h"
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

void expectEqualInt(int actual, int expected, const char* message) {
    if (actual != expected) {
        std::cerr << "[strategy map test] FAIL: " << message
                  << " (expected " << expected << ", got " << actual << ")\n";
        ++g_failures;
    }
}

// A featureless inland map: every tile grassland at elevation 0, no roads, no
// settlements. Unit-rule tests start from this and add only what they exercise.
odai::game::StrategyMap makeFlatLandMap(std::uint32_t width, std::uint32_t height) {
    using namespace odai::game;
    StrategyMap map{};
    map.resize(width, height);
    map.hexSize = 10.0f;
    for (MapTile& tile : map.tiles) {
        tile.terrain = TerrainType::Grassland;
        tile.elevation = 0;
        tile.flags = 0;
        tile.owner = 0;
        tile.visibility = TileVisibility::Visible;
    }
    return map;
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

void testMesherChunking() {
    using namespace odai::game;
    const StrategyMap map = makeSampleMap();  // 5x4
    StrategyMapMeshOptions options{};
    options.chunkSize = 2u;  // ceil(5/2)=3 cols * ceil(4/2)=2 rows = 6 chunks
    const odai::importer::ImportedScene scene = buildStrategyMapScene(map, options);

    // 6 terrain chunk draws + 1 overlay page (grid + settlements).
    expectEqualU32(scene.sourceLandscapeCellCount, 6u, "one terrain draw per chunk");
    expectEqualU32(static_cast<std::uint32_t>(scene.pageRanges.size()), 7u,
                   "6 chunk pages + 1 overlay page");

    // Renderer invariant: every non-empty packed draw is covered by exactly one page,
    // else the renderer discards all pages and culling is lost.
    bool everyDrawCovered = true;
    for (std::uint32_t d = 0; d < scene.packedDraws.size(); ++d) {
        if (scene.packedDraws[d].indexCount == 0u) continue;
        bool covered = false;
        for (const auto& page : scene.pageRanges) {
            if (d >= page.firstDraw && d < page.firstDraw + page.drawCount) { covered = true; break; }
        }
        everyDrawCovered = everyDrawCovered && covered;
    }
    expectTrue(everyDrawCovered, "every packed draw is covered by a page range");

    // Chunk pages must have non-degenerate XZ bounds for frustum culling.
    bool boundsValid = true;
    for (std::uint32_t i = 0; i < 6u && i < scene.pageRanges.size(); ++i) {
        const auto& p = scene.pageRanges[i];
        boundsValid = boundsValid && p.boundsMax[0] > p.boundsMin[0] && p.boundsMax[2] > p.boundsMin[2];
    }
    expectTrue(boundsValid, "chunk page bounds are non-degenerate");
}

void testMesherFlatMode() {
    using namespace odai::game;
    const StrategyMap map = makeSampleMap();
    StrategyMapMeshOptions base{};
    base.drawGridOverlay = false;   // isolate terrain so all verts are tile tops
    base.drawSettlements = false;

    StrategyMapMeshOptions flat = base;
    flat.extruded = false;
    const odai::importer::ImportedScene flatScene = buildStrategyMapScene(map, flat);

    StrategyMapMeshOptions relief = base;
    relief.extruded = true;
    const odai::importer::ImportedScene reliefScene = buildStrategyMapScene(map, relief);

    expectTrue(flatScene.packedVertices.size() < reliefScene.packedVertices.size(),
               "flat mode omits side-skirt geometry");

    bool allFlat = true;
    for (const auto& v : flatScene.packedVertices) {
        if (std::fabs(v.position[1]) > 1e-4f) { allFlat = false; break; }
    }
    expectTrue(allFlat, "flat mode places every terrain vertex on the y=0 plane");
}

void testMesherWaterPatches() {
    using namespace odai::game;
    const StrategyMap map = makeSampleMap();

    std::uint32_t waterTiles = 0;
    for (const auto& tile : map.tiles) {
        if (terrainIsWater(tile.terrain)) ++waterTiles;
    }
    expectTrue(waterTiles > 0u, "sample map has water tiles to cover");

    const odai::importer::ImportedScene withWater = buildStrategyMapScene(map);
    expectEqualU32(static_cast<std::uint32_t>(withWater.waterPatches.size()), waterTiles,
                   "one water patch per ocean/coast tile");
    bool patchesSized = true;
    for (const auto& patch : withWater.waterPatches) {
        if (patch.sizeX <= 0.0f || patch.sizeZ <= 0.0f) { patchesSized = false; break; }
    }
    expectTrue(patchesSized, "water patches have positive extent");

    StrategyMapMeshOptions noWater{};
    noWater.emitWaterPatches = false;
    const odai::importer::ImportedScene dry = buildStrategyMapScene(map, noWater);
    expectTrue(dry.waterPatches.empty(), "emitWaterPatches=false suppresses water");
}

void testHexDistance() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(7, 7);

    expectEqualInt(hexDistance(3, 3, 3, 3), 0, "distance from a tile to itself is zero");

    // Every in-bounds immediate neighbor sits exactly one step away.
    bool neighborsAreOne = true;
    bool symmetric = true;
    for (int direction = 0; direction < 6; ++direction) {
        int nc = 0;
        int nr = 0;
        if (!tileNeighbor(map, 3, 3, direction, nc, nr)) {
            continue;
        }
        neighborsAreOne = neighborsAreOne && hexDistance(3, 3, nc, nr) == 1;
        symmetric = symmetric && hexDistance(nc, nr, 3, 3) == hexDistance(3, 3, nc, nr);
    }
    expectTrue(neighborsAreOne, "every immediate neighbor is distance 1");
    expectTrue(symmetric, "hex distance is symmetric");

    // Hand-checked odd-r results (two rows straight down, and a diagonal).
    expectEqualInt(hexDistance(0, 0, 0, 2), 2, "two rows down is distance 2");
    expectEqualInt(hexDistance(0, 0, 2, 2), 3, "down-and-across is distance 3");
}

void testSupplyDrainOnWilderness() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(6, 6);
    GameState gs{};
    Unit& unit = gs.spawnUnit("warrior", 2, 2, /*owner=*/1);
    const int startSupply = unit.supply;

    int nc = 0;
    int nr = 0;
    expectTrue(tileNeighbor(map, 2, 2, 0, nc, nr), "test map has a neighbor to march onto");
    const MoveResult result =
        moveUnitStep(gs, map, unit, static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr));
    expectTrue(result == MoveResult::Ok, "marching onto open land succeeds");
    expectEqualInt(unit.supply, startSupply - 1, "open wilderness step costs one supply");
    expectEqualInt(static_cast<int>(unit.col), nc, "unit advanced to the destination column");
    expectEqualInt(static_cast<int>(unit.row), nr, "unit advanced to the destination row");
}

void testRoadIsFree() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(6, 6);
    int nc = 0;
    int nr = 0;
    (void)tileNeighbor(map, 2, 2, 0, nc, nr);
    map.at(static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr)).flags |= TileFlag_Road;

    GameState gs{};
    Unit& unit = gs.spawnUnit("warrior", 2, 2, 1);
    const int startSupply = unit.supply;
    const MoveResult result =
        moveUnitStep(gs, map, unit, static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr));
    expectTrue(result == MoveResult::Ok, "marching onto a road succeeds");
    expectEqualInt(unit.supply, startSupply, "a road step costs no supply");
}

void testMountainCostsTwo() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(6, 6);
    int nc = 0;
    int nr = 0;
    (void)tileNeighbor(map, 2, 2, 0, nc, nr);
    map.at(static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr)).terrain = TerrainType::Mountains;

    GameState gs{};
    Unit& unit = gs.spawnUnit("warrior", 2, 2, 1);
    const int startSupply = unit.supply;
    const MoveResult result =
        moveUnitStep(gs, map, unit, static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr));
    expectTrue(result == MoveResult::Ok, "crossing into mountains succeeds");
    expectEqualInt(unit.supply, startSupply - 2, "a mountain step costs two supply");
}

void testRefillAtSettlement() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(6, 6);
    int nc = 0;
    int nr = 0;
    (void)tileNeighbor(map, 2, 2, 0, nc, nr);
    // Friendly settlement sitting on the destination tile.
    map.settlements.push_back(
        Settlement{"Aid Station", static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr), 1, /*owner=*/1});

    GameState gs{};
    Unit& unit = gs.spawnUnit("warrior", 2, 2, 1);
    unit.supply = 1;  // nearly out of provisions
    const MoveResult result =
        moveUnitStep(gs, map, unit, static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr));
    expectTrue(result == MoveResult::Ok, "marching into a friendly settlement succeeds");
    expectEqualInt(unit.supply, unit.maxSupply, "passing through a friendly settlement refills supply");
}

void testAttritionAtZeroSupply() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(6, 6);  // no settlements anywhere
    GameState gs{};
    Unit& spawned = gs.spawnUnit("warrior", 2, 2, 1);
    const std::uint32_t unitId = spawned.id;
    const int maxHp = spawned.maxHp;
    spawned.supply = 0;  // stranded without provisions

    advanceTurn(gs, map);
    Unit* afterOne = gs.findUnit(unitId);
    expectTrue(afterOne != nullptr, "starving unit survives the first turn");
    if (afterOne != nullptr) {
        expectEqualInt(afterOne->hp, maxHp - 10, "zero-supply unit loses 10 HP per turn");
    }

    // Keep advancing; the unit must eventually starve to death and be removed.
    int guard = 0;
    while (gs.findUnit(unitId) != nullptr && guard < 20) {
        advanceTurn(gs, map);
        ++guard;
    }
    expectTrue(gs.findUnit(unitId) == nullptr, "a unit at zero supply eventually dies and is removed");
}

void testRegenAtSettlement() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(6, 6);
    map.settlements.push_back(Settlement{"Home", 2, 2, 1, /*owner=*/1});

    GameState gs{};
    Unit& unit = gs.spawnUnit("warrior", 2, 2, 1);
    unit.hp = 5;
    unit.supply = 0;
    const std::uint32_t unitId = unit.id;

    advanceTurn(gs, map);
    Unit* afterOne = gs.findUnit(unitId);
    expectTrue(afterOne != nullptr, "unit resting at a settlement is not removed");
    if (afterOne != nullptr) {
        expectEqualInt(afterOne->supply, afterOne->maxSupply, "resting at a settlement refills supply");
        expectEqualInt(afterOne->hp, 10, "resting at a settlement heals 5 HP per turn");
    }

    // Healing must cap at maxHp, never overshoot.
    for (int i = 0; i < 20; ++i) {
        advanceTurn(gs, map);
    }
    Unit* healed = gs.findUnit(unitId);
    expectTrue(healed != nullptr, "fully healed unit still present");
    if (healed != nullptr) {
        expectEqualInt(healed->hp, healed->maxHp, "regen caps at max HP");
    }
}

void testMovementPointsResetAndExhaust() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(8, 4);  // no settlements: supply drains, no refill
    GameState gs{};
    Unit& unit = gs.spawnUnit("warrior", 1, 1, 1);  // warrior has 2 movement
    const int movement = unitStatsFor("warrior").movement;
    expectEqualInt(unit.movementLeft, movement, "unit starts the turn with full movement");

    // March straight east (direction 0 keeps the same row): 'movement' steps, then dry.
    int col = 1;
    const int row = 1;
    for (int step = 0; step < movement; ++step) {
        int nc = 0;
        int nr = 0;
        expectTrue(tileNeighbor(map, col, row, 0, nc, nr), "room to keep marching east");
        const MoveResult r =
            moveUnitStep(gs, map, unit, static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr));
        expectTrue(r == MoveResult::Ok, "each step within movement budget succeeds");
        col = nc;
    }
    int nc = 0;
    int nr = 0;
    (void)tileNeighbor(map, col, row, 0, nc, nr);
    const MoveResult overshoot =
        moveUnitStep(gs, map, unit, static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr));
    expectTrue(overshoot == MoveResult::OutOfMoves, "stepping past the movement budget is rejected");
    expectEqualInt(static_cast<int>(unit.col), col, "rejected move leaves the unit in place");

    // A fresh turn restores the full movement allowance.
    advanceTurn(gs, map);
    Unit* fresh = gs.findUnit(unit.id);
    expectTrue(fresh != nullptr && fresh->movementLeft == movement, "advanceTurn resets movement points");
}

void testPathReachesGoal() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(9, 6);
    GameState gs{};
    const std::vector<std::array<std::uint32_t, 2>> path = findHexPath(map, gs, 1, 1, 5, 3);
    expectTrue(!path.empty(), "A* finds a path across open land");
    if (!path.empty()) {
        expectEqualU32(path.back()[0], 5u, "path ends at the goal column");
        expectEqualU32(path.back()[1], 3u, "path ends at the goal row");

        int pc = 1;
        int pr = 1;
        bool adjacentChain = true;
        bool noWater = true;
        for (const auto& wp : path) {
            adjacentChain = adjacentChain &&
                hexDistance(pc, pr, static_cast<int>(wp[0]), static_cast<int>(wp[1])) == 1;
            noWater = noWater && !terrainIsWater(map.at(wp[0], wp[1]).terrain);
            pc = static_cast<int>(wp[0]);
            pr = static_cast<int>(wp[1]);
        }
        expectTrue(adjacentChain, "every path step is to an adjacent hex");
        expectTrue(noWater, "path stays on land");
        expectEqualU32(static_cast<std::uint32_t>(path.size()),
                       static_cast<std::uint32_t>(hexDistance(1, 1, 5, 3)),
                       "open-land path is minimal length");
    }
}

void testPathAvoidsWater() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(9, 6);
    // A vertical ocean wall down column 4, with a single land gap at the top edge
    // (4,0); any path must detour through that gap.
    for (std::uint32_t row = 1; row < map.height; ++row) {
        map.at(4, row).terrain = TerrainType::Ocean;
    }
    GameState gs{};
    const std::vector<std::array<std::uint32_t, 2>> path = findHexPath(map, gs, 1, 2, 7, 2);
    expectTrue(!path.empty(), "A* routes around a water barrier");
    if (!path.empty()) {
        expectEqualU32(path.back()[0], 7u, "detour still ends at the goal column");
        expectEqualU32(path.back()[1], 2u, "detour still ends at the goal row");
        bool noWater = true;
        for (const auto& wp : path) {
            noWater = noWater && !terrainIsWater(map.at(wp[0], wp[1]).terrain);
        }
        expectTrue(noWater, "no path waypoint sits on water");
    }
}

void testPathPrefersCheaperTerrain() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(6, 7);
    // (2,2)->(2,4) has two equal-length routes, through (2,3) or (1,3). Make the
    // first a mountain; A* must take the cheaper open detour and skip the mountain.
    map.at(2, 3).terrain = TerrainType::Mountains;
    GameState gs{};
    const std::vector<std::array<std::uint32_t, 2>> path = findHexPath(map, gs, 2, 2, 2, 4);
    expectTrue(!path.empty(), "A* finds a path past the mountain");
    if (!path.empty()) {
        expectEqualU32(path.back()[0], 2u, "diamond path ends at the goal column");
        expectEqualU32(path.back()[1], 4u, "diamond path ends at the goal row");
        bool avoidsMountain = true;
        for (const auto& wp : path) {
            if (wp[0] == 2u && wp[1] == 3u) avoidsMountain = false;
        }
        expectTrue(avoidsMountain, "A* routes around the costly mountain tile");
    }
}

void testMoveOrderTraversesOverTurns() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(9, 4);  // no settlements: nothing masks supply use
    GameState gs{};
    Unit& unit = gs.spawnUnit("warrior", 1, 1, 1);  // movement 2
    const std::uint32_t uid = unit.id;

    issueMoveOrder(gs, map, unit, 6, 1);  // distance 5 > one turn's movement
    {
        const Unit* u = gs.findUnit(uid);
        expectTrue(u != nullptr, "unit survives the move order");
        expectTrue(u != nullptr && (u->col != 1u || u->row != 1u), "unit starts marching immediately");
        expectTrue(u != nullptr && !u->path.empty(), "a long march leaves waypoints for later turns");
        expectTrue(u != nullptr && u->movementLeft == 0, "the first turn's movement is fully spent");
    }

    int guard = 0;
    while (guard < 12) {
        const Unit* u = gs.findUnit(uid);
        if (u == nullptr || (u->col == 6u && u->row == 1u && u->path.empty())) {
            break;
        }
        advanceTurn(gs, map);
        ++guard;
    }
    const Unit* arrived = gs.findUnit(uid);
    expectTrue(arrived != nullptr, "marching unit is still alive on arrival");
    if (arrived != nullptr) {
        expectEqualU32(arrived->col, 6u, "unit reaches the goal column over several turns");
        expectEqualU32(arrived->row, 1u, "unit reaches the goal row over several turns");
        expectTrue(arrived->path.empty(), "the path is cleared once the unit arrives");
    }
}

// Map with a single friendly city at (3,3) for production/combat tests.
odai::game::StrategyMap makeCityMap(std::uint32_t width, std::uint32_t height,
                                    std::uint32_t cityCol, std::uint32_t cityRow,
                                    std::uint8_t owner) {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(width, height);
    map.settlements.push_back(Settlement{"Testopolis", cityCol, cityRow, 2, owner});
    return map;
}

void testCityProducesUnitOnNeighbor() {
    using namespace odai::game;
    StrategyMap map = makeCityMap(8, 8, 3, 3, /*owner=*/1);
    GameState gs{};
    gs.initCities(map);
    expectEqualU32(static_cast<std::uint32_t>(gs.cities.size()), 1u, "initCities makes one city per settlement");

    CityState& city = gs.cities[0];
    city.buildings = {"barracks"};
    city.perTurn = 100;       // finish a spearman (cost 60) in a single turn
    city.producing = "spearman";
    city.accumulated = 0;

    advanceTurn(gs, map);
    expectEqualU32(static_cast<std::uint32_t>(gs.units.size()), 1u, "city produced one unit");
    if (!gs.units.empty()) {
        const Unit& u = gs.units.front();
        expectTrue(u.typeId == "spearman", "produced the queued unit type");
        expectEqualU32(u.owner, 1u, "produced unit inherits the city owner");
        expectTrue(hexDistance(static_cast<int>(u.col), static_cast<int>(u.row), 3, 3) <= 1,
                   "produced unit is placed on or beside the city");
        expectTrue(gs.unitAt(u.col, u.row) != nullptr, "produced unit occupies its tile");
    }
    expectEqualU32(static_cast<std::uint32_t>(gs.cities[0].unitsProduced), 1u, "city counts units produced");
}

void testProductionRequiresBuilding() {
    using namespace odai::game;
    StrategyMap map = makeCityMap(8, 8, 3, 3, 1);
    GameState gs{};
    gs.initCities(map);
    CityState& city = gs.cities[0];

    city.buildings = {"barracks"};
    expectTrue(cityCanProduce(city, "spearman"), "a barracks enables melee units");
    expectTrue(!cityCanProduce(city, "archer"), "no fletcher blocks archers");
    expectTrue(cityCanProduce(city, "scout"), "recon units need no building");
    expectTrue(nextAutoProduction(city) == "spearman", "a barracks-only city auto-builds melee");

    city.buildings = {"fletcher"};
    expectTrue(cityCanProduce(city, "archer"), "a fletcher enables archers");
    expectTrue(!cityCanProduce(city, "warrior"), "no barracks blocks melee");
    expectTrue(nextAutoProduction(city) == "archer", "a fletcher-only city auto-builds archers");

    city.buildings.clear();
    expectTrue(nextAutoProduction(city).empty(), "a city with no military building builds nothing");
}

void testSmithyGrantsArmor() {
    using namespace odai::game;
    StrategyMap map = makeCityMap(8, 8, 3, 3, 1);

    GameState withSmithy{};
    withSmithy.initCities(map);
    withSmithy.cities[0].buildings = {"barracks", "smithy"};
    withSmithy.cities[0].perTurn = 100;
    withSmithy.cities[0].producing = "spearman";
    advanceTurn(withSmithy, map);
    expectTrue(!withSmithy.units.empty() && withSmithy.units.front().armor == 3,
               "a smithy forges armor onto produced units");

    GameState noSmithy{};
    noSmithy.initCities(map);
    noSmithy.cities[0].buildings = {"barracks"};
    noSmithy.cities[0].perTurn = 100;
    noSmithy.cities[0].producing = "spearman";
    advanceTurn(noSmithy, map);
    expectTrue(!noSmithy.units.empty() && noSmithy.units.front().armor == 0,
               "without a smithy produced units have no armor");
}

void testOnePerTilePlacement() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(6, 6);
    GameState gs{};
    const std::uint32_t aId = gs.spawnUnit("warrior", 2, 2, 1).id;
    int nc = 0;
    int nr = 0;
    (void)tileNeighbor(map, 2, 2, 0, nc, nr);
    gs.spawnUnit("warrior", static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr), 1);

    Unit* a = gs.findUnit(aId);
    expectTrue(a != nullptr, "attacker exists");
    if (a != nullptr) {
        const MoveResult r = moveUnitStep(gs, map, *a,
            static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr));
        expectTrue(r == MoveResult::Occupied, "a unit cannot move onto an occupied tile");
    }

    const FreeTile spot = findFreeNeighbor(map, gs, 2, 2);
    expectTrue(spot.found, "an open neighbor exists for placement");
    expectTrue(!(spot.col == static_cast<std::uint32_t>(nc) && spot.row == static_cast<std::uint32_t>(nr)),
               "the chosen free tile is not the occupied one");
    expectTrue(gs.unitAt(spot.col, spot.row) == nullptr, "the chosen free tile is unoccupied");
}

void testRangedAttackInRange() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(12, 6);
    GameState gs{};
    const std::uint32_t archerId = gs.spawnUnit("archer", 2, 2, 1).id;   // ranged 5, range 2
    const std::uint32_t enemyId = gs.spawnUnit("warrior", 4, 2, 2).id;   // distance 2
    const int hpBefore = gs.findUnit(enemyId)->hp;

    const AttackResult r = resolveAttack(gs, map, archerId, enemyId);
    expectTrue(r == AttackResult::Ok, "archer fires at a target in range");
    const Unit* enemy = gs.findUnit(enemyId);
    expectTrue(enemy != nullptr, "target survives a single volley");
    if (enemy != nullptr) {
        expectEqualInt(enemy->hp, hpBefore - 5, "ranged damage = ranged strength (no armor, no screen)");
    }
    const Unit* archer = gs.findUnit(archerId);
    expectTrue(archer != nullptr && archer->movementLeft == 0, "firing ends the archer's turn");
}

void testArcherAdjacencyBonus() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(12, 6);
    GameState gs{};
    const std::uint32_t archerId = gs.spawnUnit("archer", 2, 2, 1).id;
    int nc = 0;
    int nr = 0;
    (void)tileNeighbor(map, 2, 2, 0, nc, nr);                 // a tile next to the archer
    gs.spawnUnit("spearman", static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr), 1);  // friendly screen
    const std::uint32_t enemyId = gs.spawnUnit("warrior", 4, 2, 2).id;  // distance 2 from archer
    const int hpBefore = gs.findUnit(enemyId)->hp;

    const AttackResult r = resolveAttack(gs, map, archerId, enemyId);
    expectTrue(r == AttackResult::Ok, "screened archer fires");
    const Unit* enemy = gs.findUnit(enemyId);
    if (enemy != nullptr) {
        expectEqualInt(enemy->hp, hpBefore - (5 + 2), "an archer behind a pikeman deals +2 ranged damage");
    }
}

void testRangedOutOfRange() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(14, 6);
    GameState gs{};
    const std::uint32_t archerId = gs.spawnUnit("archer", 2, 2, 1).id;
    const std::uint32_t enemyId = gs.spawnUnit("warrior", 6, 2, 2).id;  // distance 4 > range 2
    const int hpBefore = gs.findUnit(enemyId)->hp;

    const AttackResult r = resolveAttack(gs, map, archerId, enemyId);
    expectTrue(r == AttackResult::OutOfRange, "a target beyond range cannot be hit");
    expectEqualInt(gs.findUnit(enemyId)->hp, hpBefore, "out-of-range fire deals no damage");
    expectTrue(gs.findUnit(archerId)->movementLeft > 0, "a rejected attack does not spend the turn");
}

void testMeleeAttackRetaliationAndArmor() {
    using namespace odai::game;
    StrategyMap map = makeFlatLandMap(8, 6);
    GameState gs{};
    const std::uint32_t attId = gs.spawnUnit("warrior", 2, 2, 1).id;    // attack 5
    int nc = 0;
    int nr = 0;
    (void)tileNeighbor(map, 2, 2, 0, nc, nr);
    const std::uint32_t defId = gs.spawnUnit("spearman", static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr), 2).id;  // attack 7
    const int attHp = gs.findUnit(attId)->hp;
    const int defHp = gs.findUnit(defId)->hp;

    const AttackResult r = resolveAttack(gs, map, attId, defId);
    expectTrue(r == AttackResult::Ok, "melee strikes an adjacent enemy");
    expectTrue(gs.findUnit(defId) != nullptr && gs.findUnit(defId)->hp == defHp - 5,
               "defender takes the attacker's strength as damage");
    expectTrue(gs.findUnit(attId) != nullptr && gs.findUnit(attId)->hp == attHp - 7,
               "a surviving melee defender retaliates");

    // Armor blunts incoming damage (clamped to at least 1).
    GameState armored{};
    const std::uint32_t aId = armored.spawnUnit("warrior", 2, 2, 1).id;  // attack 5
    armored.spawnUnit("warrior", static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr), 2);
    Unit* armoredDef = armored.unitAt(static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr));
    armoredDef->armor = 3;
    const std::uint32_t armoredDefId = armoredDef->id;
    const int armoredHp = armoredDef->hp;
    resolveAttack(armored, map, aId, armoredDefId);
    expectEqualInt(armored.findUnit(armoredDefId)->hp, armoredHp - 2, "armor reduces damage (5 - 3 = 2)");
}

// A varied relief map: elevation rises west->east, two mountain tiles, and a small
// patch of water in the north-west corner. Used by the hex-terrain builder tests.
odai::game::StrategyMap makeReliefMap() {
    using namespace odai::game;
    StrategyMap map{};
    map.resize(6, 5);
    map.hexSize = 12.0f;
    map.elevationStep = 5.0f;
    for (std::uint32_t row = 0; row < map.height; ++row) {
        for (std::uint32_t col = 0; col < map.width; ++col) {
            MapTile& tile = map.at(col, row);
            tile.terrain = TerrainType::Grassland;
            tile.elevation = static_cast<std::int16_t>(col);  // 0 (west) .. 5 (east)
            tile.visibility = TileVisibility::Visible;
        }
    }
    map.at(5, 0).terrain = TerrainType::Mountains;
    map.at(5, 4).terrain = TerrainType::Mountains;
    map.at(0, 0).terrain = TerrainType::Ocean;
    map.at(0, 1).terrain = TerrainType::Coast;
    return map;
}

std::uint32_t countLandTiles(const odai::game::StrategyMap& map) {
    using namespace odai::game;
    std::uint32_t land = 0;
    for (const MapTile& tile : map.tiles) {
        if (!terrainIsWater(tile.terrain)) ++land;
    }
    return land;
}

void testHexTerrainBaseMesh() {
    using namespace odai::game;
    const StrategyMap map = makeReliefMap();
    const odai::importer::HexTerrainData data = buildHexTerrain(map);

    expectEqualU32(static_cast<std::uint32_t>(data.baseVertices.size()), 7u,
                   "base hex fan has center + 6 corners");
    expectEqualU32(static_cast<std::uint32_t>(data.baseIndices.size()), 18u,
                   "base hex fan has six triangle patches (18 indices)");
    expectNear(data.baseVertices[0].localXZ[0], 0.0f, 1e-4f, "base center X is at the origin");
    expectNear(data.baseVertices[0].localXZ[1], 0.0f, 1e-4f, "base center Z is at the origin");

    bool indicesValid = true;
    for (const std::uint32_t index : data.baseIndices) {
        if (index >= data.baseVertices.size()) { indicesValid = false; break; }
    }
    expectTrue(indicesValid, "every base index references a base vertex");

    bool cornersAtRadius = true;
    for (std::size_t i = 1; i < data.baseVertices.size(); ++i) {
        const float x = data.baseVertices[i].localXZ[0];
        const float z = data.baseVertices[i].localXZ[1];
        if (std::fabs(std::sqrt((x * x) + (z * z)) - map.hexSize) > 1e-3f) { cornersAtRadius = false; break; }
    }
    expectTrue(cornersAtRadius, "every base corner sits one circumradius from the local origin");
}

void testHexTerrainInstancesAndPages() {
    using namespace odai::game;
    const StrategyMap map = makeReliefMap();
    HexTerrainOptions options{};
    options.chunkSize = 2u;
    const odai::importer::HexTerrainData data = buildHexTerrain(map, options);

    expectEqualU32(static_cast<std::uint32_t>(data.instances.size()), countLandTiles(map),
                   "one instance per land tile");

    bool noWaterInstances = true;
    for (const auto& inst : data.instances) {
        const TerrainType terrain = static_cast<TerrainType>(inst.classFlags & 0xFFu);
        if (terrainIsWater(terrain)) { noWaterInstances = false; break; }
    }
    expectTrue(noWaterInstances, "no instance is emitted for a water tile");

    // Pages must form a contiguous cover of the instance array (firstInstance slicing).
    std::uint32_t running = 0;
    bool contiguous = true;
    bool boundsValid = true;
    for (const auto& page : data.pages) {
        contiguous = contiguous && page.firstInstance == running && page.instanceCount > 0u;
        boundsValid = boundsValid &&
            page.boundsMax[0] > page.boundsMin[0] && page.boundsMax[2] > page.boundsMin[2];
        running += page.instanceCount;
    }
    expectTrue(contiguous, "pages are contiguous slices starting at instance 0");
    expectTrue(boundsValid, "page XZ bounds are non-degenerate for culling");
    expectEqualU32(running, static_cast<std::uint32_t>(data.instances.size()),
                   "pages cover every instance exactly once");

    bool hexSizeSet = true;
    for (const auto& inst : data.instances) {
        if (std::fabs(inst.hexSize - map.hexSize) > 1e-4f) { hexSizeSet = false; break; }
    }
    expectTrue(hexSizeSet, "each instance carries the map hexSize");
}

void testHexTerrainCrackFreeCorners() {
    using namespace odai::game;
    // All-land map in a single chunk so instance[row*W+col] addresses every tile.
    StrategyMap map{};
    map.resize(5, 5);
    map.hexSize = 12.0f;
    map.elevationStep = 5.0f;
    for (std::uint32_t row = 0; row < map.height; ++row) {
        for (std::uint32_t col = 0; col < map.width; ++col) {
            MapTile& tile = map.at(col, row);
            tile.terrain = TerrainType::Hills;
            tile.elevation = static_cast<std::int16_t>((col * 2) + row);  // varied relief
            tile.visibility = TileVisibility::Visible;
        }
    }
    HexTerrainOptions options{};
    options.chunkSize = 8u;  // one chunk -> instance index == row*width + col
    const odai::importer::HexTerrainData data = buildHexTerrain(map, options);
    expectEqualU32(static_cast<std::uint32_t>(data.instances.size()), map.width * map.height,
                   "all-land single-chunk map instances every tile in order");

    auto instAt = [&](std::uint32_t col, std::uint32_t row) -> const odai::importer::HexTileInstance& {
        return data.instances[(static_cast<std::size_t>(row) * map.width) + col];
    };
    // Mirror the shader's continuous-base corner height: corner k averages this tile
    // with the two edge-neighbours adjacent to that corner.
    auto cornerHeight = [](const odai::importer::HexTileInstance& inst, int k) -> float {
        return (inst.ownElevY + inst.neighborElevY[(k + 5) % 6] + inst.neighborElevY[k]) / 3.0f;
    };

    // Brute force: any two tiles whose corners coincide in world XZ must compute the
    // same corner height (the seam invariant). Check an interior tile against the map.
    bool cracksFree = true;
    for (int k = 0; k < 6 && cracksFree; ++k) {
        const odai::math::Vector3 cornerA = tileCornerWorld(map, 2, 2, k);
        const float heightA = cornerHeight(instAt(2, 2), k);
        for (std::uint32_t row = 0; row < map.height && cracksFree; ++row) {
            for (std::uint32_t col = 0; col < map.width && cracksFree; ++col) {
                for (int j = 0; j < 6; ++j) {
                    const odai::math::Vector3 cornerB = tileCornerWorld(map, col, row, j);
                    const float dx = cornerB.x - cornerA.x;
                    const float dz = cornerB.z - cornerA.z;
                    if (((dx * dx) + (dz * dz)) > 1e-2f) continue;  // not the same corner
                    if (std::fabs(cornerHeight(instAt(col, row), j) - heightA) > 1e-3f) {
                        cracksFree = false;
                        break;
                    }
                }
            }
        }
    }
    expectTrue(cracksFree, "tiles sharing a world corner agree on its height (no cracks)");

    // The per-tile detail window must close before any shared edge: edge points are at
    // least one inradius (hexSize * sqrt3/2) from the center.
    const float inradius = map.hexSize * 0.8660254f;
    bool windowClosesInsideEdge = true;
    for (const auto& inst : data.instances) {
        if (inst.detailParams[2] >= inradius) { windowClosesInsideEdge = false; break; }
    }
    expectTrue(windowClosesInsideEdge,
               "detail window closes inside the inradius so shared edges stay seamless");
}

void testHexTerrainDegenerateMaps() {
    using namespace odai::game;
    const odai::importer::HexTerrainData emptyData = buildHexTerrain(StrategyMap{});
    expectTrue(emptyData.instances.empty(), "an empty map yields no instances");

    StrategyMap allWater{};
    allWater.resize(3, 3);
    allWater.hexSize = 10.0f;
    for (MapTile& tile : allWater.tiles) tile.terrain = TerrainType::Ocean;
    const odai::importer::HexTerrainData waterData = buildHexTerrain(allWater);
    expectTrue(waterData.instances.empty(), "an all-water map yields no land instances");
    expectTrue(waterData.pages.empty(), "an all-water map yields no pages");
}

void testMesherSkipsLandSurface() {
    using namespace odai::game;
    const StrategyMap map = makeReliefMap();

    StrategyMapMeshOptions withLand{};
    const odai::importer::ImportedScene full = buildStrategyMapScene(map, withLand);

    StrategyMapMeshOptions noLand{};
    noLand.emitLandSurface = false;
    const odai::importer::ImportedScene lean = buildStrategyMapScene(map, noLand);

    expectTrue(lean.packedVertices.size() < full.packedVertices.size(),
               "emitLandSurface=false drops the land top/skirt geometry");
    expectEqualU32(static_cast<std::uint32_t>(lean.waterPatches.size()),
                   static_cast<std::uint32_t>(full.waterPatches.size()),
                   "skipping the land surface leaves water patches untouched");
}

}  // namespace

int main() {
    testModelIndexingAndBounds();
    testHexNeighbors();
    testHexGeometry();
    testSerializationRoundTrip();
    testLoadRejectsGarbage();
    testMesherProducesRenderableScene();
    testMesherChunking();
    testMesherFlatMode();
    testMesherWaterPatches();
    testHexDistance();
    testSupplyDrainOnWilderness();
    testRoadIsFree();
    testMountainCostsTwo();
    testRefillAtSettlement();
    testAttritionAtZeroSupply();
    testRegenAtSettlement();
    testMovementPointsResetAndExhaust();
    testPathReachesGoal();
    testPathAvoidsWater();
    testPathPrefersCheaperTerrain();
    testMoveOrderTraversesOverTurns();
    testCityProducesUnitOnNeighbor();
    testProductionRequiresBuilding();
    testSmithyGrantsArmor();
    testOnePerTilePlacement();
    testRangedAttackInRange();
    testArcherAdjacencyBonus();
    testRangedOutOfRange();
    testMeleeAttackRetaliationAndArmor();
    testHexTerrainBaseMesh();
    testHexTerrainInstancesAndPages();
    testHexTerrainCrackFreeCorners();
    testHexTerrainDegenerateMaps();
    testMesherSkipsLandSurface();

    if (g_failures != 0) {
        std::cerr << "[strategy map test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[strategy map test] all checks passed\n";
    return 0;
}
