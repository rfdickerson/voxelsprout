// Correctness tests for the strategic economy + headless simulation. No Vulkan,
// no GTest -- same lightweight harness style as strategy_map_tests.cc.

#include "game/economy.h"
#include "game/game_sim.h"
#include "game/strategy_map.h"

#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace odai::game;

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[economy test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectEqualInt(int actual, int expected, const char* message) {
    if (actual != expected) {
        std::cerr << "[economy test] FAIL: " << message
                  << " (expected " << expected << ", got " << actual << ")\n";
        ++g_failures;
    }
}

void testTerrainYields() {
    expectEqualInt(terrainYields(TerrainType::Grassland, 0).food, 3, "grassland yields 3 food");
    expectEqualInt(terrainYields(TerrainType::Hills, 0).production, 2, "hills yield 2 production");
    expectEqualInt(terrainYields(TerrainType::Coast, 0).gold, 2, "coast yields 2 gold");
    // River + road each add a gold of trade.
    const Yields riverRoad = terrainYields(TerrainType::Plains, TileFlag_River | TileFlag_Road);
    expectEqualInt(riverRoad.gold, 2, "river + road add 2 gold to a plains tile");
    expectTrue(!tileIsWorkable(TerrainType::Mountains), "mountains are unworkable");
    expectTrue(!tileIsWorkable(TerrainType::Snow), "snow is unworkable");
    expectTrue(tileIsWorkable(TerrainType::Grassland), "grassland is workable");
}

void testCatalogIntegrity() {
    // Every tech prereq names a real tech; every unlock names a real building.
    for (const TechDef& t : techTree()) {
        for (const std::string& p : t.prereqs)
            expectTrue(findTech(p) != nullptr, "tech prereq references an existing tech");
        for (const std::string& u : t.unlocks)
            expectTrue(findBuildingDef(u) != nullptr, "tech unlock references an existing building");
    }
    // Every wonder is unlocked by exactly one tech (so it is reachable + unique).
    int wonderCount = 0;
    for (const BuildingDef& d : buildingDefs()) {
        if (!d.isWonder) continue;
        ++wonderCount;
        int unlockedBy = 0;
        for (const TechDef& t : techTree())
            for (const std::string& u : t.unlocks)
                if (u == d.id) ++unlockedBy;
        expectEqualInt(unlockedBy, 1, "each wonder is unlocked by exactly one tech");
        expectTrue(d.score > 0, "each wonder has a positive score value");
        expectTrue(isWonder(d.id), "isWonder agrees with the catalog flag");
    }
    expectTrue(wonderCount >= 8, "catalog has a healthy number of wonders");
}

void testWorldGenIsFair() {
    WorldConfig cfg{};
    cfg.seed = 4242u;
    cfg.empireCount = 4;
    World world = makeWorld(cfg);
    expectEqualInt(static_cast<int>(world.empires.size()), 4, "makeWorld seats all four empires");
    for (const Empire& e : world.empires) {
        expectTrue(!e.cityIndices.empty(), "every empire starts with a capital");
        const City& cap = world.cities[e.cityIndices.front()];
        expectTrue(cap.population >= 1, "a capital starts populated");
        expectTrue(!terrainIsWater(world.map.at(cap.col, cap.row).terrain), "capitals are founded on land");
    }
}

void testSimSmokeAndInvariants() {
    WorldConfig cfg{};
    cfg.seed = 99u;
    cfg.empireCount = 4;
    World world = makeWorld(cfg);
    std::vector<TurnSample> samples;
    for (int t = 0; t < 200; ++t) stepTurn(world, samples);

    expectEqualInt(static_cast<int>(samples.size()), 200, "one metrics row per turn");

    // Population invariant: no city ever drops below 1.
    for (const City& c : world.cities) {
        expectTrue(c.population >= 1, "no city falls below population 1");
    }

    // Wonder uniqueness: each wonder owned by at most one empire, and the global
    // list matches the sum of per-empire ownership with no duplicates.
    std::set<std::string> seen;
    int totalOwned = 0;
    for (const Empire& e : world.empires) {
        totalOwned += static_cast<int>(e.wonders.size());
        for (const std::string& w : e.wonders) {
            expectTrue(isWonder(w), "owned wonder is a real wonder");
            expectTrue(seen.insert(w).second, "a wonder is owned by only one empire");
        }
    }
    expectEqualInt(static_cast<int>(world.builtWonders.size()), totalOwned,
                   "global wonder list matches per-empire ownership");
    int catalogWonders = 0;
    for (const BuildingDef& d : buildingDefs()) if (d.isWonder) ++catalogWonders;
    expectTrue(static_cast<int>(world.builtWonders.size()) <= catalogWonders,
               "never more wonders built than exist");

    // Everyone should have made progress (this is a working economy, not a stall).
    for (const Empire& e : world.empires) {
        expectTrue(e.score > 0, "every empire has a positive score");
        expectTrue(!e.researched.empty(), "every empire researches at least one tech");
    }
}

void testDeterminism() {
    auto runFinalScores = [](std::uint32_t seed) {
        WorldConfig cfg{};
        cfg.seed = seed;
        cfg.empireCount = 4;
        World world = makeWorld(cfg);
        std::vector<TurnSample> samples;
        for (int t = 0; t < 120; ++t) stepTurn(world, samples);
        std::vector<int> scores;
        for (const Empire& e : world.empires) scores.push_back(e.score);
        return scores;
    };
    const std::vector<int> a = runFinalScores(7u);
    const std::vector<int> b = runFinalScores(7u);
    expectTrue(a == b, "same seed produces identical final scores (deterministic)");
}

}  // namespace

int main() {
    testTerrainYields();
    testCatalogIntegrity();
    testWorldGenIsFair();
    testSimSmokeAndInvariants();
    testDeterminism();

    if (g_failures != 0) {
        std::cerr << "[economy test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[economy test] all checks passed\n";
    return 0;
}
