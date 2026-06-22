// Correctness tests for the data-driven content system (mods/base). Confirms the
// base game loads cleanly from JSON, referential integrity holds, the loaded
// values match the pre-migration hardcoded tables (parity oracle), and the
// headless sim stays deterministic on the JSON-driven content. No Vulkan, no
// GTest -- same lightweight harness style as economy_tests.cc.
//
// The exhaustive numeric parity check is the odai_civ_sim --sweep diff in the
// verification step; this test locks the invariants so regressions surface in CI.

#include "content/content_database.h"
#include "game/buildable.h"
#include "game/economy.h"
#include "game/game_sim.h"
#include "game/great_people.h"
#include "game/strategy_map.h"
#include "game/units.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace odai::game;

int g_failures = 0;

void expectTrue(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[content test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectEqInt(long long actual, long long expected, const std::string& message) {
    if (actual != expected) {
        std::cerr << "[content test] FAIL: " << message
                  << " (expected " << expected << ", got " << actual << ")\n";
        ++g_failures;
    }
}

void expectEqStr(const std::string& actual, const std::string& expected, const std::string& message) {
    if (actual != expected) {
        std::cerr << "[content test] FAIL: " << message
                  << " (expected '" << expected << "', got '" << actual << "')\n";
        ++g_failures;
    }
}

void expectNear(float actual, float expected, const std::string& message) {
    if (std::fabs(actual - expected) > 1e-5f) {
        std::cerr << "[content test] FAIL: " << message
                  << " (expected " << expected << ", got " << actual << ")\n";
        ++g_failures;
    }
}

void testCleanLoad() {
    const odai::content::ContentDatabase& db = odai::content::activeContent();
    for (const std::string& e : db.errors()) std::cerr << "  load error: " << e << "\n";
    expectTrue(db.ok(), "base game content loads without errors");

    expectEqInt(static_cast<long long>(techTree().size()), 20, "tech count");
    expectEqInt(static_cast<long long>(buildingDefs().size()), 29, "building count");
    expectEqInt(static_cast<long long>(defaultUnitStats().size()), 6, "unit count");
    expectEqInt(static_cast<long long>(defaultBuildables().size()), 13, "buildable count");
    expectEqInt(static_cast<long long>(db.leaders().size()), 6, "leader count");
    expectEqInt(static_cast<long long>(db.greatPeople().size()), 12, "great people count");

    int wonders = 0;
    for (const BuildingDef& d : buildingDefs()) if (d.isWonder) ++wonders;
    expectEqInt(wonders, 12, "wonder count");
}

// Referential integrity, ported from economy_tests::testCatalogIntegrity so the
// content layer enforces the same invariants.
void testIntegrity() {
    for (const TechDef& t : techTree()) {
        expectTrue(!t.era.empty(), "tech '" + t.id + "' has a non-empty era");
        for (const std::string& p : t.prereqs)
            expectTrue(findTech(p) != nullptr, "tech prereq '" + p + "' references an existing tech");
        for (const std::string& u : t.unlocks)
            expectTrue(findBuildingDef(u) != nullptr, "tech unlock '" + u + "' references an existing building");
    }
    for (const BuildingDef& d : buildingDefs()) {
        if (!d.isWonder) continue;
        int unlockedBy = 0;
        for (const TechDef& t : techTree())
            for (const std::string& u : t.unlocks)
                if (u == d.id) ++unlockedBy;
        expectEqInt(unlockedBy, 1, "wonder '" + d.id + "' unlocked by exactly one tech");
        expectTrue(d.score > 0, "wonder '" + d.id + "' has a positive score");
        expectTrue(d.effects.present && d.effects.scope == BuildingEffects::Scope::Empire,
                   "wonder '" + d.id + "' has empire-scoped effects");
    }
    // Note: not every buildable has a CivPedia article in the base game (e.g.
    // "fletcher" has none), so we only assert presence for the known set below.
}

void testBalanceParity() {
    const Balance& b = balance();
    expectEqInt(b.citizenFoodUpkeep, 2, "balance.citizenFoodUpkeep");
    expectEqInt(b.foodBoxBase, 12, "balance.foodBoxBase");
    expectEqInt(b.foodBoxPerPop, 7, "balance.foodBoxPerPop");
    expectEqInt(b.baseHappyCap, 8, "balance.baseHappyCap");
    expectEqInt(b.settlerPopCost, 1, "balance.settlerPopCost");
    expectEqInt(b.settlerFoundPop, 1, "balance.settlerFoundPop");
    expectEqInt(b.settlerCost, 72, "balance.settlerCost");
    expectEqInt(b.capitalStartPop, 2, "balance.capitalStartPop");
    expectEqInt(b.rushGoldPerShield, 4, "balance.rushGoldPerShield");
    expectEqInt(b.cityWorkRadius, 2, "balance.cityWorkRadius");
    expectEqInt(b.settleMinSpacing, 3, "balance.settleMinSpacing");
    expectEqInt(b.sciencePerPopDiv, 3, "balance.sciencePerPopDiv");
    expectEqInt(b.maxGrowthBonus, 80, "balance.maxGrowthBonus");
    expectEqInt(b.civicUpkeepPerCity, 2, "balance.civicUpkeepPerCity");
    expectEqInt(b.starveBuffer, 8, "balance.starveBuffer");
    expectEqInt(b.greatPersonBaseCost, 200, "balance.greatPersonBaseCost");
    expectEqInt(b.greatPersonCostGrowth, 120, "balance.greatPersonCostGrowth");
    expectEqInt(b.greatPersonSciencePointDiv, 2, "balance.greatPersonSciencePointDiv");
}

void testTerrainParity() {
    struct TE { TerrainType t; const char* name; int f, p, g, s, c; bool work; };
    const TE table[] = {
        {TerrainType::Ocean,     "Ocean",     1, 0, 1, 0, 0, true},
        {TerrainType::Coast,     "Coast",     1, 0, 2, 0, 0, true},
        {TerrainType::Grassland, "Grassland", 3, 0, 0, 0, 0, true},
        {TerrainType::Plains,    "Plains",    1, 1, 0, 0, 0, true},
        {TerrainType::Forest,    "Forest",    1, 2, 0, 0, 0, true},
        {TerrainType::Jungle,    "Jungle",    1, 0, 0, 1, 0, true},
        {TerrainType::Hills,     "Hills",     0, 2, 0, 0, 0, true},
        {TerrainType::Mountains, "Mountains", 0, 0, 0, 0, 0, false},
        {TerrainType::Desert,    "Desert",    0, 0, 0, 0, 0, true},
        {TerrainType::Tundra,    "Tundra",    1, 0, 0, 0, 0, true},
        {TerrainType::Snow,      "Snow",      0, 0, 0, 0, 0, false},
    };
    for (const TE& e : table) {
        const Yields y = terrainYields(e.t, 0);
        const std::string n = e.name;
        expectEqInt(y.food, e.f, n + " food");
        expectEqInt(y.production, e.p, n + " production");
        expectEqInt(y.gold, e.g, n + " gold");
        expectEqInt(y.science, e.s, n + " science");
        expectEqInt(y.culture, e.c, n + " culture");
        expectTrue(tileIsWorkable(e.t) == e.work, n + " workability");
    }
    // River + road each add a gold of trade (TileFlag-driven).
    expectEqInt(terrainYields(TerrainType::Plains, TileFlag_River | TileFlag_Road).gold, 2,
                "river + road add 2 gold to plains");
    expectEqInt(terrainYields(TerrainType::Grassland, TileFlag_River).gold, 1,
                "river adds 1 gold to grassland");
    const Yields cc = cityCenterYields();
    expectEqInt(cc.food, 2, "city center food");
    expectEqInt(cc.production, 1, "city center production");
    expectEqInt(cc.gold, 1, "city center gold");
    expectEqInt(cc.science, 1, "city center science");
    expectEqInt(cc.culture, 0, "city center culture");
}

void checkTech(const std::string& id, const std::string& name, int cost, const std::string& era,
               GateKind kind, const std::string& cond, int amount, int boostPct) {
    const TechDef* t = findTech(id);
    if (t == nullptr) { expectTrue(false, "tech '" + id + "' exists"); return; }
    expectEqStr(t->name, name, id + ".name");
    expectEqInt(t->cost, cost, id + ".cost");
    expectEqStr(t->era, era, id + ".era");
    expectTrue(t->gate.kind == kind, id + ".gate.kind");
    expectEqStr(t->gate.condition, cond, id + ".gate.condition");
    expectEqInt(t->gate.amount, amount, id + ".gate.amount");
    expectEqInt(t->gate.boostPct, boostPct, id + ".gate.boostPct");
}

void testTechParity() {
    // Spot-check across all three gate kinds and eras (full numeric coverage comes
    // from the sweep diff). Order is also checked: the first/last ids must match.
    expectEqStr(techTree().front().id, "pottery", "first tech id");
    expectEqStr(techTree().back().id, "algebra", "last tech id");
    checkTech("pottery", "Pottery", 22, "Ancient Era", GateKind::Open, "", 0, 50);
    checkTech("writing", "Writing", 34, "Ancient Era", GateKind::Boost, "meet_rival", 0, 50);
    checkTech("sailing", "Sailing", 48, "Ancient Era", GateKind::Locked, "coastal_city", 0, 0);
    checkTech("masonry", "Masonry", 54, "Ancient Era", GateKind::Boost, "work_terrain:hills", 0, 50);
    checkTech("currency", "Currency", 72, "Classical Era", GateKind::Boost, "treasury", 120, 50);
    checkTech("construction", "Construction", 160, "Classical Era", GateKind::Locked, "own_wonder", 0, 0);
    checkTech("astronomy", "Astronomy", 250, "Medieval Era", GateKind::Locked, "building:harbor", 1, 0);
    checkTech("guilds", "Guilds", 290, "Medieval Era", GateKind::Boost, "building:bank", 1, 50);
    // Prereqs/unlocks for a representative branching tech.
    const TechDef* math = findTech("mathematics");
    expectTrue(math != nullptr && math->prereqs.size() == 2, "mathematics has 2 prereqs");
    if (math != nullptr) {
        expectEqStr(math->prereqs[0], "masonry", "mathematics prereq 0");
        expectEqStr(math->prereqs[1], "currency", "mathematics prereq 1");
        expectTrue(math->unlocks.size() == 1 && math->unlocks[0] == "aqueduct", "mathematics unlocks aqueduct");
    }
}

void checkBuildingFlat(const std::string& id, int cost, int maint, int f, int p, int g, int s, int c,
                       int prodPct, int goldPct, int sciPct, int happy, int grow, const std::string& tech) {
    const BuildingDef* d = findBuildingDef(id);
    if (d == nullptr) { expectTrue(false, "building '" + id + "' exists"); return; }
    expectEqInt(d->productionCost, cost, id + ".cost");
    expectEqInt(d->maintenance, maint, id + ".maintenance");
    expectEqInt(d->flat.food, f, id + ".flat.food");
    expectEqInt(d->flat.production, p, id + ".flat.production");
    expectEqInt(d->flat.gold, g, id + ".flat.gold");
    expectEqInt(d->flat.science, s, id + ".flat.science");
    expectEqInt(d->flat.culture, c, id + ".flat.culture");
    expectEqInt(d->prodPct, prodPct, id + ".prodPct");
    expectEqInt(d->goldPct, goldPct, id + ".goldPct");
    expectEqInt(d->sciencePct, sciPct, id + ".sciencePct");
    expectEqInt(d->happiness, happy, id + ".happiness");
    expectEqInt(d->growthBonus, grow, id + ".growthBonus");
    expectEqStr(d->requiredTech, tech, id + ".requiredTech");
    expectTrue(!d->isWonder, id + " is not a wonder");
}

void checkWonderEffect(const std::string& id, int f, int p, int g, int s, int c,
                       int prodPct, int goldPct, int sciPct, int happy, int grow) {
    const BuildingDef* d = findBuildingDef(id);
    if (d == nullptr) { expectTrue(false, "wonder '" + id + "' exists"); return; }
    expectTrue(d->isWonder, id + " is a wonder");
    const BuildingEffects& e = d->effects;
    expectTrue(e.present && e.scope == BuildingEffects::Scope::Empire, id + " empire-scoped effects");
    expectEqInt(e.flat.food, f, id + ".eff.food");
    expectEqInt(e.flat.production, p, id + ".eff.production");
    expectEqInt(e.flat.gold, g, id + ".eff.gold");
    expectEqInt(e.flat.science, s, id + ".eff.science");
    expectEqInt(e.flat.culture, c, id + ".eff.culture");
    expectEqInt(e.prodPct, prodPct, id + ".eff.prodPct");
    expectEqInt(e.goldPct, goldPct, id + ".eff.goldPct");
    expectEqInt(e.sciencePct, sciPct, id + ".eff.sciencePct");
    expectEqInt(e.happiness, happy, id + ".eff.happiness");
    expectEqInt(e.growthBonus, grow, id + ".eff.growthBonus");
}

void testBuildingParity() {
    //                 id           cost mt  f  p  g  s  c  pP gP sP hp gr  tech
    checkBuildingFlat("granary",    58, 1,  2, 0, 0, 0, 0,  0, 0, 0, 1, 50, "pottery");
    checkBuildingFlat("library",    76, 1,  0, 0, 0, 1, 1,  0, 0, 50, 0, 0, "writing");
    checkBuildingFlat("smithy",     72, 1,  0, 2, 0, 0, 0, 25, 0, 0, 0, 0, "bronze_working");
    checkBuildingFlat("market",     96, 2,  0, 0, 2, 0, 0,  0, 40, 0, 0, 0, "currency");
    checkBuildingFlat("temple",     82, 1,  0, 0, 0, 0, 2,  0, 0, 0, 3, 0, "philosophy");
    checkBuildingFlat("aqueduct",  108, 2,  1, 0, 0, 0, 0,  0, 0, 0, 2, 25, "mathematics");
    checkBuildingFlat("guildhall", 150, 2,  0, 1, 2, 0, 0,  0, 20, 0, 0, 0, "guilds");

    // Wonder empire-wide effects (formerly the hardcoded if/else in game_sim.cc).
    //                  id                 f  p  g  s  c  pP gP sP hp gr
    checkWonderEffect("pyramids",         0, 0, 0, 0, 0, 25, 0, 0, 0, 0);
    checkWonderEffect("hanging_gardens",  1, 0, 0, 0, 0,  0, 0, 0, 2, 0);
    checkWonderEffect("colossus",         0, 0, 2, 0, 0,  0, 0, 0, 0, 0);
    checkWonderEffect("oracle",           0, 0, 0, 2, 1,  0, 0, 0, 0, 0);
    checkWonderEffect("great_library",    0, 0, 0, 3, 0,  0, 0, 25, 0, 0);
    checkWonderEffect("parthenon",        0, 0, 0, 0, 3,  0, 0, 0, 0, 0);
    checkWonderEffect("colosseum",        0, 0, 0, 0, 0,  0, 0, 0, 3, 0);
    checkWonderEffect("great_wall",       0, 0, 0, 0, 0,  0, 0, 0, 2, 0);
    checkWonderEffect("grand_temple",     0, 0, 0, 0, 2,  0, 0, 0, 1, 0);
    checkWonderEffect("copernicus",       0, 0, 0, 3, 0,  0, 0, 25, 0, 0);
    checkWonderEffect("grand_bazaar",     0, 0, 3, 0, 0,  0, 25, 0, 0, 0);
    expectEqStr(findBuildingDef("copernicus")->name, "Copernicus' Observatory", "copernicus display name");
}

void checkUnit(const std::string& id, int hp, int move, int supply, int atk, int ranged, int range,
               bool melee, const std::string& building) {
    const UnitStats& u = unitStatsFor(id);
    expectEqStr(u.id, id, id + ".id");
    expectEqInt(u.maxHp, hp, id + ".maxHp");
    expectEqInt(u.movement, move, id + ".movement");
    expectEqInt(u.maxSupply, supply, id + ".maxSupply");
    expectEqInt(u.attack, atk, id + ".attack");
    expectEqInt(u.rangedAttack, ranged, id + ".rangedAttack");
    expectEqInt(u.range, range, id + ".range");
    expectTrue(u.melee == melee, id + ".melee");
    expectEqStr(u.requiredBuilding, building, id + ".requiredBuilding");
}

void testUnitParity() {
    checkUnit("warrior",  30, 2, 5, 5, 0, 0, true,  "barracks");
    checkUnit("spearman", 30, 2, 5, 7, 0, 0, true,  "barracks");
    checkUnit("archer",   25, 2, 5, 4, 5, 2, false, "fletcher");
    checkUnit("scout",    20, 3, 8, 2, 0, 0, false, "");
    checkUnit("settler",  20, 2, 4, 0, 0, 0, false, "");
    checkUnit("builder",  20, 2, 4, 0, 0, 0, false, "");
}

void testLeaderParity() {
    const std::vector<LeaderDef>& leaders = odai::content::activeContent().leaders();
    if (leaders.size() < 6) { expectTrue(false, "6 leaders present"); return; }
    expectEqStr(leaders[0].civName, "Egypt", "leader 0 civ");
    expectEqStr(leaders[0].leaderName, "Ramesses", "leader 0 name");
    expectEqStr(leaders[0].personality.name, "Builder / Religion", "leader 0 personality name");
    expectNear(leaders[0].personality.expansion, 0.8f, "Egypt expansion");
    expectNear(leaders[0].personality.wonderLove, 1.8f, "Egypt wonderLove");
    expectNear(leaders[0].personality.religion, 2.0f, "Egypt religion");
    expectNear(leaders[0].personality.culture, 1.2f, "Egypt culture");
    expectEqStr(leaders[1].civName, "Mongols", "leader 1 civ");
    expectNear(leaders[1].personality.expansion, 1.8f, "Mongols expansion");
    expectEqStr(leaders[5].civName, "India", "leader 5 civ");
    expectNear(leaders[5].personality.religion, 2.3f, "India religion");
    for (const LeaderDef& l : leaders)
        expectEqInt(static_cast<long long>(l.cityNames.size()), 8, l.civName + " has 8 city names");
    expectEqStr(leaders[0].cityNames[0], "Memphis", "Egypt capital name");
}

void testCivpediaPresence() {
    const char* ids[] = {"spearman", "warrior", "archer", "granary", "smithy", "market",
                         "pottery", "writing", "sailing", "guilds"};
    for (const char* id : ids)
        expectTrue(!getPediaArticle(id).empty(), std::string("civpedia article '") + id + "' present");
    expectTrue(getPediaArticle("nonexistent_xyz").empty(), "unknown civpedia id returns empty");
}

// Sim smoke + determinism: the JSON-driven content runs a full match and is
// reproducible from a seed (a no-op precondition for the sweep parity diff).
void testSimDeterminism() {
    auto run = [](std::uint32_t seed) {
        WorldConfig cfg{};
        cfg.seed = seed;
        cfg.empireCount = 4;
        World w = makeWorld(cfg);
        std::vector<TurnSample> samples;
        for (int t = 0; t < 80; ++t) stepTurn(w, samples);
        return samples;
    };
    const std::vector<TurnSample> a = run(1337u);
    const std::vector<TurnSample> b = run(1337u);
    expectTrue(!a.empty(), "sim produced turn samples");
    expectTrue(a.size() == b.size(), "deterministic sample count");
    bool identical = a.size() == b.size();
    for (std::size_t i = 0; identical && i < a.size(); ++i) {
        identical = (a[i].score == b[i].score) && (a[i].leader == b[i].leader) &&
                    (a[i].population == b[i].population) && (a[i].wonders == b[i].wonders);
    }
    expectTrue(identical, "two runs with the same seed are bit-identical");
}

}  // namespace

int main() {
    testCleanLoad();
    testIntegrity();
    testBalanceParity();
    testTerrainParity();
    testTechParity();
    testBuildingParity();
    testUnitParity();
    testLeaderParity();
    testCivpediaPresence();
    testSimDeterminism();

    if (g_failures == 0) {
        std::cout << "[content test] all checks passed\n";
        return 0;
    }
    std::cerr << "[content test] " << g_failures << " failure(s)\n";
    return 1;
}
