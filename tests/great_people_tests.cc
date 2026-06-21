// Correctness tests for the Great People feature: the data-driven catalog, the
// city-scoped bonus application in computeCityYields, global uniqueness, and the
// great-person-point birth/integration logic in the headless sim. No Vulkan, no
// GTest -- same lightweight harness style as economy_tests.cc / advisor_tests.cc.

#include "game/economy.h"
#include "game/game_sim.h"
#include "game/great_people.h"
#include "game/strategy_map.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace odai::game;

int g_failures = 0;

void expectTrue(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[great people test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectEqualInt(int actual, int expected, const std::string& message) {
    if (actual != expected) {
        std::cerr << "[great people test] FAIL: " << message
                  << " (expected " << expected << ", got " << actual << ")\n";
        ++g_failures;
    }
}

bool bonusHasAnyEffect(const BuildingEffects& b) {
    return b.flat.food || b.flat.production || b.flat.gold || b.flat.science ||
           b.flat.culture || b.prodPct || b.goldPct || b.sciencePct || b.happiness ||
           b.growthBonus;
}

// --- catalog ---------------------------------------------------------------

void testCatalogLoads() {
    const std::vector<GreatPersonDef>& cat = greatPeopleCatalog();
    expectEqualInt(static_cast<int>(cat.size()), 12, "twelve great people in the catalog");

    std::set<std::string> ids;
    std::set<std::pair<int, int>> cells;
    for (const GreatPersonDef& g : cat) {
        expectTrue(!g.id.empty(), "great person id non-empty");
        expectTrue(!g.name.empty(), g.id + " has a name");
        expectTrue(!g.title.empty(), g.id + " has a title");
        expectTrue(!g.bonusSummary.empty(), g.id + " has a bonus summary");
        expectTrue(!g.description.empty(), g.id + " has a description");
        expectTrue(g.cls != GreatPersonClass::Count, g.id + " has a known class");
        expectTrue(g.portraitCol >= 0 && g.portraitCol < 4, g.id + " portrait col in range");
        expectTrue(g.portraitRow >= 0 && g.portraitRow < 3, g.id + " portrait row in range");
        expectTrue(g.bonus.present, g.id + " has a bonus block");
        expectTrue(g.bonus.scope == BuildingEffects::Scope::City, g.id + " bonus is city-scoped");
        expectTrue(bonusHasAnyEffect(g.bonus), g.id + " bonus actually does something");
        expectTrue(findGreatPerson(g.id) == &g, g.id + " findGreatPerson round-trips");
        ids.insert(g.id);
        cells.insert({g.portraitCol, g.portraitRow});
    }
    expectEqualInt(static_cast<int>(ids.size()), static_cast<int>(cat.size()),
                   "great person ids are unique");
    expectEqualInt(static_cast<int>(cells.size()), static_cast<int>(cat.size()),
                   "portrait cells are unique (no two share an atlas cell)");
    expectTrue(findGreatPerson("does_not_exist") == nullptr, "unknown id -> nullptr");

    // The known anchors from great_people.json (locks the art<->code contract).
    const GreatPersonDef* euclid = findGreatPerson("euclid");
    expectTrue(euclid != nullptr && euclid->cls == GreatPersonClass::Scientist,
               "euclid is a Great Scientist");
    expectTrue(euclid != nullptr && euclid->bonus.flat.science == 2 && euclid->bonus.sciencePct == 20,
               "euclid grants +2 science and +20% science");
    const GreatPersonDef* sunTzu = findGreatPerson("sun_tzu");
    expectTrue(sunTzu != nullptr && sunTzu->cls == GreatPersonClass::General,
               "sun_tzu is a Great General");
}

void testClassNameRoundTrip() {
    for (int i = 0; i < static_cast<int>(GreatPersonClass::Count); ++i) {
        const auto cls = static_cast<GreatPersonClass>(i);
        const std::string name = greatPersonClassName(cls);
        expectTrue(!name.empty(), "class name non-empty");
        expectTrue(greatPersonClassFromName(name) == cls, "class name round-trips: " + name);
    }
    expectTrue(greatPersonClassFromName("Not A Class") == GreatPersonClass::Count,
               "unknown class label -> Count sentinel");
}

// --- bonus application -----------------------------------------------------

// Settle a figure into a fresh capital and confirm computeCityYields folds in its
// bonus. Asserts use deltas off a clean baseline so they don't depend on map RNG.
void testBonusAppliedToCity() {
    WorldConfig cfg{};
    cfg.seed = 1337u;
    cfg.empireCount = 4;
    World world = makeWorld(cfg);
    expectTrue(!world.cities.empty(), "world has cities");
    if (world.cities.empty()) return;

    City& city = world.cities[0];
    expectTrue(city.greatPeople.empty(), "a fresh capital hosts no great people");

    // Flat-only bonus (Homer: +3 culture, +1 happiness) -> exact, since culture and
    // happiness are never percentage-scaled.
    computeCityYields(world, city);
    const Yields before = city.yields;
    const int beforeHappy = city.happyCap;

    integrateGreatPerson(world, city, "homer");
    expectEqualInt(static_cast<int>(city.greatPeople.size()), 1, "homer settled into the city");
    computeCityYields(world, city);
    expectEqualInt(city.yields.culture, before.culture + 3, "homer adds exactly +3 culture");
    expectEqualInt(city.happyCap, beforeHappy + 1, "homer adds exactly +1 happiness cap");

    // Percentage bonus (Euclid: +2 science flat then +20%) on a second fresh city.
    City& city2 = world.cities[1];
    computeCityYields(world, city2);
    const int rawSci = city2.yields.science;  // no buildings/great people yet -> no sci%
    integrateGreatPerson(world, city2, "euclid");
    computeCityYields(world, city2);
    const int expectedSci = (rawSci + 2) + ((rawSci + 2) * 20) / 100;
    expectEqualInt(city2.yields.science, expectedSci,
                   "euclid applies +2 science then a 20% multiplier");
}

void testIntegrationUniqueness() {
    WorldConfig cfg{};
    cfg.seed = 99u;
    cfg.empireCount = 4;
    World world = makeWorld(cfg);
    if (world.cities.size() < 2) { expectTrue(false, "need two cities"); return; }

    City& a = world.cities[0];
    expectTrue(!world.greatPersonTaken("archimedes"), "archimedes unborn at start");
    integrateGreatPerson(world, a, "archimedes");
    expectTrue(world.greatPersonTaken("archimedes"), "archimedes is taken once settled");

    // Double-integrate is idempotent (no duplicate on the same city).
    const std::size_t n = a.greatPeople.size();
    integrateGreatPerson(world, a, "archimedes");
    expectEqualInt(static_cast<int>(a.greatPeople.size()), static_cast<int>(n),
                   "re-settling the same figure does not duplicate it");

    // Pending removal: stash one as pending, then integrating it clears the queue.
    Empire* emp = world.empireById(a.owner);
    if (emp != nullptr) {
        emp->pendingGreatPeople.push_back("homer");
        integrateGreatPerson(world, a, "homer");
        expectTrue(std::find(emp->pendingGreatPeople.begin(), emp->pendingGreatPeople.end(),
                             "homer") == emp->pendingGreatPeople.end(),
                   "integrating a pending figure removes it from the queue");
    }
}

void testCostFormula() {
    Empire emp{};
    emp.greatPeopleBorn = 0;
    const int base = greatPersonCost(emp);
    expectEqualInt(base, balance().greatPersonBaseCost, "first great person costs the base");
    emp.greatPeopleBorn = 3;
    expectEqualInt(greatPersonCost(emp),
                   balance().greatPersonBaseCost + 3 * balance().greatPersonCostGrowth,
                   "cost rises by growth per figure already born");
}

// --- generation in the full sim --------------------------------------------

void testGenerationInSim() {
    WorldConfig cfg{};
    cfg.seed = 2026u;
    cfg.empireCount = 4;
    World world = makeWorld(cfg);
    std::vector<TurnSample> samples;
    for (int t = 0; t < 250; ++t) stepTurn(world, samples);

    expectTrue(!world.bornGreatPeople.empty(),
               "at least one great person is born over a long match");

    // Global uniqueness: no figure born twice, and each names a real catalog entry.
    std::set<std::string> seen;
    for (const std::string& id : world.bornGreatPeople) {
        expectTrue(findGreatPerson(id) != nullptr, "born figure '" + id + "' is in the catalog");
        expectTrue(seen.insert(id).second, "figure '" + id + "' born at most once");
    }
    const int catalogSize = static_cast<int>(greatPeopleCatalog().size());
    expectTrue(static_cast<int>(world.bornGreatPeople.size()) <= catalogSize,
               "never more figures born than exist in the catalog");

    // Conservation: every born figure is accounted for -- settled in some city or
    // waiting in some empire's pending queue. In an all-AI world they all settle.
    std::set<std::string> settled;
    for (const City& c : world.cities)
        for (const std::string& id : c.greatPeople) settled.insert(id);
    std::set<std::string> pending;
    for (const Empire& e : world.empires)
        for (const std::string& id : e.pendingGreatPeople) pending.insert(id);

    for (const std::string& id : world.bornGreatPeople) {
        expectTrue(settled.count(id) > 0 || pending.count(id) > 0,
                   "born figure '" + id + "' is either settled or pending");
    }
    // makeWorld marks every empire aiManaged, so the AI settles each at birth.
    expectTrue(pending.empty(), "all-AI match settles every figure (nothing left pending)");
    expectEqualInt(static_cast<int>(settled.size()), static_cast<int>(world.bornGreatPeople.size()),
                   "settled figures match the born roster in an all-AI match");

    // The great-person event log actually fires.
    int gpEvents = 0;
    for (const GameEvent& e : world.events)
        if (e.kind == GameEvent::GreatPerson) ++gpEvents;
    expectTrue(gpEvents > 0, "great-person events are logged");
}

void testGenerationDeterminism() {
    auto run = [](std::uint32_t seed) {
        WorldConfig cfg{};
        cfg.seed = seed;
        cfg.empireCount = 4;
        World world = makeWorld(cfg);
        std::vector<TurnSample> samples;
        for (int t = 0; t < 150; ++t) stepTurn(world, samples);
        return world.bornGreatPeople;  // ordered birth sequence
    };
    const std::vector<std::string> a = run(55u);
    const std::vector<std::string> b = run(55u);
    expectTrue(a == b, "same seed yields the identical great-person birth sequence");
}

}  // namespace

int main() {
    testCatalogLoads();
    testClassNameRoundTrip();
    testBonusAppliedToCity();
    testIntegrationUniqueness();
    testCostFormula();
    testGenerationInSim();
    testGenerationDeterminism();

    if (g_failures == 0) {
        std::cout << "[great people test] all checks passed\n";
        return 0;
    }
    std::cerr << "[great people test] " << g_failures << " check(s) failed\n";
    return 1;
}
