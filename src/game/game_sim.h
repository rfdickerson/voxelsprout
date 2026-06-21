#pragma once

#include "game/economy.h"
#include "game/strategy_map.h"

#include <cstdint>
#include <string>
#include <vector>

// Headless strategic-game simulation: a whole 4X match (multiple AI empires,
// expansion, research, wonders, the gold squeeze) advanced one turn at a time
// with no renderer. This is the harness used to playtest the economy's "fun
// factor" -- it reuses the real hex StrategyMap and the economy rules in
// economy.h, and records rich per-turn metrics for analysis.
namespace odai::game {

// One city. Production is a single queue; focus is a single knob -- those two
// choices per turn are the whole strategic game.
struct City {
    std::string name;
    std::uint32_t col = 0;
    std::uint32_t row = 0;
    std::uint8_t owner = 0;           // empire id (1-based)
    int population = 1;
    int foodStored = 0;
    CityFocus focus = CityFocus::Balanced;
    std::vector<std::string> buildings;
    std::vector<std::string> greatPeople;  // ids of great people settled here (permanent bonuses)
    std::string producing;            // building / wonder / "settler" in progress
    int accumulated = 0;              // production banked toward `producing`
    bool foundedThisGame = true;      // false for capitals (cosmetic)

    // Cached for reporting / AI (recomputed each turn).
    Yields yields{};                  // food field is NET of citizen upkeep
    int happyCap = 0;
    int growthBonusPct = 0;           // % of the food box kept after growing
    bool inDisorder = false;
    int turnsToFinish = 0;            // est. turns to complete `producing`

    [[nodiscard]] bool hasBuilding(const std::string& id) const;
};

// A simple AI personality: weights that bias research, expansion, and the build
// queue so different empires feel different and contend over wonders.
struct Personality {
    std::string name;
    float expansion = 1.0f;   // appetite for settlers / new cities
    float wonderLove = 1.0f;  // willingness to commit to wonders
    float science = 1.0f;     // research-tree weighting
    float gold = 1.0f;        // economy weighting
    float religion = 1.0f;    // proactive happiness buildings; biases toward happiness wonders
    float culture = 1.0f;     // culture buildings and culture-granting wonders
};

// A playable civilization preset: the named civ/leader, their AI personality, and
// the ordered list of city names new settlements draw from. Loaded from
// leaders.json (formerly a hardcoded presets table in game_sim.cc) so mods can add
// or retune civilizations.
struct LeaderDef {
    std::string civName;                 // e.g. "Egypt"
    std::string leaderName;              // e.g. "Ramesses"
    Personality personality{};
    std::vector<std::string> cityNames;  // capital first, then expansion names
};

struct Empire {
    std::uint8_t id = 0;
    std::string name;
    std::string leaderName;                     // the named ruler (e.g., "Ramesses")
    std::vector<std::string> cityNames;         // ordered city names for new settlements
    int nextCityName = 1;                       // index of the next unused name (0 = capital)
    Personality personality{};
    int treasury = 20;
    int sciencePool = 0;
    int culturePoints = 0;                    // accumulated culture (feeds score)
    std::string researching;                 // tech id currently researched
    std::vector<std::string> researched;     // completed tech ids
    std::vector<std::string> unlockedTechs;  // Locked-gate techs whose deed is done (now researchable)
    std::vector<std::string> boostedTechs;   // Boost-gate techs whose deed is done (discount active)
    std::vector<std::string> wonders;        // wonder ids this empire owns
    std::vector<std::size_t> cityIndices;    // indices into World::cities
    int futureTechs = 0;                      // repeatable techs past the tree (score sink)
    // Great People: points accrue each turn; at the rising threshold a globally
    // unique figure is born. AI empires settle theirs at once; the human player's
    // births wait in `pendingGreatPeople` for the app to place into a chosen city.
    int greatPersonPoints = 0;               // banked great-person points
    int greatPeopleBorn = 0;                 // count birthed (raises the next cost)
    std::vector<std::string> pendingGreatPeople;  // born, awaiting a host city
    bool alive = true;
    bool aiManaged = true;                    // false for the human player's empire: the AI
                                              // will not pick its research target, city focus,
                                              // or production queue (the app drives those).

    // running totals (recomputed each turn for reporting)
    int score = 0;
    int totalPopulation = 0;

    [[nodiscard]] bool knows(const std::string& techId) const;
    [[nodiscard]] bool techUnlocked(const std::string& techId) const;  // Locked gate satisfied
    [[nodiscard]] bool techBoosted(const std::string& techId) const;   // Boost gate satisfied
    [[nodiscard]] bool ownsWonder(const struct World& world, const std::string& id) const;
};

// One notable thing that happened on a turn -- the stuff a player would see in
// the event log. Used to measure the reward cadence ("one more turn").
struct GameEvent {
    int turn = 0;
    std::uint8_t empire = 0;
    std::string text;
    enum Kind { Growth, Building, Wonder, WonderLost, Tech, Founded, FireSale, Starve, Disorder, Conquest, Unlock, Eureka, GreatPerson } kind = Building;
};

struct World {
    StrategyMap map;
    std::vector<City> cities;
    std::vector<Empire> empires;
    std::vector<std::string> builtWonders;   // global: each wonder owned once
    std::vector<std::string> bornGreatPeople; // global: each great person born once
    int turn = 0;
    std::uint32_t rng = 0x1234567u;

    std::vector<GameEvent> events;            // chronological event log

    [[nodiscard]] bool wonderTaken(const std::string& id) const;
    [[nodiscard]] bool greatPersonTaken(const std::string& id) const;  // already born this game
    [[nodiscard]] Empire* empireById(std::uint8_t id);
    [[nodiscard]] int cityCount(std::uint8_t empireId) const;
};

// --- Per-turn metrics snapshot (one row per turn) --------------------------
struct TurnSample {
    int turn = 0;
    std::vector<int> score;        // per empire (index = empire id - 1)
    std::vector<int> population;
    std::vector<int> cities;
    std::vector<int> techs;
    std::vector<int> treasury;
    std::vector<int> wonders;
    std::vector<int> disorder;      // per empire: # cities in disorder this turn
    std::uint8_t leader = 0;       // empire id with the top score this turn
};

// --- World construction ----------------------------------------------------

struct WorldConfig {
    std::uint32_t width = 26;
    std::uint32_t height = 20;
    std::uint32_t seed = 1337u;
    int empireCount = 4;
};

// Generate a fresh continent, place one capital per empire on good land spaced
// apart, claim starting territory, and seed personalities.
World makeWorld(const WorldConfig& config);

// Claim the workable tiles within a city's work radius for its owner (sets
// MapTile::owner where currently unowned). Exposed so a host (e.g. the app) can
// seed territory for cities it places without duplicating the radius rule.
void claimCityTerritory(World& world, const City& city);

// --- Simulation ------------------------------------------------------------

// Advance the whole world one turn: AI decisions, yields, growth, production,
// research, expansion, the gold squeeze, wonder resolution. Appends a metrics
// row to `samples`.
void stepTurn(World& world, std::vector<TurnSample>& samples);

// Recompute an empire's score and totals from its cities (also refreshed inside
// stepTurn; exposed for tests / reporting).
void recomputeScore(World& world, Empire& empire);

// Compute a city's yields this turn given its focus, buildings, settled great
// people, the empire's wonders, and happiness. Fills city.yields / happyCap /
// inDisorder.
void computeCityYields(World& world, City& city);

// Settle a (already-born) great person into a city: removes it from the owning
// empire's pending list, marks it taken globally, and records it on the city so its
// bonus applies from next yield computation. Used by the AI at birth and by the app
// when the human player places a pending figure. No-op if the city already hosts it.
void integrateGreatPerson(World& world, City& city, const std::string& greatPersonId);

// The great-person point cost an empire must clear to birth its next figure. Rises
// with each one already birthed (balance.greatPersonBaseCost + born * growth).
[[nodiscard]] int greatPersonCost(const Empire& empire);

}  // namespace odai::game
