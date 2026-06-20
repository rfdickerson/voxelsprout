#pragma once

#include "game/strategy_map.h"

#include <cstdint>
#include <string>
#include <vector>

// Strategic economy model: the empire-building layer that sits on top of the
// hex StrategyMap. This is the "4X economy" -- yields, population growth,
// happiness, a research tree, buildings, and world wonders. Pure CPU data with
// no Vulkan / renderer / UI types so it can be driven by a headless simulation
// and unit-tested in isolation.
//
// Design intent (the "one more turn" loop): every city, every turn, faces a
// single production queue and a single citizen-focus knob. Spending a turn
// growing is a turn not spent building; banking a wonder is shields not spent on
// the temple your unhappy city needs. The tension between food / production /
// gold / science / happiness is the whole game.
namespace odai::game {

// The five core per-turn yields a city produces.
struct Yields {
    int food = 0;
    int production = 0;
    int gold = 0;
    int science = 0;
    int culture = 0;
};

// How a city assigns its citizens to surrounding tiles. The single most
// important per-turn decision: grow now, or build now?
enum class CityFocus : std::uint8_t {
    Balanced = 0,
    Food,        // grow population as fast as possible
    Production,  // rush a building / wonder
    Gold,        // fill the treasury
    Count
};
const char* cityFocusName(CityFocus focus);

// Base yield a single citizen extracts from a worked tile of this terrain,
// including the +gold bonuses rivers and roads add. The city center tile is
// handled separately (see kCityCenterYields).
Yields terrainYields(TerrainType terrain, std::uint8_t tileFlags);

// True if a tile can be worked by a citizen at all (mountains/snow/deep desert
// are dead weight -- this is what makes city placement matter).
bool tileIsWorkable(TerrainType terrain);

// --- Research tree ---------------------------------------------------------

struct TechDef {
    std::string id;
    std::string name;
    int cost = 0;                       // science points to complete
    std::vector<std::string> prereqs;   // tech ids that must be known first
    std::vector<std::string> unlocks;   // building / wonder ids this enables
};

// The full (small, hand-authored) tech tree, in no particular order.
const std::vector<TechDef>& techTree();
const TechDef* findTech(const std::string& id);

// --- Buildings & wonders ---------------------------------------------------

struct BuildingDef {
    std::string id;
    std::string name;
    int productionCost = 0;
    int maintenance = 0;            // gold/turn drained from the treasury
    Yields flat;                    // flat yields added to the city
    int prodPct = 0;               // % bonus to the city's production
    int goldPct = 0;               // % bonus to the city's gold
    int sciencePct = 0;            // % bonus to the city's science
    int happiness = 0;             // raises the city's happiness cap
    int growthBonus = 0;           // % of the food box kept after a city grows
    std::string requiredTech;      // "" == available from the first turn
    bool isWonder = false;         // world-unique; only one civ may own it
    int score = 0;                 // points the wonder adds to its owner's score
};

// Catalog of economic buildings and wonders (keyed by the same ids the UI /
// CivPedia use where they overlap, e.g. "granary", "library").
const std::vector<BuildingDef>& buildingDefs();
const BuildingDef* findBuildingDef(const std::string& id);

// True if `id` names a wonder in the catalog.
bool isWonder(const std::string& id);

// --- Tunable balance knobs (one place to tweak the whole feel) -------------

namespace balance {
constexpr int kCitizenFoodUpkeep = 2;     // food each population point eats/turn
constexpr int kFoodBoxBase = 12;          // food to grow at population 1
constexpr int kFoodBoxPerPop = 7;         // extra food per existing population
constexpr int kBaseHappyCap = 8;          // happiness a brand-new city starts with
constexpr int kSettlerPopCost = 1;        // population a settler removes when built
constexpr int kSettlerFoundPop = 1;       // population a founded city starts at
}  // namespace balance

// City-center tile always yields this regardless of terrain (so a 1-pop city is
// never completely dead).
inline constexpr Yields kCityCenterYields{2, 1, 1, 1, 0};

}  // namespace odai::game
