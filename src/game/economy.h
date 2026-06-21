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
// handled separately (see cityCenterYields()).
Yields terrainYields(TerrainType terrain, std::uint8_t tileFlags);

// True if a tile can be worked by a citizen at all (mountains/snow/deep desert
// are dead weight -- this is what makes city placement matter).
bool tileIsWorkable(TerrainType terrain);

// --- Research tree ---------------------------------------------------------

// How a tech becomes available. The twist over a flat Civ tree: progress on the
// map can open whole branches.
//   Open   -- standard prereq tech; researchable as soon as its prereqs are known.
//   Boost  -- researchable anytime, but doing the gated deed first discounts its
//             science cost (a Civ-style "Eureka" / "Inspiration").
//   Locked -- CANNOT be researched at all until the deed is done. Entire branches
//             stay dark on the tree until the empire earns them in play.
enum class GateKind : std::uint8_t { Open = 0, Boost, Locked };
const char* gateKindName(GateKind kind);

// A condition tied to in-game accomplishment that opens (Locked) or discounts
// (Boost) a tech. `condition` is a tiny DSL the simulation evaluates against the
// live World; recognized forms:
//   "coastal_city"        own a city adjacent to water
//   "own_wonder"          own any world wonder
//   "meet_rival"          your borders touch another empire's
//   "treasury"            empire treasury >= amount
//   "culture"             accumulated culture >= amount
//   "cities"              number of cities >= amount
//   "pop"                 some city's population >= amount
//   "building:<id>"       own >= amount cities that have building <id>
//   "work_terrain:<name>" own a tile of that terrain (e.g. "hills")
// A condition latches: once satisfied it stays satisfied for the rest of the game.
struct TechGate {
    GateKind kind = GateKind::Open;
    std::string condition;   // "" for Open
    int amount = 0;          // threshold for counted conditions
    int boostPct = 50;       // science discount applied when a Boost is earned
};

// Human-readable phrase for a gate's requirement ("found a coastal city"), used in
// event-log messages and the tech pedia. Empty for Open gates.
std::string gateRequirement(const TechGate& gate);

struct TechDef {
    std::string id;
    std::string name;
    int cost = 0;                       // science points to complete
    std::string era;                    // era bucket for the tree UI (e.g. "Ancient Era")
    std::vector<std::string> prereqs;   // tech ids that must be known first
    std::vector<std::string> unlocks;   // building / wonder ids this enables
    TechGate gate{};                    // how this tech unlocks (defaults to Open)
    std::string description;            // optional rich-text blurb (empty == synthesize from fields)
};

// The full (small, hand-authored) tech tree, in no particular order.
const std::vector<TechDef>& techTree();
const TechDef* findTech(const std::string& id);

// --- Buildings & wonders ---------------------------------------------------

// A declarative bundle of yield bonuses a building (typically a wonder) confers.
// Wonder effects used to be a hardcoded if/else keyed by id in game_sim.cc; they
// now live in data so mods can author new wonders without touching the engine.
//   scope == City   : applies only to the city that owns the building.
//   scope == Empire : applies to every city of the owning empire (wonder behavior).
struct BuildingEffects {
    enum class Scope : std::uint8_t { City = 0, Empire };
    bool present = false;          // false == no effects block authored (skip entirely)
    Scope scope = Scope::City;
    Yields flat;                   // flat yields added per affected city
    int prodPct = 0;               // % bonus to the city's production
    int goldPct = 0;               // % bonus to the city's gold
    int sciencePct = 0;            // % bonus to the city's science
    int happiness = 0;             // raises the city's happiness cap
    int growthBonus = 0;           // % of the food box kept after a city grows
};

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
    BuildingEffects effects{};     // empire-/city-wide bonuses (mainly wonders)
};

// Catalog of economic buildings and wonders (keyed by the same ids the UI /
// CivPedia use where they overlap, e.g. "granary", "library").
const std::vector<BuildingDef>& buildingDefs();
const BuildingDef* findBuildingDef(const std::string& id);

// True if `id` names a wonder in the catalog.
bool isWonder(const std::string& id);

// --- Tunable balance knobs (one place to tweak the whole feel) -------------

// All the integer tuning knobs for the economy + headless simulation, loaded
// from balance.json so mods can rebalance the whole game without recompiling.
// Defaults match the original hardcoded values so a default-constructed Balance
// reproduces the base game.
struct Balance {
    // Economy rules (formerly namespace balance in economy.h).
    int citizenFoodUpkeep = 2;     // food each population point eats/turn
    int foodBoxBase = 12;          // food to grow at population 1
    int foodBoxPerPop = 7;         // extra food per existing population
    int baseHappyCap = 8;          // happiness a brand-new city starts with
    int settlerPopCost = 1;        // population a settler removes when built
    int settlerFoundPop = 1;       // population a founded city starts at
    // Simulation-driver knobs (formerly constexpr in game_sim.cc).
    int settlerCost = 72;          // production for a settler
    int capitalStartPop = 2;       // founding population of a capital
    int rushGoldPerShield = 4;     // gold to rush-buy one production point
    int cityWorkRadius = 2;        // hex radius a city works tiles within
    int settleMinSpacing = 3;      // min hex distance between any two cities
    int sciencePerPopDiv = 3;      // science = base + population / this
    int maxGrowthBonus = 80;       // cap on stacked food-box carryover %
    int civicUpkeepPerCity = 2;    // gold/turn per city past the first
    int starveBuffer = 8;          // food debt a city absorbs before losing pop
    // Great People: an empire accrues great-person points each turn (its cities'
    // culture, plus their science divided by the div below) and births a globally
    // unique figure once the points clear the rising cost.
    int greatPersonBaseCost = 200;        // points to birth the first great person
    int greatPersonCostGrowth = 120;      // added cost per figure this empire already birthed
    int greatPersonSciencePointDiv = 2;   // science contributes science/this to the point pool
};

// The active balance tunables (from the loaded ContentDatabase).
const Balance& balance();

// City-center tile always yields this regardless of terrain (so a 1-pop city is
// never completely dead). Data-driven via terrain.json.
Yields cityCenterYields();

}  // namespace odai::game
