#pragma once

#include "game/strategy_map.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Live units that march across a StrategyMap, plus the supply/attrition rules
// that make long marches costly. Pure CPU data with no Vulkan or renderer types;
// units are intentionally NOT part of StrategyMap (which is static terrain) and
// are NOT serialized into .smap -- they are spawned at runtime / in tests.
namespace odai::game {

// Machine-readable per-type gameplay stats. The CivPedia text in buildable.cc has
// these numbers in prose only; this table is the authoritative gameplay source.
struct UnitStats {
    std::string id;        // matches BuildableItem::id, e.g. "warrior".
    int maxHp = 30;
    int movement = 2;      // movement points (open-terrain hexes) per turn.
    int maxSupply = 5;     // hexes of wilderness travel before provisions run out.
    int attack = 0;        // melee combat strength (0 == non-combatant).
    int rangedAttack = 0;  // ranged combat strength (0 == cannot fire).
    int range = 0;         // ranged reach in hexes (0 == melee only).
    bool melee = false;    // a melee soldier: can screen archers and needs a barracks.
    std::string requiredBuilding;  // building the producing city must have ("" == none).
};

// Stats table for the playable unit types (warrior/spearman/archer/scout/...).
const std::vector<UnitStats>& defaultUnitStats();

// Stats for a unit type id; returns a safe default if the id is unknown.
const UnitStats& unitStatsFor(const std::string& id);

// A live unit standing on one hex of the strategy map.
struct Unit {
    std::uint32_t id = 0;    // stable handle for selection; 0 == invalid.
    std::string typeId;      // indexes unitStatsFor(), e.g. "warrior".
    std::uint32_t col = 0;
    std::uint32_t row = 0;
    std::uint8_t owner = 0;  // matches MapTile::owner / Settlement::owner.
    int hp = 30;
    int maxHp = 30;
    int supply = 5;          // provisions remaining; drains as the unit marches.
    int maxSupply = 5;
    int movementLeft = 2;    // resets to UnitStats::movement each turn.
    int armor = 0;           // flat damage reduction; granted by a Smithy at build time.

    // Remaining waypoints of an in-progress move order (each adjacent to the last,
    // goal last). Empty when the unit is idle or has arrived. The unit advances
    // along this each turn until it runs out of movement or the path is consumed.
    std::vector<std::array<std::uint32_t, 2>> path;

    [[nodiscard]] bool alive() const { return hp > 0; }
};

// Per-city mutable state, parallel to map.settlements (built by initCities). Holds
// the completed buildings, the item under production, and accumulated shields.
struct CityState {
    std::uint32_t settlementIndex = 0;
    std::uint32_t col = 0;
    std::uint32_t row = 0;
    std::uint8_t owner = 0;
    std::vector<std::string> buildings;  // completed building ids (e.g. "barracks").
    std::string producing;               // buildable id in progress ("" == idle).
    int accumulated = 0;                 // production shields banked toward `producing`.
    int perTurn = 5;                     // shields produced per turn.
    int unitsProduced = 0;               // lifetime count (used to cap demo output).

    [[nodiscard]] bool hasBuilding(const std::string& id) const;
};

// Owns the live units and cities. References (does not own) the map for rules.
struct GameState {
    std::vector<Unit> units;
    std::vector<CityState> cities;
    std::uint32_t nextUnitId = 1;

    // Spawn a unit of the given type at a tile, fully provisioned and healed.
    Unit& spawnUnit(const std::string& typeId, std::uint32_t col, std::uint32_t row, std::uint8_t owner);

    // First live unit on a tile, or nullptr.
    [[nodiscard]] Unit* unitAt(std::uint32_t col, std::uint32_t row);
    [[nodiscard]] const Unit* unitAt(std::uint32_t col, std::uint32_t row) const;

    // Live unit by id, or nullptr.
    [[nodiscard]] Unit* findUnit(std::uint32_t id);

    // Populate `cities` from the map's settlements (clears any existing entries).
    void initCities(const StrategyMap& map);

    // City on a tile, or nullptr.
    [[nodiscard]] CityState* cityAt(std::uint32_t col, std::uint32_t row);
};

// --- Supply / movement rules ------------------------------------------------

// Provisions spent stepping from a tile to an ADJACENT tile:
//   0 if either endpoint carries a road, 2 onto hills/mountains, 1 otherwise.
[[nodiscard]] int supplyCostForStep(const StrategyMap& map,
                                    std::uint32_t fromCol, std::uint32_t fromRow,
                                    std::uint32_t toCol, std::uint32_t toRow);

// True if the tile is on, or adjacent to, a friendly settlement (refill range).
[[nodiscard]] bool isNearFriendlySettlement(const StrategyMap& map,
                                            std::uint32_t col, std::uint32_t row,
                                            std::uint8_t owner);

enum class MoveResult {
    Ok,
    NotAdjacent,
    OutOfMoves,
    OffMap,
    NotLand,
    Occupied,
};

// Move a unit one hex to an adjacent tile. Debits one movement point and the
// supply cost, then refills to max if the destination is in settlement range.
// Does NOT apply attrition -- that happens in advanceTurn at end of turn.
MoveResult moveUnitStep(GameState& gs, const StrategyMap& map, Unit& unit,
                        std::uint32_t toCol, std::uint32_t toRow);

// --- Pathfinding / move orders ----------------------------------------------

// A* path of adjacent hexes from (startCol,startRow) to (goalCol,goalRow) over
// land tiles, preferring roads and bending around rough terrain (and never through
// water or tiles occupied by another unit). The returned path excludes the start
// and ends at the goal; empty if start==goal or the goal is unreachable/occupied.
[[nodiscard]] std::vector<std::array<std::uint32_t, 2>> findHexPath(
    const StrategyMap& map, const GameState& gs,
    std::uint32_t startCol, std::uint32_t startRow,
    std::uint32_t goalCol, std::uint32_t goalRow);

// Advance a unit along its stored path while it has movement and each step is
// legal. Stops (keeping the path) when movement runs out so the order resumes next
// turn; clears the path if a step becomes blocked or on arrival.
void followPath(GameState& gs, const StrategyMap& map, Unit& unit);

// Plan an A* path to the goal, store it on the unit, and begin moving immediately
// with the unit's current movement allowance.
void issueMoveOrder(GameState& gs, const StrategyMap& map, Unit& unit,
                    std::uint32_t goalCol, std::uint32_t goalRow);

// --- Cities / production ----------------------------------------------------

// A free, in-bounds land tile with no unit on it (for placing a freshly built unit).
struct FreeTile {
    bool found = false;
    std::uint32_t col = 0;
    std::uint32_t row = 0;
};

// First unoccupied land neighbor of (col,row); falls back to (col,row) itself if it
// is free land. Used to place a city's newly produced unit (one unit per tile).
[[nodiscard]] FreeTile findFreeNeighbor(const StrategyMap& map, const GameState& gs,
                                        std::uint32_t col, std::uint32_t row);

// True if a city may produce the given unit type: its required building (Barracks
// for melee, Fletcher for archers) must be present. Civilians/recon need none.
[[nodiscard]] bool cityCanProduce(const CityState& city, const std::string& unitId);

// Next military unit a city should auto-produce given its buildings, cycling so a
// well-equipped city alternates types. Empty string if it can build none.
[[nodiscard]] std::string nextAutoProduction(const CityState& city);

// --- Combat -----------------------------------------------------------------

enum class AttackResult {
    Ok,
    OutOfRange,
    NotAdjacent,
    NotEnemy,
    NoAttack,
    AlreadyActed,
    Invalid,
};

// Archer formation bonus: +bonus to ranged strength when a friendly melee soldier
// stands adjacent (archers firing from behind pikemen). 0 for non-archers.
[[nodiscard]] int archerAdjacencyBonus(const GameState& gs, const StrategyMap& map, const Unit& archer);

// True if `attacker` could strike `defender` right now: a living enemy, in range
// (ranged) or adjacent (melee), with the attacker's action still available.
[[nodiscard]] bool canAttack(const GameState& gs, const Unit& attacker, const Unit& defender);

// Resolve an attack by unit id. Ranged fire draws no retaliation; melee provokes a
// counterattack if the defender survives. Armor reduces incoming damage; archers
// add their adjacency bonus. Spends the attacker's turn and removes the dead.
AttackResult resolveAttack(GameState& gs, const StrategyMap& map,
                           std::uint32_t attackerId, std::uint32_t defenderId);

// --- Turn resolution --------------------------------------------------------

struct TurnConfig {
    int attritionPerTurn = 10;  // HP lost per turn at zero supply in the wild.
    int regenPerTurn = 5;       // HP healed per turn resting at a friendly settlement.
};

// Advance one turn for every unit: resupply + regen at friendly settlements,
// else attrition at zero supply, then reset movement. Dead units are removed.
// Resupply/regen are applied before attrition so a unit that reaches a
// settlement this turn is saved.
void advanceTurn(GameState& gs, const StrategyMap& map, TurnConfig cfg = {});

}  // namespace odai::game
