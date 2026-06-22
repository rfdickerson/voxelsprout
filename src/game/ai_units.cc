#include "game/ai_units.h"

#include "game/buildable.h"
#include "game/strategy_map.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace odai::game {

namespace {

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

int countUnits(const GameState& gs, std::uint8_t owner, const std::string& typeId) {
    int n = 0;
    for (const Unit& u : gs.units)
        if (u.alive() && u.owner == owner && u.typeId == typeId) ++n;
    return n;
}

int countMilitaryUnits(const GameState& gs, std::uint8_t owner) {
    int n = 0;
    for (const Unit& u : gs.units) {
        if (!u.alive() || u.owner != owner) continue;
        const UnitStats& s = unitStatsFor(u.typeId);
        if (s.attack > 0 || s.rangedAttack > 0) ++n;
    }
    return n;
}

struct TilePos {
    std::uint32_t col = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t row = std::numeric_limits<std::uint32_t>::max();
    bool valid() const { return col != std::numeric_limits<std::uint32_t>::max(); }
};

TilePos nearestPlayerCity(const World& world,
                          std::uint32_t fromCol, std::uint32_t fromRow,
                          std::uint8_t playerOwner) {
    TilePos best{};
    int bestDist = std::numeric_limits<int>::max();
    for (const City& c : world.cities) {
        if (c.owner != playerOwner) continue;
        const int d = hexDistance(static_cast<int>(fromCol), static_cast<int>(fromRow),
                                  static_cast<int>(c.col), static_cast<int>(c.row));
        if (d < bestDist) { bestDist = d; best = {c.col, c.row}; }
    }
    return best;
}

TilePos nearestOwnCity(const World& world,
                       std::uint32_t fromCol, std::uint32_t fromRow,
                       std::uint8_t owner) {
    TilePos best{};
    int bestDist = std::numeric_limits<int>::max();
    for (const City& c : world.cities) {
        if (c.owner != owner) continue;
        const int d = hexDistance(static_cast<int>(fromCol), static_cast<int>(fromRow),
                                  static_cast<int>(c.col), static_cast<int>(c.row));
        if (d < bestDist) { bestDist = d; best = {c.col, c.row}; }
    }
    return best;
}

// Find an own-territory tile that sits on the border with another empire.
// Returns the caller's position if no border tile is found.
TilePos borderAdvanceTarget(const World& world,
                            std::uint32_t fromCol, std::uint32_t fromRow,
                            std::uint8_t owner) {
    TilePos best{ fromCol, fromRow };
    int bestDist = std::numeric_limits<int>::max();
    for (std::uint32_t row = 0; row < world.map.height; ++row) {
        for (std::uint32_t col = 0; col < world.map.width; ++col) {
            if (world.map.at(col, row).owner != owner) continue;
            bool isBorder = false;
            for (int dir = 0; dir < 6; ++dir) {
                int nc = 0, nr = 0;
                if (!tileNeighbor(world.map, static_cast<int>(col),
                                  static_cast<int>(row), dir, nc, nr)) continue;
                const std::uint8_t nb = world.map.at(
                    static_cast<std::uint32_t>(nc),
                    static_cast<std::uint32_t>(nr)).owner;
                if (nb != 0 && nb != owner) { isBorder = true; break; }
            }
            if (!isBorder) continue;
            const int d = hexDistance(static_cast<int>(fromCol),
                                      static_cast<int>(fromRow),
                                      static_cast<int>(col), static_cast<int>(row));
            if (d < bestDist) { bestDist = d; best = {col, row}; }
        }
    }
    return best;
}

// Nearest land tile that is still Hidden (used by scouts to pick an explore target).
TilePos nearestHiddenTile(const StrategyMap& map,
                          std::uint32_t fromCol, std::uint32_t fromRow) {
    TilePos best{};
    int bestDist = std::numeric_limits<int>::max();
    for (std::uint32_t row = 0; row < map.height; ++row) {
        for (std::uint32_t col = 0; col < map.width; ++col) {
            const MapTile& t = map.at(col, row);
            if (t.visibility != TileVisibility::Hidden) continue;
            if (terrainIsWater(t.terrain)) continue;
            const int d = hexDistance(static_cast<int>(fromCol),
                                      static_cast<int>(fromRow),
                                      static_cast<int>(col), static_cast<int>(row));
            if (d < bestDist) { bestDist = d; best = {col, row}; }
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Production decisions
// ---------------------------------------------------------------------------

void decideAiMilitaryProduction(World& world, GameState& gs, const Empire& emp) {
    const int cityCount = world.cityCount(emp.id);
    const float exp = emp.personality.expansion;

    // Desired counts: aggressive civs want warriors proportional to city count;
    // peaceful civs keep a minimal garrison.
    const int desiredWarriors = std::max(
        1,
        static_cast<int>(std::lround(
            static_cast<float>(cityCount) *
            std::clamp(exp * 0.8f, 0.5f, 2.0f))));
    const int desiredScouts = (exp > 1.0f) ? 1 : 0;

    const int actualWarriors = countUnits(gs, emp.id, "warrior") +
                               countUnits(gs, emp.id, "spearman");
    const int actualScouts   = countUnits(gs, emp.id, "scout");

    // Override idle city queues with military production when under target.
    // Never interrupt an in-progress build; only fill empty slots.
    for (std::size_t ci : emp.cityIndices) {
        City& city = world.cities[ci];
        if (!city.producing.empty()) continue;

        std::string unit;
        if (actualWarriors < desiredWarriors) {
            unit = "warrior";
        } else if (actualScouts < desiredScouts) {
            unit = "scout";
        }
        if (unit.empty()) continue;

        // Check barracks/fletcher requirement against the GameState city.
        const CityState* cs = gs.cityAt(city.col, city.row);
        const UnitStats& stats = unitStatsFor(unit);
        if (cs != nullptr && !stats.requiredBuilding.empty() &&
            !cs->hasBuilding(stats.requiredBuilding)) {
            continue;
        }
        city.producing = unit;
        city.accumulated = 0;
        break;  // one override per empire per call — avoid spam
    }
}

// ---------------------------------------------------------------------------
// Movement and attack orders
// ---------------------------------------------------------------------------

void issueAiMovementOrders(World& world, GameState& gs,
                           const Empire& emp, std::uint8_t playerOwner) {
    const float aggressionScore = emp.personality.expansion * 0.7f;

    // Collect unit IDs first so resolveAttack() can erase dead units without
    // invalidating the outer iterator.
    std::vector<std::uint32_t> aiUnitIds;
    aiUnitIds.reserve(gs.units.size());
    for (const Unit& u : gs.units)
        if (u.owner == emp.id && u.alive()) aiUnitIds.push_back(u.id);

    for (std::uint32_t uid : aiUnitIds) {
        Unit* unit = gs.findUnit(uid);
        if (unit == nullptr || !unit->alive()) continue;

        const UnitStats& stats = unitStatsFor(unit->typeId);

        // Combat check: attack an adjacent (or in-range) player unit if possible.
        if (stats.attack > 0 || stats.rangedAttack > 0) {
            bool attacked = false;
            for (const Unit& target : gs.units) {
                if (!target.alive() || target.owner != playerOwner) continue;
                if (canAttack(gs, *unit, target)) {
                    resolveAttack(gs, world.map, unit->id, target.id);
                    attacked = true;
                    break;
                }
            }
            if (attacked) continue;
        }

        // Don't re-order units already on a multi-turn march.
        if (!unit->path.empty()) continue;

        // Scouts: explore toward nearest unseen land tile.
        if (unit->typeId == "scout") {
            const TilePos target = nearestHiddenTile(world.map, unit->col, unit->row);
            if (target.valid()) {
                issueMoveOrder(gs, world.map, *unit, target.col, target.row);
            }
            continue;
        }

        // Military units: personality-driven behavior.
        if (stats.attack > 0 || stats.rangedAttack > 0) {
            if (aggressionScore > 1.0f) {
                // Aggressive (e.g. Genghis, expansion=1.8 → score=1.26):
                // march toward the nearest player city.
                const TilePos target = nearestPlayerCity(world, unit->col, unit->row, playerOwner);
                if (target.valid()) {
                    issueMoveOrder(gs, world.map, *unit, target.col, target.row);
                }
            } else if (aggressionScore > 0.6f) {
                // Moderate (e.g. Augustus 1.05, Hiram 0.84):
                // advance to the nearest own-territory border tile.
                const TilePos target = borderAdvanceTarget(world, unit->col, unit->row, emp.id);
                if (target.col != unit->col || target.row != unit->row) {
                    issueMoveOrder(gs, world.map, *unit, target.col, target.row);
                }
            } else {
                // Passive (Ramesses 0.56, Ashoka 0.42, Pericles 0.35):
                // garrison — stay within 3 hexes of the nearest own city.
                const TilePos home = nearestOwnCity(world, unit->col, unit->row, emp.id);
                if (home.valid()) {
                    const int dist = hexDistance(
                        static_cast<int>(unit->col), static_cast<int>(unit->row),
                        static_cast<int>(home.col), static_cast<int>(home.row));
                    if (dist > 3) {
                        issueMoveOrder(gs, world.map, *unit, home.col, home.row);
                    }
                }
            }
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

void stepAiUnits(World& world, GameState& gs, std::uint8_t playerOwner) {
    for (const Empire& emp : world.empires) {
        if (!emp.alive || !emp.aiManaged) continue;
        decideAiMilitaryProduction(world, gs, emp);
        issueAiMovementOrders(world, gs, emp, playerOwner);
    }
}

}  // namespace odai::game
