#include "game/units.h"

#include "content/content_database.h"
#include "game/buildable.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <utility>

namespace odai::game {

namespace {

constexpr int kArcherScreenBonus = 2;  // ranged bonus when a melee friend is adjacent.
constexpr int kSmithyArmor = 3;        // armor granted to units built with a Smithy.
constexpr int kCityUnitCap = 4;        // demo: stop a city auto-producing past this.

// Route-planning cost of stepping onto `to` (independent of the provisions debit):
// roads are cheapest so units hug the network, rough terrain is dear so paths bend
// around mountains, and open land sits in between. Min cost is 1 (a road), which is
// also the per-step lower bound the A* heuristic assumes, keeping it admissible.
int pathStepCost(const StrategyMap& map,
                 std::uint32_t fromCol, std::uint32_t fromRow,
                 std::uint32_t toCol, std::uint32_t toRow) {
    const MapTile& from = map.at(fromCol, fromRow);
    const MapTile& to = map.at(toCol, toRow);
    if (((from.flags | to.flags) & TileFlag_Road) != 0u) {
        return 1;
    }
    if (to.terrain == TerrainType::Hills || to.terrain == TerrainType::Mountains) {
        return 3;
    }
    return 2;
}

}  // namespace

const std::vector<UnitStats>& defaultUnitStats() {
    // Data-driven: the unit stats table lives in mods/base/data/units.json.
    return content::activeContent().units();
}

const UnitStats& unitStatsFor(const std::string& id) {
    return content::activeContent().unitStatsFor(id);
}

Unit& GameState::spawnUnit(const std::string& typeId, std::uint32_t col, std::uint32_t row, std::uint8_t owner) {
    const UnitStats& stats = unitStatsFor(typeId);
    Unit unit{};
    unit.id = nextUnitId++;
    unit.typeId = typeId;
    unit.col = col;
    unit.row = row;
    unit.owner = owner;
    unit.hp = stats.maxHp;
    unit.maxHp = stats.maxHp;
    unit.supply = stats.maxSupply;
    unit.maxSupply = stats.maxSupply;
    unit.movementLeft = stats.movement;
    units.push_back(std::move(unit));
    return units.back();
}

Unit* GameState::unitAt(std::uint32_t col, std::uint32_t row) {
    for (Unit& unit : units) {
        if (unit.alive() && unit.col == col && unit.row == row) {
            return &unit;
        }
    }
    return nullptr;
}

const Unit* GameState::unitAt(std::uint32_t col, std::uint32_t row) const {
    for (const Unit& unit : units) {
        if (unit.alive() && unit.col == col && unit.row == row) {
            return &unit;
        }
    }
    return nullptr;
}

Unit* GameState::findUnit(std::uint32_t id) {
    for (Unit& unit : units) {
        if (unit.id == id) {
            return &unit;
        }
    }
    return nullptr;
}

bool CityState::hasBuilding(const std::string& id) const {
    return std::find(buildings.begin(), buildings.end(), id) != buildings.end();
}

void GameState::initCities(const StrategyMap& map) {
    cities.clear();
    cities.reserve(map.settlements.size());
    for (std::size_t i = 0; i < map.settlements.size(); ++i) {
        const Settlement& s = map.settlements[i];
        CityState city{};
        city.settlementIndex = static_cast<std::uint32_t>(i);
        city.col = s.col;
        city.row = s.row;
        city.owner = s.owner;
        cities.push_back(std::move(city));
    }
}

CityState* GameState::cityAt(std::uint32_t col, std::uint32_t row) {
    for (CityState& city : cities) {
        if (city.col == col && city.row == row) {
            return &city;
        }
    }
    return nullptr;
}

int supplyCostForStep(const StrategyMap& map,
                      std::uint32_t fromCol, std::uint32_t fromRow,
                      std::uint32_t toCol, std::uint32_t toRow) {
    const MapTile& from = map.at(fromCol, fromRow);
    const MapTile& to = map.at(toCol, toRow);
    // Roads are free: travelling along the road network costs no provisions.
    if (((from.flags | to.flags) & TileFlag_Road) != 0u) {
        return 0;
    }
    // Rough terrain drains provisions twice as fast -- the Alps toll.
    if (to.terrain == TerrainType::Hills || to.terrain == TerrainType::Mountains) {
        return 2;
    }
    return 1;
}

bool isNearFriendlySettlement(const StrategyMap& map,
                              std::uint32_t col, std::uint32_t row,
                              std::uint8_t owner) {
    for (const Settlement& settlement : map.settlements) {
        if (settlement.owner != owner) {
            continue;
        }
        if (hexDistance(static_cast<int>(col), static_cast<int>(row),
                        static_cast<int>(settlement.col), static_cast<int>(settlement.row)) <= 1) {
            return true;
        }
    }
    return false;
}

MoveResult moveUnitStep(GameState& gs, const StrategyMap& map, Unit& unit,
                        std::uint32_t toCol, std::uint32_t toRow) {
    if (!map.inBounds(static_cast<int>(toCol), static_cast<int>(toRow))) {
        return MoveResult::OffMap;
    }

    // Destination must be one of the six immediate neighbors (odd-r topology).
    bool adjacent = false;
    for (int direction = 0; direction < 6; ++direction) {
        int nc = 0;
        int nr = 0;
        if (tileNeighbor(map, static_cast<int>(unit.col), static_cast<int>(unit.row), direction, nc, nr) &&
            nc == static_cast<int>(toCol) && nr == static_cast<int>(toRow)) {
            adjacent = true;
            break;
        }
    }
    if (!adjacent) {
        return MoveResult::NotAdjacent;
    }
    if (terrainIsWater(map.at(toCol, toRow).terrain)) {
        return MoveResult::NotLand;
    }
    if (unit.movementLeft <= 0) {
        return MoveResult::OutOfMoves;
    }

    const Unit* occupant = gs.unitAt(toCol, toRow);
    if (occupant != nullptr && occupant->id != unit.id) {
        return MoveResult::Occupied;
    }

    const int cost = supplyCostForStep(map, unit.col, unit.row, toCol, toRow);
    unit.movementLeft -= 1;
    unit.supply = std::max(0, unit.supply - cost);
    unit.col = toCol;
    unit.row = toRow;

    // Passing through a friendly settlement tops the wagons back up.
    if (isNearFriendlySettlement(map, unit.col, unit.row, unit.owner)) {
        unit.supply = unit.maxSupply;
    }
    return MoveResult::Ok;
}

std::vector<std::array<std::uint32_t, 2>> findHexPath(
        const StrategyMap& map, const GameState& gs,
        std::uint32_t startCol, std::uint32_t startRow,
        std::uint32_t goalCol, std::uint32_t goalRow) {
    std::vector<std::array<std::uint32_t, 2>> path;
    if (!map.inBounds(static_cast<int>(startCol), static_cast<int>(startRow)) ||
        !map.inBounds(static_cast<int>(goalCol), static_cast<int>(goalRow))) {
        return path;
    }
    if (startCol == goalCol && startRow == goalRow) {
        return path;
    }
    // Can't end a march on water or on top of another unit.
    if (terrainIsWater(map.at(goalCol, goalRow).terrain) || gs.unitAt(goalCol, goalRow) != nullptr) {
        return path;
    }

    const std::uint32_t W = map.width;
    const std::size_t N = static_cast<std::size_t>(W) * static_cast<std::size_t>(map.height);
    const auto idxOf = [W](std::uint32_t c, std::uint32_t r) {
        return static_cast<std::size_t>(r) * static_cast<std::size_t>(W) + static_cast<std::size_t>(c);
    };
    const std::size_t startIdx = idxOf(startCol, startRow);
    const std::size_t goalIdx = idxOf(goalCol, goalRow);

    constexpr int kInf = std::numeric_limits<int>::max();
    std::vector<int> gScore(N, kInf);
    std::vector<std::size_t> cameFrom(N, N);  // N == no parent yet.
    std::vector<char> closed(N, 0);

    const auto heuristic = [&](std::uint32_t c, std::uint32_t r) {
        return hexDistance(static_cast<int>(c), static_cast<int>(r),
                           static_cast<int>(goalCol), static_cast<int>(goalRow));
    };

    // Min-heap on (f = g + h, tile index). std::greater makes the smallest f pop first.
    using Node = std::pair<int, std::size_t>;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;
    gScore[startIdx] = 0;
    open.push({heuristic(startCol, startRow), startIdx});

    bool found = false;
    while (!open.empty()) {
        const std::size_t current = open.top().second;
        open.pop();
        if (current == goalIdx) {
            found = true;
            break;
        }
        if (closed[current]) {
            continue;  // stale heap entry
        }
        closed[current] = 1;

        const int curCol = static_cast<int>(current % W);
        const int curRow = static_cast<int>(current / W);
        for (int dir = 0; dir < 6; ++dir) {
            int nc = 0;
            int nr = 0;
            if (!tileNeighbor(map, curCol, curRow, dir, nc, nr)) {
                continue;
            }
            const std::uint32_t ncu = static_cast<std::uint32_t>(nc);
            const std::uint32_t nru = static_cast<std::uint32_t>(nr);
            const std::size_t nIdx = idxOf(ncu, nru);
            if (closed[nIdx] || terrainIsWater(map.at(ncu, nru).terrain)) {
                continue;
            }
            // Never path through a tile occupied by another unit (no stacking).
            if (nIdx != goalIdx && gs.unitAt(ncu, nru) != nullptr) {
                continue;
            }
            const int tentative = gScore[current] +
                pathStepCost(map, static_cast<std::uint32_t>(curCol), static_cast<std::uint32_t>(curRow), ncu, nru);
            if (tentative < gScore[nIdx]) {
                gScore[nIdx] = tentative;
                cameFrom[nIdx] = current;
                open.push({tentative + heuristic(ncu, nru), nIdx});
            }
        }
    }

    if (!found) {
        return path;
    }
    // Walk parents from goal back to start, then reverse into forward order.
    for (std::size_t node = goalIdx; node != startIdx; node = cameFrom[node]) {
        path.push_back({static_cast<std::uint32_t>(node % W), static_cast<std::uint32_t>(node / W)});
        if (cameFrom[node] == N) {  // broken chain (shouldn't happen): bail safely
            path.clear();
            return path;
        }
    }
    std::reverse(path.begin(), path.end());
    return path;
}

void followPath(GameState& gs, const StrategyMap& map, Unit& unit) {
    while (!unit.path.empty() && unit.movementLeft > 0) {
        const std::array<std::uint32_t, 2> next = unit.path.front();
        const MoveResult result = moveUnitStep(gs, map, unit, next[0], next[1]);
        if (result == MoveResult::Ok) {
            unit.path.erase(unit.path.begin());
            continue;
        }
        if (result == MoveResult::OutOfMoves) {
            break;  // keep the path; resume next turn with fresh movement
        }
        unit.path.clear();  // blocked or now-illegal: cancel the order
        break;
    }
}

void issueMoveOrder(GameState& gs, const StrategyMap& map, Unit& unit,
                    std::uint32_t goalCol, std::uint32_t goalRow) {
    unit.path = findHexPath(map, gs, unit.col, unit.row, goalCol, goalRow);
    followPath(gs, map, unit);  // begin moving immediately this turn
}

FreeTile findFreeNeighbor(const StrategyMap& map, const GameState& gs,
                          std::uint32_t col, std::uint32_t row) {
    for (int dir = 0; dir < 6; ++dir) {
        int nc = 0;
        int nr = 0;
        if (!tileNeighbor(map, static_cast<int>(col), static_cast<int>(row), dir, nc, nr)) {
            continue;
        }
        const std::uint32_t ncu = static_cast<std::uint32_t>(nc);
        const std::uint32_t nru = static_cast<std::uint32_t>(nr);
        if (terrainIsWater(map.at(ncu, nru).terrain) || gs.unitAt(ncu, nru) != nullptr) {
            continue;
        }
        return FreeTile{true, ncu, nru};
    }
    // Fall back to the city's own tile if it happens to be free land.
    if (map.inBounds(static_cast<int>(col), static_cast<int>(row)) &&
        !terrainIsWater(map.at(col, row).terrain) && gs.unitAt(col, row) == nullptr) {
        return FreeTile{true, col, row};
    }
    return FreeTile{};
}

bool cityCanProduce(const CityState& city, const std::string& unitId) {
    const UnitStats& stats = unitStatsFor(unitId);
    if (stats.requiredBuilding.empty()) {
        return true;
    }
    return city.hasBuilding(stats.requiredBuilding);
}

std::string nextAutoProduction(const CityState& city) {
    // Military rotation; a city builds whichever of these its buildings allow.
    static const std::array<const char*, 3> kRotation = {"spearman", "archer", "warrior"};
    std::vector<std::string> options;
    for (const char* id : kRotation) {
        if (cityCanProduce(city, id)) {
            options.push_back(id);
        }
    }
    if (options.empty()) {
        return std::string{};
    }
    return options[static_cast<std::size_t>(city.unitsProduced) % options.size()];
}

int archerAdjacencyBonus(const GameState& gs, const StrategyMap& map, const Unit& archer) {
    if (unitStatsFor(archer.typeId).rangedAttack <= 0) {
        return 0;
    }
    for (int dir = 0; dir < 6; ++dir) {
        int nc = 0;
        int nr = 0;
        if (!tileNeighbor(map, static_cast<int>(archer.col), static_cast<int>(archer.row), dir, nc, nr)) {
            continue;
        }
        const Unit* friendUnit = gs.unitAt(static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr));
        if (friendUnit != nullptr && friendUnit->owner == archer.owner &&
            unitStatsFor(friendUnit->typeId).melee) {
            return kArcherScreenBonus;  // shielded from behind a pikeman: fire freely
        }
    }
    return 0;
}

bool canAttack(const GameState& gs, const Unit& attacker, const Unit& defender) {
    (void)gs;
    if (!attacker.alive() || !defender.alive() || attacker.owner == defender.owner ||
        attacker.movementLeft <= 0) {
        return false;
    }
    const UnitStats& as = unitStatsFor(attacker.typeId);
    const int dist = hexDistance(static_cast<int>(attacker.col), static_cast<int>(attacker.row),
                                 static_cast<int>(defender.col), static_cast<int>(defender.row));
    if (as.rangedAttack > 0 && as.range > 0) {
        return dist >= 1 && dist <= as.range;
    }
    if (as.attack > 0) {
        return dist == 1;
    }
    return false;
}

AttackResult resolveAttack(GameState& gs, const StrategyMap& map,
                           std::uint32_t attackerId, std::uint32_t defenderId) {
    Unit* att = gs.findUnit(attackerId);
    Unit* def = gs.findUnit(defenderId);
    if (att == nullptr || def == nullptr || !att->alive() || !def->alive()) {
        return AttackResult::Invalid;
    }
    if (att->owner == def->owner) {
        return AttackResult::NotEnemy;
    }
    if (att->movementLeft <= 0) {
        return AttackResult::AlreadyActed;
    }

    const UnitStats& as = unitStatsFor(att->typeId);
    const int dist = hexDistance(static_cast<int>(att->col), static_cast<int>(att->row),
                                 static_cast<int>(def->col), static_cast<int>(def->row));
    const bool ranged = as.rangedAttack > 0 && as.range > 0;
    if (ranged) {
        if (dist < 1 || dist > as.range) {
            return AttackResult::OutOfRange;
        }
        const int dmg = std::max(1, as.rangedAttack + archerAdjacencyBonus(gs, map, *att) - def->armor);
        def->hp -= dmg;  // ranged fire: no retaliation
    } else {
        if (as.attack <= 0) {
            return AttackResult::NoAttack;
        }
        if (dist != 1) {
            return AttackResult::NotAdjacent;
        }
        def->hp -= std::max(1, as.attack - def->armor);
        if (def->alive()) {
            const UnitStats& ds = unitStatsFor(def->typeId);
            if (ds.attack > 0) {  // melee defender hits back
                att->hp -= std::max(1, ds.attack - att->armor);
            }
        }
    }

    att->movementLeft = 0;  // attacking ends the unit's turn
    att->path.clear();
    std::erase_if(gs.units, [](const Unit& u) { return u.hp <= 0; });
    return AttackResult::Ok;
}

void advanceTurn(GameState& gs, const StrategyMap& map, TurnConfig cfg) {
    for (Unit& unit : gs.units) {
        if (!unit.alive()) {
            continue;
        }
        if (isNearFriendlySettlement(map, unit.col, unit.row, unit.owner)) {
            // Resupplied and resting: refill provisions, mend wounds.
            unit.supply = unit.maxSupply;
            unit.hp = std::min(unit.maxHp, unit.hp + cfg.regenPerTurn);
        } else if (unit.supply == 0) {
            // Starving in the wilderness: the march takes its toll.
            unit.hp -= cfg.attritionPerTurn;
        }
        unit.movementLeft = unitStatsFor(unit.typeId).movement;
    }
    std::erase_if(gs.units, [](const Unit& unit) { return unit.hp <= 0; });

    // Continue queued move orders with the refreshed movement allowance so long
    // marches traverse over several turns (draining supply the whole way).
    for (Unit& unit : gs.units) {
        if (unit.alive() && !unit.path.empty()) {
            followPath(gs, map, unit);
        }
    }

    // City production: bank shields, and when an item completes either add a
    // building or place a freshly built unit on a free neighbor (one per tile).
    // Cities with no production queued are left idle so the player can choose.
    for (CityState& city : gs.cities) {
        if (city.producing.empty()) {
            continue;
        }
        city.accumulated += city.perTurn;
        const BuildableItem* item = findBuildable(city.producing);
        if (item == nullptr) {
            city.producing.clear();
            city.accumulated = 0;
            continue;
        }
        if (city.accumulated < item->productionCost) {
            continue;  // still under construction
        }
        if (item->kind == BuildableKind::Building) {
            if (!city.hasBuilding(item->id)) {
                city.buildings.push_back(item->id);
            }
            city.accumulated -= item->productionCost;
            city.producing.clear();
        } else {
            if (!cityCanProduce(city, item->id)) {
                city.producing.clear();
                city.accumulated = 0;
                continue;
            }
            const FreeTile spot = findFreeNeighbor(map, gs, city.col, city.row);
            if (!spot.found) {
                continue;  // no room around the city; hold shields until a tile frees
            }
            Unit& produced = gs.spawnUnit(item->id, spot.col, spot.row, city.owner);
            if (city.hasBuilding("smithy")) {
                produced.armor = kSmithyArmor;  // smithy forges armor for new units
            }
            city.accumulated -= item->productionCost;
            city.unitsProduced += 1;
            city.producing.clear();
        }
    }
}

}  // namespace odai::game
