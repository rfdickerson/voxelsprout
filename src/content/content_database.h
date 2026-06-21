#pragma once

#include "game/buildable.h"
#include "game/economy.h"
#include "game/game_sim.h"
#include "game/great_people.h"
#include "game/strategy_map.h"
#include "game/units.h"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// The content layer: all gameplay data (techs, buildings, units, terrain yields,
// balance, CivPedia, leaders) loaded from JSON instead of being hardcoded in C++.
// Pure CPU, no Vulkan. The base game ships as data under mods/base and is loaded
// through the same path user mods will use.
//
// The free accessor functions in game/economy.cc, units.cc, and buildable.cc
// (techTree(), buildingDefs(), getPediaArticle(), terrainYields(), balance(), ...)
// delegate to the process-wide activeContent() database, so existing call sites
// are unchanged.
namespace odai::content {

// Per-terrain base yields + workability (from terrain.json). The river/road gold
// bonuses are applied on top by terrainYields().
struct TerrainYieldDef {
    odai::game::Yields base{};
    bool workable = true;
};

// A single ruleset's worth of authored content. Built by the loader from one or
// more mod directories; later mods override entries by id (Phase 3).
class ContentDatabase {
public:
    // --- accessors (mirror the legacy free functions) -----------------------
    const std::vector<odai::game::TechDef>& techs() const { return m_techs; }
    const odai::game::TechDef* findTech(const std::string& id) const;

    const std::vector<odai::game::BuildingDef>& buildings() const { return m_buildings; }
    const odai::game::BuildingDef* findBuilding(const std::string& id) const;

    const std::vector<odai::game::UnitStats>& units() const { return m_units; }
    const odai::game::UnitStats& unitStatsFor(const std::string& id) const;

    const std::vector<odai::game::BuildableItem>& buildables() const { return m_buildables; }
    const odai::game::BuildableItem* findBuildable(const std::string& id) const;

    // Rich-text CivPedia article for an id, or an empty string when none exists.
    const std::string& pediaArticle(const std::string& id) const;

    const std::vector<odai::game::LeaderDef>& leaders() const { return m_leaders; }

    const std::vector<odai::game::GreatPersonDef>& greatPeople() const { return m_greatPeople; }
    const odai::game::GreatPersonDef* findGreatPerson(const std::string& id) const;

    const odai::game::Balance& balance() const { return m_balance; }

    odai::game::Yields terrainYields(odai::game::TerrainType terrain, std::uint8_t tileFlags) const;
    bool tileIsWorkable(odai::game::TerrainType terrain) const;
    odai::game::Yields cityCenterYields() const { return m_cityCenter; }

    // Non-fatal load diagnostics; empty == clean load.
    const std::vector<std::string>& errors() const { return m_errors; }
    bool ok() const { return m_errors.empty(); }

    // --- mutators used by the loader (keep nlohmann out of this header) ------
    void setBalance(const odai::game::Balance& b) { m_balance = b; }
    void addTech(odai::game::TechDef t) { m_techs.push_back(std::move(t)); }
    void addBuilding(odai::game::BuildingDef d) { m_buildings.push_back(std::move(d)); }
    void addUnit(odai::game::UnitStats u) { m_units.push_back(std::move(u)); }
    void addBuildable(odai::game::BuildableItem b) { m_buildables.push_back(std::move(b)); }
    void setPediaArticle(const std::string& id, std::string text) { m_pedia[id] = std::move(text); }
    void addLeader(odai::game::LeaderDef l) { m_leaders.push_back(std::move(l)); }
    void addGreatPerson(odai::game::GreatPersonDef g) { m_greatPeople.push_back(std::move(g)); }
    void setTerrain(odai::game::TerrainType t, const TerrainYieldDef& def);
    void setCityCenterYields(const odai::game::Yields& y) { m_cityCenter = y; }
    void setRiverGold(int g) { m_riverGold = g; }
    void setRoadGold(int g) { m_roadGold = g; }
    void addError(std::string msg) { m_errors.push_back(std::move(msg)); }

private:
    static constexpr std::size_t kTerrainCount =
        static_cast<std::size_t>(odai::game::TerrainType::Count);

    std::vector<odai::game::TechDef> m_techs;
    std::vector<odai::game::BuildingDef> m_buildings;
    std::vector<odai::game::UnitStats> m_units;
    std::vector<odai::game::BuildableItem> m_buildables;
    std::unordered_map<std::string, std::string> m_pedia;
    std::vector<odai::game::LeaderDef> m_leaders;
    std::vector<odai::game::GreatPersonDef> m_greatPeople;
    odai::game::Balance m_balance{};
    std::array<TerrainYieldDef, kTerrainCount> m_terrain{};
    odai::game::Yields m_cityCenter{2, 1, 1, 1, 0};
    int m_riverGold = 1;
    int m_roadGold = 1;
    std::vector<std::string> m_errors;
};

// --- process-wide active content -------------------------------------------

// The active ruleset. On first use it lazily loads the base game (mods/base) so
// every existing call site keeps working with zero initialization. If a custom
// database has been installed via setActiveContent it is returned instead.
const ContentDatabase& activeContent();

// Override the active ruleset (e.g. for tests or, later, a stacked mod load).
// Pass nullptr to revert to the lazily-loaded base game. The pointed-to database
// must outlive its use.
void setActiveContent(const ContentDatabase* db);

}  // namespace odai::content
