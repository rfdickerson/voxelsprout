#include "content/content_database.h"

#include "content/mod_loader.h"

namespace odai::content {

using odai::game::BuildableItem;
using odai::game::BuildingDef;
using odai::game::GreatPersonDef;
using odai::game::ReligionDef;
using odai::game::TechDef;
using odai::game::TerrainType;
using odai::game::UnitStats;
using odai::game::Yields;

void ContentDatabase::setTerrain(TerrainType t, const TerrainYieldDef& def) {
    const auto idx = static_cast<std::size_t>(t);
    if (idx < m_terrain.size()) {
        m_terrain[idx] = def;
    }
}

const TechDef* ContentDatabase::findTech(const std::string& id) const {
    for (const TechDef& t : m_techs) {
        if (t.id == id) return &t;
    }
    return nullptr;
}

const BuildingDef* ContentDatabase::findBuilding(const std::string& id) const {
    for (const BuildingDef& d : m_buildings) {
        if (d.id == id) return &d;
    }
    return nullptr;
}

const UnitStats& ContentDatabase::unitStatsFor(const std::string& id) const {
    for (const UnitStats& stats : m_units) {
        if (stats.id == id) return stats;
    }
    static const UnitStats kFallback{"", 30, 2, 5};
    return kFallback;
}

const BuildableItem* ContentDatabase::findBuildable(const std::string& id) const {
    for (const BuildableItem& item : m_buildables) {
        if (item.id == id) return &item;
    }
    return nullptr;
}

const GreatPersonDef* ContentDatabase::findGreatPerson(const std::string& id) const {
    for (const GreatPersonDef& g : m_greatPeople) {
        if (g.id == id) return &g;
    }
    return nullptr;
}

const ReligionDef* ContentDatabase::findReligion(const std::string& id) const {
    for (const ReligionDef& r : m_religions) {
        if (r.id == id) return &r;
    }
    return nullptr;
}

const std::string& ContentDatabase::pediaArticle(const std::string& id) const {
    static const std::string kEmpty;
    auto it = m_pedia.find(id);
    return it != m_pedia.end() ? it->second : kEmpty;
}

Yields ContentDatabase::terrainYields(TerrainType terrain, std::uint8_t tileFlags) const {
    const auto idx = static_cast<std::size_t>(terrain);
    Yields y = (idx < m_terrain.size()) ? m_terrain[idx].base : Yields{};
    if ((tileFlags & odai::game::TileFlag_River) != 0u) y.gold += m_riverGold;
    if ((tileFlags & odai::game::TileFlag_Road) != 0u) y.gold += m_roadGold;
    return y;
}

bool ContentDatabase::tileIsWorkable(TerrainType terrain) const {
    const auto idx = static_cast<std::size_t>(terrain);
    return (idx < m_terrain.size()) ? m_terrain[idx].workable : true;
}

// --- process-wide active content -------------------------------------------

namespace {
const ContentDatabase* g_active = nullptr;
}  // namespace

const ContentDatabase& activeContent() {
    // Lazily loaded base game; thread-safe static initialization. Built once on
    // first access so tools/tests/app need no explicit init step.
    static const ContentDatabase base = loadBaseContent();
    return g_active != nullptr ? *g_active : base;
}

void setActiveContent(const ContentDatabase* db) {
    g_active = db;
}

}  // namespace odai::content
