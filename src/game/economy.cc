#include "game/economy.h"

#include "content/content_database.h"

// The economy catalog is now data-driven: tech tree, building/wonder catalog,
// terrain yields, and balance tunables all live in JSON under mods/base/data and
// are served by the active ContentDatabase. These accessors keep their original
// signatures so existing call sites are unchanged; they simply delegate to the
// loaded content. The pure helper functions (name lookups, the gate-requirement
// phrase) stay here because they are logic, not data.
namespace odai::game {

namespace {
const content::ContentDatabase& db() { return content::activeContent(); }
}  // namespace

const char* cityFocusName(CityFocus focus) {
    switch (focus) {
        case CityFocus::Balanced:   return "Balanced";
        case CityFocus::Food:       return "Food";
        case CityFocus::Production: return "Production";
        case CityFocus::Gold:       return "Gold";
        case CityFocus::Count:      break;
    }
    return "?";
}

bool tileIsWorkable(TerrainType terrain) {
    return db().tileIsWorkable(terrain);
}

Yields terrainYields(TerrainType terrain, std::uint8_t tileFlags) {
    return db().terrainYields(terrain, tileFlags);
}

const std::vector<TechDef>& techTree() {
    return db().techs();
}

const TechDef* findTech(const std::string& id) {
    return db().findTech(id);
}

const char* gateKindName(GateKind kind) {
    switch (kind) {
        case GateKind::Open:   return "Open";
        case GateKind::Boost:  return "Boost";
        case GateKind::Locked: return "Locked";
    }
    return "?";
}

std::string gateRequirement(const TechGate& gate) {
    const std::string& c = gate.condition;
    const std::string n = std::to_string(gate.amount);
    if (c.empty())                  return "";
    if (c == "coastal_city")        return "found a coastal city";
    if (c == "own_wonder")          return "own a world wonder";
    if (c == "meet_rival")          return "meet another empire";
    if (c == "treasury")            return "bank " + n + " gold";
    if (c == "culture")             return "earn " + n + " culture";
    if (c == "cities")              return "found " + n + " cities";
    if (c == "pop")                 return "grow a city to " + n + " pop";
    if (c.starts_with("building:"))
        return "build " + n + "x " + c.substr(9);
    if (c.starts_with("work_terrain:"))
        return "work a " + c.substr(13) + " tile";
    return c;
}

const std::vector<BuildingDef>& buildingDefs() {
    return db().buildings();
}

const BuildingDef* findBuildingDef(const std::string& id) {
    return db().findBuilding(id);
}

bool isWonder(const std::string& id) {
    const BuildingDef* d = findBuildingDef(id);
    return d != nullptr && d->isWonder;
}

const Balance& balance() {
    return db().balance();
}

Yields cityCenterYields() {
    return db().cityCenterYields();
}

}  // namespace odai::game
