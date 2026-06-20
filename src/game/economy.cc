#include "game/economy.h"

namespace odai::game {

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
    switch (terrain) {
        case TerrainType::Mountains:
        case TerrainType::Snow:
            return false;
        default:
            return true;
    }
}

Yields terrainYields(TerrainType terrain, std::uint8_t tileFlags) {
    Yields y{};
    switch (terrain) {
        case TerrainType::Grassland: y.food = 3; break;
        case TerrainType::Plains:    y.food = 1; y.production = 1; break;
        case TerrainType::Forest:    y.food = 1; y.production = 2; break;
        case TerrainType::Jungle:    y.food = 1; y.science = 1; break;
        case TerrainType::Hills:     y.production = 2; break;
        case TerrainType::Tundra:    y.food = 1; break;
        case TerrainType::Desert:    y.production = 0; break;  // bare desert: dead weight
        case TerrainType::Coast:     y.food = 1; y.gold = 2; break;
        case TerrainType::Ocean:     y.food = 1; y.gold = 1; break;
        case TerrainType::Mountains:                          // unworkable; here for completeness
        case TerrainType::Snow:
        case TerrainType::Count:     break;
    }
    // Rivers bring trade and fresh water; roads carry commerce.
    if ((tileFlags & TileFlag_River) != 0u) y.gold += 1;
    if ((tileFlags & TileFlag_Road) != 0u)  y.gold += 1;
    return y;
}

// ---------------------------------------------------------------------------
// Research tree -- a compact ancient-era beeline with a couple of branches.
// Costs climb so research keeps pace with a growing empire's science output.
// ---------------------------------------------------------------------------
const std::vector<TechDef>& techTree() {
    static const std::vector<TechDef> kTechs = {
        {"pottery",        "Pottery",         22,  {},                          {"granary", "hanging_gardens"}},
        {"writing",        "Writing",         34,  {},                          {"library"}},
        {"bronze_working", "Bronze Working",  40,  {},                          {"smithy", "barracks"}},
        {"sailing",        "Sailing",         48,  {},                          {"harbor", "colossus"}},
        {"masonry",        "Masonry",         54,  {"pottery"},                 {"walls", "pyramids"}},
        {"currency",       "Currency",        72,  {"bronze_working"},          {"market"}},
        {"philosophy",     "Philosophy",      90,  {"writing"},                 {"temple", "oracle"}},
        {"mathematics",    "Mathematics",     108, {"masonry", "currency"},     {"aqueduct"}},
        {"literature",     "Literature",      128, {"philosophy"},              {"great_library", "parthenon"}},
        {"construction",   "Construction",    160, {"mathematics"},             {"colosseum", "great_wall"}},
        {"theology",       "Theology",        165, {"philosophy"},              {"cathedral", "grand_temple"}},
        {"engineering",    "Engineering",     190, {"mathematics"},             {"university"}},
        {"banking",        "Banking",         215, {"currency", "mathematics"}, {"bank"}},
        {"astronomy",      "Astronomy",       250, {"literature", "sailing"},   {"observatory", "copernicus"}},
        {"guilds",         "Guilds",          290, {"banking"},                 {"guildhall", "grand_bazaar"}},
    };
    return kTechs;
}

const TechDef* findTech(const std::string& id) {
    for (const TechDef& t : techTree()) {
        if (t.id == id) return &t;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Buildings & wonders.
//   id            name          cost maint  flat{f,p,g,s,c}   pPct gPct sPct happy grow  tech            wonder score
// Wonder global effects (empire-wide bonuses) are applied by id in game_sim.cc.
// ---------------------------------------------------------------------------
const std::vector<BuildingDef>& buildingDefs() {
    static const std::vector<BuildingDef> kDefs = {
        // --- ordinary buildings ---
        {"granary",   "Granary",    58, 1, {2,0,0,0,0},  0,  0,  0,  1, 50, "pottery",        false, 0},
        {"library",   "Library",    76, 1, {0,0,0,1,1},  0,  0, 50,  0,  0, "writing",        false, 0},
        {"smithy",    "Smithy",     72, 1, {0,2,0,0,0}, 25,  0,  0,  0,  0, "bronze_working", false, 0},
        {"barracks",  "Barracks",   64, 1, {0,0,0,0,0},  0,  0,  0,  0,  0, "bronze_working", false, 0},
        {"harbor",    "Harbor",     66, 1, {1,0,1,0,0},  0,  0,  0,  0,  0, "sailing",        false, 0},
        {"walls",     "Walls",      60, 1, {0,0,0,0,0},  0,  0,  0,  2,  0, "masonry",        false, 0},
        {"market",    "Market",     96, 2, {0,0,2,0,0},  0, 40,  0,  0,  0, "currency",       false, 0},
        {"temple",    "Temple",     82, 1, {0,0,0,0,2},  0,  0,  0,  3,  0, "philosophy",     false, 0},
        {"aqueduct",  "Aqueduct",  108, 2, {1,0,0,0,0},  0,  0,  0,  2, 25, "mathematics",    false, 0},
        {"cathedral", "Cathedral", 120, 2, {0,0,0,0,2},  0,  0,  0,  3,  0, "theology",       false, 0},
        {"university","University",150, 2, {0,0,0,2,0},  0,  0, 50,  0,  0, "engineering",    false, 0},
        {"bank",      "Bank",      170, 2, {0,0,3,0,0},  0, 40,  0,  0,  0, "banking",        false, 0},
        {"observatory","Observatory",180,2,{0,0,0,3,0},  0,  0, 25,  0,  0, "astronomy",      false, 0},
        {"guildhall", "Guild Hall",150, 2, {0,1,2,0,0},  0, 20,  0,  0,  0, "guilds",         false, 0},

        // --- world wonders (one owner each; strong empire-wide pull). The tech
        //     column spreads them from the first era to the last so the wonder
        //     race never stops. Empire-wide effects are applied by id in game_sim.
        {"hanging_gardens","Hanging Gardens",230, 0, {0,0,0,0,0}, 0, 0, 0, 0, 0, "pottery",      true, 22},
        {"pyramids",      "Pyramids",        260, 0, {0,0,0,0,0}, 0, 0, 0, 0, 0, "masonry",      true, 25},
        {"colossus",      "Colossus",        200, 0, {0,0,0,0,0}, 0, 0, 0, 0, 0, "sailing",      true, 20},
        {"oracle",        "Oracle",          240, 0, {0,0,0,0,0}, 0, 0, 0, 0, 0, "philosophy",   true, 24},
        {"great_library", "Great Library",   300, 0, {0,0,0,0,0}, 0, 0, 0, 0, 0, "literature",   true, 30},
        {"parthenon",     "Parthenon",       240, 0, {0,0,0,0,0}, 0, 0, 0, 0, 0, "literature",   true, 24},
        {"colosseum",     "Colosseum",       220, 0, {0,0,0,0,0}, 0, 0, 0, 0, 0, "construction", true, 22},
        {"great_wall",    "Great Wall",      210, 0, {0,0,0,0,0}, 0, 0, 0, 0, 0, "construction", true, 20},
        {"grand_temple",  "Grand Temple",    250, 0, {0,0,0,0,0}, 0, 0, 0, 0, 0, "theology",     true, 24},
        {"copernicus",    "Copernicus' Observatory", 300, 0, {0,0,0,0,0}, 0, 0, 0, 0, 0, "astronomy", true, 30},
        {"grand_bazaar",  "Grand Bazaar",    280, 0, {0,0,0,0,0}, 0, 0, 0, 0, 0, "guilds",       true, 28},
    };
    return kDefs;
}

const BuildingDef* findBuildingDef(const std::string& id) {
    for (const BuildingDef& d : buildingDefs()) {
        if (d.id == id) return &d;
    }
    return nullptr;
}

bool isWonder(const std::string& id) {
    const BuildingDef* d = findBuildingDef(id);
    return d != nullptr && d->isWonder;
}

}  // namespace odai::game
