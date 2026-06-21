#include "content/content_loader.h"

#include "content/content_database.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <system_error>

namespace odai::content {

namespace {

using json = nlohmann::json;
using namespace odai::game;

// Read one JSON file. A missing file is not an error (mods may omit files); a
// file that exists but fails to parse records an error and returns false.
bool tryReadJson(const std::filesystem::path& path, json& out, ContentDatabase& db) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return false;
    }
    std::ifstream file(path);
    if (!file) {
        const std::string msg = "content: cannot open " + path.string();
        db.addError(msg);
        std::cerr << msg << "\n";
        return false;
    }
    try {
        file >> out;
    } catch (const json::exception& e) {
        const std::string msg = "content: JSON parse error in " + path.string() + ": " + e.what();
        db.addError(msg);
        std::cerr << msg << "\n";
        return false;
    }
    return true;
}

Yields parseYields(const json& j) {
    Yields y{};
    if (!j.is_object()) return y;
    y.food = j.value("food", 0);
    y.production = j.value("production", 0);
    y.gold = j.value("gold", 0);
    y.science = j.value("science", 0);
    y.culture = j.value("culture", 0);
    return y;
}

std::vector<std::string> parseStringArray(const json& parent, const char* key) {
    std::vector<std::string> out;
    if (parent.contains(key) && parent[key].is_array()) {
        for (const json& s : parent[key]) {
            if (s.is_string()) out.push_back(s.get<std::string>());
        }
    }
    return out;
}

TerrainType terrainFromName(const std::string& name) {
    if (name == "Ocean") return TerrainType::Ocean;
    if (name == "Coast") return TerrainType::Coast;
    if (name == "Grassland") return TerrainType::Grassland;
    if (name == "Plains") return TerrainType::Plains;
    if (name == "Forest") return TerrainType::Forest;
    if (name == "Jungle") return TerrainType::Jungle;
    if (name == "Hills") return TerrainType::Hills;
    if (name == "Mountains") return TerrainType::Mountains;
    if (name == "Desert") return TerrainType::Desert;
    if (name == "Tundra") return TerrainType::Tundra;
    if (name == "Snow") return TerrainType::Snow;
    return TerrainType::Count;  // sentinel: unknown
}

void loadBalance(const std::filesystem::path& dir, ContentDatabase& db) {
    json j;
    if (!tryReadJson(dir / "balance.json", j, db)) return;
    Balance b{};  // defaults match the base game
    b.citizenFoodUpkeep = j.value("citizenFoodUpkeep", b.citizenFoodUpkeep);
    b.foodBoxBase = j.value("foodBoxBase", b.foodBoxBase);
    b.foodBoxPerPop = j.value("foodBoxPerPop", b.foodBoxPerPop);
    b.baseHappyCap = j.value("baseHappyCap", b.baseHappyCap);
    b.settlerPopCost = j.value("settlerPopCost", b.settlerPopCost);
    b.settlerFoundPop = j.value("settlerFoundPop", b.settlerFoundPop);
    b.settlerCost = j.value("settlerCost", b.settlerCost);
    b.capitalStartPop = j.value("capitalStartPop", b.capitalStartPop);
    b.rushGoldPerShield = j.value("rushGoldPerShield", b.rushGoldPerShield);
    b.cityWorkRadius = j.value("cityWorkRadius", b.cityWorkRadius);
    b.settleMinSpacing = j.value("settleMinSpacing", b.settleMinSpacing);
    b.sciencePerPopDiv = j.value("sciencePerPopDiv", b.sciencePerPopDiv);
    b.maxGrowthBonus = j.value("maxGrowthBonus", b.maxGrowthBonus);
    b.civicUpkeepPerCity = j.value("civicUpkeepPerCity", b.civicUpkeepPerCity);
    b.starveBuffer = j.value("starveBuffer", b.starveBuffer);
    b.greatPersonBaseCost = j.value("greatPersonBaseCost", b.greatPersonBaseCost);
    b.greatPersonCostGrowth = j.value("greatPersonCostGrowth", b.greatPersonCostGrowth);
    b.greatPersonSciencePointDiv = j.value("greatPersonSciencePointDiv", b.greatPersonSciencePointDiv);
    db.setBalance(b);
}

void loadTerrain(const std::filesystem::path& dir, ContentDatabase& db) {
    json j;
    if (!tryReadJson(dir / "terrain.json", j, db)) return;
    if (j.contains("cityCenter")) db.setCityCenterYields(parseYields(j["cityCenter"]));
    db.setRiverGold(j.value("riverGold", 1));
    db.setRoadGold(j.value("roadGold", 1));
    if (j.contains("terrain") && j["terrain"].is_object()) {
        for (const auto& [name, val] : j["terrain"].items()) {
            const TerrainType t = terrainFromName(name);
            if (t == TerrainType::Count) {
                db.addError("content: unknown terrain '" + name + "' in terrain.json");
                continue;
            }
            TerrainYieldDef def{};
            def.base = parseYields(val);
            def.workable = val.value("workable", true);
            db.setTerrain(t, def);
        }
    }
}

void loadTechs(const std::filesystem::path& dir, ContentDatabase& db) {
    json j;
    if (!tryReadJson(dir / "techs.json", j, db)) return;
    if (!j.contains("techs") || !j["techs"].is_array()) return;
    for (const json& tj : j["techs"]) {
        TechDef t{};
        t.id = tj.value("id", "");
        t.name = tj.value("name", "");
        t.cost = tj.value("cost", 0);
        t.era = tj.value("era", "");
        t.prereqs = parseStringArray(tj, "prereqs");
        t.unlocks = parseStringArray(tj, "unlocks");
        t.description = tj.value("description", "");
        if (tj.contains("gate") && tj["gate"].is_object()) {
            const json& g = tj["gate"];
            const std::string kind = g.value("kind", "open");
            if (kind == "boost") t.gate.kind = GateKind::Boost;
            else if (kind == "locked") t.gate.kind = GateKind::Locked;
            else t.gate.kind = GateKind::Open;
            t.gate.condition = g.value("condition", "");
            t.gate.amount = g.value("amount", 0);
            t.gate.boostPct = g.value("boostPct", 50);
        }
        db.addTech(std::move(t));
    }
}

BuildingEffects parseEffects(const json& e) {
    BuildingEffects fx{};
    fx.present = true;
    const std::string scope = e.value("scope", "city");
    fx.scope = (scope == "empire") ? BuildingEffects::Scope::Empire : BuildingEffects::Scope::City;
    if (e.contains("flat")) fx.flat = parseYields(e["flat"]);
    fx.prodPct = e.value("prodPct", 0);
    fx.goldPct = e.value("goldPct", 0);
    fx.sciencePct = e.value("sciencePct", 0);
    fx.happiness = e.value("happiness", 0);
    fx.growthBonus = e.value("growthBonus", 0);
    return fx;
}

void loadBuildings(const std::filesystem::path& dir, ContentDatabase& db) {
    json j;
    if (!tryReadJson(dir / "buildings.json", j, db)) return;
    if (!j.contains("buildings") || !j["buildings"].is_array()) return;
    for (const json& bj : j["buildings"]) {
        BuildingDef d{};
        d.id = bj.value("id", "");
        d.name = bj.value("name", "");
        d.productionCost = bj.value("productionCost", 0);
        d.maintenance = bj.value("maintenance", 0);
        if (bj.contains("flat")) d.flat = parseYields(bj["flat"]);
        d.prodPct = bj.value("prodPct", 0);
        d.goldPct = bj.value("goldPct", 0);
        d.sciencePct = bj.value("sciencePct", 0);
        d.happiness = bj.value("happiness", 0);
        d.growthBonus = bj.value("growthBonus", 0);
        d.requiredTech = bj.value("requiredTech", "");
        d.isWonder = bj.value("isWonder", false);
        d.score = bj.value("score", 0);
        if (bj.contains("effects") && bj["effects"].is_object()) {
            d.effects = parseEffects(bj["effects"]);
        }
        db.addBuilding(std::move(d));
    }
}

void loadUnits(const std::filesystem::path& dir, ContentDatabase& db) {
    json j;
    if (!tryReadJson(dir / "units.json", j, db)) return;
    if (!j.contains("units") || !j["units"].is_array()) return;
    for (const json& uj : j["units"]) {
        UnitStats u{};
        u.id = uj.value("id", "");
        u.maxHp = uj.value("maxHp", 30);
        u.movement = uj.value("movement", 2);
        u.maxSupply = uj.value("maxSupply", 5);
        u.attack = uj.value("attack", 0);
        u.rangedAttack = uj.value("rangedAttack", 0);
        u.range = uj.value("range", 0);
        u.melee = uj.value("melee", false);
        u.requiredBuilding = uj.value("requiredBuilding", "");
        db.addUnit(std::move(u));
    }
}

void loadBuildables(const std::filesystem::path& dir, ContentDatabase& db) {
    json j;
    if (!tryReadJson(dir / "buildables.json", j, db)) return;
    if (!j.contains("buildables") || !j["buildables"].is_array()) return;
    for (const json& bj : j["buildables"]) {
        BuildableItem b{};
        b.id = bj.value("id", "");
        b.name = bj.value("name", "");
        b.iconName = bj.value("iconName", "");
        const std::string kind = bj.value("kind", "Building");
        b.kind = (kind == "Unit") ? BuildableKind::Unit : BuildableKind::Building;
        b.productionCost = bj.value("productionCost", 0);
        b.civpediaId = bj.value("civpediaId", b.id);
        db.addBuildable(std::move(b));
    }
}

void loadCivpedia(const std::filesystem::path& dir, ContentDatabase& db) {
    json j;
    if (!tryReadJson(dir / "civpedia.json", j, db)) return;
    const json& articles = j.contains("articles") ? j["articles"] : j;
    if (!articles.is_object()) return;
    for (const auto& [id, text] : articles.items()) {
        if (text.is_string()) db.setPediaArticle(id, text.get<std::string>());
    }
}

void loadLeaders(const std::filesystem::path& dir, ContentDatabase& db) {
    json j;
    if (!tryReadJson(dir / "leaders.json", j, db)) return;
    if (!j.contains("leaders") || !j["leaders"].is_array()) return;
    for (const json& lj : j["leaders"]) {
        LeaderDef l{};
        l.civName = lj.value("civName", "");
        l.leaderName = lj.value("leaderName", "");
        if (lj.contains("personality") && lj["personality"].is_object()) {
            const json& p = lj["personality"];
            l.personality.name = p.value("name", "");
            l.personality.expansion = p.value("expansion", 1.0f);
            l.personality.wonderLove = p.value("wonderLove", 1.0f);
            l.personality.science = p.value("science", 1.0f);
            l.personality.gold = p.value("gold", 1.0f);
            l.personality.religion = p.value("religion", 1.0f);
            l.personality.culture = p.value("culture", 1.0f);
        }
        l.cityNames = parseStringArray(lj, "cityNames");
        db.addLeader(std::move(l));
    }
}

void loadGreatPeople(const std::filesystem::path& dir, ContentDatabase& db) {
    json j;
    if (!tryReadJson(dir / "great_people.json", j, db)) return;
    if (!j.contains("greatPeople") || !j["greatPeople"].is_array()) return;
    for (const json& gj : j["greatPeople"]) {
        GreatPersonDef g{};
        g.id = gj.value("id", "");
        g.name = gj.value("name", "");
        g.title = gj.value("title", "");
        const std::string cls = gj.value("class", "");
        g.cls = greatPersonClassFromName(cls);
        if (g.cls == GreatPersonClass::Count) {
            db.addError("content: great person '" + g.id + "' has unknown class '" + cls + "'");
            g.cls = GreatPersonClass::Scientist;  // keep it loadable
        }
        if (gj.contains("portraitCell") && gj["portraitCell"].is_array() &&
            gj["portraitCell"].size() == 2) {
            g.portraitCol = gj["portraitCell"][0].get<int>();
            g.portraitRow = gj["portraitCell"][1].get<int>();
        }
        // A great person's bonus is always city-scoped (it lands on the one city the
        // figure is settled into), so reuse parseEffects but force the scope.
        if (gj.contains("bonus") && gj["bonus"].is_object()) {
            g.bonus = parseEffects(gj["bonus"]);
            g.bonus.scope = BuildingEffects::Scope::City;
        }
        g.bonusSummary = gj.value("bonusSummary", "");
        g.description = gj.value("description", "");
        db.addGreatPerson(std::move(g));
    }
}

}  // namespace

void loadModData(ContentDatabase& db, const std::filesystem::path& modDir) {
    const std::filesystem::path dataDir = modDir / "data";
    loadBalance(dataDir, db);
    loadTerrain(dataDir, db);
    loadTechs(dataDir, db);
    loadBuildings(dataDir, db);
    loadUnits(dataDir, db);
    loadBuildables(dataDir, db);
    loadCivpedia(dataDir, db);
    loadLeaders(dataDir, db);
    loadGreatPeople(dataDir, db);
}

}  // namespace odai::content
