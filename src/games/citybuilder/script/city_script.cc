#include "games/citybuilder/script/city_script.h"

#include "content/content_paths.h"
#include "procgen/rng.h"
#include "script/script_core.h"

#include <sol/sol.hpp>

#include <array>
#include <iostream>
#include <unordered_map>
#include <utility>

namespace odai::citybuilder {

namespace {

// The Rng handle passed to Lua name generators. Wraps the shared procgen LCG
// so the same seed produces the same name on every run. Scripts receive it as
// an argument and must not retain it past the call.
struct ScriptRng {
    odai::procgen::Rng rng;

    explicit ScriptRng(std::uint32_t seed) : rng(seed) {}

    int rangeInt(int lo, int hi) {
        if (hi < lo) std::swap(lo, hi);
        return rng.range(lo, hi);
    }
    double number() { return static_cast<double>(rng.uniform(0.0f, 1.0f)); }
    bool chance(double p) { return rng.chance(static_cast<float>(p)); }
    sol::object pick(sol::table pool) {
        const int n = static_cast<int>(pool.size());
        if (n <= 0) return sol::nil;
        return pool[1 + static_cast<int>(rng.next() % static_cast<std::uint32_t>(n))];
    }
};

// ── Compiled-in fallbacks ────────────────────────────────────────────────────
// Deliberately small: enough for the game to read well scriptless or past a
// broken script. The real content lives in mods/citybuilder/scripts/.

const std::array<const char*, 12> kCityPre = {"Cedar",  "Marrow", "Ash",   "Harlow",
                                              "Alder",  "Stone",  "Maple", "Norwood",
                                              "Copper", "Wren",   "Fall",  "Bright"};
const std::array<const char*, 8> kCitySuf = {"field", "brook", " Falls",  " Springs",
                                             "ford",  "dale",  " Grove", "port"};
const std::array<const char*, 10> kStreetName = {"Maple",  "Oak",    "Elm",    "Birch", "Main",
                                                 "Harbor", "Willow", "Juniper", "Hill",  "Lake"};
const std::array<const char*, 4> kStreetType = {" St", " Ave", " Rd", " Lane"};
const std::array<const char*, 10> kFirstF = {"Marla", "June",  "Opal",  "Cora", "Hazel",
                                             "Ruth",  "Vera",  "Ida",   "Nell", "Pearl"};
const std::array<const char*, 10> kFirstM = {"Ed",   "Gus",   "Ray",  "Cal",  "Ned",
                                             "Amos", "Frank", "Walt", "Hank", "Roy"};
const std::array<const char*, 12> kLast = {"Voss",   "Krane",  "Mercer", "Holt", "Bram",
                                           "Tully",  "Ashford", "Pike",  "Snell", "Dunmore",
                                           "Fairley", "Ostrander"};
const std::array<const char*, 3> kBlockByTier = {"Trailer Court", "Rowhouses", "Estates"};
const std::array<const char*, 8> kComCats = {"cafe",    "grocery", "diner", "salon",
                                             "bookshop", "gym",     "yoga",  "arcade"};
const std::array<const char*, 5> kIndCats = {"mill", "foundry", "depot", "plant", "yard"};

const std::vector<StoryTemplate>& fallbackStories() {
    static const std::vector<StoryTemplate> kStories = {
        {"fallback_opening", "opening", 1.0f, {}, "{place} opens its doors"},
        {"fallback_arrival", "arrival", 1.0f, {}, "The {family} family moved to town"},
        {"fallback_departure", "departure", 1.0f, {}, "{a} packed up and left town"},
        {"fallback_life", "life", 1.0f, {}, "{a} was spotted at {place}"},
        {"fallback_weekend", "weekend", 1.0f, {}, "Saturday practice at {place}"},
    };
    return kStories;
}

const std::vector<NeedRule>& fallbackNeeds() {
    static const std::vector<NeedRule> kNeeds = {
        {"any", "cafe", 1.5f},
        {"any", "grocery", 1.5f},
        {"fit", "gym", 2.0f},
        {"parent", "daycare", 3.0f},
    };
    return kNeeds;
}

}  // namespace

struct CityScriptHost::Impl {
    sol::state lua;  // declared first so it is destroyed LAST, after the
                     // protected_functions that reference it

    sol::protected_function nameCity, nameStreet, nameFirst, nameLast, nameBlock, nameBusiness;
    std::vector<sol::protected_function> onMonthStep;
    std::vector<sol::protected_function> onBuildingPlaced;

    std::vector<StoryTemplate> stories;
    std::vector<NeedRule> needs;
    std::unordered_map<std::string, double> config;

    std::uint32_t rngState = 0x51C17Bu;  // global Rng table; reseeded via seedRng
    std::vector<std::string> errors;

    void recordError(const std::string& where, const std::string& what) {
        const std::string msg = "[citymod] " + where + ": " + what;
        errors.push_back(msg);
        std::cerr << msg << "\n";
    }

    // Invoke a registered name generator; returns an engaged optional only when
    // the script call succeeded and returned the expected type.
    sol::optional<sol::object> callGenerator(sol::protected_function& fn, const char* where,
                                             ScriptRng& rng, sol::object extra1 = sol::nil,
                                             sol::object extra2 = sol::nil,
                                             sol::object extra3 = sol::nil) {
        if (!fn.valid()) return sol::nullopt;
        sol::protected_function_result result = fn(&rng, extra1, extra2, extra3);
        if (!result.valid()) {
            sol::error err = result;
            recordError(where, err.what());
            return sol::nullopt;
        }
        return result.get<sol::object>();
    }

    void registerApi();
};

void CityScriptHost::Impl::registerApi() {
    // Rng handle type passed to name generators (no constructor from Lua).
    lua.new_usertype<ScriptRng>("NameRng", sol::no_constructor,
                                "int", &ScriptRng::rangeInt,
                                "number", &ScriptRng::number,
                                "chance", &ScriptRng::chance,
                                "pick", &ScriptRng::pick);

    // --- Names.register{ city=fn, street=fn, first=fn, last=fn, block=fn,
    //     business=fn } — partial tables merge over earlier registrations. ---
    sol::table names = lua.create_named_table("Names");
    names.set_function("register", [this](sol::table t) {
        const auto take = [&](const char* key, sol::protected_function& slot) {
            sol::object v = t[key];
            if (v.is<sol::protected_function>()) slot = v.as<sol::protected_function>();
            else if (v != sol::nil) recordError("Names.register", std::string(key) + " is not a function");
        };
        take("city", nameCity);
        take("street", nameStreet);
        take("first", nameFirst);
        take("last", nameLast);
        take("block", nameBlock);
        take("business", nameBusiness);
    });

    // --- Stories.register{ id=, kind=, weight=, requires={...}, text= } ---
    sol::table storiesTable = lua.create_named_table("Stories");
    storiesTable.set_function("register", [this](sol::table t) {
        StoryTemplate tpl;
        sol::object id = t["id"], kind = t["kind"], text = t["text"];
        if (!id.is<std::string>() || !kind.is<std::string>() || !text.is<std::string>()) {
            recordError("Stories.register", "id, kind, and text must be strings");
            return;
        }
        tpl.id = id.as<std::string>();
        tpl.kind = kind.as<std::string>();
        tpl.text = text.as<std::string>();
        if (sol::object w = t["weight"]; w.is<double>()) tpl.weight = static_cast<float>(w.as<double>());
        if (sol::object req = t["requires"]; req.is<sol::table>()) {
            sol::table reqTable = req.as<sol::table>();
            for (std::size_t i = 1; i <= reqTable.size(); ++i) {
                sol::object tag = reqTable[i];
                if (tag.is<std::string>()) tpl.conditions.push_back(tag.as<std::string>());
            }
        }
        stories.push_back(std::move(tpl));
    });

    // --- Needs.register{ trait=, category=, weight= } ---
    sol::table needsTable = lua.create_named_table("Needs");
    needsTable.set_function("register", [this](sol::table t) {
        sol::object trait = t["trait"], category = t["category"];
        if (!trait.is<std::string>() || !category.is<std::string>()) {
            recordError("Needs.register", "trait and category must be strings");
            return;
        }
        NeedRule rule;
        rule.trait = trait.as<std::string>();
        rule.category = category.as<std::string>();
        if (sol::object w = t["weight"]; w.is<double>()) rule.weight = static_cast<float>(w.as<double>());
        needs.push_back(std::move(rule));
    });

    // --- Config.terrain{...} / Config.scatter{...} — numeric tuning tables. ---
    sol::table configTable = lua.create_named_table("Config");
    const auto configGroup = [this](const char* prefix) {
        return [this, prefix](sol::table t) {
            for (const auto& [key, value] : t) {
                if (!key.is<std::string>()) continue;
                if (!value.is<double>()) {
                    recordError(std::string("Config.") + prefix, key.as<std::string>() + " is not a number");
                    continue;
                }
                config[std::string(prefix) + "." + key.as<std::string>()] = value.as<double>();
            }
        };
    };
    configTable.set_function("terrain", configGroup("terrain"));
    configTable.set_function("scatter", configGroup("scatter"));

    // --- Events.on(name, fn) — the behavioural escape hatch (two events). ---
    sol::table events = lua.create_named_table("Events");
    events.set_function("on", [this](const std::string& name, sol::protected_function fn) {
        if (name == "month_step") onMonthStep.push_back(std::move(fn));
        else if (name == "building_placed") onBuildingPlaced.push_back(std::move(fn));
        else recordError("Events.on", "unknown event '" + name + "'");
    });

    odai::script::registerRngTable(lua, rngState);
    odai::script::registerLogTable(lua, "[citymod] ");
}

CityScriptHost::CityScriptHost() : m_impl(std::make_unique<Impl>()) {
    odai::script::sandboxLua(m_impl->lua);
    m_impl->registerApi();
}

CityScriptHost::~CityScriptHost() = default;

bool CityScriptHost::ok() const { return m_impl->errors.empty(); }
const std::vector<std::string>& CityScriptHost::errors() const { return m_impl->errors; }

void CityScriptHost::seedRng(std::uint32_t seed) { m_impl->rngState = seed ? seed : 1u; }

bool CityScriptHost::runScriptFile(const std::filesystem::path& path) {
    sol::load_result chunk = m_impl->lua.load_file(path.string());
    if (!chunk.valid()) {
        sol::error err = chunk;
        m_impl->recordError("load " + path.filename().string(), err.what());
        return false;
    }
    sol::protected_function fn = chunk;
    sol::protected_function_result result = fn();
    if (!result.valid()) {
        sol::error err = result;
        m_impl->recordError("run " + path.filename().string(), err.what());
        return false;
    }
    return true;
}

bool CityScriptHost::runScriptString(const std::string& source, const std::string& chunkName) {
    sol::load_result chunk = m_impl->lua.load(source, "@" + chunkName);
    if (!chunk.valid()) {
        sol::error err = chunk;
        m_impl->recordError("load " + chunkName, err.what());
        return false;
    }
    sol::protected_function fn = chunk;
    sol::protected_function_result result = fn();
    if (!result.valid()) {
        sol::error err = result;
        m_impl->recordError("run " + chunkName, err.what());
        return false;
    }
    return true;
}

void CityScriptHost::loadModScripts(const std::filesystem::path& modDir) {
    for (const std::filesystem::path& f : odai::script::collectModScripts(modDir)) runScriptFile(f);
}

// ── Name generation ──────────────────────────────────────────────────────────

std::string CityScriptHost::cityName(std::uint32_t seed) {
    ScriptRng rng(seed);
    if (auto r = m_impl->callGenerator(m_impl->nameCity, "Names.city", rng); r && r->is<std::string>()) {
        return r->as<std::string>();
    }
    odai::procgen::Rng fb(seed);
    return std::string(fb.pick(kCityPre)) + fb.pick(kCitySuf);
}

std::string CityScriptHost::streetName(std::uint32_t seed) {
    ScriptRng rng(seed);
    if (auto r = m_impl->callGenerator(m_impl->nameStreet, "Names.street", rng); r && r->is<std::string>()) {
        return r->as<std::string>();
    }
    odai::procgen::Rng fb(seed);
    return std::string(fb.pick(kStreetName)) + fb.pick(kStreetType);
}

std::string CityScriptHost::firstName(std::uint32_t seed, bool feminine) {
    ScriptRng rng(seed);
    if (auto r = m_impl->callGenerator(m_impl->nameFirst, "Names.first", rng,
                                       sol::make_object(m_impl->lua, feminine));
        r && r->is<std::string>()) {
        return r->as<std::string>();
    }
    odai::procgen::Rng fb(seed);
    return feminine ? kFirstF[fb.next() % kFirstF.size()] : kFirstM[fb.next() % kFirstM.size()];
}

std::string CityScriptHost::lastName(std::uint32_t seed) {
    ScriptRng rng(seed);
    if (auto r = m_impl->callGenerator(m_impl->nameLast, "Names.last", rng); r && r->is<std::string>()) {
        return r->as<std::string>();
    }
    odai::procgen::Rng fb(seed);
    return fb.pick(kLast);
}

std::string CityScriptHost::blockName(int wealthTier, std::uint32_t seed) {
    ScriptRng rng(seed);
    if (auto r = m_impl->callGenerator(m_impl->nameBlock, "Names.block", rng,
                                       sol::make_object(m_impl->lua, wealthTier));
        r && r->is<std::string>()) {
        return r->as<std::string>();
    }
    odai::procgen::Rng fb(seed);
    const int tier = wealthTier < 0 ? 0 : (wealthTier > 2 ? 2 : wealthTier);
    return std::string(fb.pick(kLast)) + " " + kBlockByTier[static_cast<std::size_t>(tier)];
}

BusinessName CityScriptHost::businessName(bool industrial, int wealthTier, int era,
                                          std::uint32_t seed) {
    ScriptRng rng(seed);
    if (auto r = m_impl->callGenerator(m_impl->nameBusiness, "Names.business", rng,
                                       sol::make_object(m_impl->lua, industrial ? "industrial" : "commercial"),
                                       sol::make_object(m_impl->lua, wealthTier),
                                       sol::make_object(m_impl->lua, era));
        r && r->is<sol::table>()) {
        sol::table t = r->as<sol::table>();
        sol::object name = t["name"], category = t["category"];
        if (name.is<std::string>() && category.is<std::string>()) {
            return {name.as<std::string>(), category.as<std::string>()};
        }
        m_impl->recordError("Names.business", "must return {name=string, category=string}");
    }
    odai::procgen::Rng fb(seed);
    if (industrial) {
        const std::string cat = fb.pick(kIndCats);
        std::string label = cat;
        label[0] = static_cast<char>(label[0] - 'a' + 'A');
        return {std::string(fb.pick(kLast)) + " " + label, cat};
    }
    const std::string cat = fb.pick(kComCats);
    std::string label = cat;
    label[0] = static_cast<char>(label[0] - 'a' + 'A');
    return {std::string(fb.pick(kLast)) + "'s " + label, cat};
}

// ── Registered content ───────────────────────────────────────────────────────

const std::vector<StoryTemplate>& CityScriptHost::stories() const {
    return m_impl->stories.empty() ? fallbackStories() : m_impl->stories;
}

const std::vector<NeedRule>& CityScriptHost::needs() const {
    return m_impl->needs.empty() ? fallbackNeeds() : m_impl->needs;
}

double CityScriptHost::configNumber(const std::string& key, double fallback) const {
    const auto it = m_impl->config.find(key);
    return it != m_impl->config.end() ? it->second : fallback;
}

// ── Event hooks ──────────────────────────────────────────────────────────────

namespace {

sol::table makeStatsTable(sol::state& lua, const CityScriptStats& stats) {
    sol::table t = lua.create_table();
    t["population"] = stats.population;
    t["money"] = stats.money;
    t["month"] = stats.month;
    t["year"] = stats.year;
    return t;
}

}  // namespace

void CityScriptHost::fireMonthStep(const CityScriptStats& stats) {
    sol::table t = makeStatsTable(m_impl->lua, stats);
    for (sol::protected_function& fn : m_impl->onMonthStep) {
        sol::protected_function_result result = fn(t);
        if (!result.valid()) {
            sol::error err = result;
            m_impl->recordError("month_step", err.what());
        }
    }
}

void CityScriptHost::fireBuildingPlaced(int c, int r, const std::string& building,
                                        const CityScriptStats& stats) {
    sol::table t = makeStatsTable(m_impl->lua, stats);
    for (sol::protected_function& fn : m_impl->onBuildingPlaced) {
        sol::protected_function_result result = fn(c, r, building, t);
        if (!result.valid()) {
            sol::error err = result;
            m_impl->recordError("building_placed", err.what());
        }
    }
}

std::unique_ptr<CityScriptHost> createCityScriptHost() {
    auto host = std::make_unique<CityScriptHost>();
    host->loadModScripts(odai::content::resolveContentPath(
        std::filesystem::path("mods") / "citybuilder"));
    return host;
}

}  // namespace odai::citybuilder
