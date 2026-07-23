#include "script/script_engine.h"

#include "content/mod_loader.h"
#include "game/economy.h"
#include "game/game_sim.h"
#include "script/lua_internal.h"

#include <algorithm>
#include <filesystem>
#include <system_error>
#include <utility>
#include <vector>

namespace odai::script {

using odai::game::City;
using odai::game::Empire;
using odai::game::World;
using odai::game::YieldContext;

namespace {

// Invoke one Lua callback, recording (never throwing) any error so a buggy script
// can't crash or halt the simulation.
template <class... Args>
void callHook(EngineState& es, sol::protected_function& fn, const std::string& where, Args&&... args) {
    sol::protected_function_result result = fn(std::forward<Args>(args)...);
    if (!result.valid()) {
        sol::error err = result;
        recordError(es, where, err.what());
    }
}

}  // namespace

struct ScriptHost::Impl {
    EngineState es;
};

ScriptHost::ScriptHost() : m_impl(std::make_unique<Impl>()) {
    sandboxLua(m_impl->es.lua);
    registerBindings(m_impl->es);
    registerApi(m_impl->es);
}

ScriptHost::~ScriptHost() = default;

bool ScriptHost::ok() const { return m_impl->es.errors.empty(); }
const std::vector<std::string>& ScriptHost::errors() const { return m_impl->es.errors; }

bool ScriptHost::runScriptFile(const std::filesystem::path& path) {
    EngineState& es = m_impl->es;
    sol::load_result chunk = es.lua.load_file(path.string());
    if (!chunk.valid()) {
        sol::error err = chunk;
        recordError(es, "load " + path.filename().string(), err.what());
        return false;
    }
    sol::protected_function fn = chunk;
    sol::protected_function_result result = fn();
    if (!result.valid()) {
        sol::error err = result;
        recordError(es, "run " + path.filename().string(), err.what());
        return false;
    }
    return true;
}

void ScriptHost::loadModScripts(const std::filesystem::path& modDir) {
    for (const std::filesystem::path& f : collectModScripts(modDir)) runScriptFile(f);
}

// --- IModHost dispatch ------------------------------------------------------

void ScriptHost::onTurnStart(World& world) {
    EngineState& es = m_impl->es;
    es.currentWorld = &world;
    if (!es.rngSeeded) {  // seed the script RNG from the world's seed-derived state
        es.rngState = world.rng;
        es.rngSeeded = true;
    }
    for (sol::protected_function& fn : es.onTurnStart) callHook(es, fn, "turn_start", &world);
}

void ScriptHost::onTurnEnd(World& world) {
    EngineState& es = m_impl->es;
    es.currentWorld = &world;
    for (sol::protected_function& fn : es.onTurnEnd) callHook(es, fn, "turn_end", &world);
}

void ScriptHost::onCityYields(YieldContext& ctx) {
    EngineState& es = m_impl->es;
    es.currentWorld = ctx.world;

    for (sol::protected_function& fn : es.onCityYields) callHook(es, fn, "city_yields", &ctx);

    // Per-building effect callbacks fire for the city's own (non-wonder) buildings.
    if (ctx.city != nullptr) {
        for (const std::string& b : ctx.city->buildings) {
            const odai::game::BuildingDef* d = odai::game::findBuildingDef(b);
            if (d != nullptr && d->isWonder) continue;  // wonders handled empire-wide below
            auto it = es.effects.find(b);
            if (it != es.effects.end()) callHook(es, it->second, "effect:" + b, &ctx);
        }
    }
    // Wonder effect callbacks fire empire-wide (once per city of the owner).
    if (ctx.empire != nullptr) {
        for (const std::string& w : ctx.empire->wonders) {
            auto it = es.effects.find(w);
            if (it != es.effects.end()) callHook(es, it->second, "effect:" + w, &ctx);
        }
    }
}

void ScriptHost::onBuildingBuilt(World& world, City& city, const std::string& id) {
    EngineState& es = m_impl->es;
    es.currentWorld = &world;
    for (sol::protected_function& fn : es.onBuildingBuilt) callHook(es, fn, "building_built", &city, id);
}

void ScriptHost::onWonderBuilt(World& world, Empire& empire, const std::string& id) {
    EngineState& es = m_impl->es;
    es.currentWorld = &world;
    for (sol::protected_function& fn : es.onWonderBuilt) callHook(es, fn, "wonder_built", &empire, id);
}

void ScriptHost::onTechResearched(World& world, Empire& empire, const std::string& id) {
    EngineState& es = m_impl->es;
    es.currentWorld = &world;
    for (sol::protected_function& fn : es.onTechResearched) callHook(es, fn, "tech_researched", &empire, id);
}

void ScriptHost::onCityFounded(World& world, City& city) {
    EngineState& es = m_impl->es;
    es.currentWorld = &world;
    for (sol::protected_function& fn : es.onCityFounded) callHook(es, fn, "city_founded", &city);
}

std::unique_ptr<ScriptHost> createBaseScriptHost() {
    auto host = std::make_unique<ScriptHost>();
    host->loadModScripts(odai::content::baseModDir());
    return host;
}

}  // namespace odai::script
