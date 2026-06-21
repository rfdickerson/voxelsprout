#include "script/lua_internal.h"

#include "game/game_sim.h"

#include <iostream>
#include <utility>

// Registers the modder-facing global tables:
//   Events.on(name, fn)        -- subscribe to a gameplay event
//   Effects.register(id, fn)   -- per-building/wonder yield callback (Tier-B)
//   Game.turn() / empire(i) /  -- read the world being simulated
//        city(i) / num_*()
//   Rng.int(lo,hi) / number()  -- deterministic, seeded RNG (never wall-clock)
//   Log.info(msg)              -- diagnostic logging (stderr, never the event log)
namespace odai::script {

using odai::game::City;
using odai::game::Empire;
using odai::game::World;

void registerApi(EngineState& es) {
    sol::state& lua = es.lua;

    // --- Events.on(name, fn) ---
    sol::table events = lua.create_named_table("Events");
    events.set_function("on", [&es](const std::string& name, sol::protected_function fn) {
        if (name == "turn_start") es.onTurnStart.push_back(std::move(fn));
        else if (name == "turn_end") es.onTurnEnd.push_back(std::move(fn));
        else if (name == "city_yields") es.onCityYields.push_back(std::move(fn));
        else if (name == "building_built") es.onBuildingBuilt.push_back(std::move(fn));
        else if (name == "wonder_built") es.onWonderBuilt.push_back(std::move(fn));
        else if (name == "tech_researched") es.onTechResearched.push_back(std::move(fn));
        else if (name == "city_founded") es.onCityFounded.push_back(std::move(fn));
        else recordError(es, "Events.on", "unknown event '" + name + "'");
    });

    // --- Effects.register(buildingId, fn) ---
    sol::table effects = lua.create_named_table("Effects");
    effects.set_function("register", [&es](const std::string& id, sol::protected_function fn) {
        es.effects[id] = std::move(fn);
    });

    // --- Game read API (queries the world currently being simulated) ---
    sol::table game = lua.create_named_table("Game");
    game.set_function("turn", [&es]() { return es.currentWorld != nullptr ? es.currentWorld->turn : 0; });
    game.set_function("num_empires", [&es]() {
        return es.currentWorld != nullptr ? static_cast<int>(es.currentWorld->empires.size()) : 0;
    });
    game.set_function("empire", [&es](int i) -> Empire* {
        World* w = es.currentWorld;
        if (w == nullptr || i < 1 || i > static_cast<int>(w->empires.size())) return nullptr;
        return &w->empires[static_cast<std::size_t>(i - 1)];
    });
    game.set_function("num_cities", [&es]() {
        return es.currentWorld != nullptr ? static_cast<int>(es.currentWorld->cities.size()) : 0;
    });
    game.set_function("city", [&es](int i) -> City* {
        World* w = es.currentWorld;
        if (w == nullptr || i < 1 || i > static_cast<int>(w->cities.size())) return nullptr;
        return &w->cities[static_cast<std::size_t>(i - 1)];
    });

    // --- Rng: deterministic LCG seeded from world.rng on the first turn, advanced
    //     independently so it never perturbs the simulation's own random stream. ---
    sol::table rng = lua.create_named_table("Rng");
    rng.set_function("int", [&es](int lo, int hi) -> int {
        if (hi < lo) std::swap(lo, hi);
        es.rngState = es.rngState * 1664525u + 1013904223u;
        const std::uint32_t span = static_cast<std::uint32_t>(hi - lo) + 1u;
        return lo + static_cast<int>((es.rngState >> 16) % span);
    });
    rng.set_function("number", [&es]() -> double {
        es.rngState = es.rngState * 1664525u + 1013904223u;
        return static_cast<double>(es.rngState >> 8) / static_cast<double>(1u << 24);
    });

    // --- Log: goes to stderr, never the in-game event log, so modded logging can't
    //     skew the sim's reward-cadence metrics. ---
    sol::table log = lua.create_named_table("Log");
    log.set_function("info", [](const std::string& msg) { std::cerr << "[mod] " << msg << "\n"; });
}

}  // namespace odai::script
