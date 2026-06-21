#pragma once

#include "game/mod_host.h"

#include <sol/sol.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Internal shared state for the Lua engine. This header DOES include sol2, so it
// is used only by the src/script/*.cc files -- never by the public script_engine.h
// (which the app includes and must stay Lua-free).
namespace odai::script {

// All mutable engine state for one ScriptHost: the sol2 VM plus the callbacks
// registered by mod scripts. `lua` is declared first so it is destroyed LAST,
// after the protected_functions that reference it.
struct EngineState {
    sol::state lua;

    std::vector<sol::protected_function> onTurnStart;
    std::vector<sol::protected_function> onTurnEnd;
    std::vector<sol::protected_function> onCityYields;
    std::vector<sol::protected_function> onBuildingBuilt;
    std::vector<sol::protected_function> onWonderBuilt;
    std::vector<sol::protected_function> onTechResearched;
    std::vector<sol::protected_function> onCityFounded;

    // Per-building/wonder yield callbacks registered via Effects.register(id, fn).
    std::unordered_map<std::string, sol::protected_function> effects;

    // Deterministic, seeded RNG for scripts (seeded from world.rng on first turn;
    // advanced independently so it never perturbs the simulation's own stream).
    std::uint32_t rngState = 0;
    bool rngSeeded = false;

    // The world currently being simulated (set at each hook dispatch) so the Game
    // table can answer queries that aren't tied to a specific callback argument.
    odai::game::World* currentWorld = nullptr;

    std::vector<std::string> errors;
};

// Set up the sandbox: open safe libraries, strip os/io/package/require/load/etc.
void sandboxLua(sol::state& lua);

// Register the C++ game types (World/Empire/City/Yields/YieldContext) as usertypes.
void registerBindings(EngineState& es);

// Register the modder-facing API tables (Events/Effects/Game/Rng/Log).
void registerApi(EngineState& es);

// Record a non-fatal script error (also echoed to stderr).
void recordError(EngineState& es, const std::string& where, const std::string& what);

}  // namespace odai::script
