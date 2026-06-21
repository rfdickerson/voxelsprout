#include "script/lua_internal.h"

#include <iostream>

// Engine-level utilities: the sandbox and error recording. The ScriptHost class
// itself lives in lua_mod_host.cc; the bindings and modder-facing API live in
// lua_bindings.cc and lua_api.cc.
namespace odai::script {

void recordError(EngineState& es, const std::string& where, const std::string& what) {
    const std::string msg = "[mod] " + where + ": " + what;
    es.errors.push_back(msg);
    std::cerr << msg << "\n";
}

void sandboxLua(sol::state& lua) {
    // Only safe, deterministic libraries. No os/io/debug -- scripts cannot touch
    // the filesystem, clock, or process, which keeps modded games reproducible.
    lua.open_libraries(sol::lib::base, sol::lib::string, sol::lib::math, sol::lib::table);

    // Strip the escape hatches the base library still exposes.
    for (const char* name : {"dofile", "loadfile", "load", "loadstring", "require",
                             "collectgarbage", "package"}) {
        lua[name] = sol::nil;
    }

    // math.random / randomseed are non-deterministic across runs; scripts must use
    // the seeded Rng table instead (see registerApi).
    sol::optional<sol::table> mathTable = lua["math"];
    if (mathTable) {
        (*mathTable)["random"] = sol::nil;
        (*mathTable)["randomseed"] = sol::nil;
    }
}

}  // namespace odai::script
