#include "script/lua_internal.h"

#include <iostream>

// Engine-level utilities: error recording. The sandbox itself lives in the
// game-agnostic script_core.cc; the ScriptHost class lives in lua_mod_host.cc;
// the bindings and modder-facing API live in lua_bindings.cc and lua_api.cc.
namespace odai::script {

void recordError(EngineState& es, const std::string& where, const std::string& what) {
    const std::string msg = "[mod] " + where + ": " + what;
    es.errors.push_back(msg);
    std::cerr << msg << "\n";
}

}  // namespace odai::script
