#pragma once

#include <sol/sol.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// Game-agnostic Lua plumbing shared by every script host in the repo (the 4X
// ScriptHost and the citybuilder CityScriptHost). This header DOES include
// sol2, so it is used only by script-host .cc files — public host headers stay
// Lua-free behind their pimpls.
namespace odai::script {

// Set up the sandbox: open only safe, deterministic libraries (base, string,
// math, table) and strip the escape hatches (dofile/loadfile/load/loadstring/
// require/collectgarbage/package, math.random/randomseed). Scripts cannot touch
// the filesystem, clock, or process, which keeps modded runs reproducible.
void sandboxLua(sol::state& lua);

// Every *.lua under modDir/scripts, sorted for deterministic load order.
std::vector<std::filesystem::path> collectModScripts(const std::filesystem::path& modDir);

// Global `Rng` table (Rng.int / Rng.number) backed by the caller-owned LCG
// state, advanced independently of any simulation stream. `state` must outlive
// the lua state's use of the table.
void registerRngTable(sol::state& lua, std::uint32_t& state);

// Global `Log` table (Log.info) — stderr only, never an in-game event log.
// `prefix` tags each line, e.g. "[mod] " or "[citymod] ".
void registerLogTable(sol::state& lua, std::string prefix);

}  // namespace odai::script
