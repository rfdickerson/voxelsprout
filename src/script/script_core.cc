#include "script/script_core.h"

#include <algorithm>
#include <iostream>
#include <system_error>
#include <utility>

namespace odai::script {

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
    // the seeded Rng table instead.
    sol::optional<sol::table> mathTable = lua["math"];
    if (mathTable) {
        (*mathTable)["random"] = sol::nil;
        (*mathTable)["randomseed"] = sol::nil;
    }
}

std::vector<std::filesystem::path> collectModScripts(const std::filesystem::path& modDir) {
    std::vector<std::filesystem::path> files;
    const std::filesystem::path dir = modDir / "scripts";
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) return files;

    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".lua") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());  // deterministic load order
    return files;
}

void registerRngTable(sol::state& lua, std::uint32_t& state) {
    sol::table rng = lua.create_named_table("Rng");
    rng.set_function("int", [&state](int lo, int hi) -> int {
        if (hi < lo) std::swap(lo, hi);
        state = state * 1664525u + 1013904223u;
        const std::uint32_t span = static_cast<std::uint32_t>(hi - lo) + 1u;
        return lo + static_cast<int>((state >> 16) % span);
    });
    rng.set_function("number", [&state]() -> double {
        state = state * 1664525u + 1013904223u;
        return static_cast<double>(state >> 8) / static_cast<double>(1u << 24);
    });
}

void registerLogTable(sol::state& lua, std::string prefix) {
    sol::table log = lua.create_named_table("Log");
    log.set_function("info", [prefix = std::move(prefix)](const std::string& msg) {
        std::cerr << prefix << msg << "\n";
    });
}

}  // namespace odai::script
