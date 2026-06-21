#pragma once

#include "game/mod_host.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Public entry point for the Lua scripting engine. This header is Lua-free (sol2
// is hidden behind a pimpl) so the app can include it without pulling in Lua.
//
// Usage:
//   auto host = odai::script::createBaseScriptHost();
//   odai::game::setModHost(host.get());   // install; sim now fires Lua hooks
//   ...
//   odai::game::setModHost(nullptr);      // uninstall before host is destroyed
namespace odai::script {

// A Lua-backed implementation of the simulation's IModHost. Owns a sandboxed sol2
// state and dispatches gameplay events to the Lua callbacks registered by mod
// scripts (Events.on / Effects.register). Script errors are recorded, never thrown
// into the simulation.
class ScriptHost final : public odai::game::IModHost {
public:
    ScriptHost();
    ~ScriptHost() override;
    ScriptHost(const ScriptHost&) = delete;
    ScriptHost& operator=(const ScriptHost&) = delete;

    // Load and execute one Lua file; its registrations take effect immediately.
    // Returns false (and records an error) on a syntax or runtime error.
    bool runScriptFile(const std::filesystem::path& path);

    // Execute every *.lua under modDir/scripts in sorted (deterministic) order.
    void loadModScripts(const std::filesystem::path& modDir);

    [[nodiscard]] bool ok() const;
    [[nodiscard]] const std::vector<std::string>& errors() const;

    // --- IModHost ---
    void onTurnStart(odai::game::World& world) override;
    void onTurnEnd(odai::game::World& world) override;
    void onCityYields(odai::game::YieldContext& ctx) override;
    void onBuildingBuilt(odai::game::World& world, odai::game::City& city, const std::string& id) override;
    void onWonderBuilt(odai::game::World& world, odai::game::Empire& empire, const std::string& id) override;
    void onTechResearched(odai::game::World& world, odai::game::Empire& empire, const std::string& id) override;
    void onCityFounded(odai::game::World& world, odai::game::City& city) override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// Build a ScriptHost and load the base game's scripts (mods/base/scripts/*.lua).
std::unique_ptr<ScriptHost> createBaseScriptHost();

}  // namespace odai::script
