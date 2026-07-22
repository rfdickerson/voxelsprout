#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Public entry point for the citybuilder's Lua content layer. This header is
// Lua-free (sol2 hidden behind a pimpl) so the game includes it without pulling
// in Lua.
//
// The API is registration-based data + pure functions, not per-tick hooks: mods
// register name generators, story templates, need rules, and tuning config at
// load time; the C++ engine calls the name generators rarely (results cached by
// seed) and rolls story/need tables itself. Two coarse gameplay events
// (month_step, building_placed) are the escape hatch for behavioural mods.
// Every registration point has a compiled-in fallback, so the game runs
// scriptless and survives script errors.
namespace odai::citybuilder {

// A generated business name plus the category the generator drew ("yoga",
// "daycare", "cafe", ...). The category is load-bearing: the citizen sim sends
// sims to the same named storefront the player sees when hovering the tile.
struct BusinessName {
    std::string name;
    std::string category;
};

// A story/event template registered from Lua. The C++ event engine rolls these
// monthly per citizen and interpolates {a} {b} {family} {place} {street} from
// the roster, destination catalog, and street names.
struct StoryTemplate {
    std::string id;
    std::string kind;                    // "life" | "drama" | "opening" | "arrival" | "departure"
    float weight = 1.0f;                 // relative pick weight within its kind
    std::vector<std::string> conditions; // required tags: "fit","parent","gossip","married","affair"
    std::string text;
};

// Trait-to-destination-category weighting registered from Lua; drives which
// business a citizen visits when a need fires. trait "any" applies to everyone.
struct NeedRule {
    std::string trait;
    std::string category;
    float weight = 1.0f;
};

// Read-only snapshot passed to the month_step / building_placed event hooks.
struct CityScriptStats {
    int population = 0;
    double money = 0.0;
    int month = 1;
    int year = 1900;
};

class CityScriptHost {
public:
    CityScriptHost();
    ~CityScriptHost();
    CityScriptHost(const CityScriptHost&) = delete;
    CityScriptHost& operator=(const CityScriptHost&) = delete;

    // Load and execute one Lua file; its registrations take effect immediately.
    // Returns false (and records an error) on a syntax or runtime error.
    bool runScriptFile(const std::filesystem::path& path);

    // Execute a Lua chunk from a string (tests and tooling).
    bool runScriptString(const std::string& source, const std::string& chunkName);

    // Execute every *.lua under modDir/scripts in sorted (deterministic) order.
    void loadModScripts(const std::filesystem::path& modDir);

    [[nodiscard]] bool ok() const;
    [[nodiscard]] const std::vector<std::string>& errors() const;

    // Seed the global Rng table available to event hooks (derive from the world
    // seed so modded runs are reproducible per map).
    void seedRng(std::uint32_t seed);

    // --- Seeded name generation (same seed => same string) ------------------
    // Lua generators registered via Names.register are preferred; compiled-in
    // fallbacks keep the game running scriptless or past a broken script.
    std::string cityName(std::uint32_t seed);
    std::string streetName(std::uint32_t seed);
    std::string firstName(std::uint32_t seed, bool feminine);
    std::string lastName(std::uint32_t seed);
    std::string blockName(int wealthTier, std::uint32_t seed);
    BusinessName businessName(bool industrial, int wealthTier, int era, std::uint32_t seed);

    // --- Content registered by scripts (with fallbacks when empty) ----------
    [[nodiscard]] const std::vector<StoryTemplate>& stories() const;
    [[nodiscard]] const std::vector<NeedRule>& needs() const;

    // Numeric tuning value registered via Config.terrain{...} / Config.scatter
    // {...}, keyed "terrain.land_min", "scatter.hydrant_per_mille", ...
    [[nodiscard]] double configNumber(const std::string& key, double fallback) const;

    // --- Gameplay event hooks (Events.on) ------------------------------------
    void fireMonthStep(const CityScriptStats& stats);
    void fireBuildingPlaced(int c, int r, const std::string& building, const CityScriptStats& stats);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// Build a CityScriptHost and load the citybuilder's base scripts
// (mods/citybuilder/scripts/*.lua).
std::unique_ptr<CityScriptHost> createCityScriptHost();

}  // namespace odai::citybuilder
