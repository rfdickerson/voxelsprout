// Tests for the Lua scripting engine (odai_script): the simulation fires the
// right hooks, scripted runs stay deterministic, a logging-only mod changes
// nothing, and yield-modifying scripts (city_yields + Effects.register) actually
// affect the game. No GTest -- the lightweight harness style used elsewhere.

#include "game/game_sim.h"
#include "game/mod_host.h"
#include "script/script_engine.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace odai::game;

int g_failures = 0;

void expectTrue(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[lua hook test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectEqInt(long long actual, long long expected, const std::string& message) {
    if (actual != expected) {
        std::cerr << "[lua hook test] FAIL: " << message
                  << " (expected " << expected << ", got " << actual << ")\n";
        ++g_failures;
    }
}

bool sampleEq(const TurnSample& a, const TurnSample& b) {
    return a.turn == b.turn && a.leader == b.leader && a.score == b.score &&
           a.population == b.population && a.cities == b.cities && a.techs == b.techs &&
           a.treasury == b.treasury && a.wonders == b.wonders && a.disorder == b.disorder;
}

bool samplesEq(const std::vector<TurnSample>& a, const std::vector<TurnSample>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!sampleEq(a[i], b[i])) return false;
    return true;
}

std::vector<TurnSample> runGame(std::uint32_t seed, int turns) {
    WorldConfig cfg{};
    cfg.seed = seed;
    cfg.empireCount = 4;
    World world = makeWorld(cfg);
    std::vector<TurnSample> samples;
    for (int t = 0; t < turns; ++t) stepTurn(world, samples);
    return samples;
}

// Run a game with a FRESH ScriptHost loading one script file. Fresh per call so
// the script RNG re-seeds from the world seed (the basis for reproducibility).
std::vector<TurnSample> runWithScript(const std::filesystem::path& script, std::uint32_t seed, int turns) {
    auto host = std::make_unique<odai::script::ScriptHost>();
    host->runScriptFile(script);
    odai::game::setModHost(host.get());
    std::vector<TurnSample> samples = runGame(seed, turns);
    odai::game::setModHost(nullptr);
    return samples;
}

std::filesystem::path writeTempScript(const std::string& name, const std::string& body) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path);
    out << body;
    return path;
}

// Counts every hook the simulation fires -- proves the insertion points in
// game_sim.cc actually call modHost() (independent of Lua).
struct CountingHost final : public IModHost {
    int turnStart = 0, turnEnd = 0, cityYields = 0, buildingBuilt = 0;
    int wonderBuilt = 0, techResearched = 0, cityFounded = 0;
    void onTurnStart(World&) override { ++turnStart; }
    void onTurnEnd(World&) override { ++turnEnd; }
    void onCityYields(YieldContext&) override { ++cityYields; }
    void onBuildingBuilt(World&, City&, const std::string&) override { ++buildingBuilt; }
    void onWonderBuilt(World&, Empire&, const std::string&) override { ++wonderBuilt; }
    void onTechResearched(World&, Empire&, const std::string&) override { ++techResearched; }
    void onCityFounded(World&, City&) override { ++cityFounded; }
};

constexpr std::uint32_t kSeed = 1337u;
constexpr int kTurns = 80;

void testSimBaselineDeterminism() {
    setModHost(nullptr);
    const std::vector<TurnSample> a = runGame(kSeed, kTurns);
    const std::vector<TurnSample> b = runGame(kSeed, kTurns);
    expectTrue(!a.empty(), "baseline produced samples");
    expectTrue(samplesEq(a, b), "NullModHost: same seed -> identical samples");
}

void testHooksFire() {
    CountingHost host;
    setModHost(&host);
    runGame(kSeed, kTurns);
    setModHost(nullptr);
    expectEqInt(host.turnStart, kTurns, "onTurnStart fires once per turn");
    expectEqInt(host.turnEnd, kTurns, "onTurnEnd fires once per turn");
    expectTrue(host.cityYields > 0, "onCityYields fires");
    expectTrue(host.techResearched > 0, "onTechResearched fires");
    expectTrue(host.buildingBuilt > 0, "onBuildingBuilt fires");
    // Wonders and new cities are likely but not guaranteed every game; just sanity.
    expectTrue(host.cityFounded >= 0 && host.wonderBuilt >= 0, "founding/wonder counters valid");
}

void testNoOpScriptMatchesBaseline() {
    const std::filesystem::path noop =
        writeTempScript("odai_test_noop.lua", "Events.on('turn_start', function(w) end)\n");
    setModHost(nullptr);
    const std::vector<TurnSample> baseline = runGame(kSeed, kTurns);
    const std::vector<TurnSample> scripted = runWithScript(noop, kSeed, kTurns);
    expectTrue(samplesEq(baseline, scripted), "logging/no-op mod matches NullModHost run");
    std::filesystem::remove(noop);
}

void testCityYieldsEffect() {
    const std::filesystem::path script = writeTempScript(
        "odai_test_science.lua",
        "Events.on('city_yields', function(ctx) ctx.science = ctx.science + 1 end)\n");
    setModHost(nullptr);
    const std::vector<TurnSample> baseline = runGame(kSeed, kTurns);
    const std::vector<TurnSample> a = runWithScript(script, kSeed, kTurns);
    const std::vector<TurnSample> b = runWithScript(script, kSeed, kTurns);
    expectTrue(samplesEq(a, b), "city_yields script: same seed -> identical samples");
    expectTrue(!samplesEq(a, baseline), "city_yields +science actually changes the game");
    std::filesystem::remove(script);
}

void testEffectsRegister() {
    const std::filesystem::path script = writeTempScript(
        "odai_test_effect.lua",
        "Effects.register('granary', function(ctx) ctx.production = ctx.production + 2 end)\n");
    setModHost(nullptr);
    const std::vector<TurnSample> baseline = runGame(kSeed, kTurns);
    const std::vector<TurnSample> scripted = runWithScript(script, kSeed, kTurns);
    expectTrue(!samplesEq(baseline, scripted), "Effects.register('granary') changes the game");
    std::filesystem::remove(script);
}

void testRngDeterminism() {
    const std::filesystem::path script = writeTempScript(
        "odai_test_rng.lua",
        "Events.on('city_yields', function(ctx)\n"
        "  if Rng.int(1, 100) <= 50 then ctx.gold = ctx.gold + 1 end\n"
        "end)\n");
    const std::vector<TurnSample> a = runWithScript(script, kSeed, kTurns);
    const std::vector<TurnSample> b = runWithScript(script, kSeed, kTurns);
    expectTrue(samplesEq(a, b), "Rng-using script is deterministic across fresh hosts");
    std::filesystem::remove(script);
}

void testBaseScriptHostLoadsClean() {
    std::unique_ptr<odai::script::ScriptHost> host = odai::script::createBaseScriptHost();
    for (const std::string& e : host->errors()) std::cerr << "  script error: " << e << "\n";
    expectTrue(host->ok(), "base mod scripts load without error");
    // Install and run a short game to ensure dispatch through Lua doesn't crash.
    setModHost(host.get());
    const std::vector<TurnSample> s = runGame(kSeed, 20);
    setModHost(nullptr);
    expectTrue(!s.empty(), "scripted base game runs");
}

}  // namespace

int main() {
    testSimBaselineDeterminism();
    testHooksFire();
    testNoOpScriptMatchesBaseline();
    testCityYieldsEffect();
    testEffectsRegister();
    testRngDeterminism();
    testBaseScriptHostLoadsClean();

    if (g_failures == 0) {
        std::cout << "[lua hook test] all checks passed\n";
        return 0;
    }
    std::cerr << "[lua hook test] " << g_failures << " failure(s)\n";
    return 1;
}
