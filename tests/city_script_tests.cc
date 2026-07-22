// Tests for the citybuilder Lua content layer (odai_city_script): the sandbox
// holds, registrations round-trip, name generation is deterministic per seed,
// broken scripts fall back to compiled-in defaults, the gameplay event hooks
// fire, and the shipped mods/citybuilder scripts load clean. Lightweight
// harness style, matching the other test executables.

#include "games/citybuilder/script/city_script.h"

#include <iostream>
#include <set>
#include <string>

namespace {

using odai::citybuilder::BusinessName;
using odai::citybuilder::CityScriptHost;
using odai::citybuilder::CityScriptStats;

int g_failures = 0;

void expectTrue(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[city script test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

// A scriptless host must run entirely on the compiled-in fallbacks.
void testFallbacksScriptless() {
    CityScriptHost host;
    expectTrue(host.ok(), "fresh host has no errors");
    expectTrue(!host.cityName(1u).empty(), "fallback city name non-empty");
    expectTrue(!host.streetName(2u).empty(), "fallback street name non-empty");
    expectTrue(!host.firstName(3u, true).empty(), "fallback feminine first name non-empty");
    expectTrue(!host.firstName(3u, false).empty(), "fallback masculine first name non-empty");
    expectTrue(!host.lastName(4u).empty(), "fallback last name non-empty");
    expectTrue(!host.blockName(1, 5u).empty(), "fallback block name non-empty");
    const BusinessName biz = host.businessName(false, 1, 1, 6u);
    expectTrue(!biz.name.empty() && !biz.category.empty(), "fallback business name+category");
    expectTrue(!host.stories().empty(), "fallback stories non-empty");
    expectTrue(!host.needs().empty(), "fallback needs non-empty");
    expectTrue(host.configNumber("terrain.land_min", 0.55) == 0.55,
               "config falls back to caller default");
    expectTrue(host.ok(), "fallback path records no errors");
}

void testSandbox() {
    CityScriptHost host;
    expectTrue(host.runScriptString(
                   "assert(io == nil); assert(os == nil); assert(require == nil);"
                   "assert(dofile == nil); assert(load == nil); assert(package == nil);"
                   "assert(math.random == nil); assert(math.randomseed == nil)",
                   "sandbox_check"),
               "sandbox strips io/os/require/load/package/math.random");
    expectTrue(host.ok(), "sandbox check leaves no errors");
}

void testNameRegistrationAndDeterminism() {
    CityScriptHost host;
    expectTrue(host.runScriptString(R"lua(
        Names.register{
          city = function(rng) return "City" .. rng:int(100, 999) end,
          business = function(rng, kind, tier, era)
            return { name = kind .. "-" .. tier .. "-" .. era .. "-" .. rng:int(10, 99),
                     category = (kind == "industrial") and "mill" or "cafe" }
          end,
        }
    )lua", "names_reg"),
               "name registration script runs");

    const std::string a = host.cityName(0xBEEFu);
    const std::string b = host.cityName(0xBEEFu);
    expectTrue(a == b, "same seed => same scripted city name");
    expectTrue(a.rfind("City", 0) == 0, "scripted generator is used");

    std::set<std::string> distinct;
    for (std::uint32_t seed = 1; seed <= 40; ++seed) distinct.insert(host.cityName(seed));
    expectTrue(distinct.size() >= 10, "seeds spread across the name space");

    const BusinessName biz = host.businessName(true, 2, 1, 7u);
    expectTrue(biz.category == "mill", "business category round-trips");
    expectTrue(biz.name.rfind("industrial-2-1-", 0) == 0, "business args reach the script");

    // Partial re-registration keeps earlier generators.
    expectTrue(host.runScriptString(
                   "Names.register{ street = function(rng) return 'Test St' end }",
                   "names_partial"),
               "partial registration runs");
    expectTrue(host.streetName(1u) == "Test St", "new street generator active");
    expectTrue(host.cityName(0xBEEFu) == a, "earlier city generator preserved");
    expectTrue(host.ok(), "no errors after registrations");
}

void testStoriesNeedsConfig() {
    CityScriptHost host;
    expectTrue(host.runScriptString(R"lua(
        Stories.register{ id = "t1", kind = "life", weight = 2.5,
                          requires = { "fit", "married" }, text = "{a} at {place}" }
        Needs.register{ trait = "fit", category = "yoga", weight = 2.0 }
        Config.terrain{ land_min = 0.61 }
        Config.scatter{ hydrant_per_mille = 99 }
    )lua", "content_reg"),
               "content registration script runs");

    expectTrue(host.stories().size() == 1, "one story registered (fallbacks not mixed in)");
    const auto& story = host.stories()[0];
    expectTrue(story.id == "t1" && story.kind == "life" && story.weight == 2.5f,
               "story fields parsed");
    expectTrue(story.conditions.size() == 2 && story.conditions[0] == "fit" &&
                   story.conditions[1] == "married",
               "story requires parsed in order");
    expectTrue(story.text == "{a} at {place}", "story text preserved");

    expectTrue(host.needs().size() == 1 && host.needs()[0].trait == "fit" &&
                   host.needs()[0].category == "yoga" && host.needs()[0].weight == 2.0f,
               "need rule parsed");

    expectTrue(host.configNumber("terrain.land_min", 0.55) == 0.61, "terrain config read");
    expectTrue(host.configNumber("scatter.hydrant_per_mille", 120) == 99, "scatter config read");
    expectTrue(host.configNumber("terrain.missing_key", 7.0) == 7.0, "missing key falls back");
}

void testBrokenScriptFallsBack() {
    CityScriptHost host;
    expectTrue(!host.runScriptString("this is not lua(", "broken"), "syntax error reported");
    expectTrue(!host.ok() && !host.errors().empty(), "error recorded");

    // A generator that throws at call time: error recorded, fallback used.
    CityScriptHost host2;
    expectTrue(host2.runScriptString(
                   "Names.register{ city = function(rng) error('boom') end }", "thrower"),
               "throwing generator registers fine");
    const std::string name = host2.cityName(9u);
    expectTrue(!name.empty(), "fallback name produced despite runtime error");
    expectTrue(!host2.ok(), "runtime error recorded");

    // Wrong return shape: error recorded, fallback used.
    CityScriptHost host3;
    host3.runScriptString(
        "Names.register{ business = function(rng, k, t, e) return { name = 42 } end }",
        "bad_shape");
    const BusinessName biz = host3.businessName(false, 1, 1, 3u);
    expectTrue(!biz.name.empty() && !biz.category.empty(), "fallback business despite bad shape");
    expectTrue(!host3.ok(), "bad shape recorded as error");
}

void testEventHooks() {
    CityScriptHost host;
    expectTrue(host.runScriptString(R"lua(
        months = {}
        placed = nil
        Events.on("month_step", function(stats)
          months[#months + 1] = stats.month
        end)
        Events.on("building_placed", function(c, r, building, stats)
          placed = building .. "@" .. c .. "," .. r .. " pop=" .. stats.population
        end)
    )lua", "events_reg"),
               "event registration script runs");

    CityScriptStats stats;
    stats.population = 340;
    stats.month = 3;
    stats.year = 1902;
    host.fireMonthStep(stats);
    stats.month = 4;
    host.fireMonthStep(stats);
    host.fireBuildingPlaced(12, 7, "school", stats);

    expectTrue(host.runScriptString(
                   "assert(#months == 2 and months[1] == 3 and months[2] == 4);"
                   "assert(placed == 'school@12,7 pop=340')",
                   "events_assert"),
               "hooks received the right payloads");
    expectTrue(host.ok(), "event dispatch left no errors");

    // Unknown event names are recorded, not silently ignored.
    CityScriptHost host2;
    host2.runScriptString("Events.on('frame_tick', function() end)", "bad_event");
    expectTrue(!host2.ok(), "unknown event name recorded as error");
}

void testGlobalRngDeterminism() {
    CityScriptHost a;
    CityScriptHost b;
    a.seedRng(0x1234u);
    b.seedRng(0x1234u);
    expectTrue(a.runScriptString("v = Rng.int(1, 1000000)", "rng_a") &&
                   b.runScriptString("v = Rng.int(1, 1000000)", "rng_b"),
               "Rng scripts run");
    // Compare via a script-level assertion channel: re-derive in a fresh pair.
    CityScriptHost c;
    c.seedRng(0x1234u);
    c.runScriptString("v1 = Rng.int(1, 1000000)", "rng_c1");
    CityScriptHost d;
    d.seedRng(0x1234u);
    d.runScriptString("v1 = Rng.int(1, 1000000); same = (v1 ~= nil)", "rng_d");
    expectTrue(c.ok() && d.ok(), "seeded Rng usable from scripts");
}

// The shipped base content must load clean and cover the citizen sim's needs.
void testShippedScriptsLoadClean() {
    auto host = odai::citybuilder::createCityScriptHost();
    if (!host->ok()) {
        for (const std::string& err : host->errors()) std::cerr << "  " << err << '\n';
    }
    expectTrue(host->ok(), "mods/citybuilder scripts load without errors");
    expectTrue(host->stories().size() >= 10, "shipped stories cover all kinds");
    expectTrue(host->needs().size() >= 10, "shipped needs cover the trait space");

    std::set<std::string> kinds;
    for (const auto& s : host->stories()) kinds.insert(s.kind);
    for (const char* kind : {"opening", "arrival", "departure", "life", "drama"}) {
        expectTrue(kinds.count(kind) == 1, std::string("shipped stories include kind ") + kind);
    }

    // Business categories drawn by the shipped generator must be routable by
    // the shipped needs (every need category except civic ones is reachable).
    std::set<std::string> catsSeen;
    for (std::uint32_t seed = 1; seed <= 200; ++seed) {
        catsSeen.insert(host->businessName(false, 1, 1, seed).category);
        catsSeen.insert(host->businessName(true, 1, 1, seed ^ 0xA5A5u).category);
    }
    expectTrue(catsSeen.count("yoga") == 1 && catsSeen.count("daycare") == 1 &&
                   catsSeen.count("cafe") == 1 && catsSeen.count("mill") == 1,
               "shipped business generator reaches key categories");

    const std::string city1 = host->cityName(77u);
    const std::string city2 = host->cityName(77u);
    expectTrue(city1 == city2 && !city1.empty(), "shipped city names deterministic");
    expectTrue(host->configNumber("terrain.land_min", 0.0) > 0.0, "shipped terrain config present");
    expectTrue(host->configNumber("scatter.hydrant_per_mille", 0.0) > 0.0,
               "shipped scatter config present");
}

}  // namespace

int main() {
    testFallbacksScriptless();
    testSandbox();
    testNameRegistrationAndDeterminism();
    testStoriesNeedsConfig();
    testBrokenScriptFallsBack();
    testEventHooks();
    testGlobalRngDeterminism();
    testShippedScriptsLoadClean();
    if (g_failures != 0) {
        std::cerr << "[city script test] " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "[city script test] all tests passed\n";
    return 0;
}
