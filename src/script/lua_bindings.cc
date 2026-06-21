#include "script/lua_internal.h"

#include "game/economy.h"
#include "game/game_sim.h"

#include <algorithm>

// Registers the C++ game types as Lua usertypes. Read access is exposed widely;
// mutation is deliberately narrow (treasury/culture via helper methods, and the
// numeric fields of YieldContext inside a city_yields callback) so scripts can
// shape the economy without corrupting invariants.
namespace odai::script {

using odai::game::City;
using odai::game::Empire;
using odai::game::World;
using odai::game::Yields;
using odai::game::YieldContext;

void registerBindings(EngineState& es) {
    sol::state& lua = es.lua;

    lua.new_usertype<Yields>(
        "Yields", sol::no_constructor,
        "food", &Yields::food,
        "production", &Yields::production,
        "gold", &Yields::gold,
        "science", &Yields::science,
        "culture", &Yields::culture);

    // The yield accumulators a city_yields / Effects callback may adjust. The five
    // yield fields are exposed directly (mapping onto the flat Yields) so a script
    // reads `ctx.science` rather than `ctx.flat.science`.
    lua.new_usertype<YieldContext>(
        "YieldContext", sol::no_constructor,
        "food", sol::property([](YieldContext& c) { return c.flat.food; },
                              [](YieldContext& c, int v) { c.flat.food = v; }),
        "production", sol::property([](YieldContext& c) { return c.flat.production; },
                                   [](YieldContext& c, int v) { c.flat.production = v; }),
        "gold", sol::property([](YieldContext& c) { return c.flat.gold; },
                              [](YieldContext& c, int v) { c.flat.gold = v; }),
        "science", sol::property([](YieldContext& c) { return c.flat.science; },
                                 [](YieldContext& c, int v) { c.flat.science = v; }),
        "culture", sol::property([](YieldContext& c) { return c.flat.culture; },
                                 [](YieldContext& c, int v) { c.flat.culture = v; }),
        "prodPct", &YieldContext::prodPct,
        "goldPct", &YieldContext::goldPct,
        "sciencePct", &YieldContext::sciencePct,
        "happy", &YieldContext::happy,
        "growBonus", &YieldContext::growBonus,
        "city", sol::readonly(&YieldContext::city),
        "empire", sol::readonly(&YieldContext::empire));

    lua.new_usertype<City>(
        "City", sol::no_constructor,
        "name", sol::readonly(&City::name),
        "population", sol::readonly(&City::population),
        "owner", sol::readonly(&City::owner),
        "producing", sol::readonly(&City::producing),
        "in_disorder", sol::readonly(&City::inDisorder),
        "num_buildings", [](City& c) { return static_cast<int>(c.buildings.size()); },
        "has_building", &City::hasBuilding);

    lua.new_usertype<Empire>(
        "Empire", sol::no_constructor,
        "id", sol::readonly(&Empire::id),
        "name", sol::readonly(&Empire::name),
        "leader_name", sol::readonly(&Empire::leaderName),
        "treasury", sol::readonly(&Empire::treasury),
        "science_pool", sol::readonly(&Empire::sciencePool),
        "culture", sol::readonly(&Empire::culturePoints),
        "num_cities", [](Empire& e) { return static_cast<int>(e.cityIndices.size()); },
        "num_wonders", [](Empire& e) { return static_cast<int>(e.wonders.size()); },
        "knows", &Empire::knows,
        "owns_wonder", [](Empire& e, const std::string& id) {
            return std::find(e.wonders.begin(), e.wonders.end(), id) != e.wonders.end();
        },
        // Narrow, replayable mutations usable outside yield callbacks.
        "add_gold", [](Empire& e, int n) { e.treasury += n; },
        "add_culture", [](Empire& e, int n) { e.culturePoints += n; });

    lua.new_usertype<World>(
        "World", sol::no_constructor,
        "turn", sol::readonly(&World::turn),
        "num_cities", [](World& w) { return static_cast<int>(w.cities.size()); },
        "city", [](World& w, int i) -> City* {
            return (i >= 1 && i <= static_cast<int>(w.cities.size())) ? &w.cities[static_cast<std::size_t>(i - 1)] : nullptr;
        },
        "num_empires", [](World& w) { return static_cast<int>(w.empires.size()); },
        "empire", [](World& w, int i) -> Empire* {
            return (i >= 1 && i <= static_cast<int>(w.empires.size())) ? &w.empires[static_cast<std::size_t>(i - 1)] : nullptr;
        });
}

}  // namespace odai::script
