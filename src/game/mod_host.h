#pragma once

#include "game/economy.h"  // Yields

#include <string>

// The scripting seam between the pure-CPU simulation (game/) and the Lua engine
// (script/). game_sim.cc calls modHost().onX(...) at gameplay events; the concrete
// host is injected at startup by the app. This header contains NO Lua: the default
// host is a no-op, so tools and tests that never install a host stay Lua-free and
// fully deterministic. The concrete sol2 implementation lives in src/script/.
namespace odai::game {

struct World;
struct Empire;
struct City;

// Mutable per-city yield accumulators handed to onCityYields, mirroring the locals
// computeCityYields builds. A script may adjust the numeric fields before the
// percentage multipliers are applied; the read-only pointers give context.
struct YieldContext {
    World* world = nullptr;
    City* city = nullptr;
    Empire* empire = nullptr;  // may be null (a city with no living owner)
    Yields flat{};             // accumulated flat yields (food/prod/gold/sci/culture)
    int prodPct = 0;           // % production bonus accumulated so far
    int goldPct = 0;
    int sciencePct = 0;
    int happy = 0;             // happiness cap accumulated so far
    int growBonus = 0;         // food-box carryover % accumulated so far
};

// Abstract host the simulation notifies at gameplay events. All methods default to
// no-ops so a partial host only overrides what it needs, and the NullModHost (the
// default) changes nothing.
class IModHost {
public:
    virtual ~IModHost() = default;
    virtual void onTurnStart(World& /*world*/) {}
    virtual void onTurnEnd(World& /*world*/) {}
    virtual void onCityYields(YieldContext& /*ctx*/) {}
    virtual void onBuildingBuilt(World& /*world*/, City& /*city*/, const std::string& /*buildingId*/) {}
    virtual void onWonderBuilt(World& /*world*/, Empire& /*empire*/, const std::string& /*wonderId*/) {}
    virtual void onTechResearched(World& /*world*/, Empire& /*empire*/, const std::string& /*techId*/) {}
    virtual void onCityFounded(World& /*world*/, City& /*city*/) {}
};

// The process-wide active host. Defaults to a no-op host until setModHost installs
// one. The pointed-to host must outlive its use; pass nullptr to revert to no-op.
IModHost& modHost();
void setModHost(IModHost* host);

}  // namespace odai::game
