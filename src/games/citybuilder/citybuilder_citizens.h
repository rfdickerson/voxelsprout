#pragma once

#include "games/citybuilder/script/city_script.h"
#include "procgen/rng.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <vector>

// The "notable citizens" layer: a sampled roster of named sims living on top
// of the aggregate census. Citizens have homes, workplaces, traits, spouses —
// and, occasionally, affairs. The C++ engine here does all the bookkeeping and
// probability rolls; every piece of *text* (story templates, weights, trait
// requirements) comes from the Lua Stories/Needs tables, so the tabloid tone
// is fully moddable without a rebuild.
//
// Deliberately decoupled from the tile grid: the app hands in flat lists of
// home sites and destinations each month, so this file never touches Tile.
namespace odai::games::citybuilder {

// Trait bits; the string tags Lua templates use in `requires` map onto these
// ("fit", "parent", "nightowl", "gossip") plus the derived "married"/"affair".
constexpr std::uint8_t kTraitFit = 1u;
constexpr std::uint8_t kTraitParent = 2u;
constexpr std::uint8_t kTraitNightOwl = 4u;
constexpr std::uint8_t kTraitGossip = 8u;

struct Citizen {
    std::uint32_t seed = 0;          // identity; names derive from it
    std::string firstName;
    std::string lastName;
    short homeC = -1, homeR = -1;    // developed residential tile
    short workC = -1, workR = -1;    // destination tile (business or civic)
    signed char spouse = -1;         // roster indices; -1 = none
    signed char affair = -1;
    std::uint8_t traits = 0;
    bool atWork = false;             // schedule state: commuted out, not yet home

    [[nodiscard]] std::string fullName() const { return firstName + " " + lastName; }
};

enum class TickerKind : std::uint8_t { Opening, Life, Drama, Arrival, Departure };

struct TickerItem {
    std::string text;
    TickerKind kind = TickerKind::Life;
    short c = -1, r = -1;   // tile the story anchors to (-1 = none); click pans
    float age = 0.0f;       // seconds since emission (the app advances this)
};

// A named place a citizen can visit. category matches the Lua needs table
// ("yoga", "daycare", "cafe", ..., plus civic "park"/"school").
struct Destination {
    short c = 0, r = 0;
    std::string category;
    std::string name;
};

struct HomeSite {
    short c = 0, r = 0;
    float develop = 0.0f;   // spawn weight
};

struct ReconcileInput {
    int population = 0;
    const std::vector<HomeSite>* homes = nullptr;
    const std::vector<Destination>* destinations = nullptr;
    // Street label for a tile (the app resolves the nearest road run).
    std::function<std::string(short c, short r)> streetName;
};

class CitizenSim {
public:
    // storyBoost scales every event probability (ODAI_CITY_STORY QA hook).
    void configure(odai::citybuilder::CityScriptHost* script, std::uint32_t worldSeed,
                   float storyBoost);

    // Monthly roster churn + event rolls, called from stepMonth after the
    // post-growth census: spawn to target, remove citizens whose homes
    // abandoned, re-home lost workplaces, roll life/drama stories.
    void reconcileMonthly(const ReconcileInput& in);

    // A commercial lot developed this month — the SimCity-newspaper beat.
    void emitOpening(const Destination& dest, const std::string& street);

    // A weekend beat (Saturday soccer, farmers market, ...), anchored to a
    // park or other destination. Rolls the Lua "weekend" story templates.
    void emitWeekendStory(const Destination& dest, const std::string& street);

    // Where the day/week clock currently stands; drives which trips make
    // sense (commute out, lunch run, commute home, night out, weekend).
    struct TripContext {
        int weekday = 0;      // 0=Mon .. 6=Sun
        float hour = 12.0f;   // 0..24
    };

    // Pick a citizen and a schedule-appropriate activity and return origin/
    // destination tiles for the app to route a car toward. Morning rush sends
    // workers out (marking them atWork), evening rush brings the same sims
    // home, lunch runs leave from the workplace, night owls hit the town, and
    // weekends swap commuting for errands and Saturday soccer at the park.
    struct Trip {
        short fromC = -1, fromR = -1, toC = -1, toR = -1;
    };
    bool rollTrip(const std::vector<Destination>& destinations, const TripContext& ctx, Trip& out);

    // Friday night: everyone clocks out, whether or not their commute-home
    // trip got sampled (called by the app at the weekend rollover).
    void endWorkWeek();

    [[nodiscard]] const std::vector<Citizen>& roster() const { return m_roster; }
    std::deque<TickerItem>& ticker() { return m_ticker; }

private:
    void pushTicker(std::string text, TickerKind kind, short c, short r);
    // All tags currently true for a citizen ("fit", "married", "affair", ...).
    [[nodiscard]] bool hasTag(const Citizen& cz, const std::string& tag) const;
    [[nodiscard]] const odai::citybuilder::StoryTemplate* pickStory(const std::string& kind,
                                                                    const Citizen& cz);
    [[nodiscard]] std::string interpolate(const std::string& tpl, const Citizen& a,
                                          const Citizen* b, const Destination* place,
                                          const std::string& street) const;

    odai::citybuilder::CityScriptHost* m_script = nullptr;
    std::uint32_t m_worldSeed = 1u;
    float m_storyBoost = 1.0f;
    odai::procgen::Rng m_rng{0xC171F0u};
    std::uint32_t m_citizenCounter = 0;
    std::vector<Citizen> m_roster;
    std::deque<TickerItem> m_ticker;
};

}  // namespace odai::games::citybuilder
