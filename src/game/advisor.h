#pragma once

#include <cstdint>
#include <string>
#include <vector>

// The Council of Houses: a Civ-3-style advisor system themed to Morrowind's Great
// Houses and Temple. Each advisor owns a gameplay domain and a personality; the
// rule engine reads a read-only snapshot of the live game and returns flavored
// advice attributed to the relevant advisor.
//
// Pure CPU data with no Vulkan / renderer / UI types so it can be unit-tested in
// isolation. The catalog + evaluateAdvisors() are deliberately UI-agnostic: the
// app maps the returned advice into its own UI structs, exactly as it does for the
// tech tree, and a future palace/throne-room screen can reuse them unchanged.
//
// The snapshot (AdvisorWorldView) mirrors the live economy model -- the app fills
// it from its game World's player Empire + that empire's cities + the world event
// log. The view is deliberately decoupled from game_sim.h (the app does the trivial
// field-copy) so the advisor module stays independently testable.
namespace odai::game {

// The gameplay area an advisor speaks for, mapped to a Great House / faction.
enum class AdvisorDomain : std::uint8_t {
    Domestic = 0,  // economy / growth / production   (House Hlaalu)
    Military,      // units / defense                 (House Redoran)
    Arcane,        // research / science              (House Telvanni)
    Cultural,      // culture / faith / the people    (Tribunal Temple)
    Count
};
const char* advisorDomainName(AdvisorDomain domain);

// A council member. Static catalog data (see advisorCatalog()).
struct Advisor {
    std::string   id;            // stable key, e.g. "hlaalu_councilor"
    std::string   displayName;   // "Councilor Dolvas Andrano"
    std::string   title;         // "Hlaalu Trade Councilor"
    AdvisorDomain domain = AdvisorDomain::Domestic;
    std::string   portraitName;  // icon-registry name into the advisor atlas
    std::string   voice;         // one-line tone descriptor (for authors / tests)
    std::string   greeting;      // rich-text line shown when no advice is pending
};

// How loudly a piece of advice should be surfaced. Urgent/Warn also fire as
// proactive toasts; Info shows only on the council screen.
enum class AdviceSeverity : std::uint8_t { Info = 0, Warn, Urgent };

// One recommendation, attributed to an advisor. `key` is a stable de-dup id (e.g.
// "idle_city:3,5") the toast layer uses to avoid re-nudging the same standing
// condition every turn.
struct Advice {
    std::string    advisorId;
    AdviceSeverity severity = AdviceSeverity::Info;
    std::string    key;
    std::string    headline;   // short, toast-sized
    std::string    body;       // rich-text (see ui/rich_text.h markup) for the screen
    std::string    category;   // optional grouping label ("Production", ...)
};

// The subset of world-event kinds advisors react to. The app maps its
// GameEvent::Kind onto these (everything it doesn't care about -> Other).
enum class WorldEventKind : std::uint8_t {
    Other = 0,
    WonderBuilt,
    WonderLost,
    Starving,
};

// A notable thing that happened recently (new since the last evaluation), so an
// advisor can react in character. `text` is the player-facing event line.
struct WorldEvent {
    WorldEventKind kind = WorldEventKind::Other;
    std::string text;
};

// A read-only snapshot of the live game the app assembles from its player Empire,
// that empire's cities, its live units, and the world event log. Pure data with no
// app/UI/game_sim types.
struct AdvisorWorldView {
    struct City {
        std::string name;
        std::uint32_t col = 0;
        std::uint32_t row = 0;
        int population = 1;
        std::vector<std::string> buildings;   // completed building/wonder ids
        std::string producing;                // id in progress, "" == idle
        std::string producingName;            // display name of `producing` (app-filled)
        int turnsToFinish = 0;                // est. turns to complete `producing`
        int happyCap = 0;                     // happiness capacity
        bool inDisorder = false;              // population exceeds happiness -> revolt
    };
    struct UnitCount {
        int military = 0;
        int settlers = 0;
        int total = 0;
    };

    int turn = 0;
    std::string leaderName;                // player ruler's name (for flavor)

    std::vector<City> playerCities;        // player-owned only
    UnitCount units;                       // player-owned live units

    // Player empire economy / totals.
    int treasury = 0;
    int culturePoints = 0;
    int totalPopulation = 0;

    // Research (the player empire's current target + banked science).
    std::string researchTechId;            // "" == nothing selected
    std::string researchName;              // display name of current target, or ""
    int researchAccumulated = 0;           // science banked toward the target
    int researchCost = 0;                  // effective science cost of the target
    int sciencePerTurn = 0;                // player empire's science output this turn
    bool canBuildLibrary = false;          // a researched tech unlocks the library (app-filled)

    // Era (tracked app-side; the engine only reacts to a transition).
    std::string eraName;
    bool eraAdvancedThisTurn = false;

    // Notable events new since the last evaluation, for in-character reactions.
    std::vector<WorldEvent> recentEvents;
};

// Static roster: exactly one advisor per AdvisorDomain, in domain order.
const std::vector<Advisor>& advisorCatalog();
const Advisor* findAdvisor(const std::string& id);
const Advisor* advisorForDomain(AdvisorDomain domain);

// The rule engine. Pure and deterministic: the same view yields the same advice,
// sorted severity-descending then by advisor domain order, so urgent advice
// surfaces first in both the council screen and the toast queue. Never returns an
// empty vector when there is a player to address (an empty empire gets a single
// Temple greeting).
std::vector<Advice> evaluateAdvisors(const AdvisorWorldView& view);

}  // namespace odai::game
