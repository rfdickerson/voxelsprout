#include "game/advisor.h"

#include <algorithm>
#include <string>

namespace odai::game {
namespace {

// Advisor ids -- one per domain. Kept as named constants so the rules and the
// catalog never drift apart.
constexpr const char* kHlaalu   = "hlaalu_councilor";
constexpr const char* kRedoran  = "redoran_warlord";
constexpr const char* kTelvanni = "telvanni_magister";
constexpr const char* kTemple   = "temple_almoner";

// Speaker colors for the rich-text bodies (one accent per House).
constexpr const char* kHlaaluCol   = "#d8b86a";  // Hlaalu amber/gold
constexpr const char* kRedoranCol  = "#c46a4a";  // Redoran rust
constexpr const char* kTelvanniCol = "#9a7bc8";  // Telvanni arcane violet
constexpr const char* kTempleCol   = "#cdb88f";  // Temple ivory

constexpr int kLowTreasury     = 12;  // gold below this draws a warning
constexpr int kAqueductPopGate = 6;   // a city this large wants an aqueduct to keep growing

// Compose a flavored advice body: a bold colored speaker, a stage direction, and
// an italic spoken line.
std::string body(const char* color, const std::string& speaker,
                 const std::string& action, const std::string& quote) {
    return "<b><color=" + std::string(color) + ">" + speaker + "</color></b> " +
           action + " <i>\"" + quote + "\"</i>";
}

std::string coordKey(const std::string& prefix, const AdvisorWorldView::City& c) {
    return prefix + ":" + std::to_string(c.col) + "," + std::to_string(c.row);
}

bool cityHas(const AdvisorWorldView::City& c, const std::string& buildingId) {
    return std::find(c.buildings.begin(), c.buildings.end(), buildingId) != c.buildings.end();
}

// A readable name for what a city is producing. The app fills `producingName`; we
// fall back to the raw id (or "a settler") if it didn't.
std::string buildDisplayName(const AdvisorWorldView::City& c) {
    if (!c.producingName.empty()) return c.producingName;
    if (c.producing == "settler") return "a settler";
    return c.producing;
}

// Index of the player's largest city by population; ties break on the lowest
// (col, row) for determinism. Returns -1 when the player has no cities.
int largestCityIndex(const AdvisorWorldView& v) {
    int best = -1;
    for (int i = 0; i < static_cast<int>(v.playerCities.size()); ++i) {
        const auto& c = v.playerCities[i];
        if (best < 0) { best = i; continue; }
        const auto& b = v.playerCities[best];
        if (c.population > b.population ||
            (c.population == b.population &&
             (c.col < b.col || (c.col == b.col && c.row < b.row)))) {
            best = i;
        }
    }
    return best;
}

int severityRank(AdviceSeverity s) { return static_cast<int>(s); }

int domainRank(const std::string& advisorId) {
    const Advisor* a = findAdvisor(advisorId);
    return a != nullptr ? static_cast<int>(a->domain) : static_cast<int>(AdvisorDomain::Count);
}

}  // namespace

const char* advisorDomainName(AdvisorDomain domain) {
    switch (domain) {
        case AdvisorDomain::Domestic: return "Domestic";
        case AdvisorDomain::Military: return "Military";
        case AdvisorDomain::Arcane:   return "Arcane";
        case AdvisorDomain::Cultural: return "Cultural";
        case AdvisorDomain::Count:    break;
    }
    return "Unknown";
}

const std::vector<Advisor>& advisorCatalog() {
    static const std::vector<Advisor> kCatalog = {
        Advisor{
            kHlaalu, "Councilor Dolvas Andrano", "Hlaalu Trade Councilor",
            AdvisorDomain::Domestic, "hlaalu",
            "Smooth, mercantile, profit-minded",
            body(kHlaaluCol, "Councilor Andrano", "spreads his ledgers and smiles.",
                 "Coin flows where industry leads, friend. Keep the forges busy and the "
                 "treasury follows.")},
        Advisor{
            kRedoran, "Warlord Brara Morvayn", "Redoran War-Councilor",
            AdvisorDomain::Military, "redoran",
            "Blunt, honor-bound, martial",
            body(kRedoranCol, "Warlord Morvayn", "stands at ease, hand on hilt.",
                 "The Houses are quiet for now. See that our walls and our blades stay "
                 "ready, and they will remain so.")},
        Advisor{
            kTelvanni, "Magister Therana Sethan", "Telvanni Magister",
            AdvisorDomain::Arcane, "telvanni",
            "Haughty, arcane, contemptuous of haste",
            body(kTelvanniCol, "Magister Sethan", "regards you with half-lidded patience.",
                 "Knowledge is the only wealth that compounds. Tell me what your scholars "
                 "should pursue, and try not to waste my time.")},
        Advisor{
            kTemple, "Almoner Tholer Dalveni", "Tribunal Temple Almoner",
            AdvisorDomain::Cultural, "temple",
            "Pious, measured, speaks for the people",
            body(kTempleCol, "Almoner Dalveni", "bows his head.",
                 "The people are content, and the Tribunal is honored. Long may it remain "
                 "so under your hand.")},
    };
    return kCatalog;
}

const Advisor* findAdvisor(const std::string& id) {
    for (const Advisor& a : advisorCatalog()) {
        if (a.id == id) return &a;
    }
    return nullptr;
}

const Advisor* advisorForDomain(AdvisorDomain domain) {
    for (const Advisor& a : advisorCatalog()) {
        if (a.domain == domain) return &a;
    }
    return nullptr;
}

std::vector<Advice> evaluateAdvisors(const AdvisorWorldView& view) {
    std::vector<Advice> out;

    // No capital yet: a single steadying word, never an empty council.
    if (view.playerCities.empty()) {
        out.push_back(Advice{
            kTemple, AdviceSeverity::Info, "await_capital",
            "Awaiting your first city",
            body(kTempleCol, "Almoner Dalveni", "gazes to the horizon.",
                 "We are a people without a hearth. Send forth a settler and raise a "
                 "capital, that the Tribunal may watch over it."),
            "Founding"});
        return out;
    }

    // === Urgent ============================================================

    // Treasury in the red -- buildings will be sold off to cover the deficit.
    if (view.treasury < 0) {
        out.push_back(Advice{
            kHlaalu, AdviceSeverity::Urgent, "treasury_empty",
            "The treasury is empty",
            body(kHlaaluCol, "Councilor Andrano", "drums his fingers, agitated.",
                 "We are spending faster than we earn, and the coffers have run dry. Cut "
                 "your upkeep or raise your trade, before we must sell the very roofs over "
                 "our heads."),
            "Economy"});
    }

    // Cities in open disorder (population has overwhelmed happiness).
    for (const auto& c : view.playerCities) {
        if (c.inDisorder) {
            const std::string who = c.name.empty() ? std::string("A city") : c.name;
            out.push_back(Advice{
                kTemple, AdviceSeverity::Urgent, coordKey("disorder", c),
                who + " is in revolt",
                body(kTempleCol, "Almoner Dalveni", "wrings his hands.",
                     "The people of " + who + " have downed their tools in anger. Raise a "
                     "temple, a colosseum, anything to ease them -- a city in revolt "
                     "produces nothing."),
                "Happiness"});
        }
    }

    // No standing army while holding cities.
    if (view.units.military == 0) {
        out.push_back(Advice{
            kRedoran, AdviceSeverity::Urgent, "no_defense",
            "No soldiers stand guard",
            body(kRedoranCol, "Warlord Morvayn", "scowls.",
                 "You hold cities and not one blade to defend them? The Redoran do not "
                 "abide such folly. Raise a garrison before the ash-raiders do it for you."),
            "Defense"});
    }

    // A wonder lost out to a rival this turn (reactive).
    for (const WorldEvent& e : view.recentEvents) {
        if (e.kind == WorldEventKind::WonderLost) {
            out.push_back(Advice{
                kRedoran, AdviceSeverity::Warn, "evt_wonder_lost:" + e.text,
                e.text.empty() ? std::string("A wonder slips away") : e.text,
                body(kRedoranCol, "Warlord Morvayn", "strikes the table.",
                     "A rival House has finished a wonder we coveted. Shields spent for "
                     "nothing. Choose the next prize with more care -- and more speed."),
                "Wonders"});
        } else if (e.kind == WorldEventKind::Starving) {
            out.push_back(Advice{
                kTemple, AdviceSeverity::Warn, "evt_starving:" + e.text,
                e.text.empty() ? std::string("Famine in the realm") : e.text,
                body(kTempleCol, "Almoner Dalveni", "lowers his voice.",
                     "There is not bread enough, and the people go hungry. Work the "
                     "farmland and raise a granary, lest we lose them to famine."),
                "Growth"});
        }
    }

    // === Warn ==============================================================

    // Idle cities.
    for (const auto& c : view.playerCities) {
        if (c.producing.empty()) {
            const std::string who = c.name.empty() ? std::string("A city") : c.name;
            out.push_back(Advice{
                kHlaalu, AdviceSeverity::Warn, coordKey("idle_city", c),
                who + " sits idle",
                body(kHlaaluCol, "Councilor Andrano", "taps his ledger.",
                     "An idle forge is coin left in the dirt. Set " + who +
                     " to work -- a building, a soldier, anything but silence."),
                "Production"});
        }
    }

    // Nothing being researched.
    if (view.researchTechId.empty()) {
        out.push_back(Advice{
            kTelvanni, AdviceSeverity::Warn, "no_research",
            "No research underway",
            body(kTelvanniCol, "Magister Sethan", "arches an eyebrow.",
                 "Your scholars sit idle and uninstructed. Choose a course of study, lest "
                 "we be left in the dark while rivals walk in the light."),
            "Research"});
    }

    // Treasury low but not yet negative.
    if (view.treasury >= 0 && view.treasury < kLowTreasury) {
        out.push_back(Advice{
            kHlaalu, AdviceSeverity::Warn, "treasury_low",
            "Coffers running low",
            body(kHlaaluCol, "Councilor Andrano", "frowns at the tally.",
                 "Our reserves grow thin. A market or a trade route would steady us before "
                 "a lean turn empties the vaults entirely."),
            "Economy"});
    }

    // A restless (but not yet revolting) city.
    for (const auto& c : view.playerCities) {
        if (!c.inDisorder && c.happyCap > 0 && c.population >= c.happyCap) {
            const std::string who = c.name.empty() ? std::string("A city") : c.name;
            out.push_back(Advice{
                kTemple, AdviceSeverity::Warn, coordKey("unhappy", c),
                who + " grows restless",
                body(kTempleCol, "Almoner Dalveni", "speaks softly.",
                     who + " has grown as large as its contentment allows. One more soul "
                     "and it tips into revolt -- give the people a temple or a holiday "
                     "before it does."),
                "Happiness"});
        }
    }

    // Still a single city well into the game.
    if (view.playerCities.size() == 1 && view.turn >= 10) {
        out.push_back(Advice{
            kHlaalu, AdviceSeverity::Warn, "single_city",
            "Still but one city",
            body(kHlaaluCol, "Councilor Andrano", "frowns at the map.",
                 "One city is a market stall, not a realm. We must spread, or be "
                 "outgrown by every House that does."),
            "Expansion"});
    }

    // === Info ==============================================================

    // Entered a new era this turn (reactive, app-tracked).
    if (view.eraAdvancedThisTurn) {
        const std::string era = view.eraName.empty() ? std::string("a new age") : view.eraName;
        out.push_back(Advice{
            kTelvanni, AdviceSeverity::Info, "era:" + view.eraName,
            "We enter " + era,
            body(kTelvanniCol, "Magister Sethan", "inclines her head, almost approving.",
                 "Our learning carries us into " + era +
                 ". New arts and wonders open to us -- see that we are first to seize them."),
            "Research"});
    }

    // Current research will finish next turn.
    if (view.researchCost > 0 &&
        view.researchAccumulated + view.sciencePerTurn >= view.researchCost) {
        const std::string name = view.researchName.empty() ? std::string("Your study")
                                                            : view.researchName;
        out.push_back(Advice{
            kTelvanni, AdviceSeverity::Info, "research_done:" + view.researchTechId,
            name + " nearly complete",
            body(kTelvanniCol, "Magister Sethan", "permits a thin smile.",
                 name + " will be ours by the next moon. Have the next inquiry ready -- "
                 "idleness ill suits a Magister."),
            "Research"});
    }

    // A wonder completed this turn (reactive congratulation).
    for (const WorldEvent& e : view.recentEvents) {
        if (e.kind == WorldEventKind::WonderBuilt) {
            out.push_back(Advice{
                kTemple, AdviceSeverity::Info, "evt_wonder_built:" + e.text,
                e.text.empty() ? std::string("A wonder rises") : e.text,
                body(kTempleCol, "Almoner Dalveni", "smiles broadly.",
                     "A great work is finished, and the people rejoice. Let its glory "
                     "draw every eye to your name."),
                "Wonders"});
        }
    }

    // Small realm and no settler in the field.
    if (view.playerCities.size() <= 2 && view.units.settlers == 0) {
        out.push_back(Advice{
            kHlaalu, AdviceSeverity::Info, "expand_more",
            "Your realm is small",
            body(kHlaaluCol, "Councilor Andrano", "unrolls a trade map.",
                 "Two cities will not fill a treasury. Train a settler and claim the "
                 "fertile ground before a rival House does."),
            "Expansion"});
    }

    const int big = largestCityIndex(view);

    // The chief city has no granary.
    if (big >= 0 && !cityHas(view.playerCities[big], "granary")) {
        const auto& c = view.playerCities[big];
        const std::string who = c.name.empty() ? std::string("Your chief city") : c.name;
        out.push_back(Advice{
            kHlaalu, AdviceSeverity::Info, coordKey("need_granary", c),
            who + " lacks a granary",
            body(kHlaaluCol, "Councilor Andrano", "consults a grain tally.",
                 "A granary in " + who + " would let it grow fat and fast. Bread before "
                 "marble, I always say."),
            "Growth"});
    }

    // A large city that will soon stop growing without an aqueduct.
    if (big >= 0 && view.playerCities[big].population >= kAqueductPopGate &&
        !cityHas(view.playerCities[big], "aqueduct")) {
        const auto& c = view.playerCities[big];
        const std::string who = c.name.empty() ? std::string("your chief city") : c.name;
        out.push_back(Advice{
            kHlaalu, AdviceSeverity::Info, coordKey("need_aqueduct", c),
            "Growth is stalling",
            body(kHlaaluCol, "Councilor Andrano", "taps a cistern plan.",
                 who + " has grown about as large as its wells allow. An aqueduct would "
                 "let it swell well beyond."),
            "Growth"});
    }

    // A library is unlocked but the chief city hasn't built one.
    if (big >= 0 && view.canBuildLibrary &&
        !cityHas(view.playerCities[big], "library")) {
        const auto& c = view.playerCities[big];
        const std::string who = c.name.empty() ? std::string("your chief city") : c.name;
        out.push_back(Advice{
            kTelvanni, AdviceSeverity::Info, coordKey("need_library", c),
            "A library awaits building",
            body(kTelvanniCol, "Magister Sethan", "taps a scroll-case.",
                 "You have the learning to raise a library in " + who +
                 ", yet you have not. Knowledge unhoused is knowledge half-wasted."),
            "Research"});
    }

    // A city's current build finishes next turn.
    for (const auto& c : view.playerCities) {
        if (c.producing.empty() || c.turnsToFinish > 1) continue;
        const std::string who  = c.name.empty() ? std::string("A city") : c.name;
        const std::string what = buildDisplayName(c);
        out.push_back(Advice{
            kHlaalu, AdviceSeverity::Info, coordKey("build_done", c),
            what + " nearly built",
            body(kHlaaluCol, "Councilor Andrano", "checks the works.",
                 "The " + what + " in " + who + " is all but finished. Have its next "
                 "project chosen so no shield is wasted."),
            "Production"});
    }

    // No culture anywhere: no points and no temple or monument.
    bool anyCulture = view.culturePoints > 0;
    for (const auto& c : view.playerCities) {
        if (cityHas(c, "temple") || cityHas(c, "monument")) { anyCulture = true; break; }
    }
    if (!anyCulture) {
        out.push_back(Advice{
            kTemple, AdviceSeverity::Info, "no_culture",
            "The people lack a temple",
            body(kTempleCol, "Almoner Dalveni", "folds his hands.",
                 "Not one temple stands in all your realm. The people grow restless "
                 "without the Tribunal's comfort -- raise a temple, and they will keep "
                 "faith with you."),
            "Culture"});
    }

    // Surface urgent advice first, then by advisor domain order. Stable so rules of
    // equal weight keep their authored order (e.g. idle cities by position).
    std::stable_sort(out.begin(), out.end(), [](const Advice& a, const Advice& b) {
        const int sa = severityRank(a.severity);
        const int sb = severityRank(b.severity);
        if (sa != sb) return sa > sb;
        return domainRank(a.advisorId) < domainRank(b.advisorId);
    });

    return out;
}

}  // namespace odai::game
