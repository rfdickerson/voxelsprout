// Correctness tests for the Council of Houses advisor rule engine. No Vulkan, no
// GTest -- same lightweight harness style as economy_tests.cc. Pure game-state
// input: synthetic AdvisorWorldViews in, attributed Advice out.

#include "game/advisor.h"

#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace odai::game;

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[advisor test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectEqualInt(int actual, int expected, const char* message) {
    if (actual != expected) {
        std::cerr << "[advisor test] FAIL: " << message
                  << " (expected " << expected << ", got " << actual << ")\n";
        ++g_failures;
    }
}

// --- helpers ---------------------------------------------------------------

bool hasKey(const std::vector<Advice>& advice, const std::string& key) {
    for (const Advice& a : advice) {
        if (a.key == key) return true;
    }
    return false;
}

const Advice* find(const std::vector<Advice>& advice, const std::string& key) {
    for (const Advice& a : advice) {
        if (a.key == key) return &a;
    }
    return nullptr;
}

// A "healthy" baseline: one contented mid-size city with a defender, a full
// treasury, and research underway but not near completion. By itself it triggers
// only the gentle "expand_more" Info; each test perturbs exactly one field.
AdvisorWorldView baseView() {
    AdvisorWorldView v;
    v.turn = 5;
    v.leaderName = "Nerevar";
    AdvisorWorldView::City c;
    c.name = "Balmora";
    c.col = 0;
    c.row = 0;
    c.population = 4;
    c.buildings = {"granary", "temple", "library"};
    c.producing = "smithy";
    c.producingName = "Smithy";
    c.turnsToFinish = 5;
    c.happyCap = 8;
    c.inDisorder = false;
    v.playerCities.push_back(c);
    v.units.military = 1;
    v.units.settlers = 0;
    v.units.total = 1;
    v.treasury = 50;
    v.culturePoints = 10;
    v.totalPopulation = 4;
    v.researchTechId = "pottery";
    v.researchName = "Pottery";
    v.researchAccumulated = 0;
    v.researchCost = 22;
    v.sciencePerTurn = 5;
    v.eraName = "Ancient";
    v.eraAdvancedThisTurn = false;
    return v;
}

// --- tests -----------------------------------------------------------------

void testCatalogIntegrity() {
    const auto& cat = advisorCatalog();
    expectEqualInt(static_cast<int>(cat.size()),
                   static_cast<int>(AdvisorDomain::Count),
                   "one advisor per domain");

    std::set<std::string> ids;
    std::set<int> domains;
    for (const Advisor& a : cat) {
        expectTrue(!a.id.empty(), "advisor id non-empty");
        expectTrue(!a.portraitName.empty(), "advisor portraitName non-empty");
        expectTrue(!a.greeting.empty(), "advisor greeting non-empty");
        ids.insert(a.id);
        domains.insert(static_cast<int>(a.domain));
        expectTrue(findAdvisor(a.id) == &a, "findAdvisor round-trips");
        expectTrue(advisorForDomain(a.domain) != nullptr, "advisorForDomain resolves");
    }
    expectEqualInt(static_cast<int>(ids.size()), static_cast<int>(cat.size()),
                   "advisor ids are unique");
    expectEqualInt(static_cast<int>(domains.size()), static_cast<int>(cat.size()),
                   "advisor domains are unique");
    expectTrue(findAdvisor("does_not_exist") == nullptr, "unknown advisor -> nullptr");
}

void testIdleCityFires() {
    AdvisorWorldView v = baseView();
    v.playerCities[0].producing = "";
    expectTrue(hasKey(evaluateAdvisors(v), "idle_city:0,0"), "idle city fires");
    expectTrue(!hasKey(evaluateAdvisors(baseView()), "idle_city:0,0"),
               "busy city does not fire idle advice");
}

void testNoResearchFires() {
    AdvisorWorldView v = baseView();
    v.researchTechId = "";
    v.researchCost = 0;
    const auto advice = evaluateAdvisors(v);
    expectTrue(hasKey(advice, "no_research"), "no research fires");
    const Advice* a = find(advice, "no_research");
    expectTrue(a != nullptr && a->advisorId == "telvanni_magister",
               "no_research attributed to the Magister");
    expectTrue(!hasKey(evaluateAdvisors(baseView()), "no_research"),
               "active research does not fire no_research");
}

void testResearchCompletesNextTurn() {
    AdvisorWorldView v = baseView();
    v.researchAccumulated = 20;  // 20 + 5 = 25 >= 22
    expectTrue(hasKey(evaluateAdvisors(v), "research_done:pottery"),
               "research about to complete fires");

    AdvisorWorldView below = baseView();
    below.researchAccumulated = 5;  // 5 + 5 = 10 < 22
    expectTrue(!hasKey(evaluateAdvisors(below), "research_done:pottery"),
               "research far from done does not fire");
}

void testNoDefenseUrgent() {
    AdvisorWorldView v = baseView();
    v.units.military = 0;
    const auto advice = evaluateAdvisors(v);
    const Advice* a = find(advice, "no_defense");
    expectTrue(a != nullptr, "no defense fires");
    expectTrue(a != nullptr && a->severity == AdviceSeverity::Urgent,
               "no_defense is Urgent");
    expectTrue(a != nullptr && a->advisorId == "redoran_warlord",
               "no_defense attributed to the Warlord");
    expectTrue(!hasKey(evaluateAdvisors(baseView()), "no_defense"),
               "a defended realm does not fire no_defense");
}

void testDisorderUrgent() {
    AdvisorWorldView v = baseView();
    v.playerCities[0].inDisorder = true;
    const auto advice = evaluateAdvisors(v);
    const Advice* a = find(advice, "disorder:0,0");
    expectTrue(a != nullptr, "disorder fires");
    expectTrue(a != nullptr && a->severity == AdviceSeverity::Urgent, "disorder is Urgent");
    expectTrue(a != nullptr && a->advisorId == "temple_almoner",
               "disorder attributed to the Temple");
    expectTrue(!hasKey(evaluateAdvisors(baseView()), "disorder:0,0"),
               "a content city does not fire disorder");
}

void testTreasuryStates() {
    AdvisorWorldView red = baseView();
    red.treasury = -5;
    const auto redAdvice = evaluateAdvisors(red);
    const Advice* empty = find(redAdvice, "treasury_empty");
    expectTrue(empty != nullptr && empty->severity == AdviceSeverity::Urgent,
               "negative treasury fires Urgent treasury_empty");
    expectTrue(!hasKey(redAdvice, "treasury_low"), "negative treasury is not 'low'");

    AdvisorWorldView low = baseView();
    low.treasury = 5;  // 0..kLowTreasury
    const auto lowAdvice = evaluateAdvisors(low);
    expectTrue(hasKey(lowAdvice, "treasury_low"), "low treasury fires treasury_low");
    expectTrue(!hasKey(lowAdvice, "treasury_empty"), "low treasury is not empty");

    const auto healthy = evaluateAdvisors(baseView());  // treasury 50
    expectTrue(!hasKey(healthy, "treasury_low") && !hasKey(healthy, "treasury_empty"),
               "healthy treasury fires no economy warning");
}

void testUnhappyWarn() {
    AdvisorWorldView v = baseView();
    v.playerCities[0].population = 8;  // == happyCap, not yet in disorder
    expectTrue(hasKey(evaluateAdvisors(v), "unhappy:0,0"), "a city at its happy cap warns");
    expectTrue(!hasKey(evaluateAdvisors(baseView()), "unhappy:0,0"),
               "a city below its cap stays quiet");
}

void testGranaryAndSingleCity() {
    AdvisorWorldView v = baseView();
    v.playerCities[0].buildings = {"temple", "library"};  // no granary
    v.turn = 15;                                          // and well into the game
    const auto advice = evaluateAdvisors(v);
    expectTrue(hasKey(advice, "need_granary:0,0"), "missing granary fires");
    expectTrue(hasKey(advice, "single_city"), "lone city late game fires");

    const auto base = evaluateAdvisors(baseView());
    expectTrue(!hasKey(base, "need_granary:0,0"), "city with granary stays quiet");
    expectTrue(!hasKey(base, "single_city"), "early single city stays quiet");
}

void testAqueductGrowth() {
    AdvisorWorldView v = baseView();
    v.playerCities[0].population = 7;  // >= aqueduct gate, no aqueduct built
    expectTrue(hasKey(evaluateAdvisors(v), "need_aqueduct:0,0"),
               "a large city without an aqueduct warns of stalling growth");
    expectTrue(!hasKey(evaluateAdvisors(baseView()), "need_aqueduct:0,0"),
               "a small city does not");
}

void testLibraryAdvice() {
    AdvisorWorldView v = baseView();
    v.canBuildLibrary = true;
    v.playerCities[0].buildings = {"granary", "temple"};  // no library
    expectTrue(hasKey(evaluateAdvisors(v), "need_library:0,0"),
               "an unbuilt-but-unlocked library is advised");

    AdvisorWorldView locked = baseView();
    locked.canBuildLibrary = false;
    locked.playerCities[0].buildings = {"granary", "temple"};
    expectTrue(!hasKey(evaluateAdvisors(locked), "need_library:0,0"),
               "no library advice before it is unlocked");
}

void testEventReactions() {
    AdvisorWorldView lost = baseView();
    lost.recentEvents.push_back(WorldEvent{WorldEventKind::WonderLost,
                                           "Lost the race for the Pyramids"});
    const auto lostAdvice = evaluateAdvisors(lost);
    const Advice* la = find(lostAdvice, "evt_wonder_lost:Lost the race for the Pyramids");
    expectTrue(la != nullptr, "wonder-lost event produces advice");
    expectTrue(la != nullptr && la->advisorId == "redoran_warlord",
               "wonder-lost lament comes from the Warlord");

    AdvisorWorldView built = baseView();
    built.recentEvents.push_back(WorldEvent{WorldEventKind::WonderBuilt,
                                            "The Pyramids are complete"});
    const auto builtAdvice = evaluateAdvisors(built);
    const Advice* ba = find(builtAdvice, "evt_wonder_built:The Pyramids are complete");
    expectTrue(ba != nullptr, "wonder-built event produces advice");
    expectTrue(ba != nullptr && ba->severity == AdviceSeverity::Info,
               "wonder-built congratulation is Info");
}

void testEraAdvance() {
    AdvisorWorldView v = baseView();
    v.eraAdvancedThisTurn = true;
    v.eraName = "Classical";
    const auto advice = evaluateAdvisors(v);
    const Advice* a = find(advice, "era:Classical");
    expectTrue(a != nullptr, "era transition fires");
    expectTrue(a != nullptr && a->advisorId == "telvanni_magister",
               "era transition heralded by the Magister");
    expectTrue(!hasKey(evaluateAdvisors(baseView()), "era:Ancient"),
               "no era transition without the flag");
}

void testSeverityOrdering() {
    AdvisorWorldView v = baseView();
    v.units.military = 0;                 // Urgent (no_defense)
    v.playerCities[0].producing = "";     // Warn   (idle_city)
    v.researchTechId = "";                // Warn   (no_research)
    v.researchCost = 0;
    v.culturePoints = 0;                  // Info   (no_culture)
    v.playerCities[0].buildings = {};     // Info   (need_granary, no_culture)
    const auto advice = evaluateAdvisors(v);

    expectTrue(advice.size() >= 3, "mixed-severity view yields several advice");
    expectTrue(advice.front().severity == AdviceSeverity::Urgent,
               "most-severe advice sorts first");
    bool monotonic = true;
    for (std::size_t i = 1; i < advice.size(); ++i) {
        if (static_cast<int>(advice[i - 1].severity) <
            static_cast<int>(advice[i].severity)) {
            monotonic = false;
        }
    }
    expectTrue(monotonic, "advice is sorted severity-descending");
}

void testEmptyEmpireGreets() {
    AdvisorWorldView v;  // no cities at all
    v.turn = 1;
    const auto advice = evaluateAdvisors(v);
    expectEqualInt(static_cast<int>(advice.size()), 1, "empty empire -> exactly one advice");
    expectTrue(!advice.empty() && advice[0].key == "await_capital",
               "empty empire greeted with await_capital");
    expectTrue(!advice.empty() && advice[0].advisorId == "temple_almoner",
               "founding advice comes from the Temple");
}

void testDeterminism() {
    const AdvisorWorldView v = baseView();
    const auto a = evaluateAdvisors(v);
    const auto b = evaluateAdvisors(v);
    expectEqualInt(static_cast<int>(a.size()), static_cast<int>(b.size()),
                   "determinism: same count");
    bool same = a.size() == b.size();
    for (std::size_t i = 0; same && i < a.size(); ++i) {
        same = a[i].advisorId == b[i].advisorId && a[i].key == b[i].key &&
               a[i].severity == b[i].severity;
    }
    expectTrue(same, "determinism: identical advisor/key/severity sequence");
}

}  // namespace

int main() {
    testCatalogIntegrity();
    testIdleCityFires();
    testNoResearchFires();
    testResearchCompletesNextTurn();
    testNoDefenseUrgent();
    testDisorderUrgent();
    testTreasuryStates();
    testUnhappyWarn();
    testGranaryAndSingleCity();
    testAqueductGrowth();
    testLibraryAdvice();
    testEventReactions();
    testEraAdvance();
    testSeverityOrdering();
    testEmptyEmpireGreets();
    testDeterminism();

    if (g_failures == 0) {
        std::cout << "[advisor test] all checks passed\n";
        return 0;
    }
    std::cerr << "[advisor test] " << g_failures << " check(s) failed\n";
    return 1;
}
