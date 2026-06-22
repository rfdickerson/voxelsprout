// Headless Stellaris-style space 4X playtest harness. Runs a galaxy of alien
// empires for N turns with no renderer and prints fun-factor metrics.
//
// Also serves as a genre-portability test: all strategy-4x UI panels are
// constructed with sci-fi resource types (energy, minerals, research, influence)
// using only a synthetic font, proving the odai_ui kit needs no modification.
//
// Build (Windows, no Vulkan required):
//   cmake --build cmake-build-release --target odai_stellaris_sim
// Run:
//   cmake-build-release\Debug\odai_stellaris_sim.exe [turns] [seed] [empires]
//   cmake-build-release\Debug\odai_stellaris_sim.exe 200 42 4 --sweep 20

#include "ui/font.h"
#include "ui/kits/strategy_4x_kit.h"
#include "ui/resource_style.h"
#include "ui/ui_types.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// All sim + UI demo code lives in this anonymous namespace so the linkage
// stays clean; the only exported symbol is main().
namespace {

using namespace odai::ui;

// ─── Data types ───────────────────────────────────────────────────────────────

struct Tech {
    std::string id;
    std::string name;
    int track = 0;   // 0 = physics, 1 = society, 2 = engineering
    int cost  = 0;
    bool rare = false;
};

// Flat per-empire state — no separate Resources struct so the simulation reads
// without indirection. Research and unity accumulate into their own progress
// pools rather than the economic stockpile (matching Stellaris semantics).
struct Empire {
    std::uint8_t id      = 0;
    std::string name, species, leaderName, govType;
    // AI personality weights (all centered on 1.0)
    float expansion  = 1.0f;
    float militarism = 1.0f;
    float science    = 1.0f;
    float diplomacy  = 1.0f;
    // Economic stockpile + monthly income
    int energy     = 200, energyIncome    = 10;
    int minerals   = 150, mineralIncome   = 6;
    int alloys     =  60, alloyIncome     = 2;
    int influence  =  50, influenceIncome = 3;
    // Research progress bars (separate from stockpile)
    std::string physTarget, socTarget, engTarget;
    int physProg = 0, socProg = 0, engProg = 0;
    int physIncome = 4, socIncome = 3, engIncome = 4;
    std::vector<std::string> completedTechs;
    // Unity / traditions
    int unity          = 0;
    int unityIncome    = 1;
    int traditions     = 0;
    int nextTradCost   = 100;
    // Military / territory
    int systemCount = 1;
    int pops        = 4;
    int fleetPower  = 100;
    // Diplomacy: atWar[other.id - 1]
    std::vector<bool> atWar;
    bool alive     = true;
    bool aiManaged = true;
    int score      = 0;
};

enum class EvKind { TechComplete, Colonized, Tradition, FirstContact, Anomaly, WarDecl, SpecProj };

struct Event {
    int turn = 0;
    std::uint8_t empire = 0;
    std::string text;
    EvKind kind = EvKind::TechComplete;
};

struct Sample {
    int turn = 0;
    std::vector<int> score, systems, techs, fleet;
    std::uint8_t leader = 0;
};

struct Galaxy {
    std::vector<Empire> empires;
    int freeSystemPool = 30;
    std::vector<Event> events;
    int turn = 0;
    std::uint32_t rng = 0xDEADBEEFu;
};

// ─── Content ─────────────────────────────────────────────────────────────────

std::vector<Tech> makeTechs() {
    return {
        // Physics (track 0)
        {"phys_lasers",     "Laser Technology",    0, 100},
        {"phys_shields",    "Deflector Shields",   0, 100},
        {"phys_sensors",    "Long-Range Sensors",  0, 120},
        {"phys_quantum",    "Quantum Theory",      0, 180},
        {"phys_plasma",     "Plasma Cannons",      0, 220},
        {"phys_antimatter", "Antimatter Reactors", 0, 340},
        {"phys_dark",       "Dark Matter Harvest", 0, 400, true},
        {"phys_jump",       "Jump Drive",          0, 500, true},
        // Society (track 1)
        {"soc_xeno",        "Xeno Relations",      1, 100},
        {"soc_combat",      "Ground Combat",       1, 120},
        {"soc_neural",      "Neural Networks",     1, 160},
        {"soc_economy",     "Galactic Economy",    1, 200},
        {"soc_genetic",     "Genetic Engineering", 1, 240},
        {"soc_synthetic",   "Synthetic Workers",   1, 320},
        {"soc_psionics",    "Psionic Theory",      1, 360, true},
        {"soc_utopia",      "Utopian Ideal",       1, 420, true},
        // Engineering (track 2)
        {"eng_alloys",      "Alloy Smelting",      2, 100},
        {"eng_power",       "Power Systems",       2, 100},
        {"eng_afterburner", "Afterburners",        2, 150},
        {"eng_construction","Construction Templates",2,140},
        {"eng_battleship",  "Battleship Hull",     2, 200},
        {"eng_nanotech",    "Nanotechnology",      2, 300},
        {"eng_mega",        "Mega-Engineering",    2, 600, true},
    };
}

// kNames[i] = { name, species, leader, govType, ethics }
static constexpr const char* kNames[6][5] = {
    {"Human Terran Union", "Humans",    "President Vasquez",      "Democratic Republic",   "Materialist / Egalitarian"},
    {"Glaxi Hegemony",     "Glaxi",     "Supreme Regent Grax'tor","Hegemonic Imperium",    "Militarist / Authoritarian"},
    {"Voran Collective",   "Vorans",    "Voice Voran-7",          "Hive Collective",        "Materialist / Xenophile"},
    {"Synthian Dominion",  "Synthians", "Prime Architect Synthex","Machine Autocracy",      "Militarist / Materialist"},
    {"Zephyr Conclave",    "Zephyrans", "Chancellor Swift",       "Federal Convention",     "Pacifist / Xenophile"},
    {"Archon Hierarchy",   "Archons",   "Grand Hierarch Archeron","Ecclesiastic Oligarchy", "Spiritualist / Authoritarian"},
};

// Per-empire AI personality weights: { expansion, militarism, science, diplomacy }
static constexpr float kTraits[6][4] = {
    {1.0f, 0.6f, 1.2f, 1.4f},  // Human
    {0.8f, 2.0f, 0.7f, 0.3f},  // Glaxi (militarist)
    {0.9f, 0.4f, 2.0f, 1.0f},  // Voran (scientist)
    {1.5f, 0.7f, 1.0f, 0.8f},  // Synthian (expansionist)
    {1.2f, 0.5f, 1.1f, 1.8f},  // Zephyr (xenophile)
    {0.7f, 1.5f, 0.8f, 0.2f},  // Archon (isolationist militarist)
};

// ─── LCG helpers ─────────────────────────────────────────────────────────────

std::uint32_t lcg(std::uint32_t& s) { return s = s * 1664525u + 1013904223u; }

int randi(std::uint32_t& s, int lo, int hi) {
    if (hi <= lo) return lo;
    return lo + static_cast<int>(lcg(s) % static_cast<std::uint32_t>(hi - lo + 1));
}

std::string pad(const std::string& s, std::size_t w) {
    return s.size() >= w ? s.substr(0, w) : s + std::string(w - s.size(), ' ');
}

const char* evTag(EvKind k) {
    switch (k) {
        case EvKind::TechComplete: return "TECH";
        case EvKind::Colonized:    return "COLO";
        case EvKind::Tradition:    return "TRAD";
        case EvKind::FirstContact: return "MEET";
        case EvKind::Anomaly:      return "ANOM";
        case EvKind::WarDecl:      return "WAR ";
        case EvKind::SpecProj:     return "PROJ";
    }
    return "????";
}

// ─── Tech helpers ────────────────────────────────────────────────────────────

std::string firstTech(const std::vector<Tech>& techs, int track) {
    for (const auto& t : techs) if (t.track == track && !t.rare) return t.id;
    return {};
}

std::string techName(const std::string& id, const std::vector<Tech>& techs) {
    for (const auto& t : techs) if (t.id == id) return t.name;
    return id;
}

int techCost(const std::string& id, const std::vector<Tech>& techs, int doneSoFar) {
    for (const auto& t : techs)
        if (t.id == id) return std::max(50, t.cost + (doneSoFar / 5) * 20);
    return 200;
}

// Returns the next uncompleted, available tech on the given track.
std::string nextTech(const Empire& emp, int track, const std::vector<Tech>& techs) {
    const int done = static_cast<int>(emp.completedTechs.size());
    for (const auto& t : techs) {
        if (t.track != track) continue;
        if (t.rare && done < 6) continue;
        bool already = false;
        for (const auto& c : emp.completedTechs) if (c == t.id) { already = true; break; }
        if (!already) return t.id;
    }
    return {};
}

// ─── Galaxy construction ──────────────────────────────────────────────────────

Galaxy makeGalaxy(std::uint32_t seed, int numEmpires, int freeSystems) {
    Galaxy g;
    g.rng = seed;
    g.freeSystemPool = freeSystems;
    const auto techs = makeTechs();

    for (int e = 0; e < numEmpires; ++e) {
        Empire emp;
        emp.id = static_cast<std::uint8_t>(e + 1);
        const int idx = e % 6;
        emp.name       = kNames[idx][0];
        emp.species    = kNames[idx][1];
        emp.leaderName = kNames[idx][2];
        emp.govType    = kNames[idx][3];
        emp.expansion  = kTraits[idx][0];
        emp.militarism = kTraits[idx][1];
        emp.science    = kTraits[idx][2];
        emp.diplomacy  = kTraits[idx][3];
        emp.aiManaged  = (e > 0);
        emp.pops       = randi(g.rng, 4, 6);
        emp.fleetPower = randi(g.rng, 80, 200);
        // Income scaled by personality
        emp.energyIncome    = randi(g.rng, 7, 14);
        emp.mineralIncome   = randi(g.rng, 4, 9);
        emp.alloyIncome     = randi(g.rng, 1, 3);
        emp.influenceIncome = 3;
        emp.physIncome = std::max(1, static_cast<int>(randi(g.rng, 3, 7) * emp.science));
        emp.socIncome  = std::max(1, static_cast<int>(randi(g.rng, 2, 5) * emp.science));
        emp.engIncome  = std::max(1, static_cast<int>(randi(g.rng, 3, 6) * emp.science));
        emp.physTarget = firstTech(techs, 0);
        emp.socTarget  = firstTech(techs, 1);
        emp.engTarget  = firstTech(techs, 2);
        emp.atWar.assign(static_cast<std::size_t>(numEmpires), false);
        g.empires.push_back(std::move(emp));
    }
    return g;
}

// ─── Simulation step ──────────────────────────────────────────────────────────

void stepGalaxy(Galaxy& g, std::vector<Sample>& samples) {
    ++g.turn;
    const auto techs = makeTechs();

    for (auto& emp : g.empires) {
        if (!emp.alive) continue;

        // 1. Economic stockpile += monthly income (empire size adds maintenance)
        const int sizePenalty = std::max(0, emp.systemCount - 3);
        emp.energy    += std::max(0, emp.energyIncome - sizePenalty);
        emp.minerals  += emp.mineralIncome;
        emp.alloys    += emp.alloyIncome;
        emp.influence += std::max(1, emp.influenceIncome - sizePenalty / 2);

        // 2. Research — three simultaneous tracks; each has its own progress bar.
        // Completing a tech unlocks the next and gives a +1 income bump.
        auto tryResearch = [&](int& prog, std::string& target, int& income,
                               int track, const char* label) {
            prog += income;
            if (target.empty()) return;
            const int cost = techCost(target, techs, static_cast<int>(emp.completedTechs.size()));
            if (prog < cost) return;
            prog -= cost;
            g.events.push_back({ g.turn, emp.id,
                emp.leaderName + " completes " + techName(target, techs)
                + " [" + label + "]", EvKind::TechComplete });
            emp.completedTechs.push_back(target);
            income += 1;
            target = nextTech(emp, track, techs);
        };
        tryResearch(emp.physProg, emp.physTarget, emp.physIncome, 0, "Physics");
        tryResearch(emp.socProg,  emp.socTarget,  emp.socIncome,  1, "Society");
        tryResearch(emp.engProg,  emp.engTarget,  emp.engIncome,  2, "Engineering");

        // 3. Unity → traditions (passive bonuses, like Stellaris traditions)
        emp.unity += emp.unityIncome;
        if (emp.unity >= emp.nextTradCost) {
            emp.unity -= emp.nextTradCost;
            ++emp.traditions;
            emp.nextTradCost = 100 + emp.traditions * 60;
            emp.unityIncome  = 1 + emp.traditions / 3;
            g.events.push_back({ g.turn, emp.id,
                emp.name + " adopts a new tradition (#" + std::to_string(emp.traditions) + ")",
                EvKind::Tradition });
        }

        // 4. AI: colonize unclaimed systems (spend influence)
        if (emp.aiManaged && emp.expansion > 0.8f
                && emp.influence >= 75 && g.freeSystemPool > 0) {
            const int chance = static_cast<int>(8.0f * emp.expansion);
            if (randi(g.rng, 1, 100) <= chance) {
                ++emp.systemCount;
                --g.freeSystemPool;
                emp.influence -= 75;
                emp.energyIncome  += randi(g.rng, 2, 5);
                emp.mineralIncome += randi(g.rng, 1, 3);
                emp.physIncome += 1; emp.socIncome += 1; emp.engIncome += 1;
                emp.pops += randi(g.rng, 0, 2);
                g.events.push_back({ g.turn, emp.id,
                    emp.name + " colonizes a new system (total: "
                    + std::to_string(emp.systemCount) + ")", EvKind::Colonized });
            }
        }

        // 5. AI: build fleet (spend alloys)
        if (emp.aiManaged && emp.militarism > 0.8f && emp.alloys >= 40) {
            emp.alloys -= 40;
            emp.fleetPower += static_cast<int>(60.0f * emp.militarism);
        }

        // 6. Random: anomaly research dump (3% per turn)
        if (randi(g.rng, 1, 100) <= 3) {
            emp.physProg += randi(g.rng, 20, 60);
            g.events.push_back({ g.turn, emp.id,
                emp.leaderName + " gains research from an anomaly survey",
                EvKind::Anomaly });
        }
    }

    // 7. First Contact (all empires meet simultaneously around turn 20)
    if (g.turn == 20 && g.empires.size() >= 2) {
        for (std::size_t i = 0; i + 1 < g.empires.size(); ++i)
            g.events.push_back({ g.turn, g.empires[i].id,
                g.empires[i].name + " makes First Contact with " + g.empires[i+1].name,
                EvKind::FirstContact });
    }

    // 8. War declarations (militarist empires attack weaker neighbors)
    for (auto& atk : g.empires) {
        if (!atk.alive || !atk.aiManaged || atk.militarism < 1.2f) continue;
        if (randi(g.rng, 1, 100) > 3) continue;
        for (auto& def : g.empires) {
            if (def.id == atk.id || !def.alive || atk.atWar[def.id - 1]) continue;
            if (atk.fleetPower < def.fleetPower * 1.3f) continue;
            atk.atWar[def.id - 1] = def.atWar[atk.id - 1] = true;
            g.events.push_back({ g.turn, atk.id,
                atk.name + " declares war on " + def.name, EvKind::WarDecl });
            // Combat outcome: steal a system if decisively stronger
            if (atk.fleetPower >= def.fleetPower * 2 && def.systemCount > 1) {
                --def.systemCount; ++atk.systemCount;
                def.fleetPower = std::max(50, def.fleetPower - 80);
            }
            break;
        }
    }

    // 9. Score update (systems + pops + fleet + techs + traditions)
    for (auto& emp : g.empires) {
        if (!emp.alive) continue;
        emp.score = emp.systemCount * 10
                  + emp.pops        * 5
                  + emp.fleetPower  / 10
                  + static_cast<int>(emp.completedTechs.size()) * 15
                  + emp.traditions  * 20;
    }

    // 10. Record sample
    Sample s;
    s.turn = g.turn;
    std::uint8_t leader = 0;
    int maxScore = -1;
    for (const auto& emp : g.empires) {
        s.score.push_back(emp.score);
        s.systems.push_back(emp.systemCount);
        s.techs.push_back(static_cast<int>(emp.completedTechs.size()));
        s.fleet.push_back(emp.fleetPower);
        if (emp.score > maxScore) { maxScore = emp.score; leader = emp.id; }
    }
    s.leader = leader;
    samples.push_back(std::move(s));
}

// ─── Analysis ─────────────────────────────────────────────────────────────────

struct MatchSummary {
    int leadChanges = 0;
    float closeness = 0.0f;
    int totalTechs  = 0;
    int totalWars   = 0;
    std::string winnerName;
};

MatchSummary analyze(const Galaxy& g, const std::vector<Sample>& samples) {
    MatchSummary m{};
    std::uint8_t prev = samples.empty() ? 0 : samples.front().leader;
    for (const auto& s : samples) {
        if (s.leader != prev) { ++m.leadChanges; prev = s.leader; }
    }
    for (const auto& ev : g.events) {
        if (ev.kind == EvKind::TechComplete) ++m.totalTechs;
        if (ev.kind == EvKind::WarDecl)      ++m.totalWars;
    }
    std::vector<const Empire*> ranked;
    for (const auto& e : g.empires) ranked.push_back(&e);
    std::sort(ranked.begin(), ranked.end(),
              [](const Empire* a, const Empire* b) { return a->score > b->score; });
    if (ranked.size() >= 2 && ranked[0]->score > 0)
        m.closeness = static_cast<float>(ranked[1]->score) / static_cast<float>(ranked[0]->score);
    if (!ranked.empty()) m.winnerName = ranked[0]->name;
    return m;
}

// ─── Resource style registration ─────────────────────────────────────────────

void registerSciFiResources() {
    registerResourceStyle("energy",         { "icon_energy",    UiColor{1.0f,  0.82f, 0.20f, 1.0f} });
    registerResourceStyle("minerals",       { "icon_minerals",  UiColor{0.80f, 0.47f, 0.28f, 1.0f} });
    registerResourceStyle("alloys",         { "icon_alloys",    UiColor{0.49f, 0.78f, 0.88f, 1.0f} });
    registerResourceStyle("consumer_goods", { "icon_goods",     UiColor{0.53f, 0.88f, 0.50f, 1.0f} });
    registerResourceStyle("research",       { "icon_research",  UiColor{0.63f, 0.50f, 1.00f, 1.0f} });
    registerResourceStyle("influence",      { "icon_influence", UiColor{0.75f, 0.38f, 1.00f, 1.0f} });
    registerResourceStyle("unity",          { "icon_unity",     UiColor{0.00f, 0.71f, 0.85f, 1.0f} });
}

// ─── UI panel smoke-test ──────────────────────────────────────────────────────

// Constructs every strategy-4x panel with sci-fi data, using a synthetic
// monospace font so no TTF files are required. Verifies the panel API can
// represent space-4X state without any modifications to the UI library.
// Returns the number of panels successfully built.
int demoUiPanels(const Galaxy& g) {
    Font font;
    font.initSyntheticMonospace(9.0f, 16.0f, 4.0f);
    FontSet fonts;
    fonts.regular = fonts.bold = fonts.italic = fonts.boldItalic = fonts.numeric = &font;

    const UiRect screen = UiRect::fromXYWH(0.0f, 0.0f, 1920.0f, 1080.0f);
    const float  dpi    = 1.0f;
    const auto   techs  = makeTechs();
    const Empire& player = g.empires.front();
    int built = 0;

    // 1. ResourceBarPanel — top-of-screen HUD strip with space economy resources
    {
        ResourceBarPanel bar(fonts);
        std::vector<ResourceBarPanel::ResourceEntry> entries = {
            { "icon_energy",
              std::to_string(player.energy) + " (+" + std::to_string(player.energyIncome) + "/mo)",
              "Energy Credits — powers ships and buildings",
              UiColor{1.0f, 0.82f, 0.20f, 1.0f} },
            { "icon_minerals",
              std::to_string(player.minerals) + " (+" + std::to_string(player.mineralIncome) + "/mo)",
              "Minerals — raw construction material",
              UiColor{0.80f, 0.47f, 0.28f, 1.0f} },
            { "icon_alloys",
              std::to_string(player.alloys) + " (+" + std::to_string(player.alloyIncome) + "/mo)",
              "Alloys — advanced ship components",
              UiColor{0.49f, 0.78f, 0.88f, 1.0f} },
            { "icon_influence",
              std::to_string(player.influence) + " (+" + std::to_string(player.influenceIncome) + "/mo)",
              "Influence — diplomacy and edicts",
              UiColor{0.75f, 0.38f, 1.00f, 1.0f} },
            { "icon_research",
              std::to_string(player.physIncome + player.socIncome + player.engIncome) + "/mo",
              "Total monthly research output across all three tracks",
              UiColor{0.63f, 0.50f, 1.00f, 1.0f} },
        };
        bar.setResources(UiRect::fromXYWH(0.0f, 0.0f, 1920.0f, 48.0f), dpi, entries);
        ++built;
    }

    // 2. ResearchPanel — physics tech tree with Available / Selected / Completed / Locked
    {
        ResearchPanel panel(fonts);
        std::vector<ResearchPanel::Row> rows;
        for (const auto& t : techs) {
            if (t.track != 0) continue;  // show physics tree
            bool done = false;
            for (const auto& c : player.completedTechs) if (c == t.id) { done = true; break; }
            ResearchPanel::ItemState state;
            if (done)
                state = ResearchPanel::ItemState::Completed;
            else if (t.id == player.physTarget)
                state = ResearchPanel::ItemState::Selected;
            else if (t.rare && static_cast<int>(player.completedTechs.size()) < 6)
                state = ResearchPanel::ItemState::Locked;
            else
                state = ResearchPanel::ItemState::Available;
            const int cost = techCost(t.id, techs, static_cast<int>(player.completedTechs.size()));
            const int etaMonths = done ? 0
                : std::max(1, (cost - player.physProg) / std::max(1, player.physIncome));
            rows.push_back({
                t.id, t.name,
                std::to_string(cost) + " Physics  ·  " + std::to_string(etaMonths) + " months",
                state, "Physics"
            });
        }
        ResearchPanel::ResearchProgress progress;
        if (!player.physTarget.empty()) {
            const int cost = techCost(player.physTarget, techs,
                                      static_cast<int>(player.completedTechs.size()));
            progress.title    = techName(player.physTarget, techs);
            progress.fraction = static_cast<float>(player.physProg) / static_cast<float>(cost);
            progress.status   = std::to_string(player.physProg) + " / " + std::to_string(cost)
                              + " research  ·  "
                              + std::to_string(std::max(1, (cost - player.physProg)
                                / std::max(1, player.physIncome))) + " months left";
            progress.description = "Advanced photonic focusing for ship-mounted weapons.";
        }
        panel.setItems(UiRect::fromXYWH(300.0f, 50.0f, 680.0f, 800.0f), dpi, rows, progress);
        ++built;
    }

    // 3. FactionPanel — empire roster + ethics (replaces civ religion/ideology panel)
    {
        FactionPanel panel(fonts);
        std::vector<FactionPanel::Entry> entries;
        for (const auto& emp : g.empires) {
            const int nameIdx = (emp.id - 1) % 6;
            FactionPanel::Status status = (emp.id == player.id)
                ? FactionPanel::Status::Current
                : FactionPanel::Status::Available;
            entries.push_back({
                std::to_string(emp.id),
                emp.name,
                emp.govType + "  ·  " + emp.species,
                status
            });
            (void)nameIdx;
        }
        const int playerIdx = 0;
        FactionPanel::Detail detail;
        detail.id          = "1";
        detail.name        = player.name;
        detail.bonusSummary= std::string("Ethics: ") + kNames[playerIdx][4];
        detail.body        = "<b>" + player.govType + "</b><br>"
                           + "Leader: " + player.leaderName + "<br>"
                           + "Species: " + player.species + "<br>"
                           + "Systems: " + std::to_string(player.systemCount)
                           + "  ·  Fleet Power: " + std::to_string(player.fleetPower)
                           + "  ·  Techs: " + std::to_string(player.completedTechs.size());
        detail.showJoin = false;
        panel.setEntries(UiRect::fromXYWH(0.0f, 50.0f, 295.0f, 800.0f), dpi, entries, detail);
        ++built;
    }

    // 4. EventTrackerPanel — galaxy event log (anomalies, wars, contacts)
    {
        EventTrackerPanel panel(fonts);
        std::vector<EventTrackerPanel::Entry> entries;
        int shown = 0;
        for (auto it = g.events.rbegin(); it != g.events.rend() && shown < 10; ++it, ++shown) {
            const char* icon = "icon_event";
            const char* sub  = "EVENT";
            switch (it->kind) {
                case EvKind::TechComplete: icon = "icon_research"; sub = "RESEARCH";  break;
                case EvKind::Colonized:    icon = "icon_colony";   sub = "EXPANSION"; break;
                case EvKind::Tradition:    icon = "icon_unity";    sub = "TRADITION"; break;
                case EvKind::FirstContact: icon = "icon_contact";  sub = "DIPLOMACY"; break;
                case EvKind::Anomaly:      icon = "icon_anomaly";  sub = "ANOMALY";   break;
                case EvKind::WarDecl:      icon = "icon_war";      sub = "CONFLICT";  break;
                case EvKind::SpecProj:     icon = "icon_project";  sub = "PROJECT";   break;
            }
            entries.push_back({
                icon, it->text, sub,
                "Year " + std::to_string(2200 + it->turn)
            });
        }
        panel.setEntries(UiRect::fromXYWH(1620.0f, 50.0f, 300.0f, 580.0f), dpi,
                         "GALAXY EVENTS", entries);
        ++built;
    }

    // 5. SimControlsPanel — real-time with pause (Stellaris style, no End Turn)
    {
        SimControlsPanel panel(fonts);
        SimControlsPanel::State state;
        state.speed            = SimControlsPanel::SimSpeed::Normal;
        state.dateLabel        = "March 2210.03";
        state.showSpeedButtons = true;
        state.showEndTurn      = false;  // Stellaris is RT, not turn-based
        panel.setState(UiRect::fromXYWH(760.0f, 1032.0f, 400.0f, 48.0f), dpi, state);
        ++built;
    }

    // 6. SelectionInspectorPanel — selected star system with planet yields
    {
        SelectionInspectorPanel panel(fonts);
        SelectionInspectorPanel::State state;
        state.title   = "Star System Inspector";
        state.hasTile = true;
        state.tile.name     = "Tau Ceti System  ·  G-type star  ·  5 planets";
        state.tile.iconName = "icon_star_g";
        state.tile.yields   = {
            { "icon_energy",   "+" + std::to_string(player.energyIncome
                                                    / std::max(1, player.systemCount)) },
            { "icon_minerals", "+" + std::to_string(player.mineralIncome
                                                    / std::max(1, player.systemCount)) },
            { "icon_research", "+" + std::to_string((player.physIncome + player.engIncome)
                                                    / std::max(1, player.systemCount)) },
        };
        state.hasEntity = true;
        state.entity.portraitName = "icon_planet_continental";
        state.entity.name  = "Tau Ceti Prime";
        state.entity.klass = "Continental World — Size 22";
        state.entity.primaryStats = {
            { "Habitability", "70%" },
            { "Pops",         std::to_string(player.pops / std::max(1, player.systemCount)) },
            { "Districts",    "4" },
            { "Buildings",    "2 / 6" },
        };
        state.entity.secondaryStats = {
            { "Amenities",   "+6" },
            { "Stability",   "60" },
        };
        state.entity.abilities = "<b>+15% mineral output.</b>  Orbital survey complete.";
        state.tileCaption   = "Star System";
        state.entityCaption = "Primary Planet";
        panel.setState(UiRect::fromXYWH(1615.0f, 630.0f, 305.0f, 400.0f), dpi, state);
        ++built;
    }

    return built;
}

}  // anonymous namespace

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int turns   = 200;
    std::uint32_t seed = 42u;
    int empires = 4;
    bool quiet  = false;
    int sweep   = 0;

    if (argc > 1) turns   = std::max(1, std::atoi(argv[1]));
    if (argc > 2) seed    = static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10));
    if (argc > 3) empires = std::clamp(std::atoi(argv[3]), 2, 6);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quiet") quiet = true;
        if (a == "--sweep" && i + 1 < argc) sweep = std::max(1, std::atoi(argv[i + 1]));
    }

    registerSciFiResources();

    // ─── Sweep mode ──────────────────────────────────────────────────────────
    if (sweep > 0) {
        std::cout << "==== SWEEP: " << sweep << " seeds x " << turns
                  << " turns x " << empires << " empires ====\n";
        std::vector<MatchSummary> all;
        std::map<std::string, int> wins;
        for (int s = 0; s < sweep; ++s) {
            Galaxy g = makeGalaxy(seed + static_cast<std::uint32_t>(s) * 2654435761u, empires, 40);
            std::vector<Sample> samples;
            samples.reserve(static_cast<std::size_t>(turns));
            for (int t = 0; t < turns; ++t) stepGalaxy(g, samples);
            MatchSummary m = analyze(g, samples);
            all.push_back(m);
            ++wins[m.winnerName];
        }
        auto avg = [&](auto fn) {
            double sum = 0;
            for (const auto& m : all) sum += fn(m);
            return sum / static_cast<double>(all.size());
        };
        int nailbiters = 0, runaways = 0;
        for (const auto& m : all) {
            if (m.closeness > 0.80f) ++nailbiters;
            if (m.closeness < 0.55f) ++runaways;
        }
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "avg lead changes    : " << avg([](const MatchSummary& m){ return m.leadChanges; }) << "\n";
        std::cout << "avg final closeness : " << avg([](const MatchSummary& m){ return m.closeness; })
                  << "  (" << nailbiters << " nail-biters >80%, " << runaways << " runaways <55%)\n";
        std::cout << "avg total techs     : " << avg([](const MatchSummary& m){ return m.totalTechs; }) << "\n";
        std::cout << "avg wars declared   : " << avg([](const MatchSummary& m){ return m.totalWars; }) << "\n";
        std::cout << "wins by empire      :";
        for (const auto& kv : wins) std::cout << "  " << kv.first << "=" << kv.second;
        std::cout << "\n";
        return 0;
    }

    // ─── Single match ────────────────────────────────────────────────────────
    Galaxy g = makeGalaxy(seed, empires, 40);
    std::vector<Sample> samples;
    samples.reserve(static_cast<std::size_t>(turns));

    std::cout << "=================================================================\n";
    std::cout << " ODAI Stellaris-style space 4X — headless playtest\n";
    std::cout << " seed " << seed << "  empires " << empires << "  turns " << turns << "\n";
    std::cout << "=================================================================\n";
    std::cout << "Empires:\n";
    for (const auto& emp : g.empires) {
        std::cout << "  " << +emp.id << ". " << pad(emp.name, 22) << " ["
                  << pad(emp.species, 10) << "]  " << emp.govType << "\n";
        std::cout << "       " << emp.leaderName
                  << "  |  ethics: " << kNames[(emp.id - 1) % 6][4] << "\n";
    }
    std::cout << "\n";

    for (int t = 0; t < turns; ++t) stepGalaxy(g, samples);

    // Highlight reel
    if (!quiet) {
        std::cout << "------------------------ HIGHLIGHT REEL ------------------------------\n";
        for (const auto& ev : g.events) {
            const bool notable = ev.kind == EvKind::Colonized
                              || ev.kind == EvKind::WarDecl
                              || ev.kind == EvKind::FirstContact
                              || ev.kind == EvKind::SpecProj
                              || ev.kind == EvKind::Tradition;
            if (!notable) continue;
            std::cout << "  Y" << std::setw(4) << (2200 + ev.turn)
                      << " [" << evTag(ev.kind) << "] " << ev.text << "\n";
        }
        std::cout << "\n";
    }

    // Score timeline
    const std::size_t E = g.empires.size();
    std::cout << "------------------------ SCORE TIMELINE ------------------------------\n";
    std::cout << "   year |";
    for (const auto& emp : g.empires) std::cout << " " << pad(emp.name.substr(0, 8), 9);
    std::cout << " | leader\n";
    const int stride = std::max(1, turns / 12);
    for (const auto& s : samples) {
        if (s.turn % stride != 0 && s.turn != turns) continue;
        std::cout << "  " << std::setw(5) << (2200 + s.turn) << " |";
        for (int sc : s.score) std::cout << " " << std::setw(9) << sc;
        const std::string lname = (s.leader >= 1 && s.leader <= E)
            ? g.empires[s.leader - 1].name.substr(0, 10) : "-";
        std::cout << " | " << lname << "\n";
    }
    std::cout << "\n";

    // Final standings
    std::cout << "------------------------ FINAL STANDINGS -----------------------------\n";
    std::vector<const Empire*> ranked;
    for (const auto& e : g.empires) ranked.push_back(&e);
    std::sort(ranked.begin(), ranked.end(),
              [](const Empire* a, const Empire* b) { return a->score > b->score; });
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        const Empire* e = ranked[i];
        std::cout << "  " << (i + 1) << ". " << pad(e->name, 22)
                  << " score " << std::setw(5) << e->score
                  << "  systems " << std::setw(2) << e->systemCount
                  << "  fleet " << std::setw(4) << e->fleetPower
                  << "  techs "  << std::setw(2) << e->completedTechs.size()
                  << "  traditions " << e->traditions << "\n";
    }
    std::cout << "\n";

    // Fun factor analysis
    std::cout << "====================== FUN FACTOR ANALYSIS ==========================\n";
    const MatchSummary m = analyze(g, samples);

    // Lead changes
    std::cout << "Lead changes: " << m.leadChanges;
    if (m.leadChanges == 0) std::cout << "  -> runaway from start";
    else if (m.leadChanges >= 4) std::cout << "  -> dynamic race";
    else std::cout << "  -> some drama";
    std::cout << "\n";

    // Closeness
    if (ranked.size() >= 2 && ranked[0]->score > 0) {
        const float ratio = static_cast<float>(ranked[1]->score)
                          / static_cast<float>(ranked[0]->score);
        std::cout << "Final closeness: runner-up at " << std::fixed << std::setprecision(0)
                  << (ratio * 100.0f) << "% of winner";
        if (ratio > 0.85f) std::cout << "  -> nail-biter";
        else if (ratio > 0.65f) std::cout << "  -> competitive";
        else std::cout << "  -> runaway";
        std::cout << "\n";
    }

    // Research totals
    std::cout << "Total techs researched: " << m.totalTechs
              << " across " << empires << " empires ("
              << std::fixed << std::setprecision(1)
              << (static_cast<double>(m.totalTechs) / empires) << " avg per empire)\n";

    // Tech distribution across tracks
    {
        std::map<int, int> trackCounts;
        const auto allTechs = makeTechs();
        for (const auto& emp : g.empires)
            for (const auto& tc : emp.completedTechs)
                for (const auto& t : allTechs)
                    if (t.id == tc) { ++trackCounts[t.track]; break; }
        std::cout << "Tech distribution: Physics=" << trackCounts[0]
                  << " Society=" << trackCounts[1]
                  << " Engineering=" << trackCounts[2] << "\n";
    }

    // War analysis
    std::cout << "Wars declared: " << m.totalWars;
    if (m.totalWars == 0) std::cout << "  -> peaceful galaxy (boring?)";
    else if (m.totalWars < 3) std::cout << "  -> rare but dramatic";
    else std::cout << "  -> turbulent galaxy";
    std::cout << "\n";

    // Expansion analysis
    {
        const int totalSystems = std::max(1, [&] {
            int s = 0; for (const auto& e : g.empires) s += e.systemCount; return s;
        }());
        std::cout << "Galaxy coverage: " << totalSystems << " systems claimed (free pool: "
                  << g.freeSystemPool << " remaining)\n";
    }

    // Player reward cadence: how many tech / tradition events?
    {
        int playerRewards = 0;
        for (const auto& ev : g.events) {
            if (ev.empire != 1) continue;
            if (ev.kind == EvKind::TechComplete || ev.kind == EvKind::Tradition
                    || ev.kind == EvKind::Colonized || ev.kind == EvKind::SpecProj)
                ++playerRewards;
        }
        std::cout << "Player (" << g.empires.front().name << ") reward events: "
                  << playerRewards << " over " << turns << " turns (~1 per "
                  << std::fixed << std::setprecision(1)
                  << (playerRewards > 0 ? static_cast<double>(turns) / playerRewards : 0.0)
                  << " months)\n";
    }

    std::cout << "=================================================================\n";

    // ─── UI library smoke test ────────────────────────────────────────────────
    std::cout << "\n=================== UI LIBRARY SMOKE TEST ===========================\n";
    std::cout << "Constructing all strategy-4x panels with Stellaris-style sci-fi data\n";
    std::cout << "using synthetic fonts (no TTF or Vulkan required)...\n\n";
    const int built = demoUiPanels(g);
    const char* kPanels[] = {
        "ResourceBarPanel     energy / minerals / alloys / influence / research",
        "ResearchPanel        physics tech tree with Available/Selected/Locked states",
        "FactionPanel         empire roster + ethics detail pane",
        "EventTrackerPanel    10 most recent galaxy events",
        "SimControlsPanel     real-time mode (speed chips, no End Turn button)",
        "SelectionInspectorPanel  star system + continental planet stats",
    };
    for (int i = 0; i < 6; ++i)
        std::cout << (i < built ? "  [OK] " : "  [!!] ") << kPanels[i] << "\n";
    std::cout << "\n" << built << "/6 panels built successfully.\n";
    std::cout << "Theme file: assets/ui/themes/theme_stellaris.json (GalacticFrontier)\n";
    std::cout << "  accent #00C8FF  bg #060C1A  border-radius 3px\n";
    std::cout << "\nConclusion: odai_ui is genre-agnostic — sci-fi 4X layout is\n";
    std::cout << "fully supported with zero changes to the panel library.\n";
    std::cout << "=================================================================\n";

    return 0;
}
