// Headless playtest harness for the strategic economy. Runs a full multi-empire
// match for N turns with no renderer and prints (a) a turn-by-turn highlight reel
// and (b) a "fun factor" report: lead changes, wonder contention, the reward
// cadence that drives "one more turn", and the sacrifice moments that create
// tension. This is Sid Meier's playtest seat.
//
// Build (headless, no Vulkan):
//   g++ -std=c++20 -I src src/game/strategy_map.cc src/game/economy.cc \
//       src/game/game_sim.cc src/tools/civ_sim_main.cc -o civ_sim
// Usage: civ_sim [turns] [seed] [empires] [--quiet]

#include "game/economy.h"
#include "game/game_sim.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace odai::game;

namespace {

int median(std::vector<int> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::string pad(const std::string& s, std::size_t w) {
    if (s.size() >= w) return s.substr(0, w);
    return s + std::string(w - s.size(), ' ');
}

// Headline metrics for one finished match -- the numbers a designer reads to
// decide whether the match "felt" good.
struct MatchSummary {
    int leadChanges = 0;
    float closeness = 0.0f;      // runner-up score / winner score
    int wondersBuilt = 0;
    int wonderRacesLost = 0;
    int winnerEmpire = 0;        // empire id of the winner
    std::string winnerPersonality;
    int playerRewardEvents = 0;
    float playerCadence = 0.0f;  // turns per reward event for empire 1
    int playerMaxDrought = 0;
    int playerDisorderTurns = 0;
    int playerBrokeTurns = 0;
    int playerFireSales = 0;
    int playerStarves = 0;
};

MatchSummary analyze(const World& world, const std::vector<TurnSample>& samples, int turns) {
    MatchSummary m{};
    std::uint8_t prevLeader = samples.empty() ? 0 : samples.front().leader;
    for (const TurnSample& s : samples) {
        if (s.leader != prevLeader) { ++m.leadChanges; prevLeader = s.leader; }
    }
    std::vector<const Empire*> ranked;
    for (const Empire& e : world.empires) ranked.push_back(&e);
    std::sort(ranked.begin(), ranked.end(), [](const Empire* a, const Empire* b) { return a->score > b->score; });
    if (ranked.size() >= 2 && ranked[0]->score > 0)
        m.closeness = static_cast<float>(ranked[1]->score) / static_cast<float>(ranked[0]->score);
    if (!ranked.empty()) { m.winnerEmpire = ranked[0]->id; m.winnerPersonality = ranked[0]->personality.name; }
    for (const GameEvent& ev : world.events) {
        if (ev.kind == GameEvent::Wonder) ++m.wondersBuilt;
        if (ev.kind == GameEvent::WonderLost) ++m.wonderRacesLost;
    }
    std::vector<int> rewardTurns;
    for (const GameEvent& ev : world.events) {
        if (ev.empire != 1) continue;
        if (ev.kind == GameEvent::Growth || ev.kind == GameEvent::Building || ev.kind == GameEvent::Wonder ||
            ev.kind == GameEvent::Tech || ev.kind == GameEvent::Founded || ev.kind == GameEvent::Unlock)
            rewardTurns.push_back(ev.turn);
        if (ev.kind == GameEvent::FireSale) ++m.playerFireSales;
        if (ev.kind == GameEvent::Starve) ++m.playerStarves;
    }
    std::sort(rewardTurns.begin(), rewardTurns.end());
    int prev = 0;
    for (int rt : rewardTurns) { m.playerMaxDrought = std::max(m.playerMaxDrought, rt - prev); prev = rt; }
    m.playerRewardEvents = static_cast<int>(rewardTurns.size());
    m.playerCadence = rewardTurns.empty() ? 0.0f : static_cast<float>(turns) / rewardTurns.size();
    for (const TurnSample& s : samples) {
        if (!s.disorder.empty() && s.disorder[0] > 0) ++m.playerDisorderTurns;
        if (!s.treasury.empty() && s.treasury[0] < 5) ++m.playerBrokeTurns;
    }
    return m;
}

const char* kindTag(GameEvent::Kind k) {
    switch (k) {
        case GameEvent::Growth:     return "GROW";
        case GameEvent::Building:   return "BLDG";
        case GameEvent::Wonder:     return "WNDR";
        case GameEvent::WonderLost: return "LOST";
        case GameEvent::Tech:       return "TECH";
        case GameEvent::Founded:    return "CITY";
        case GameEvent::FireSale:   return "SELL";
        case GameEvent::Starve:     return "STRV";
        case GameEvent::Disorder:   return "DISO";
        case GameEvent::Conquest:   return "WAR ";
        case GameEvent::Unlock:     return "OPEN";
        case GameEvent::Eureka:     return "EURK";
        case GameEvent::GreatPerson: return "GRTP";
    }
    return "????";
}

}  // namespace

int main(int argc, char** argv) {
    int turns = 300;
    std::uint32_t seed = 1337u;
    int empires = 4;
    bool quiet = false;
    int sweep = 0;
    if (argc > 1) turns = std::max(1, std::atoi(argv[1]));
    if (argc > 2) seed = static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10));
    if (argc > 3) empires = std::clamp(std::atoi(argv[3]), 1, 6);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quiet") quiet = true;
        if (a == "--sweep" && i + 1 < argc) sweep = std::max(1, std::atoi(argv[i + 1]));
    }

    // ---- Sweep mode: run many seeds and report aggregate balance/fun metrics.
    if (sweep > 0) {
        std::cout << "==== SWEEP: " << sweep << " seeds x " << turns << " turns x "
                  << empires << " empires ====\n";
        std::vector<MatchSummary> all;
        std::map<std::string, int> winsByPersonality;
        for (int s = 0; s < sweep; ++s) {
            WorldConfig cfg{};
            cfg.seed = seed + static_cast<std::uint32_t>(s) * 2654435761u;
            cfg.empireCount = empires;
            World w = makeWorld(cfg);
            std::vector<TurnSample> samples;
            for (int t = 0; t < turns; ++t) stepTurn(w, samples);
            MatchSummary m = analyze(w, samples, turns);
            all.push_back(m);
            winsByPersonality[m.winnerPersonality]++;
        }
        auto avg = [&](auto f) {
            double sum = 0;
            for (const MatchSummary& m : all) sum += f(m);
            return sum / static_cast<double>(all.size());
        };
        int nailbiters = 0, runaways = 0;
        for (const MatchSummary& m : all) {
            if (m.closeness > 0.80f) ++nailbiters;
            if (m.closeness < 0.60f) ++runaways;
        }
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "avg lead changes      : " << avg([](const MatchSummary& m){return m.leadChanges;}) << "\n";
        std::cout << "avg final closeness   : " << avg([](const MatchSummary& m){return m.closeness;})
                  << "   (" << nailbiters << " nail-biters >80%, " << runaways << " runaways <60%)\n";
        std::cout << "avg wonders built     : " << avg([](const MatchSummary& m){return m.wondersBuilt;}) << "\n";
        std::cout << "avg wonder races lost : " << avg([](const MatchSummary& m){return m.wonderRacesLost;}) << "\n";
        std::cout << "avg player reward gap : " << avg([](const MatchSummary& m){return m.playerCadence;})
                  << " turns  (avg longest drought " << avg([](const MatchSummary& m){return m.playerMaxDrought;}) << ")\n";
        std::cout << "avg player disorder   : " << avg([](const MatchSummary& m){return m.playerDisorderTurns;}) << " turns\n";
        std::cout << "avg player broke turns: " << avg([](const MatchSummary& m){return m.playerBrokeTurns;})
                  << "  (fire-sales " << avg([](const MatchSummary& m){return m.playerFireSales;})
                  << ", starves " << avg([](const MatchSummary& m){return m.playerStarves;}) << ")\n";
        std::cout << "wins by personality   :";
        for (const auto& kv : winsByPersonality) std::cout << " " << kv.first << "=" << kv.second;
        std::cout << "\n";
        return 0;
    }

    WorldConfig cfg{};
    cfg.seed = seed;
    cfg.empireCount = empires;
    World world = makeWorld(cfg);

    std::cout << "=====================================================================\n";
    std::cout << " ODAI strategic economy -- headless playtest\n";
    std::cout << " map " << world.map.width << "x" << world.map.height
              << "  seed " << seed << "  empires " << world.empires.size()
              << "  turns " << turns << "\n";
    std::cout << "=====================================================================\n";
    std::cout << "Empires:\n";
    for (const Empire& e : world.empires) {
        const City& cap = world.cities[e.cityIndices.front()];
        const std::string label = e.leaderName + " of " + e.name;
        std::cout << "  " << e.id << ". " << pad(label, 20) << " [" << pad(e.personality.name, 20) << "]"
                  << " capital: " << cap.name << " (" << cap.col << "," << cap.row << ")"
                  << "  rel=" << e.personality.religion << " cul=" << e.personality.culture << "\n";
    }
    std::cout << "\n";

    std::vector<TurnSample> samples;
    samples.reserve(static_cast<std::size_t>(turns));
    for (int t = 0; t < turns; ++t) {
        stepTurn(world, samples);
    }

    const std::size_t E = world.empires.size();

    // ---- Highlight reel: wonders, lost races, foundings, fire-sales, starves ----
    if (!quiet) {
        std::cout << "------------------------ HIGHLIGHT REEL -----------------------------\n";
        for (const GameEvent& ev : world.events) {
            const bool notable = ev.kind == GameEvent::Wonder || ev.kind == GameEvent::WonderLost ||
                                 ev.kind == GameEvent::Founded || ev.kind == GameEvent::FireSale ||
                                 ev.kind == GameEvent::Starve || ev.kind == GameEvent::Unlock ||
                                 ev.kind == GameEvent::GreatPerson;
            if (!notable) continue;
            std::cout << "  T" << std::setw(3) << ev.turn << " [" << kindTag(ev.kind) << "] " << ev.text << "\n";
        }
        std::cout << "\n";
    }

    // ---- Score timeline (every ~10% of the game) ----
    std::cout << "------------------------ SCORE TIMELINE -----------------------------\n";
    std::cout << "  turn |";
    for (const Empire& e : world.empires) std::cout << " " << pad(e.name, 9);
    std::cout << " | leader\n";
    const int stride = std::max(1, turns / 15);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const TurnSample& s = samples[i];
        if (s.turn % stride != 0 && i + 1 != samples.size()) continue;
        std::cout << "  " << std::setw(4) << s.turn << " |";
        for (std::size_t e = 0; e < E; ++e) std::cout << " " << std::setw(9) << s.score[e];
        std::cout << " | " << (s.leader >= 1 ? world.empires[s.leader - 1].name : std::string("-")) << "\n";
    }
    std::cout << "\n";

    // ---- Final standings ----
    std::cout << "------------------------ FINAL STANDINGS ----------------------------\n";
    std::vector<const Empire*> ranked;
    for (const Empire& e : world.empires) ranked.push_back(&e);
    std::sort(ranked.begin(), ranked.end(), [](const Empire* a, const Empire* b) { return a->score > b->score; });
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        const Empire* e = ranked[i];
        const std::string label = e->leaderName + " of " + e->name;
        std::cout << "  " << (i + 1) << ". " << pad(label, 20) << " score " << std::setw(5) << e->score
                  << "  pop " << std::setw(3) << e->totalPopulation
                  << "  cities " << std::setw(2) << world.cityCount(e->id)
                  << "  techs " << std::setw(2) << e->researched.size()
                  << "  wonders " << std::setw(2) << e->wonders.size()
                  << "  gold " << std::setw(4) << e->treasury;
        if (!e->wonders.empty()) {
            std::cout << "  [";
            for (std::size_t w = 0; w < e->wonders.size(); ++w)
                std::cout << (w ? "," : "") << e->wonders[w];
            std::cout << "]";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // =================== FUN FACTOR ANALYSIS ===================
    std::cout << "======================= FUN FACTOR ANALYSIS =========================\n";

    // 1. Lead changes -- a runaway leader is boring; swings are exciting.
    int leadChanges = 0;
    std::uint8_t prevLeader = samples.empty() ? 0 : samples.front().leader;
    std::vector<int> leadChangeTurns;
    for (const TurnSample& s : samples) {
        if (s.leader != prevLeader) {
            ++leadChanges;
            leadChangeTurns.push_back(s.turn);
            prevLeader = s.leader;
        }
    }
    std::cout << "Lead changes: " << leadChanges;
    if (!leadChangeTurns.empty()) {
        std::cout << "  (turns:";
        for (std::size_t i = 0; i < leadChangeTurns.size() && i < 12; ++i) std::cout << " " << leadChangeTurns[i];
        if (leadChangeTurns.size() > 12) std::cout << " ...";
        std::cout << ")";
    }
    std::cout << "\n";

    // 2. Closeness at the end (winner vs runner-up).
    if (ranked.size() >= 2 && ranked[0]->score > 0) {
        const float ratio = static_cast<float>(ranked[1]->score) / static_cast<float>(ranked[0]->score);
        std::cout << "Final closeness: runner-up is " << std::fixed << std::setprecision(0) << (ratio * 100.0f)
                  << "% of the winner";
        if (ratio > 0.85f) std::cout << "  -> nail-biter";
        else if (ratio > 0.65f) std::cout << "  -> competitive";
        else std::cout << "  -> runaway";
        std::cout << "\n";
    }

    // 3. Wonders: how many built, and how contested (lost races = real drama).
    int wondersBuilt = 0, wonderRacesLost = 0;
    for (const GameEvent& ev : world.events) {
        if (ev.kind == GameEvent::Wonder) ++wondersBuilt;
        if (ev.kind == GameEvent::WonderLost) ++wonderRacesLost;
    }
    const int totalWonders = static_cast<int>([] {
        int n = 0;
        for (const BuildingDef& d : buildingDefs()) if (d.isWonder) ++n;
        return n;
    }());
    std::cout << "Wonders: " << wondersBuilt << "/" << totalWonders << " built, "
              << wonderRacesLost << " races lost (shields sunk into a wonder a rival finished first)";
    if (wonderRacesLost == 0) std::cout << "  -> NO contention (wonders feel free)";
    else std::cout << "  -> wonders are contested";
    std::cout << "\n";

    // 4. Reward cadence for the player (empire 1): the "one more turn" hook is a
    //    steady drip of completions. Measure the gap between reward-turns.
    {
        const std::uint8_t player = 1;
        std::vector<int> rewardTurns;
        for (const GameEvent& ev : world.events) {
            if (ev.empire != player) continue;
            const bool reward = ev.kind == GameEvent::Growth || ev.kind == GameEvent::Building ||
                                ev.kind == GameEvent::Wonder || ev.kind == GameEvent::Tech ||
                                ev.kind == GameEvent::Founded || ev.kind == GameEvent::Unlock;
            if (reward) rewardTurns.push_back(ev.turn);
        }
        std::sort(rewardTurns.begin(), rewardTurns.end());
        std::vector<int> gaps;
        int maxGap = 0, prev = 0;
        for (int rt : rewardTurns) {
            const int gap = rt - prev;
            gaps.push_back(gap);
            maxGap = std::max(maxGap, gap);
            prev = rt;
        }
        std::cout << "Player (" << world.empires[0].leaderName << " of " << world.empires[0].name << ") reward events: " << rewardTurns.size()
                  << " over " << turns << " turns\n";
        std::cout << "  reward cadence: ~1 every " << std::fixed << std::setprecision(1)
                  << (rewardTurns.empty() ? 0.0 : static_cast<double>(turns) / rewardTurns.size())
                  << " turns (median gap " << median(gaps) << ", longest drought " << maxGap << " turns)";
        if (!rewardTurns.empty() && static_cast<double>(turns) / rewardTurns.size() <= 3.0 && maxGap <= 12)
            std::cout << "  -> strong drip";
        else
            std::cout << "  -> droughts hurt the hook";
        std::cout << "\n";
    }

    // 5. Build variety -> were there real choices, or one obvious line?
    {
        std::map<std::string, int> builtTypes;
        for (const GameEvent& ev : world.events) {
            if (ev.empire != 1) continue;
            if (ev.kind == GameEvent::Building || ev.kind == GameEvent::Wonder) {
                // crude: bucket by the building name appearing in the text
                builtTypes[ev.text]++;
            }
        }
        std::cout << "Player build decisions realized: " << builtTypes.size() << " distinct outcomes\n";
    }

    // 6. Tradeoff bite: disorder + fire-sales + starvation -> the sacrifices.
    {
        int disorderTurns = 0, brokeTurns = 0, minTreasury = 1 << 30;
        for (const TurnSample& s : samples) {
            if (!s.disorder.empty() && s.disorder[0] > 0) ++disorderTurns;
            if (!s.treasury.empty()) {
                if (s.treasury[0] < 5) ++brokeTurns;
                minTreasury = std::min(minTreasury, s.treasury[0]);
            }
        }
        int fireSales = 0, starves = 0;
        for (const GameEvent& ev : world.events) {
            if (ev.empire != 1) continue;
            if (ev.kind == GameEvent::FireSale) ++fireSales;
            if (ev.kind == GameEvent::Starve) ++starves;
        }
        std::cout << "Tradeoff bite (player): " << disorderTurns << " turns with an unhappy city, "
                  << brokeTurns << " turns near-broke (min gold " << minTreasury << "), "
                  << fireSales << " fire-sales, " << starves << " starvations\n";
        if (disorderTurns + fireSales + starves == 0)
            std::cout << "  -> no sacrifices ever forced (economy too forgiving)\n";
        else if (disorderTurns > turns / 2)
            std::cout << "  -> constant pain (economy too punishing)\n";
        else
            std::cout << "  -> meaningful pressure without grinding to a halt\n";
    }

    std::cout << "=====================================================================\n";
    return 0;
}
