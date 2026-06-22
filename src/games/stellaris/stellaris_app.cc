#include "games/stellaris/stellaris_app.h"

#include "ui/resource_style.h"
#include "ui/widgets/panel.h"
#include "ui/widgets/window.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <sstream>

namespace odai::games::stellaris {

// ---------------------------------------------------------------------------
// Palette
// ---------------------------------------------------------------------------

static constexpr ui::UiColor kSpaceBg   = ui::UiColor{0.008f, 0.031f, 0.063f, 1.0f};
static constexpr ui::UiColor kHyperlane = ui::UiColor{0.22f,  0.42f,  0.68f,  0.65f};
static constexpr ui::UiColor kPanelBg   = ui::UiColor{0.02f,  0.06f,  0.12f,  0.88f};
static constexpr ui::UiColor kPanelBorder = ui::UiColor{0.00f, 0.78f, 1.00f, 0.18f};

static const ui::UiColor kEmpireColors[4] = {
    ui::UiColor::fromRgbHex(0x00C8FF),   // cyan    — player (Human United Earth)
    ui::UiColor::fromRgbHex(0xFF5040),   // red     — Voran Collective
    ui::UiColor::fromRgbHex(0x60E860),   // green   — Elynthian Concord
    ui::UiColor::fromRgbHex(0xFFD050),   // gold    — Keth Remnant
};

static const char* kEmpireNames[4] = {
    "Human United Earth", "Voran Collective", "Elynthian Concord", "Keth Remnant"
};
static const char* kLeaderNames[4] = {
    "Pres. Elena Vasquez", "High-Voice Skarl", "First-Among Aethi", "Elder Ketharan"
};

// Tech trees per track (physics / society / engineering)
static const char* kPhysTechs[] = {
    "Propulsion I", "Propulsion II", "Shields I",
    "Lasers I", "Particle Lance", nullptr
};
static const char* kSocTechs[] = {
    "Colonization I", "Xeno Diplomacy", "Neural Uplinks",
    "Synthetic Pops", "Ascension Theory", nullptr
};
static const char* kEngTechs[] = {
    "Mining I", "Alloy Smelting", "Mega-Engineering I",
    "Orbital Stations", "Ring World Segment", nullptr
};

static const char** kTechTracks[3] = { kPhysTechs, kSocTechs, kEngTechs };
static const int    kTechCosts[]   = { 80, 150, 250, 400, 700 };

// ---------------------------------------------------------------------------
// LCG helpers
// ---------------------------------------------------------------------------

static uint32_t lcgStep(uint32_t& s) { return (s = s * 1664525u + 1013904223u); }
static float    lcgFloat(uint32_t& s) {
    return static_cast<float>(lcgStep(s) >> 8) / static_cast<float>(1 << 24);
}

// ---------------------------------------------------------------------------
// Galaxy initialisation
// ---------------------------------------------------------------------------

void StellarisApp::initGalaxy() {
    m_galaxy = {};

    // 4 empires in 4 quadrants
    for (int e = 0; e < 4; ++e) {
        StelEmpire emp;
        emp.id         = e;
        emp.name       = kEmpireNames[e];
        emp.leaderName = kLeaderNames[e];
        emp.color      = kEmpireColors[e];
        emp.isPlayer   = (e == 0);
        emp.physIncome = 5 + (e == 1 ? 3 : 0);  // Voran gets bonus physics
        emp.socIncome  = 5 + (e == 2 ? 3 : 0);  // Elynthian gets bonus society
        emp.engIncome  = 5 + (e == 3 ? 3 : 0);  // Keth gets bonus engineering
        emp.physTarget = kPhysTechs[0];
        emp.socTarget  = kSocTechs[0];
        emp.engTarget  = kEngTechs[0];
        m_galaxy.empires.push_back(std::move(emp));
    }

    // Star systems: 8 per empire, placed in quadrant clusters
    uint32_t rng = 0xDEADCAFE;
    const float quadX[4] = {0.25f, 0.75f, 0.25f, 0.75f};
    const float quadY[4] = {0.25f, 0.25f, 0.75f, 0.75f};

    static const char* kSystemPrefixes[] = {
        "Sol", "Tau", "Vega", "Keth", "Elun", "Vor", "Oma", "Nyx", "Astra"
    };
    int sysSuffix = 1;

    for (int e = 0; e < 4; ++e) {
        for (int s = 0; s < 8; ++s) {
            StarSystem sys;
            sys.empireId  = e;
            sys.isCapital = (s == 0);
            sys.x = std::clamp(quadX[e] + (lcgFloat(rng) - 0.5f) * 0.38f, 0.06f, 0.94f);
            sys.y = std::clamp(quadY[e] + (lcgFloat(rng) - 0.5f) * 0.38f, 0.06f, 0.94f);
            sys.name = std::string(kSystemPrefixes[lcgStep(rng) % 9]) +
                       " " + std::to_string(sysSuffix++);
            m_galaxy.systems.push_back(std::move(sys));
        }
    }

    m_galaxy.log.push_back("Galaxy initialised. Four empires make contact.");
    m_galaxy.log.push_back(std::string(kLeaderNames[0]) + " surveys the local cluster.");
}

std::string StellarisApp::pickNextTech(const StelEmpire& emp, int track) const {
    const char** list = kTechTracks[track];
    int idx = emp.techsCompleted / 3;  // rough approximation of depth
    for (int i = 0; list[i] != nullptr; ++i) {
        // Skip any tech already in the completed set (simple name check)
        bool done = false;
        // We track techsCompleted count, not names — just pick by depth index
        if (i == std::min(idx + 1, 4)) return list[i];
    }
    return "";
}

// ---------------------------------------------------------------------------
// Simulation step
// ---------------------------------------------------------------------------

void StellarisApp::advanceTurn() {
    m_galaxy.turn += 5;  // each "turn" is 5 years

    for (auto& emp : m_galaxy.empires) {
        // Resource income
        const int mineralIncome  = emp.systemCount * 4;
        const int energyIncome   = emp.systemCount * 3;
        const int alloyIncome    = emp.systemCount;
        emp.minerals  += mineralIncome;
        emp.energy    += energyIncome;
        emp.alloys    += alloyIncome;
        emp.influence += 2;

        // Upkeep
        emp.minerals  = std::max(0, emp.minerals  - emp.systemCount * 2);
        emp.energy    = std::max(0, emp.energy    - emp.systemCount * 1);

        // Research all three tracks
        auto tryResearch = [&](int& prog, std::string& target, int& income,
                               int track, const char* trackLabel) {
            if (target.empty()) return;
            prog += income;
            const int idx  = emp.techsCompleted / 3;
            const int cost = kTechCosts[std::min(idx, 4)];
            if (prog < cost) return;
            prog -= cost;
            emp.techsCompleted++;
            const std::string msg = emp.leaderName + " completes [" + target +
                                    "] (" + trackLabel + ")";
            m_galaxy.log.push_back(msg);
            if (m_galaxy.log.size() > 60) m_galaxy.log.erase(m_galaxy.log.begin());
            income = std::min(income + 2, 25);
            // Pick next tech
            const char** list = kTechTracks[track];
            const int nextIdx = std::min(emp.techsCompleted / 3 + 1, 4);
            target = (list[nextIdx] != nullptr) ? list[nextIdx] : "";
        };

        tryResearch(emp.physProg, emp.physTarget, emp.physIncome, 0, "Physics");
        tryResearch(emp.socProg,  emp.socTarget,  emp.socIncome,  1, "Society");
        tryResearch(emp.engProg,  emp.engTarget,  emp.engIncome,  2, "Engineering");
    }

    m_panelsDirty = true;
}

// ---------------------------------------------------------------------------
// Panel layout helpers
// ---------------------------------------------------------------------------

static void styleGalacticPanel(ui::Panel* p, float s) {
    // Dark navy, slightly transparent — lets star-field bleed through at ~22%
    p->background        = ui::UiColor{0.02f, 0.04f, 0.09f, 0.78f};
    // Outer border: faint teal line
    p->borderColor       = ui::UiColor{0.00f, 0.72f, 1.00f, 0.30f};
    p->borderThicknessPx = 1.0f * s;
    p->cornerRadiusPx    = 2.0f * s;
    // Inner glow line — this is the "emitting" feel; drawn 2px inside the border
    p->innerBorderColor    = ui::UiColor{0.00f, 0.85f, 1.00f, 0.22f};
    p->innerBorderInsetPx  = 2.0f * s;
    // Teal-tinted drop shadow for the floating-panel look
    p->showShadow   = true;
    p->shadowColor  = ui::UiColor{0.00f, 0.40f, 0.70f, 0.18f};
    p->shadowBlurPx = 14.0f * s;
    p->shadowOffsetY = 3.0f * s;
}

void StellarisApp::rebuildPanels(int fbW, int fbH) {
    const float W = static_cast<float>(fbW);
    const float H = static_cast<float>(fbH);
    const float s = m_uiScale;

    static constexpr float kBarH     = 50.0f;
    static constexpr float kLeftW    = 290.0f;
    static constexpr float kRightW   = 310.0f;
    static constexpr float kBotH     = 64.0f;
    static constexpr float kGap      = 4.0f;

    const float barH   = kBarH   * s;
    const float leftW  = kLeftW  * s;
    const float rightW = kRightW * s;
    const float botH   = kBotH   * s;
    const float gap    = kGap    * s;

    // Resource bar — full width along top
    {
        const ui::UiRect rect = ui::UiRect::fromXYWH(0.0f, 0.0f, W, barH);
        const StelEmpire& p = m_galaxy.empires[0];

        using E = ui::ResourceBarPanel::ResourceEntry;
        std::vector<E> entries;

        auto fmt = [](int v) {
            if (v >= 10000) return std::to_string(v / 1000) + "k";
            return std::to_string(v);
        };

        entries.push_back({"energy",   fmt(p.energy),   "Energy credits",
                            ui::UiColor::fromRgbHex(0xFFD050)});
        entries.push_back({"minerals", fmt(p.minerals), "Minerals",
                            ui::UiColor::fromRgbHex(0xC87848)});
        entries.push_back({"alloys",   fmt(p.alloys),   "Alloys",
                            ui::UiColor::fromRgbHex(0x7EC8E0)});
        entries.push_back({"research", fmt(p.physIncome + p.socIncome + p.engIncome) + "/yr",
                            "Research output",
                            ui::UiColor::fromRgbHex(0xA080FF)});
        entries.push_back({"influence", fmt(p.influence),  "Influence",
                            ui::UiColor::fromRgbHex(0xC060FF)});

        m_resourceBar->setResources(rect, s, entries);
        if (auto* bg = m_resourceBar->bgPanel()) styleGalacticPanel(bg, s);
    }

    // Event tracker — left rail
    {
        const float x = gap;
        const float y = barH + gap;
        const float w = leftW - gap * 2.0f;
        const float h = H - barH - botH - gap * 3.0f;
        const ui::UiRect rect = ui::UiRect::fromXYWH(x, y, w, h);

        std::vector<ui::EventTrackerPanel::Entry> entries;
        // Show last 10 events, most recent first
        int count = 0;
        for (int i = static_cast<int>(m_galaxy.log.size()) - 1;
             i >= 0 && count < 10; --i, ++count) {
            ui::EventTrackerPanel::Entry e;
            e.iconName = "star";
            e.title    = std::to_string(m_galaxy.turn - count * 5) + " AD";
            e.description = m_galaxy.log[static_cast<std::size_t>(i)];
            entries.push_back(std::move(e));
        }

        m_eventTracker->setEntries(rect, s, "Galaxy Log", entries);
        if (auto* bg = m_eventTracker->bgPanel()) styleGalacticPanel(bg, s);
    }

    // Research panel — right rail
    {
        const float x = W - rightW + gap;
        const float y = barH + gap;
        const float w = rightW - gap * 2.0f;
        const float h = H - barH - botH - gap * 3.0f;
        const ui::UiRect rect = ui::UiRect::fromXYWH(x, y, w, h);

        const StelEmpire& p = m_galaxy.empires[0];
        std::vector<ui::ResearchPanel::Row> rows;

        auto addTrack = [&](const char** techList, const std::string& current,
                            int prog, int income, const std::string& section) {
            for (int i = 0; techList[i] != nullptr; ++i) {
                ui::ResearchPanel::Row row;
                row.id      = techList[i];
                row.name    = techList[i];
                row.section = section;
                const int cost = kTechCosts[std::min(i, 4)];
                row.info = std::to_string(income) + "/yr · " +
                           std::to_string(cost) + " total";
                if (techList[i] == current) {
                    row.state = ui::ResearchPanel::ItemState::Selected;
                } else if (i <= p.techsCompleted / 3) {
                    row.state = ui::ResearchPanel::ItemState::Completed;
                } else {
                    row.state = ui::ResearchPanel::ItemState::Locked;
                }
                rows.push_back(std::move(row));
            }
        };

        addTrack(kPhysTechs, p.physTarget, p.physProg, p.physIncome, "Physics");
        addTrack(kSocTechs,  p.socTarget,  p.socProg,  p.socIncome,  "Society");
        addTrack(kEngTechs,  p.engTarget,  p.engProg,  p.engIncome,  "Engineering");

        const int depth     = p.techsCompleted / 3;
        const int techCost  = kTechCosts[std::min(depth, 4)];
        const int avgProg   = (p.physProg + p.socProg + p.engProg) / 3;
        const int avgIncome = (p.physIncome + p.socIncome + p.engIncome) / 3;
        const int turns     = (avgIncome > 0) ? (techCost - avgProg) / avgIncome + 1 : 999;

        ui::ResearchPanel::ResearchProgress prog;
        prog.title       = p.physTarget.empty() ? "(none)" : p.physTarget;
        prog.fraction    = (techCost > 0)
                           ? static_cast<float>(p.physProg) / static_cast<float>(techCost)
                           : 1.0f;
        prog.status      = std::to_string(p.physProg) + " / " + std::to_string(techCost) +
                           " · ~" + std::to_string(turns) + " yrs";
        prog.description = "Physics · Society · Engineering advancing in parallel.";

        m_researchPanel->setItems(rect, s, rows, prog);
        if (auto* bg = m_researchPanel->bgPanel()) styleGalacticPanel(bg, s);
    }

    // Sim controls — bottom center
    {
        const float ctrlW = 460.0f * s;
        const float x = (W - ctrlW) * 0.5f;
        const float y = H - botH;
        const ui::UiRect rect = ui::UiRect::fromXYWH(x, y, ctrlW, botH);

        const std::string yearLabel = std::to_string(m_galaxy.turn) + " AD";
        const auto speed = m_galaxy.paused
                           ? ui::SimControlsPanel::SimSpeed::Paused
                           : ui::SimControlsPanel::SimSpeed::Normal;

        ui::SimControlsPanel::State state;
        state.speed            = speed;
        state.dateLabel        = yearLabel;
        state.showSpeedButtons = true;
        state.showEndTurn      = true;
        state.endTurnLabel     = "Advance 5 Years";
        state.endTurnEnabled   = true;
        state.onSpeedChange    = [this](ui::SimControlsPanel::SimSpeed sp) {
            m_galaxy.paused = (sp == ui::SimControlsPanel::SimSpeed::Paused);
            m_simControls->setSpeed(sp);
        };
        state.onEndTurn = [this]() {
            advanceTurn();
        };

        m_simControls->setState(rect, s, state);
        if (auto* bg = m_simControls->bgPanel()) styleGalacticPanel(bg, s);
    }

    m_panelsDirty = false;
}

// ---------------------------------------------------------------------------
// Galaxy-map drawing (into the pre-draw UiDrawList layer)
// ---------------------------------------------------------------------------

void StellarisApp::drawHyperlanes(float mapL, float mapT, float mapW, float mapH) {
    // Nearest-neighbor lane network rendered as dense sub-pixel dots (simulates solid lines).
    const float sc = m_uiScale;
    const int n = static_cast<int>(m_galaxy.systems.size());
    if (n < 2) return;

    // Screen positions
    std::vector<float> sx(n), sy(n);
    for (int i = 0; i < n; ++i) {
        sx[i] = mapL + m_galaxy.systems[i].x * mapW;
        sy[i] = mapT + m_galaxy.systems[i].y * mapH;
    }

    // Each system connects to its 3 nearest neighbours within 38% of map width
    const float kMaxDistSq = (mapW * 0.38f) * (mapW * 0.38f);
    std::set<std::pair<int,int>> lanes;
    for (int i = 0; i < n; ++i) {
        std::vector<std::pair<float,int>> dists;
        dists.reserve(n - 1);
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            float dx = sx[j]-sx[i], dy = sy[j]-sy[i];
            float d2 = dx*dx + dy*dy;
            if (d2 < kMaxDistSq) dists.push_back({d2, j});
        }
        std::sort(dists.begin(), dists.end());
        int links = std::min(3, (int)dists.size());
        for (int k = 0; k < links; ++k) {
            int a = std::min(i, dists[k].second);
            int b = std::max(i, dists[k].second);
            lanes.insert({a, b});
        }
    }

    // Render each lane as a dense run of overlapping dots — looks like a solid line
    for (const auto& [ia, ib] : lanes) {
        float ax = sx[ia], ay = sy[ia], bx = sx[ib], by = sy[ib];
        float dx = bx-ax, dy = by-ay;
        float len = std::sqrt(dx*dx + dy*dy);
        if (len < 2.0f * sc) continue;

        float flowPhase = std::fmod(m_animTime * 0.5f + ia * 0.11f + ib * 0.07f, 1.0f);
        int steps = static_cast<int>(len / (2.2f * sc));
        for (int k = 1; k < steps; ++k) {
            float t = static_cast<float>(k) / static_cast<float>(steps);
            float distFromWave = std::fabs(t - flowPhase);
            if (distFromWave > 0.5f) distFromWave = 1.0f - distFromWave;
            float wave = std::max(0.0f, 1.0f - distFromWave * 8.0f);
            float dotR  = (0.70f + 0.55f * wave) * sc;
            float alpha = std::min(1.0f, kHyperlane.a * (1.0f + 0.5f * wave));
            float px = ax + dx * t, py = ay + dy * t;
            m_uiDrawList.addCircleFilled({px, py}, dotR,
                {kHyperlane.r, kHyperlane.g, kHyperlane.b, alpha});
        }
    }
}

void StellarisApp::drawSystems(float mapL, float mapT, float mapW, float mapH) {
    const float sc = m_uiScale;
    for (const auto& sys : m_galaxy.systems) {
        const float px = mapL + sys.x * mapW;
        const float py = mapT + sys.y * mapH;
        const ui::UiColor& col = (sys.empireId >= 0)
                                  ? kEmpireColors[sys.empireId]
                                  : ui::UiColor{0.5f, 0.5f, 0.5f, 0.8f};

        // Glow ring for capitals — breathes in size and alpha
        if (sys.isCapital) {
            float pulse = 0.55f + 0.45f * std::sin(m_animTime * 1.4f + sys.empireId * 1.1f);
            m_uiDrawList.addCircle({px, py}, (10.0f + 4.0f * pulse) * sc,
                                   ui::UiColor{col.r, col.g, col.b, 0.35f * pulse}, 4.0f * sc);
            // Outer slow halo
            float halo = 0.5f + 0.5f * std::sin(m_animTime * 0.65f + sys.empireId * 0.9f);
            m_uiDrawList.addCircle({px, py}, (18.0f + 5.0f * halo) * sc,
                                   ui::UiColor{col.r, col.g, col.b, 0.10f * halo}, 2.0f * sc);
        }

        // System dot — player-empire outlines shimmer slightly
        const float radius = (sys.isCapital ? 7.0f : 4.5f) * sc;
        m_uiDrawList.addCircleFilled({px, py}, radius, col);
        float outlineAlpha = (sys.empireId == 0)
            ? 0.30f + 0.20f * std::sin(m_animTime * 1.8f + px * 0.008f)
            : 0.45f;
        m_uiDrawList.addCircle({px, py}, radius + 1.5f * sc,
                               ui::UiColor{col.r, col.g, col.b, outlineAlpha}, 1.0f * sc);

        // System name label
        if (!sys.name.empty()) {
            float nameW = m_uiFont.measureText(sys.name);
            float nameY = py + radius + 3.0f * sc;
            // Drop shadow for legibility over hyperlanes
            m_uiDrawList.addText(m_uiFont, sys.name,
                {px - nameW * 0.5f + sc, nameY + sc}, {0.0f,0.0f,0.0f,0.55f});
            m_uiDrawList.addText(m_uiFont, sys.name,
                {px - nameW * 0.5f, nameY},
                sys.isCapital
                    ? ui::UiColor{col.r*0.9f+0.1f, col.g*0.9f+0.1f, col.b*0.9f+0.1f, 0.95f}
                    : ui::UiColor{0.68f, 0.78f, 0.90f, 0.70f});
        }
    }
}

void StellarisApp::drawGalaxyMap(int fbW, int fbH) {
    const float s     = m_uiScale;
    const float W     = static_cast<float>(fbW);
    const float H     = static_cast<float>(fbH);
    const float barH  = 50.0f * s;
    const float leftW = 290.0f * s;
    const float rightW = 310.0f * s;
    const float botH  = 64.0f * s;
    const float gap   = 4.0f * s;

    const float mapL = leftW;
    const float mapT = barH;
    const float mapR = W - rightW;
    const float mapB = H - botH;
    const float mapW = mapR - mapL;
    const float mapH = mapB - mapT;

    // Dark space background
    m_uiDrawList.addRectFilled(
        ui::UiRect::fromXYWH(mapL, mapT, mapW, mapH), kSpaceBg);
    // Subtle V-gradient: deep-blue horizon glow toward the bottom
    m_uiDrawList.addRectFilledVGradient(
        ui::UiRect::fromXYWH(mapL, mapT, mapW, mapH),
        {0.00f, 0.02f, 0.06f, 0.00f},
        {0.00f, 0.04f, 0.14f, 0.22f});

    // Starfield: procedural fixed stars with subtle per-star twinkle
    {
        uint32_t seed = 0xCA5CADEu;
        for (int i = 0; i < 220; ++i) {
            float fx = mapL + lcgFloat(seed) * mapW;
            float fy = mapT + lcgFloat(seed) * mapH;
            float brightness = 0.18f + lcgFloat(seed) * 0.72f;
            float size       = 0.45f + lcgFloat(seed) * 1.05f;
            float twinkle = brightness * (0.82f + 0.18f *
                std::sin(m_animTime * (0.8f + brightness * 2.2f) + fx * 0.009f + fy * 0.007f));
            m_uiDrawList.addCircleFilled({fx, fy}, size * s,
                {twinkle, twinkle, twinkle + 0.06f, twinkle * 0.85f});
        }
    }

    // Primary nebula — breathes slowly
    float nebulaPulse = 0.28f + 0.10f * std::sin(m_animTime * 0.35f);
    m_uiDrawList.addCircleFilled(
        {mapL + mapW * 0.5f, mapT + mapH * 0.5f},
        std::min(mapW, mapH) * 0.35f,
        ui::UiColor{0.05f, 0.08f, 0.20f, nebulaPulse});
    // Secondary offset nebula with a different phase
    float pulse2 = 0.12f + 0.06f * std::sin(m_animTime * 0.55f + 1.8f);
    m_uiDrawList.addCircleFilled(
        {mapL + mapW * 0.42f, mapT + mapH * 0.46f},
        std::min(mapW, mapH) * 0.18f,
        ui::UiColor{0.09f, 0.04f, 0.22f, pulse2});

    // Territory blobs: soft empire-colored regions around claimed systems
    // Three concentric circles per system approximate a radial falloff.
    {
        const float blobR = 58.0f * s;
        for (const auto& sys : m_galaxy.systems) {
            if (sys.empireId < 0) continue;
            const ui::UiColor& col = kEmpireColors[sys.empireId];
            float px = mapL + sys.x * mapW;
            float py = mapT + sys.y * mapH;
            m_uiDrawList.addCircleFilled({px, py}, blobR,        {col.r,col.g,col.b, 0.045f});
            m_uiDrawList.addCircleFilled({px, py}, blobR * 0.6f, {col.r,col.g,col.b, 0.045f});
            m_uiDrawList.addCircleFilled({px, py}, blobR * 0.3f, {col.r,col.g,col.b, 0.045f});
        }
    }

    drawHyperlanes(mapL, mapT, mapW, mapH);
    drawSystems(mapL, mapT, mapW, mapH);

    // Empire legend — bottom-left of map area
    {
        float lx = mapL + 12.0f * s;
        float ly = mapB - 80.0f * s;
        for (int e = 0; e < 4; ++e) {
            m_uiDrawList.addCircleFilled({lx + 6.0f * s, ly + 6.0f * s},
                                         5.0f * s, kEmpireColors[e]);
            ly += 16.0f * s;
        }
    }

    // Panel-edge atmosphere: teal glow halos bleeding from each panel into the map.
    // Drawn last so they sit on top of all map content.
    {
        float glowPulse = 1.0f + 0.06f * std::sin(m_animTime * 0.55f);
        const float fw = 38.0f * s;  // how far the fade extends into the map
        const float fh = 28.0f * s;
        const float ga = 0.20f * glowPulse;
        const ui::UiColor gFull = {0.00f, 0.72f, 1.00f, ga};
        const ui::UiColor gNone = {0.00f, 0.72f, 1.00f, 0.00f};

        // Left panel → right fade
        m_uiDrawList.addRectFilledHGradient(
            ui::UiRect::fromXYWH(mapL, mapT, fw, mapH), gFull, gNone);
        // Right panel → left fade
        m_uiDrawList.addRectFilledHGradient(
            ui::UiRect::fromXYWH(mapR - fw, mapT, fw, mapH), gNone, gFull);
        // Top bar → downward fade
        m_uiDrawList.addRectFilledVGradient(
            ui::UiRect::fromXYWH(mapL, mapT, mapW, fh), gFull, gNone);
        // Bottom bar → upward fade
        m_uiDrawList.addRectFilledVGradient(
            ui::UiRect::fromXYWH(mapL, mapB - fh, mapW, fh), gNone, gFull);

        // Bright hairline precisely at each panel boundary
        const float la = 0.40f * glowPulse;
        const float lw = 1.5f * s;
        const ui::UiColor line = {0.00f, 0.85f, 1.00f, la};
        m_uiDrawList.addRectFilled(ui::UiRect::fromXYWH(mapL,      mapT, lw, mapH), line);
        m_uiDrawList.addRectFilled(ui::UiRect::fromXYWH(mapR - lw, mapT, lw, mapH), line);
        m_uiDrawList.addRectFilled(ui::UiRect::fromXYWH(mapL, mapT,      mapW, lw), line);
        m_uiDrawList.addRectFilled(ui::UiRect::fromXYWH(mapL, mapB - lw, mapW, lw), line);
    }
}

// ---------------------------------------------------------------------------
// GameApp overrides
// ---------------------------------------------------------------------------

bool StellarisApp::onInit() {
    m_uiScale = contentScale();

    // Fonts — resolved relative to project source dir or CWD parents
    if (!loadFonts(
            resolveAssetPath("assets/fonts/Exo2-Regular.ttf"),
            resolveAssetPath("assets/fonts/Inter-Bold.ttf"),   // no static Exo2-Bold; Inter-Bold weights bold correctly
            resolveAssetPath("assets/fonts/Exo2-Italic.ttf"),
            resolveAssetPath("assets/fonts/SpaceMono-Regular.ttf"),
            std::round(16.0f * m_uiScale), std::round(15.0f * m_uiScale))) {
        return false;
    }

    // Register sci-fi resource palette
    using R = ui::ResourceStyle;
    ui::registerResourceStyle("energy",    R{"energy",    ui::UiColor::fromRgbHex(0xFFD050)});
    ui::registerResourceStyle("minerals",  R{"minerals",  ui::UiColor::fromRgbHex(0xC87848)});
    ui::registerResourceStyle("alloys",    R{"alloys",    ui::UiColor::fromRgbHex(0x7EC8E0)});
    ui::registerResourceStyle("research",  R{"research",  ui::UiColor::fromRgbHex(0xA080FF)});
    ui::registerResourceStyle("influence", R{"influence", ui::UiColor::fromRgbHex(0xC060FF)});

    initGalaxy();

    // Camera: orthographic top-down, looking straight down
    m_camera.x           = 0.0f;
    m_camera.y           = 10000.0f;
    m_camera.z           = 0.0f;
    m_camera.yawDegrees  = -90.0f;
    m_camera.pitchDegrees = -89.0f;
    m_camera.fovDegrees  = 60.0f;
    m_camera.orthographic = true;
    m_camera.orthoHalfHeight = 5000.0f;

    // Build the widget tree
    auto root = std::make_unique<ui::Widget>();
    root->mousePassthrough = true;

    // Panels are owned by the widget tree; store raw ptrs for updates.
    m_resourceBar   = static_cast<ui::ResourceBarPanel*>(
        root->addChild(std::make_unique<ui::ResourceBarPanel>(m_uiFonts)));
    m_eventTracker  = static_cast<ui::EventTrackerPanel*>(
        root->addChild(std::make_unique<ui::EventTrackerPanel>(m_uiFonts)));
    m_researchPanel = static_cast<ui::ResearchPanel*>(
        root->addChild(std::make_unique<ui::ResearchPanel>(m_uiFonts)));
    m_simControls   = static_cast<ui::SimControlsPanel*>(
        root->addChild(std::make_unique<ui::SimControlsPanel>(m_uiFonts)));

    m_uiContext.setRoot(std::move(root));
    m_panelsDirty = true;
    return true;
}

void StellarisApp::onTick(float dt) {
    m_animTime += dt;
    if (!m_galaxy.paused) {
        m_galaxy.tickAccum += dt;
        if (m_galaxy.tickAccum >= kTurnIntervalSec) {
            m_galaxy.tickAccum -= kTurnIntervalSec;
            advanceTurn();
        }
    }

    // Escape to close
    if (m_window) {
        // (No GLFW key check needed; window close button works via GLFW)
    }
}

void StellarisApp::onRender(float dt) {
    (void)dt;
    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);
    if (fbW <= 0 || fbH <= 0) return;

    // Rebuild panels if viewport changed or data is dirty
    if (m_panelsDirty || fbW != m_lastFbW || fbH != m_lastFbH) {
        m_lastFbW = fbW;
        m_lastFbH = fbH;

        // Update root rect to match framebuffer
        if (auto* root = m_uiContext.root()) {
            root->setRect(ui::UiRect::fromXYWH(
                0.0f, 0.0f, static_cast<float>(fbW), static_cast<float>(fbH)));
        }
        rebuildPanels(fbW, fbH);
    }

    // Pre-draw: galaxy map background (goes under the UI panels)
    beginFrameDraw();
    drawGalaxyMap(fbW, fbH);

    // Append UI panels on top, then present
    submitFrame(m_camera);
}

} // namespace odai::games::stellaris
