#include "games/citybuilder/citybuilder_app.h"

#include "ui/font.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

namespace odai::games::citybuilder {

using ui::UiColor;
using ui::UiRect;
using ui::UiVec2;
using ui::UiMouseButton;

namespace {

// ── Palette (clean-modern flat HUD direction) ────────────────────────────────
constexpr UiColor kAppBg     = UiColor::fromRgbHex(0x0E1217);
constexpr UiColor kVoid      = UiColor::fromRgbHex(0x0B0F14);
constexpr UiColor kGrass     = UiColor::fromRgbHex(0x5E8C3E);
constexpr UiColor kGrassAlt  = UiColor::fromRgbHex(0x547E37);
constexpr UiColor kWater     = UiColor::fromRgbHex(0x2D5C8C);
constexpr UiColor kWaterAlt  = UiColor::fromRgbHex(0x23507E);
constexpr UiColor kRoad      = UiColor::fromRgbHex(0x393B43);
constexpr UiColor kRoadLine  = UiColor::fromRgbHex(0xCDBA63);
constexpr UiColor kPanel     = UiColor::fromRgbHex(0x161B22, 0.97f);
constexpr UiColor kPanelRise = UiColor::fromRgbHex(0x1E2530, 0.98f);
constexpr UiColor kBtn       = UiColor::fromRgbHex(0x232B36, 0.98f);
constexpr UiColor kBtnHover  = UiColor::fromRgbHex(0x2E3845, 0.98f);
constexpr UiColor kEdge      = UiColor(1.0f, 1.0f, 1.0f, 0.09f);
constexpr UiColor kText      = UiColor::fromRgbHex(0xE7EBF1);
constexpr UiColor kTextDim   = UiColor::fromRgbHex(0x8B94A3);
constexpr UiColor kTextFaint = UiColor::fromRgbHex(0x5A6472);
constexpr UiColor kGood      = UiColor::fromRgbHex(0x4BC56C);
constexpr UiColor kBad       = UiColor::fromRgbHex(0xE0564C);
constexpr UiColor kGold      = UiColor::fromRgbHex(0xF0C24A);
constexpr UiColor kZoneR     = UiColor::fromRgbHex(0x4CB95E);
constexpr UiColor kZoneC     = UiColor::fromRgbHex(0x3B90E0);
constexpr UiColor kZoneI     = UiColor::fromRgbHex(0xE0A22E);
constexpr UiColor kAccent    = UiColor::fromRgbHex(0x4BA0E0);

constexpr float kMonthInterval = 0.55f;  // real seconds per simulated month at 1x
constexpr float kDevEps        = 0.06f;
constexpr int   kHistMax       = 180;

struct ToolMeta {
    const char* tag;
    const char* name;
    const char* key;
    double      cost;
    UiColor     color;
};

// Indexed by CityBuilderApp::Tool, in declaration order.
const ToolMeta kTools[] = {
    {"DEMO", "Bulldoze",    "X", 4.0,    kBad},
    {"R",    "Residential", "1", 25.0,   kZoneR},
    {"C",    "Commercial",  "2", 25.0,   kZoneC},
    {"I",    "Industrial",  "3", 25.0,   kZoneI},
    {"ROAD", "Road",        "R", 12.0,   UiColor::fromRgbHex(0x6B7079)},
    {"POL",  "Police",      "4", 500.0,  UiColor::fromRgbHex(0x2F6BD6)},
    {"FIRE", "Fire Dept",   "5", 500.0,  UiColor::fromRgbHex(0xC0392B)},
    {"CLN",  "Clinic",      "6", 450.0,  UiColor::fromRgbHex(0x21A89A)},
    {"SCH",  "School",      "7", 650.0,  UiColor::fromRgbHex(0xE0852E)},
    {"PRK",  "Park",        "8", 120.0,  UiColor::fromRgbHex(0x35863A)},
    {"PWR",  "Power Plant", "9", 1200.0, UiColor::fromRgbHex(0xC9A227)},
};

const char* kMonths[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

using Tool = CityBuilderApp::Tool;

bool isPaintTool(Tool t) {
    return t == Tool::Bulldoze || t == Tool::Road || t == Tool::ZoneR ||
           t == Tool::ZoneC || t == Tool::ZoneI;
}

Building toolBuilding(Tool t) {
    switch (t) {
        case Tool::Police: return Building::Police;
        case Tool::Fire:   return Building::Fire;
        case Tool::Clinic: return Building::Clinic;
        case Tool::School: return Building::School;
        case Tool::Park:   return Building::Park;
        case Tool::Power:  return Building::Power;
        default:           return Building::None;
    }
}

int footprintOf(Building b) { return b == Building::Park ? 1 : 2; }

UiColor buildingRoof(Building b) {
    switch (b) {
        case Building::Police: return UiColor::fromRgbHex(0x2F6BD6);
        case Building::Fire:   return UiColor::fromRgbHex(0xC0392B);
        case Building::Clinic: return UiColor::fromRgbHex(0x21A89A);
        case Building::School: return UiColor::fromRgbHex(0xE0852E);
        case Building::Park:   return UiColor::fromRgbHex(0x35863A);
        case Building::Power:  return UiColor::fromRgbHex(0xC9A227);
        default:               return kBtn;
    }
}

const char* buildingTag(Building b) {
    switch (b) {
        case Building::Police: return "POL";
        case Building::Fire:   return "FIRE";
        case Building::Clinic: return "CLN";
        case Building::School: return "SCH";
        case Building::Power:  return "PWR";
        default:               return "";
    }
}

float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
float lerpf(float a, float b, float t) { return a + (b - a) * t; }

UiColor mix(const UiColor& a, const UiColor& b, float t) {
    return {lerpf(a.r, b.r, t), lerpf(a.g, b.g, t), lerpf(a.b, b.b, t), lerpf(a.a, b.a, t)};
}
UiColor withA(const UiColor& c, float a) { return {c.r, c.g, c.b, a}; }

std::string commaInt(long long v) {
    const bool neg = v < 0;
    unsigned long long n = neg ? static_cast<unsigned long long>(-(v + 1)) + 1ull
                               : static_cast<unsigned long long>(v);
    std::string digits = std::to_string(n);
    std::string out;
    int count = 0;
    for (int i = static_cast<int>(digits.size()) - 1; i >= 0; --i) {
        out.push_back(digits[static_cast<std::size_t>(i)]);
        if (++count % 3 == 0 && i > 0) out.push_back(',');
    }
    std::reverse(out.begin(), out.end());
    return (neg ? "-" : "") + out;
}

std::string moneyStr(double v) {
    const long long c = static_cast<long long>(std::llround(v));
    return (c < 0 ? "-$" : "$") + commaInt(c < 0 ? -c : c);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────
bool CityBuilderApp::onInit() {
    const float s = contentScale();
    if (!loadFonts(resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
                   resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
                   resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
                   resolveAssetPath("assets/fonts/JetBrainsMono-Regular.ttf"),
                   std::round(15.0f * s), std::round(15.0f * s))) {
        return false;
    }

    auto root = std::make_unique<ui::Widget>();
    root->mousePassthrough = true;
    m_uiContext.setRoot(std::move(root));

    generateTerrain();
    seedCity();
    recomputeStats();   // snap initial city stats to their targets
    pushHistory();      // first sample so the report charts open with data
    return true;
}

void CityBuilderApp::generateTerrain() {
    auto rnd = [&]() -> float {
        m_rng = m_rng * 1664525u + 1013904223u;
        return static_cast<float>((m_rng >> 8) & 0xFFFFu) / 65535.0f;
    };

    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            Tile& t = tile(c, r);
            t = Tile{};
            t.scenicPhase = rnd();
        }
    }

    // A meandering river down the right third of the map.
    for (int r = 0; r < kGridH; ++r) {
        const float cx = kGridW * 0.66f + 5.0f * std::sin(r * 0.26f) + 2.5f * std::sin(r * 0.11f);
        const int half = 2;
        for (int c = static_cast<int>(cx) - half; c <= static_cast<int>(cx) + half; ++c) {
            if (inBounds(c, r)) tile(c, r).terrain = Terrain::Water;
        }
    }

    // A lake in the lower-left.
    const float lakeC = 13.0f, lakeR = 41.0f, lakeRad = 6.0f;
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            const float dx = c - lakeC, dy = r - lakeR;
            if (dx * dx + dy * dy < lakeRad * lakeRad) tile(c, r).terrain = Terrain::Water;
        }
    }
}

void CityBuilderApp::seedCity() {
    auto landRoad = [&](int c, int r) {
        if (inBounds(c, r) && tile(c, r).terrain == Terrain::Grass) {
            Tile& t = tile(c, r);
            t.road = true;
            t.zone = Zone::None;
        }
    };
    auto landZone = [&](int c, int r, Zone z, float dev) {
        if (inBounds(c, r) && tile(c, r).terrain == Terrain::Grass) {
            Tile& t = tile(c, r);
            if (t.road || t.building != Building::None) return;
            t.zone = z;
            t.develop = dev;
        }
    };

    // A simple road grid around the city centre.
    for (int c = 16; c <= 34; ++c) { landRoad(c, 24); landRoad(c, 32); }
    for (int r = 24; r <= 32; ++r) { landRoad(20, r); landRoad(26, r); landRoad(32, r); }

    // Residential to the west, commercial in the middle, industry to the east.
    for (int r = 25; r <= 31; ++r) {
        for (int c = 16; c <= 19; ++c) landZone(c, r, Zone::Residential, 1.2f);
        for (int c = 21; c <= 25; ++c) landZone(c, r, Zone::Commercial, 0.9f);
        for (int c = 27; c <= 31; ++c) landZone(c, r, Zone::Industrial, 0.8f);
    }

    placeBuilding(22, 34, Building::Power);
    placeBuilding(29, 34, Building::School);
    m_money = 50000.0;  // placeBuilding charged; restore the starting grant
}

// ─────────────────────────────────────────────────────────────────────────────
// Simulation
// ─────────────────────────────────────────────────────────────────────────────
void CityBuilderApp::recomputeStats() {
    // Pass 1: clear coverage flags.
    for (Tile& t : m_tiles) { t.powered = false; t.nearRoad = false; }

    m_numRoad = m_numPolice = m_numFire = m_numClinic = m_numSchool = m_numPark = m_numPower = 0;
    float residents = 0.0f, comJobs = 0.0f, indJobs = 0.0f;

    constexpr int kRoadReach   = 3;   // Chebyshev tiles a road services
    constexpr int kPowerRadius = 9;

    // Pass 2: counts, census, and stamp road / power coverage outward.
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            Tile& t = tile(c, r);
            if (t.road) {
                ++m_numRoad;
                for (int dr = -kRoadReach; dr <= kRoadReach; ++dr)
                    for (int dc = -kRoadReach; dc <= kRoadReach; ++dc)
                        if (inBounds(c + dc, r + dr)) tile(c + dc, r + dr).nearRoad = true;
            }
            if (t.bldgOrigin) {
                switch (t.building) {
                    case Building::Police: ++m_numPolice; break;
                    case Building::Fire:   ++m_numFire;   break;
                    case Building::Clinic: ++m_numClinic; break;
                    case Building::School: ++m_numSchool; break;
                    case Building::Park:   ++m_numPark;   break;
                    case Building::Power:  ++m_numPower;  break;
                    default: break;
                }
                if (t.building == Building::Power) {
                    for (int dr = -kPowerRadius; dr <= kPowerRadius; ++dr)
                        for (int dc = -kPowerRadius; dc <= kPowerRadius; ++dc)
                            if (inBounds(c + dc, r + dr)) tile(c + dc, r + dr).powered = true;
                }
            }
            if (t.zone == Zone::Residential) residents += t.develop * 34.0f;
            else if (t.zone == Zone::Commercial) comJobs += t.develop * 30.0f;
            else if (t.zone == Zone::Industrial) indJobs += t.develop * 26.0f;
        }
    }
    const float jobs = comJobs + indJobs;

    // Pass 3: powered coverage of developed land.
    int developed = 0, poweredDeveloped = 0;
    for (const Tile& t : m_tiles) {
        if (t.zone != Zone::None && t.develop > kDevEps) {
            ++developed;
            if (t.powered) ++poweredDeveloped;
        }
    }
    m_powerCoverage = developed > 0 ? static_cast<float>(poweredDeveloped) / developed : 1.0f;

    m_population = static_cast<int>(std::lround(residents));
    m_jobs       = static_cast<int>(std::lround(jobs));

    // Demand: residents want jobs, businesses want customers/workers. A small
    // baseline keeps a fresh city growing.
    m_resDemand = clamp01(0.34f + (jobs - residents) / 1600.0f);
    m_comDemand = clamp01(0.30f + (residents * 0.55f - comJobs) / 1300.0f);
    m_indDemand = clamp01(0.27f + (residents * 0.50f - indJobs) / 1300.0f);

    const float pop = std::max(residents, 1.0f);
    const float eduTarget    = clamp01(m_numSchool * 700.0f / pop) * 100.0f;
    const float healthTarget = clamp01(m_numClinic * 900.0f / pop) * 100.0f;
    const float safety       = clamp01((m_numPolice + m_numFire) * 650.0f / pop);
    const float parkCov      = clamp01(m_numPark * 450.0f / pop);

    m_education = lerpf(m_education, eduTarget, m_statEase);
    m_health    = lerpf(m_health, healthTarget, m_statEase);

    float happyTarget = 46.0f + 0.15f * m_education + 0.15f * m_health +
                        20.0f * parkCov + 12.0f * safety + (m_powerCoverage - 1.0f) * 35.0f - 4.0f;
    happyTarget = std::clamp(happyTarget, 0.0f, 100.0f);
    m_happiness = lerpf(m_happiness, happyTarget, m_statEase);
}

void CityBuilderApp::pushHistory() {
    auto push = [](std::vector<float>& v, float x) {
        v.push_back(x);
        if (static_cast<int>(v.size()) > kHistMax) v.erase(v.begin());
    };
    push(m_histPop, static_cast<float>(m_population));
    push(m_histMoney, static_cast<float>(m_money));
    push(m_histEdu, m_education);
    push(m_histHealth, m_health);
    push(m_histHappy, m_happiness);
}

void CityBuilderApp::stepMonth() {
    // Pre-growth census so growth reacts to this month's demand and coverage.
    m_statEase = 0.0f;
    recomputeStats();

    // Grow / abandon each zoned parcel.
    for (Tile& t : m_tiles) {
        if (t.zone == Zone::None) continue;
        const float dem = t.zone == Zone::Residential ? m_resDemand
                          : t.zone == Zone::Commercial ? m_comDemand
                                                       : m_indDemand;
        if (t.powered && t.nearRoad) {
            const float target = dem * 3.0f;
            t.develop += (target - t.develop) * 0.16f;
        } else {
            t.develop += (0.0f - t.develop) * 0.10f;  // decay toward abandonment
        }
        t.develop = std::clamp(t.develop, 0.0f, 3.0f);
    }

    // Post-growth census, easing the city quality stats toward their targets.
    m_statEase = 0.12f;
    recomputeStats();

    // Monthly budget.
    const double income = m_population * 0.10 + m_jobs * 0.07;
    const double upkeep = m_numRoad * 0.6 + m_numPolice * 45.0 + m_numFire * 45.0 +
                          m_numClinic * 38.0 + m_numSchool * 48.0 + m_numPark * 9.0 +
                          m_numPower * 75.0;
    m_lastNet = income - upkeep;
    m_money += m_lastNet;

    if (++m_month >= 12) { m_month = 0; ++m_year; }
    pushHistory();
}

bool CityBuilderApp::charge(double cost) {
    if (m_money >= cost) { m_money -= cost; return true; }
    flash("Insufficient funds");
    return false;
}

void CityBuilderApp::flash(std::string msg) {
    m_flashMsg = std::move(msg);
    m_flashTimer = 1.8f;
}

void CityBuilderApp::applyTool(int c, int r) {
    if (!inBounds(c, r)) return;
    Tile& t = tile(c, r);
    const double cost = kTools[static_cast<int>(m_tool)].cost;

    switch (m_tool) {
        case Tool::Bulldoze:
            if (t.building != Building::None || t.road || t.zone != Zone::None) {
                if (charge(cost)) bulldoze(c, r);
            }
            break;
        case Tool::ZoneR:
        case Tool::ZoneC:
        case Tool::ZoneI: {
            const Zone z = m_tool == Tool::ZoneR ? Zone::Residential
                           : m_tool == Tool::ZoneC ? Zone::Commercial
                                                    : Zone::Industrial;
            if (t.terrain == Terrain::Water) { flash("Can't zone water"); break; }
            if (t.building != Building::None || t.road) { flash("Bulldoze first"); break; }
            if (t.zone == z) break;  // idempotent — no charge while dragging
            if (charge(cost)) { t.zone = z; t.develop = 0.0f; }
            break;
        }
        case Tool::Road:
            if (t.terrain == Terrain::Water) { flash("Can't pave water"); break; }
            if (t.building != Building::None) { flash("Bulldoze first"); break; }
            if (t.road) break;
            if (charge(cost)) { t.road = true; t.zone = Zone::None; t.develop = 0.0f; }
            break;
        default:
            placeBuilding(c, r, toolBuilding(m_tool));
            break;
    }
}

void CityBuilderApp::bulldoze(int c, int r) {
    Tile& t = tile(c, r);
    if (t.building != Building::None) {
        int oc = t.bOriginC >= 0 ? t.bOriginC : c;
        int orr = t.bOriginR >= 0 ? t.bOriginR : r;
        int fp = inBounds(oc, orr) ? std::max<int>(1, tile(oc, orr).footprint) : 1;
        for (int dy = 0; dy < fp; ++dy) {
            for (int dx = 0; dx < fp; ++dx) {
                if (!inBounds(oc + dx, orr + dy)) continue;
                Tile& cell = tile(oc + dx, orr + dy);
                cell.building = Building::None;
                cell.bldgOrigin = false;
                cell.footprint = 0;
                cell.bOriginC = cell.bOriginR = -1;
                cell.develop = 0.0f;
            }
        }
    } else {
        t.road = false;
        t.zone = Zone::None;
        t.develop = 0.0f;
    }
}

bool CityBuilderApp::placeBuilding(int c, int r, Building b) {
    if (b == Building::None) return false;
    const int fp = footprintOf(b);
    for (int dy = 0; dy < fp; ++dy) {
        for (int dx = 0; dx < fp; ++dx) {
            if (!inBounds(c + dx, r + dy)) { flash("Off the map"); return false; }
            const Tile& cell = tile(c + dx, r + dy);
            if (cell.terrain == Terrain::Water) { flash("Can't build on water"); return false; }
            if (cell.building != Building::None) { flash("Already occupied"); return false; }
        }
    }
    // Cost lookup by matching tool color table is awkward; charge by building type.
    double cost = 0.0;
    switch (b) {
        case Building::Police: cost = 500.0;  break;
        case Building::Fire:   cost = 500.0;  break;
        case Building::Clinic: cost = 450.0;  break;
        case Building::School: cost = 650.0;  break;
        case Building::Park:   cost = 120.0;  break;
        case Building::Power:  cost = 1200.0; break;
        default: break;
    }
    if (!charge(cost)) return false;

    for (int dy = 0; dy < fp; ++dy) {
        for (int dx = 0; dx < fp; ++dx) {
            Tile& cell = tile(c + dx, r + dy);
            cell.zone = Zone::None;
            cell.road = false;
            cell.develop = 0.0f;
            cell.building = b;
            cell.bldgOrigin = false;
            cell.footprint = 0;
            cell.bOriginC = static_cast<short>(c);
            cell.bOriginR = static_cast<short>(r);
        }
    }
    Tile& origin = tile(c, r);
    origin.bldgOrigin = true;
    origin.footprint = static_cast<std::uint8_t>(fp);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Input
// ─────────────────────────────────────────────────────────────────────────────
bool CityBuilderApp::edgeDown(int key) {
    const bool down = glfwGetKey(m_window, key) == GLFW_PRESS;
    const bool prev = m_keyPrev[key];
    m_keyPrev[key] = down;
    return down && !prev;
}

void CityBuilderApp::onTick(float dt) {
    m_time += dt;
    if (m_flashTimer > 0.0f) m_flashTimer -= dt;

    if (edgeDown(GLFW_KEY_X)) m_tool = Tool::Bulldoze;
    if (edgeDown(GLFW_KEY_1)) m_tool = Tool::ZoneR;
    if (edgeDown(GLFW_KEY_2)) m_tool = Tool::ZoneC;
    if (edgeDown(GLFW_KEY_3)) m_tool = Tool::ZoneI;
    if (edgeDown(GLFW_KEY_R)) m_tool = Tool::Road;
    if (edgeDown(GLFW_KEY_4)) m_tool = Tool::Police;
    if (edgeDown(GLFW_KEY_5)) m_tool = Tool::Fire;
    if (edgeDown(GLFW_KEY_6)) m_tool = Tool::Clinic;
    if (edgeDown(GLFW_KEY_7)) m_tool = Tool::School;
    if (edgeDown(GLFW_KEY_8)) m_tool = Tool::Park;
    if (edgeDown(GLFW_KEY_9)) m_tool = Tool::Power;

    if (edgeDown(GLFW_KEY_SPACE)) m_paused = !m_paused;
    if (edgeDown(GLFW_KEY_G)) m_reportsOpen = !m_reportsOpen;

    if (edgeDown(GLFW_KEY_ESCAPE)) {
        if (m_reportsOpen) m_reportsOpen = false;
        else glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }

    if (!m_paused) {
        m_simAccum += dt * static_cast<float>(m_speed);
        int guard = 0;
        while (m_simAccum >= kMonthInterval && guard++ < 8) {
            m_simAccum -= kMonthInterval;
            stepMonth();
        }
    }
}

CityBuilderApp::Layout CityBuilderApp::computeLayout() const {
    Layout lo;
    lo.s = contentScale();
    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);
    lo.fw = static_cast<float>(fbW);
    lo.fh = static_cast<float>(fbH);
    const float s = lo.s;

    const float topH = 54.0f * s;
    const float palW = 196.0f * s;
    lo.topBar  = UiRect::fromXYWH(0.0f, 0.0f, lo.fw, topH);
    lo.palette = UiRect::fromXYWH(0.0f, topH, palW, lo.fh - topH);
    lo.map     = UiRect::fromXYWH(palW, topH, lo.fw - palW, lo.fh - topH);

    const float ctlW = 452.0f * s, ctlH = 46.0f * s;
    lo.controls = UiRect::fromXYWH(lo.map.minX + 14.0f * s, lo.fh - ctlH - 14.0f * s, ctlW, ctlH);

    const float mmS = 190.0f * s;
    lo.minimap = UiRect::fromXYWH(lo.fw - mmS - 14.0f * s, lo.fh - mmS - 14.0f * s, mmS, mmS);

    const float rW = 470.0f * s, rH = 544.0f * s;
    lo.reports = UiRect::fromXYWH(lo.map.minX + (lo.map.width() - rW) * 0.5f,
                                  lo.map.minY + 22.0f * s, rW, rH);
    return lo;
}

void CityBuilderApp::clampCamera(const UiRect& map) {
    const float gw = kGridW * m_tilePx;
    const float gh = kGridH * m_tilePx;
    auto clampAxis = [](float cam, float vMin, float vMax, float gridLen) {
        const float vLen = vMax - vMin;
        if (gridLen <= vLen) return vMin + (vLen - gridLen) * 0.5f;  // smaller than view → centre
        return std::clamp(cam, vMax - gridLen, vMin);
    };
    m_camX = clampAxis(m_camX, map.minX, map.maxX, gw);
    m_camY = clampAxis(m_camY, map.minY, map.maxY, gh);
}

void CityBuilderApp::handleCamera(const Layout& lo) {
    const bool overMap = lo.map.contains(m_uiInput.mousePx) && !m_mouseOverUi;

    if (overMap && m_uiInput.scrollDelta != 0.0f) {
        const float oldT = m_tilePx;
        const float nt = std::clamp(oldT * (1.0f + 0.12f * m_uiInput.scrollDelta),
                                    11.0f * lo.s, 36.0f * lo.s);
        if (nt != oldT) {
            const UiVec2 m = m_uiInput.mousePx;
            m_camX = m.x - (m.x - m_camX) * nt / oldT;
            m_camY = m.y - (m.y - m_camY) * nt / oldT;
            m_tilePx = nt;
        }
    }

    if (overMap && m_uiInput.button(UiMouseButton::Right).down) {
        m_camX += m_uiInput.mouseDeltaPx.x;
        m_camY += m_uiInput.mouseDeltaPx.y;
    }
    clampCamera(lo.map);
}

void CityBuilderApp::handleMapPaint(const Layout& lo) {
    if (m_mouseOverUi || !lo.map.contains(m_uiInput.mousePx) || m_hoverC < 0) return;
    if (isPaintTool(m_tool)) {
        if (m_uiInput.button(UiMouseButton::Left).down) applyTool(m_hoverC, m_hoverR);
    } else {
        if (m_uiInput.button(UiMouseButton::Left).pressed) applyTool(m_hoverC, m_hoverR);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Render
// ─────────────────────────────────────────────────────────────────────────────
void CityBuilderApp::onRender(float /*dt*/) {
    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);
    if (fbW <= 0 || fbH <= 0) return;

    const Layout lo = computeLayout();

    if (!m_camInit) {
        // Cover the viewport (the larger ratio) so the square grid fills the wide
        // map area instead of letterboxing; the camera clamps/pans from there.
        m_tilePx = std::clamp(std::max(lo.map.width() / kGridW, lo.map.height() / kGridH),
                              14.0f * lo.s, 40.0f * lo.s);
        m_camX = lo.map.minX + (lo.map.width() - kGridW * m_tilePx) * 0.5f;
        m_camY = lo.map.minY + (lo.map.height() - kGridH * m_tilePx) * 0.5f;
        m_camInit = true;
    }

    // Hover tile.
    m_hoverC = m_hoverR = -1;
    if (lo.map.contains(m_uiInput.mousePx)) {
        const int c = static_cast<int>(std::floor((m_uiInput.mousePx.x - m_camX) / m_tilePx));
        const int r = static_cast<int>(std::floor((m_uiInput.mousePx.y - m_camY) / m_tilePx));
        if (inBounds(c, r)) { m_hoverC = c; m_hoverR = r; }
    }

    m_mouseOverUi = lo.topBar.contains(m_uiInput.mousePx) ||
                    lo.palette.contains(m_uiInput.mousePx) ||
                    lo.controls.contains(m_uiInput.mousePx) ||
                    lo.minimap.contains(m_uiInput.mousePx) ||
                    (m_reportsOpen && lo.reports.contains(m_uiInput.mousePx));

    handleCamera(lo);
    handleMapPaint(lo);

    beginFrameDraw();
    m_uiDrawList.addRectFilled(UiRect::fromXYWH(0, 0, lo.fw, lo.fh), kAppBg);
    drawMap(lo);
    drawTopBar(lo);
    drawPalette(lo);
    drawControls(lo);
    drawMinimap(lo);
    if (m_reportsOpen) drawReports(lo);
    drawFlash(lo);

    submitFrame(m_camera);
}

void CityBuilderApp::drawMap(const Layout& lo) {
    const float tp = m_tilePx;
    const float s = lo.s;
    m_uiDrawList.pushClip(lo.map);
    m_uiDrawList.addRectFilled(lo.map, kVoid);

    const int c0 = std::max(0, static_cast<int>(std::floor((lo.map.minX - m_camX) / tp)));
    const int c1 = std::min(kGridW - 1, static_cast<int>(std::floor((lo.map.maxX - m_camX) / tp)));
    const int r0 = std::max(0, static_cast<int>(std::floor((lo.map.minY - m_camY) / tp)));
    const int r1 = std::min(kGridH - 1, static_cast<int>(std::floor((lo.map.maxY - m_camY) / tp)));

    for (int r = r0; r <= r1; ++r) {
        for (int c = c0; c <= c1; ++c) {
            const Tile& t = tile(c, r);
            const float x = m_camX + c * tp;
            const float y = m_camY + r * tp;
            const UiRect cell = UiRect::fromXYWH(x, y, tp, tp);
            const bool checker = ((c + r) & 1) != 0;

            if (t.terrain == Terrain::Water) {
                m_uiDrawList.addRectFilled(cell, checker ? kWater : kWaterAlt);
                continue;
            }
            m_uiDrawList.addRectFilled(cell, checker ? kGrass : kGrassAlt);

            if (t.road) {
                const UiRect rd = UiRect::fromXYWH(x + tp * 0.06f, y + tp * 0.06f,
                                                   tp * 0.88f, tp * 0.88f);
                m_uiDrawList.addRectFilled(rd, kRoad);
                m_uiDrawList.addRectFilled(
                    UiRect::fromXYWH(x + tp * 0.46f, y + tp * 0.2f, tp * 0.08f, tp * 0.6f),
                    withA(kRoadLine, 0.5f));
                continue;
            }

            if (t.building != Building::None) {
                if (!t.bldgOrigin) continue;  // drawn by the origin cell
                const int fp = std::max<int>(1, t.footprint);
                const UiRect br = UiRect::fromXYWH(x, y, tp * fp, tp * fp);
                const float pad = tp * 0.12f;
                const UiRect roof = UiRect{br.minX + pad, br.minY + pad, br.maxX - pad, br.maxY - pad};
                const UiColor roofCol = buildingRoof(t.building);
                m_uiDrawList.addRoundRectFilled(roof, mix(roofCol, UiColor(0, 0, 0, 1), 0.15f),
                                                3.0f * s);
                if (t.building == Building::Park) {
                    for (int i = 0; i < 4; ++i) {
                        const float px = roof.minX + roof.width() * (0.28f + 0.45f * (i & 1));
                        const float py = roof.minY + roof.height() * (0.28f + 0.45f * (i >> 1));
                        m_uiDrawList.addCircleFilled({px, py}, tp * 0.12f,
                                                     UiColor::fromRgbHex(0x2A6F2E));
                    }
                } else {
                    m_uiDrawList.addRoundRect(roof, withA(UiColor(1, 1, 1, 1), 0.18f), 3.0f * s, s);
                    if (tp * fp > 26.0f * s) {
                        textCenter(m_uiFontBold, buildingTag(t.building),
                                   (roof.minX + roof.maxX) * 0.5f, (roof.minY + roof.maxY) * 0.5f,
                                   kText);
                    }
                }
                continue;
            }

            if (t.zone != Zone::None) {
                const UiColor zc = t.zone == Zone::Residential ? kZoneR
                                   : t.zone == Zone::Commercial ? kZoneC
                                                                : kZoneI;
                if (t.develop > kDevEps) {
                    const float lvl = clamp01(t.develop / 3.0f);
                    const float pad = tp * (0.18f - 0.08f * lvl);
                    const UiRect b = UiRect{x + pad, y + pad, x + tp - pad, y + tp - pad};
                    m_uiDrawList.addRoundRectFilled(b, mix(zc, UiColor(1, 1, 1, 1), 0.10f * lvl),
                                                    2.0f * s);
                    m_uiDrawList.addRoundRect(b, withA(UiColor(0, 0, 0, 1), 0.25f), 2.0f * s, s);
                    if (!t.powered) {  // brown-out marker
                        m_uiDrawList.addCircleFilled({x + tp * 0.5f, y + tp * 0.5f}, tp * 0.13f,
                                                     withA(kBad, 0.9f));
                    }
                } else {
                    // Zoned but undeveloped: translucent tint + dashed border.
                    m_uiDrawList.addRectFilled(UiRect::fromXYWH(x + 1, y + 1, tp - 2, tp - 2),
                                               withA(zc, 0.22f));
                    m_uiDrawList.addRoundRect(UiRect::fromXYWH(x + tp * 0.16f, y + tp * 0.16f,
                                                               tp * 0.68f, tp * 0.68f),
                                              withA(zc, 0.7f), 1.0f, std::max(1.0f, s));
                }
            }
        }
    }

    // Grid lines when zoomed in.
    if (tp > 15.0f * s) {
        const UiColor gl = withA(UiColor(0, 0, 0, 1), 0.10f);
        for (int c = c0; c <= c1 + 1; ++c) {
            const float x = m_camX + c * tp;
            m_uiDrawList.addRectFilled(UiRect{x, lo.map.minY, x + 1.0f, lo.map.maxY}, gl);
        }
        for (int r = r0; r <= r1 + 1; ++r) {
            const float y = m_camY + r * tp;
            m_uiDrawList.addRectFilled(UiRect{lo.map.minX, y, lo.map.maxX, y + 1.0f}, gl);
        }
    }

    // Hover footprint preview.
    if (!m_mouseOverUi && m_hoverC >= 0) {
        const Building b = toolBuilding(m_tool);
        const int fp = b != Building::None ? footprintOf(b) : 1;
        const float x = m_camX + m_hoverC * tp;
        const float y = m_camY + m_hoverR * tp;
        bool valid = true;
        for (int dy = 0; dy < fp && valid; ++dy)
            for (int dx = 0; dx < fp && valid; ++dx)
                valid = inBounds(m_hoverC + dx, m_hoverR + dy) &&
                        !(tile(m_hoverC + dx, m_hoverR + dy).terrain == Terrain::Water &&
                          m_tool != Tool::Bulldoze);
        const UiColor hc = valid ? kTools[static_cast<int>(m_tool)].color : kBad;
        const UiRect hr = UiRect::fromXYWH(x, y, tp * fp, tp * fp);
        m_uiDrawList.addRectFilled(hr, withA(hc, 0.22f));
        m_uiDrawList.addRoundRect(hr, withA(hc, 0.95f), 2.0f * s, std::max(2.0f, 2.0f * s));
    }

    m_uiDrawList.popClip();
}

void CityBuilderApp::drawTopBar(const Layout& lo) {
    const float s = lo.s;
    m_uiDrawList.addRectFilled(lo.topBar, kPanel);
    m_uiDrawList.addRectFilled(UiRect{lo.topBar.minX, lo.topBar.maxY - 1.0f, lo.topBar.maxX,
                                      lo.topBar.maxY}, kEdge);
    const float cy = lo.topBar.minY + lo.topBar.height() * 0.5f;

    textLeft(m_uiFontBold, "OdaiCity", 16.0f * s, cy - 9.0f * s, kText);
    const std::string date = std::string(kMonths[m_month]) + " · Year " + std::to_string(m_year);
    textLeft(m_uiFont, date, 16.0f * s, cy + 9.0f * s, kTextDim);

    auto chip = [&](float x, std::string_view label, std::string_view value, const UiColor& col) {
        textLeft(m_uiFont, label, x, cy - 9.0f * s, kTextDim);
        textLeft(m_uiFontNumeric, value, x, cy + 9.0f * s, col);
    };

    float x = 168.0f * s;
    const float gap = 138.0f * s;
    chip(x, "Treasury", moneyStr(m_money), kGold);
    {  // monthly net under treasury label, to the right of the value
        const std::string net = (m_lastNet >= 0 ? "+" : "") + moneyStr(m_lastNet) + "/mo";
        textLeft(m_uiFont, net, x, cy + 24.0f * s, m_lastNet >= 0 ? kGood : kBad);
    }
    x += gap;
    chip(x, "Population", commaInt(m_population), kText);
    x += gap * 0.74f;
    chip(x, "Jobs", commaInt(m_jobs), kText);
    x += gap * 0.74f;
    {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d%%", static_cast<int>(std::lround(m_powerCoverage * 100.0f)));
        chip(x, "Power", buf, m_powerCoverage > 0.95f ? kGood : kBad);
    }
    x += gap * 0.6f;
    {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d%%", static_cast<int>(std::lround(m_happiness)));
        const UiColor hc = m_happiness >= 55 ? kGood : (m_happiness >= 35 ? kGold : kBad);
        chip(x, "Approval", buf, hc);
    }

    // RCI demand bars on the far right.
    const float rciX = lo.topBar.maxX - 132.0f * s;
    textLeft(m_uiFont, "Demand", rciX, cy - 12.0f * s, kTextDim);
    const float base = cy + 16.0f * s;
    const float barW = 16.0f * s, barH = 26.0f * s, step = 30.0f * s;
    const float dem[3] = {m_resDemand, m_comDemand, m_indDemand};
    const UiColor demc[3] = {kZoneR, kZoneC, kZoneI};
    const char* deml[3] = {"R", "C", "I"};
    for (int i = 0; i < 3; ++i) {
        const float bx = rciX + i * step;
        m_uiDrawList.addRectFilled(UiRect{bx, base - barH, bx + barW, base}, withA(demc[i], 0.18f));
        const float h = barH * clamp01(dem[i]);
        m_uiDrawList.addRectFilled(UiRect{bx, base - h, bx + barW, base}, demc[i]);
        textCenter(m_uiFont, deml[i], bx + barW * 0.5f, base + 9.0f * s, kTextDim);
    }
}

void CityBuilderApp::drawPalette(const Layout& lo) {
    const float s = lo.s;
    m_uiDrawList.addRectFilled(lo.palette, kPanel);
    m_uiDrawList.addRectFilled(UiRect{lo.palette.maxX - 1.0f, lo.palette.minY, lo.palette.maxX,
                                      lo.palette.maxY}, kEdge);

    const float padX = 10.0f * s;
    float y = lo.palette.minY + 12.0f * s;
    const float btnH = 40.0f * s;
    const float btnW = lo.palette.width() - padX * 2.0f;
    const float gap = 6.0f * s;

    auto header = [&](const char* label) {
        textLeft(m_uiFontBold, label, lo.palette.minX + padX, y + 8.0f * s, kTextFaint);
        y += 20.0f * s;
    };

    auto toolRow = [&](Tool tool) {
        const ToolMeta& m = kTools[static_cast<int>(tool)];
        const UiRect r = UiRect::fromXYWH(lo.palette.minX + padX, y, btnW, btnH);
        const bool hover = r.contains(m_uiInput.mousePx);
        const bool active = m_tool == tool;
        const UiColor bg = active ? mix(kBtn, m.color, 0.32f) : (hover ? kBtnHover : kBtn);
        m_uiDrawList.addRoundRectFilled(r, bg, 7.0f * s);
        m_uiDrawList.addRoundRect(r, active ? withA(m.color, 0.95f) : kEdge, 7.0f * s,
                                  active ? 1.6f * s : s);
        // colour swatch with tag
        const UiRect sw = UiRect::fromXYWH(r.minX + 7.0f * s, r.minY + 6.0f * s,
                                           btnH - 12.0f * s, btnH - 12.0f * s);
        m_uiDrawList.addRoundRectFilled(sw, m.color, 5.0f * s);
        textCenter(m_uiFontBold, m.tag, (sw.minX + sw.maxX) * 0.5f, (sw.minY + sw.maxY) * 0.5f,
                   UiColor::fromRgbHex(0x0E1217));
        // name + cost
        const float tx = sw.maxX + 9.0f * s;
        textLeft(m_uiFontBold, m.name, tx, r.minY + btnH * 0.36f, active ? kText : kText);
        char cost[24];
        std::snprintf(cost, sizeof(cost), "$%d", static_cast<int>(m.cost));
        textLeft(m_uiFont, cost, tx, r.minY + btnH * 0.68f, kTextDim);
        // hotkey chip
        const UiRect key = UiRect::fromXYWH(r.maxX - 22.0f * s, r.minY + 6.0f * s, 16.0f * s,
                                            16.0f * s);
        m_uiDrawList.addRoundRectFilled(key, withA(UiColor(1, 1, 1, 1), 0.08f), 3.0f * s);
        textCenter(m_uiFont, m.key, (key.minX + key.maxX) * 0.5f, (key.minY + key.maxY) * 0.5f,
                   kTextDim);

        if (hover && m_uiInput.button(UiMouseButton::Left).pressed) m_tool = tool;
        y += btnH + gap;
    };

    toolRow(Tool::Bulldoze);
    y += 4.0f * s;
    header("ZONES");
    toolRow(Tool::ZoneR);
    toolRow(Tool::ZoneC);
    toolRow(Tool::ZoneI);
    header("NETWORK");
    toolRow(Tool::Road);
    header("SERVICES");
    toolRow(Tool::Police);
    toolRow(Tool::Fire);
    toolRow(Tool::Clinic);
    toolRow(Tool::School);
    toolRow(Tool::Park);
    toolRow(Tool::Power);
}

void CityBuilderApp::drawControls(const Layout& lo) {
    const float s = lo.s;
    const UiRect& r = lo.controls;
    m_uiDrawList.addDropShadow(r, withA(UiColor(0, 0, 0, 1), 0.4f), 8.0f * s, 0.0f, 3.0f * s);
    m_uiDrawList.addRoundRectFilled(r, kPanelRise, 9.0f * s);
    m_uiDrawList.addRoundRect(r, kEdge, 9.0f * s, s);

    const float pad = 8.0f * s;
    const float by = r.minY + pad;
    const float bh = r.height() - pad * 2.0f;
    float x = r.minX + pad;

    if (uiButton(UiRect::fromXYWH(x, by, 80.0f * s, bh), m_paused ? "Play" : "Pause",
                 !m_paused, kAccent)) {
        m_paused = !m_paused;
    }
    x += 86.0f * s;

    for (int sp = 1; sp <= 3; ++sp) {
        char lbl[4];
        std::snprintf(lbl, sizeof(lbl), "%dx", sp);
        if (uiButton(UiRect::fromXYWH(x, by, 36.0f * s, bh), lbl,
                     !m_paused && m_speed == sp, kAccent)) {
            m_speed = sp;
            m_paused = false;
        }
        x += 40.0f * s;
    }

    x += 8.0f * s;
    const std::string date = std::string(kMonths[m_month]) + " Yr " + std::to_string(m_year);
    textLeft(m_uiFontBold, date, x, r.minY + r.height() * 0.5f, kText);
    x += 96.0f * s;

    if (uiButton(UiRect::fromXYWH(r.maxX - 96.0f * s, by, 88.0f * s, bh), "Reports",
                 m_reportsOpen, kGold)) {
        m_reportsOpen = !m_reportsOpen;
    }
}

void CityBuilderApp::drawMinimap(const Layout& lo) {
    const float s = lo.s;
    const UiRect& r = lo.minimap;
    m_uiDrawList.addDropShadow(r, withA(UiColor(0, 0, 0, 1), 0.4f), 8.0f * s, 0.0f, 3.0f * s);
    m_uiDrawList.addRoundRectFilled(r, kPanelRise, 9.0f * s);
    m_uiDrawList.addRoundRect(r, kEdge, 9.0f * s, s);

    const float titleH = 22.0f * s;
    textLeft(m_uiFontBold, "City Map", r.minX + 10.0f * s, r.minY + titleH * 0.5f + 2.0f * s,
             kTextDim);

    const float pad = 8.0f * s;
    const float availW = r.width() - pad * 2.0f;
    const float availH = r.height() - titleH - pad * 2.0f;
    const int maxG = std::max(kGridW, kGridH);
    const float cell = std::min(availW, availH) / maxG;
    const float mapW = kGridW * cell, mapH = kGridH * cell;
    const float ox = r.minX + pad + (availW - mapW) * 0.5f;
    const float oy = r.minY + titleH + pad + (availH - mapH) * 0.5f;

    m_uiDrawList.pushClip(UiRect::fromXYWH(ox, oy, mapW, mapH));
    for (int gr = 0; gr < kGridH; ++gr) {
        for (int gc = 0; gc < kGridW; ++gc) {
            const Tile& t = tile(gc, gr);
            UiColor col;
            if (t.terrain == Terrain::Water) col = kWater;
            else if (t.building != Building::None) col = buildingRoof(t.building);
            else if (t.road) col = UiColor::fromRgbHex(0x55595F);
            else if (t.zone == Zone::Residential) col = withA(kZoneR, t.develop > kDevEps ? 1.0f : 0.4f);
            else if (t.zone == Zone::Commercial) col = withA(kZoneC, t.develop > kDevEps ? 1.0f : 0.4f);
            else if (t.zone == Zone::Industrial) col = withA(kZoneI, t.develop > kDevEps ? 1.0f : 0.4f);
            else col = UiColor::fromRgbHex(0x35471F);
            m_uiDrawList.addRectFilled(UiRect::fromXYWH(ox + gc * cell, oy + gr * cell,
                                                        cell + 0.5f, cell + 0.5f), col);
        }
    }

    // Camera viewport rectangle.
    const float vx0 = ox + std::clamp((lo.map.minX - m_camX) / m_tilePx, 0.0f, float(kGridW)) * cell;
    const float vy0 = oy + std::clamp((lo.map.minY - m_camY) / m_tilePx, 0.0f, float(kGridH)) * cell;
    const float vx1 = ox + std::clamp((lo.map.maxX - m_camX) / m_tilePx, 0.0f, float(kGridW)) * cell;
    const float vy1 = oy + std::clamp((lo.map.maxY - m_camY) / m_tilePx, 0.0f, float(kGridH)) * cell;
    m_uiDrawList.addRect(UiRect{vx0, vy0, vx1, vy1}, withA(UiColor(1, 1, 1, 1), 0.85f),
                         std::max(1.0f, 1.5f * s));
    m_uiDrawList.popClip();

    // Click to recentre the camera.
    const UiRect inner = UiRect::fromXYWH(ox, oy, mapW, mapH);
    if (inner.contains(m_uiInput.mousePx) && m_uiInput.button(UiMouseButton::Left).down) {
        const float nx = (m_uiInput.mousePx.x - ox) / mapW;
        const float ny = (m_uiInput.mousePx.y - oy) / mapH;
        m_camX = lo.map.minX + lo.map.width() * 0.5f - nx * kGridW * m_tilePx;
        m_camY = lo.map.minY + lo.map.height() * 0.5f - ny * kGridH * m_tilePx;
        clampCamera(lo.map);
    }
}

const std::vector<float>& CityBuilderApp::history(Metric m) const {
    switch (m) {
        case Metric::Population: return m_histPop;
        case Metric::Treasury:   return m_histMoney;
        case Metric::Education:  return m_histEdu;
        case Metric::Health:     return m_histHealth;
        default:                 return m_histHappy;
    }
}

void CityBuilderApp::drawReports(const Layout& lo) {
    const float s = lo.s;
    const UiRect& r = lo.reports;
    m_uiDrawList.addDropShadow(r, withA(UiColor(0, 0, 0, 1), 0.5f), 14.0f * s, 0.0f, 6.0f * s);
    m_uiDrawList.addRoundRectFilled(r, kPanel, 10.0f * s);
    m_uiDrawList.addRoundRect(r, kEdge, 10.0f * s, s);

    // Title bar.
    const float titleH = 38.0f * s;
    m_uiDrawList.addRoundRectFilled(UiRect{r.minX, r.minY, r.maxX, r.minY + titleH}, kPanelRise,
                                    10.0f * s);
    textLeft(m_uiFontBold, "City Reports", r.minX + 14.0f * s, r.minY + titleH * 0.5f, kText);
    if (uiButton(UiRect::fromXYWH(r.maxX - 30.0f * s, r.minY + 7.0f * s, 24.0f * s, 24.0f * s),
                 "X", false, kBad)) {
        m_reportsOpen = false;
    }

    // Metric tabs.
    const char* names[5] = {"Population", "Treasury", "Education", "Health", "Approval"};
    const UiColor cols[5] = {kAccent, kGold, UiColor::fromRgbHex(0x6FA0F0),
                             UiColor::fromRgbHex(0xE06A8A), kGood};
    const float tabY = r.minY + titleH + 8.0f * s;
    const float tabW = (r.width() - 16.0f * s - 4.0f * 6.0f * s) / 5.0f;
    for (int i = 0; i < 5; ++i) {
        const UiRect tr = UiRect::fromXYWH(r.minX + 8.0f * s + i * (tabW + 6.0f * s), tabY, tabW,
                                           26.0f * s);
        if (uiButton(tr, names[i], static_cast<int>(m_metric) == i, cols[i])) {
            m_metric = static_cast<Metric>(i);
        }
    }

    const UiColor mc = cols[static_cast<int>(m_metric)];
    const std::vector<float>& h = history(m_metric);

    // Current value readout.
    const float valY = tabY + 26.0f * s + 26.0f * s;
    float cur = h.empty() ? 0.0f : h.back();
    std::string curStr;
    if (m_metric == Metric::Treasury) curStr = moneyStr(cur);
    else if (m_metric == Metric::Population) curStr = commaInt(static_cast<long long>(std::lround(cur)));
    else { char b[16]; std::snprintf(b, sizeof(b), "%.0f%%", cur); curStr = b; }
    textLeft(m_uiFont, names[static_cast<int>(m_metric)], r.minX + 16.0f * s, valY - 8.0f * s,
             kTextDim);
    textLeft(m_uiFontBold, curStr, r.minX + 16.0f * s, valY + 14.0f * s, mc);

    // Chart area.
    const UiRect chart = UiRect::fromXYWH(r.minX + 16.0f * s, valY + 34.0f * s,
                                          r.width() - 32.0f * s, r.maxY - (valY + 34.0f * s) - 34.0f * s);
    m_uiDrawList.addRoundRectFilled(chart, withA(UiColor(0, 0, 0, 1), 0.25f), 6.0f * s);
    m_uiDrawList.addRoundRect(chart, kEdge, 6.0f * s, s);

    if (h.size() < 2) {
        textCenter(m_uiFont, "Collecting data… run the simulation",
                   (chart.minX + chart.maxX) * 0.5f, (chart.minY + chart.maxY) * 0.5f, kTextDim);
        textLeft(m_uiFont, "Switch metric: keys 1-5 on tabs, or G to toggle",
                 r.minX + 16.0f * s, r.maxY - 18.0f * s, kTextFaint);
        return;
    }

    // Range.
    float mn = h[0], mx = h[0];
    for (float v : h) { mn = std::min(mn, v); mx = std::max(mx, v); }
    const bool pct = m_metric == Metric::Education || m_metric == Metric::Health ||
                     m_metric == Metric::Happiness;
    if (pct) { mn = 0.0f; mx = 100.0f; }
    else {
        const float padR = std::max(1.0f, (mx - mn) * 0.12f);
        mn -= padR; mx += padR;
        if (m_metric != Metric::Treasury && mn < 0.0f) mn = 0.0f;
    }
    if (mx - mn < 1e-3f) mx = mn + 1.0f;

    const float pad = 8.0f * s;
    const UiRect area = UiRect{chart.minX + pad + 34.0f * s, chart.minY + pad,
                               chart.maxX - pad, chart.maxY - pad - 14.0f * s};
    auto valToY = [&](float v) { return area.maxY - (v - mn) / (mx - mn) * area.height(); };
    auto idxToX = [&](float i) {
        return area.minX + (i / static_cast<float>(h.size() - 1)) * area.width();
    };

    m_uiDrawList.pushClip(chart);
    // Gridlines + y labels.
    for (int g = 0; g <= 4; ++g) {
        const float t = g / 4.0f;
        const float yy = area.minY + t * area.height();
        m_uiDrawList.addRectFilled(UiRect{area.minX, yy - 0.5f, area.maxX, yy + 0.5f},
                                   withA(UiColor(1, 1, 1, 1), 0.06f));
        const float v = mx - t * (mx - mn);
        char lab[20];
        if (m_metric == Metric::Treasury) {
            std::snprintf(lab, sizeof(lab), "%lldk", static_cast<long long>(std::lround(v / 1000.0f)));
        } else if (pct) {
            std::snprintf(lab, sizeof(lab), "%.0f%%", v);
        } else {
            std::snprintf(lab, sizeof(lab), "%lld", static_cast<long long>(std::lround(v)));
        }
        textRight(m_uiFont, lab, area.minX - 6.0f * s, yy, kTextFaint);
    }

    // Area fill (smooth, per-pixel interpolation).
    const int px0 = static_cast<int>(area.minX);
    const int px1 = static_cast<int>(area.maxX);
    for (int px = px0; px < px1; ++px) {
        const float fi = (static_cast<float>(px) - area.minX) / area.width() * (h.size() - 1);
        const int i0 = std::clamp(static_cast<int>(fi), 0, static_cast<int>(h.size()) - 2);
        const float frac = fi - i0;
        const float v = lerpf(h[static_cast<std::size_t>(i0)], h[static_cast<std::size_t>(i0) + 1], frac);
        const float yy = valToY(v);
        m_uiDrawList.addRectFilled(UiRect{static_cast<float>(px), yy, static_cast<float>(px) + 1.0f,
                                          area.maxY}, withA(mc, 0.12f));
    }

    // Line.
    for (std::size_t i = 0; i + 1 < h.size(); ++i) {
        strokeLine({idxToX(static_cast<float>(i)), valToY(h[i])},
                   {idxToX(static_cast<float>(i + 1)), valToY(h[i + 1])}, 2.2f * s, mc);
    }
    // Last point marker.
    const UiVec2 last = {idxToX(static_cast<float>(h.size() - 1)), valToY(h.back())};
    m_uiDrawList.addCircleFilled(last, 3.5f * s, mc);
    m_uiDrawList.addCircle(last, 5.5f * s, withA(mc, 0.5f), 1.5f * s);
    m_uiDrawList.popClip();

    textLeft(m_uiFont, std::to_string(h.size()) + " months", area.minX, chart.maxY - 7.0f * s,
             kTextFaint);
    textRight(m_uiFont, "now", area.maxX, chart.maxY - 7.0f * s, kTextFaint);
}

void CityBuilderApp::drawFlash(const Layout& lo) {
    if (m_flashTimer <= 0.0f) return;
    const float s = lo.s;
    const float a = std::min(1.0f, m_flashTimer / 0.6f);
    const float w = std::max(180.0f * s, m_uiFontBold.measureText(m_flashMsg) + 40.0f * s);
    const float h = 34.0f * s;
    const UiRect r = UiRect::fromXYWH(lo.map.minX + (lo.map.width() - w) * 0.5f,
                                      lo.map.maxY - 90.0f * s, w, h);
    m_uiDrawList.addRoundRectFilled(r, withA(UiColor::fromRgbHex(0x2A1A1A), 0.92f * a), 8.0f * s);
    m_uiDrawList.addRoundRect(r, withA(kBad, 0.8f * a), 8.0f * s, s);
    textCenter(m_uiFontBold, m_flashMsg, (r.minX + r.maxX) * 0.5f, (r.minY + r.maxY) * 0.5f,
               withA(kText, a));
}

// ─────────────────────────────────────────────────────────────────────────────
// Small immediate-mode helpers
// ─────────────────────────────────────────────────────────────────────────────
bool CityBuilderApp::uiButton(const UiRect& r, std::string_view label, bool active,
                              const UiColor& accent, const ui::Font* font, bool enabled) {
    const ui::Font* f = font ? font : &m_uiFontBold;
    const bool hover = enabled && r.contains(m_uiInput.mousePx);
    const float radius = std::min(8.0f, r.height() * 0.28f);
    UiColor bg = !enabled ? withA(kBtn, 0.5f)
                          : active ? mix(kBtn, accent, 0.45f) : (hover ? kBtnHover : kBtn);
    m_uiDrawList.addRoundRectFilled(r, bg, radius);
    m_uiDrawList.addRoundRect(r, active ? withA(accent, 0.95f) : kEdge, radius, 1.0f);
    textCenter(*f, label, (r.minX + r.maxX) * 0.5f, (r.minY + r.maxY) * 0.5f,
               enabled ? kText : kTextFaint);
    return enabled && hover && m_uiInput.button(UiMouseButton::Left).pressed;
}

void CityBuilderApp::strokeLine(UiVec2 a, UiVec2 b, float widthPx, const UiColor& c) {
    const float dx = b.x - a.x, dy = b.y - a.y;
    const float len = std::sqrt(dx * dx + dy * dy);
    const float r = std::max(0.6f, widthPx * 0.5f);
    const int steps = std::max(1, static_cast<int>(len / r));
    for (int i = 0; i <= steps; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(steps);
        m_uiDrawList.addCircleFilled({a.x + dx * t, a.y + dy * t}, r, c);
    }
}

void CityBuilderApp::textLeft(const ui::Font& f, std::string_view str, float x, float cy,
                              const UiColor& c) {
    m_uiDrawList.addText(f, str, {x, cy - f.lineHeightPx() * 0.5f}, c);
}
void CityBuilderApp::textCenter(const ui::Font& f, std::string_view str, float cx, float cy,
                                const UiColor& c) {
    const float w = f.measureText(str);
    m_uiDrawList.addText(f, str, {cx - w * 0.5f, cy - f.lineHeightPx() * 0.5f}, c);
}
void CityBuilderApp::textRight(const ui::Font& f, std::string_view str, float rx, float cy,
                               const UiColor& c) {
    const float w = f.measureText(str);
    m_uiDrawList.addText(f, str, {rx - w, cy - f.lineHeightPx() * 0.5f}, c);
}

}  // namespace odai::games::citybuilder
