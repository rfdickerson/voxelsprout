#include "games/citybuilder/citybuilder_app.h"

#include "math/math.h"
#include "procgen/props.h"
#include "ui/font.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace odai::games::citybuilder {

using ui::UiColor;
using ui::UiRect;
using ui::UiVec2;
using ui::UiMouseButton;
using odai::importer::ImportedScene;
using odai::importer::ImportedScenePackedDraw;
using odai::importer::ImportedScenePackedVertex;
using odai::math::Vector3;

namespace {

// ── Palette (clean-modern flat HUD direction) ────────────────────────────────
constexpr UiColor kGrass     = UiColor::fromRgbHex(0x5E8C3E);
constexpr UiColor kGrassAlt  = UiColor::fromRgbHex(0x547E37);
constexpr UiColor kWater     = UiColor::fromRgbHex(0x2D5C8C);
constexpr UiColor kWaterAlt  = UiColor::fromRgbHex(0x23507E);
constexpr std::uint32_t kTreeVariants = 6;
constexpr std::uint32_t kCarVariants = 8;
constexpr std::uint32_t kPoleVariants = 3;
constexpr std::uint32_t kLampVariants = 3;
constexpr UiColor kAsphalt   = UiColor::fromRgbHex(0x303237);
constexpr UiColor kSidewalk  = UiColor::fromRgbHex(0x8E9092);
constexpr UiColor kLaneDash  = UiColor::fromRgbHex(0xD9C15A);
constexpr UiColor kCrosswalk = UiColor::fromRgbHex(0xC9CCCE);
constexpr UiColor kWire      = UiColor::fromRgbHex(0x1C1E22);
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

// Fixed isometric camera tilt (yaw rotates in 90 deg steps via Q/E).
constexpr float kCamPitchDeg = -52.0f;
// Diorama camera: a long lens from far away keeps the isometric composition
// but adds the subtle vanishing-line convergence that makes model cities read
// as miniatures. 20 deg is the renderer's minimum FOV clamp.
constexpr float kDioramaFovDeg = 20.0f;
constexpr float kCamMinZoom  = 6.0f;
constexpr float kCamMaxZoom  = 70.0f;

// ── Tuning knobs ─────────────────────────────────────────────────────────────
// The systemic feel of the city lives in these numbers — balance by turning
// them while playing, not by editing formulas inline.
constexpr int   kPowerRadius          = 9;       // a plant's direct glow, roads or not
constexpr int   kFireProtectRadius    = 6;       // fire dept coverage splat (also its amenity ring)
constexpr float kFireBaseChance       = 0.0001f; // per developed tile per month
constexpr float kFireIndustrialMul    = 3.0f;    // industry burns easiest
constexpr float kFireOldEraMul        = 2.0f;    // 1890s wood/brick (low develop) is tinder
constexpr float kFireDrySummerMul     = 1.8f;    // clear summer months are fire season
constexpr float kFireCoverageCut      = 0.75f;   // ignition removed by full fire coverage
constexpr float kFireSpreadChance     = 0.50f;   // per burning neighbour per month
constexpr float kFireSpreadWetCut     = 0.55f;   // rain/snow damps spread
constexpr float kFireSpreadCoverCut   = 0.70f;   // fire dept coverage damps spread
constexpr int   kFireBurnMonths       = 4;       // uncovered burn duration
constexpr int   kFireBurnMonthsCovered = 2;      // fire dept nearby: knocked down fast
constexpr int   kCharClearMonths      = 10;      // rubble self-clears after ~a year (+hash jitter)
constexpr int   kCharNuisanceRadius   = 3;       // burnt lots drag the neighbourhood
constexpr float kCharNuisancePeak     = 0.30f;
constexpr float kCongestionStart      = 1.2f;    // trafficLoad where a road reads as jammed
constexpr float kCongestionDecay      = 0.25f;   // per-second EMA decay of trafficLoad
constexpr int   kCongestionNuisanceRadius = 2;   // jammed arterials hurt frontage land value
constexpr float kCongestionNuisancePeak   = 0.12f;

struct ToolMeta {
    const char* tag;
    const char* name;
    const char* key;
    double      cost;
    UiColor     color;
};

// Indexed by CityBuilderApp::Tool, in declaration order.
const ToolMeta kTools[] = {
    {"DEMO", "Bulldoze",     "X", 4.0,    kBad},
    {"R",    "Residential",  "1", 25.0,   kZoneR},
    {"C",    "Commercial",   "2", 25.0,   kZoneC},
    {"I",    "Industrial",   "3", 25.0,   kZoneI},
    {"ROAD", "Road",         "R", 12.0,   UiColor::fromRgbHex(0x6B7079)},
    {"POL",  "Police",       "4", 500.0,  UiColor::fromRgbHex(0x2F6BD6)},
    {"FIRE", "Fire Dept",    "5", 500.0,  UiColor::fromRgbHex(0xC0392B)},
    {"CLN",  "Clinic",       "6", 450.0,  UiColor::fromRgbHex(0x21A89A)},
    {"SCH",  "School",       "7", 650.0,  UiColor::fromRgbHex(0xE0852E)},
    {"PRK",  "Park",         "8", 120.0,  UiColor::fromRgbHex(0x35863A)},
    {"LIB",  "Library",      "9", 550.0,  UiColor::fromRgbHex(0x8A5C3E)},
    {"AMPH", "Amphitheater", "0", 800.0,  UiColor::fromRgbHex(0xB08CD6)},
    {"PWR",  "Power Plant",  "-", 1200.0, UiColor::fromRgbHex(0xC9A227)},
};

// ── Building "character" flavor: deterministic per-tile hash so a parcel's
// business name / neighbourhood class is stable across frames without needing
// to store extra per-tile state — position + a per-table salt is enough.
std::uint32_t tileHash(int c, int r, std::uint32_t salt) {
    std::uint32_t h = static_cast<std::uint32_t>(c) * 374761393u ^
                       static_cast<std::uint32_t>(r) * 668265263u ^ salt;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}

struct FlavorEntry {
    const char* name;
    UiColor     tint;
};

// Commercial parcels: a small, silly, memorable business per tile.
constexpr FlavorEntry kCommercialFlavors[] = {
    {"Lotus Yoga Studio",           UiColor::fromRgbHex(0xC9A6D6)},
    {"Glazed & Confused Donuts",    UiColor::fromRgbHex(0xE8A6C4)},
    {"Llama Laundry Cleaners",      UiColor::fromRgbHex(0xA6D6C9)},
    {"Boo Crew Ghost Exterminators",UiColor::fromRgbHex(0x8E8CC4)},
    {"Madame Zora's Fortunes",      UiColor::fromRgbHex(0x7A4FA0)},
    {"Rusty Wrench Auto",           UiColor::fromRgbHex(0x9AA0A8)},
    {"Pixel Palace Arcade",         UiColor::fromRgbHex(0x4FC4E0)},
    {"Bark Ave Pet Spa",            UiColor::fromRgbHex(0xE0B24F)},
    {"Slurp Noodle House",          UiColor::fromRgbHex(0xE0714F)},
    {"Ink & Iron Tattoo",           UiColor::fromRgbHex(0x4A4A57)},
    {"Corner Bookshop",             UiColor::fromRgbHex(0x8A5C3E)},
    {"Sunrise Diner",               UiColor::fromRgbHex(0xE0C24A)},
    {"The Daily Grind Coffee",      UiColor::fromRgbHex(0x6B4A32)},
    {"Moonlight Lanes Bowling",     UiColor::fromRgbHex(0x4A5FE0)},
    {"Thrift & Vintage",            UiColor::fromRgbHex(0xA0824F)},
};
constexpr int kNumCommercialFlavors = static_cast<int>(sizeof(kCommercialFlavors) / sizeof(kCommercialFlavors[0]));

// Industrial parcels: less silly, but still varied instead of one flat block.
constexpr FlavorEntry kIndustrialFlavors[] = {
    {"Ironclad Steel Works",  UiColor::fromRgbHex(0x8A7A6B)},
    {"Assembly Plant",        UiColor::fromRgbHex(0xA0A6AA)},
    {"Freight Depot",         UiColor::fromRgbHex(0xB0862E)},
    {"Cascade Chem Yard",     UiColor::fromRgbHex(0x7A9A5C)},
    {"Timberline Lumber Mill",UiColor::fromRgbHex(0x9A6B3E)},
};
constexpr int kNumIndustrialFlavors = static_cast<int>(sizeof(kIndustrialFlavors) / sizeof(kIndustrialFlavors[0]));

const FlavorEntry& pickFlavor(const FlavorEntry* table, int count, int c, int r, std::uint32_t salt) {
    return table[tileHash(c, r, salt) % static_cast<std::uint32_t>(count)];
}

// Residential parcels don't get individual "shop names" — instead the whole
// block reads as a class of neighbourhood, driven by the existing land-value
// (desirability) field: poor land stays a trailer park / RV court even after
// it develops, prime land becomes an estate. tier 0 = low, 1 = mid, 2 = high.
int residentialTier(float desirability) {
    return desirability < 0.35f ? 0 : (desirability < 0.65f ? 1 : 2);
}

const char* residentialFlavorName(int c, int r, float desirability) {
    static const char* const kNames[3][3] = {
        {"Trailer Park", "RV Court", "Mobile Homes"},
        {"Suburbia", "Split-Level Homes", "Rowhouses"},
        {"Uptown Estates", "Hillside Manors", "Luxury Condos"},
    };
    const int tier = residentialTier(desirability);
    const int idx = static_cast<int>(tileHash(c, r, 0x51DE17u) % 3u);
    return kNames[tier][idx];
}

const char* kMonths[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

procgen::Season seasonForMonth(int month) {
    if (month <= 1 || month == 11) return procgen::Season::Winter;   // Dec-Feb
    if (month <= 4) return procgen::Season::Spring;                  // Mar-May
    if (month <= 7) return procgen::Season::Summer;                  // Jun-Aug
    return procgen::Season::Autumn;                                  // Sep-Nov
}

const char* seasonName(procgen::Season s) {
    switch (s) {
        case procgen::Season::Spring: return "Spring";
        case procgen::Season::Summer: return "Summer";
        case procgen::Season::Autumn: return "Autumn";
        default:                      return "Winter";
    }
}

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
        case Tool::Library: return Building::Library;
        case Tool::Amphitheater: return Building::Amphitheater;
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
        case Building::Library: return UiColor::fromRgbHex(0x8A5C3E);
        case Building::Amphitheater: return UiColor::fromRgbHex(0xB08CD6);
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
        case Building::Library: return "LIB";
        case Building::Amphitheater: return "AMPH";
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

// Red → amber → green ramp for the land-value overlay (0 = poor, 1 = prime).
UiColor heat(float d) {
    const UiColor lo  = UiColor::fromRgbHex(0xD64B3E);
    const UiColor mid = UiColor::fromRgbHex(0xE7C24A);
    const UiColor hi  = UiColor::fromRgbHex(0x46C46B);
    return d < 0.5f ? mix(lo, mid, d * 2.0f) : mix(mid, hi, (d - 0.5f) * 2.0f);
}

// World-unit roof height for each building's extruded box (see buildCityScene).
float buildingHeight(Building b) {
    switch (b) {
        case Building::Police: return 1.5f;
        case Building::Fire:   return 1.5f;
        case Building::Clinic: return 1.6f;
        case Building::School: return 1.8f;
        case Building::Park:   return 0.12f;
        case Building::Library: return 1.6f;
        case Building::Amphitheater: return 0.35f;
        case Building::Power:  return 2.6f;
        default:                return 1.0f;
    }
}

// Spatial influence each municipal building splats into the desirability
// fields (computeDesirability) — shared with the placement-time ring preview
// in drawWorldOverlay so what the player is shown is exactly what the sim
// uses. radius 0 = no influence.
struct InfluenceSpec {
    int   radius = 0;
    float peak = 0.0f;
    bool  nuisance = false;
};

constexpr InfluenceSpec buildingInfluence(Building b) {
    switch (b) {
        case Building::Park:         return {5, 0.55f, false};
        case Building::Police:       return {6, 0.28f, false};
        case Building::Fire:         return {kFireProtectRadius, 0.24f, false};
        case Building::Clinic:       return {6, 0.24f, false};
        case Building::School:       return {6, 0.30f, false};
        case Building::Library:      return {5, 0.26f, false};
        case Building::Amphitheater: return {7, 0.50f, false};  // culture draw
        case Building::Power:        return {5, 0.45f, true};   // dirty neighbour
        default:                     return {};
    }
}

// Accumulates flat-shaded packed geometry into an ImportedScene — the same
// "MeshBuilder" pattern strategy_map_mesh.cc uses to feed the renderer's
// packed vertex-color path (textureIndex left at its 0xFFFFFFFF default, so
// the imported-static shader uses per-vertex color instead of sampling).
struct CityMeshBuilder {
    ImportedScene& scene;
    explicit CityMeshBuilder(ImportedScene& target) : scene(target) {}

    std::uint32_t addVertex(const Vector3& p, const Vector3& n, const UiColor& c) {
        ImportedScenePackedVertex v{};
        v.position[0] = p.x; v.position[1] = p.y; v.position[2] = p.z;
        v.normal[0] = n.x;   v.normal[1] = n.y;   v.normal[2] = n.z;
        v.color[0] = c.r;    v.color[1] = c.g;    v.color[2] = c.b;
        const auto index = static_cast<std::uint32_t>(scene.packedVertices.size());
        scene.packedVertices.push_back(v);
        scene.boundsMin[0] = std::min(scene.boundsMin[0], p.x);
        scene.boundsMin[1] = std::min(scene.boundsMin[1], p.y);
        scene.boundsMin[2] = std::min(scene.boundsMin[2], p.z);
        scene.boundsMax[0] = std::max(scene.boundsMax[0], p.x);
        scene.boundsMax[1] = std::max(scene.boundsMax[1], p.y);
        scene.boundsMax[2] = std::max(scene.boundsMax[2], p.z);
        return index;
    }
    void addTriangle(std::uint32_t a, std::uint32_t b, std::uint32_t c) {
        scene.packedIndices.push_back(a);
        scene.packedIndices.push_back(b);
        scene.packedIndices.push_back(c);
    }
    void addFlatTriangle(const Vector3& a, const Vector3& b, const Vector3& c, const UiColor& color) {
        const Vector3 n = odai::math::normalize(odai::math::cross(b - a, c - a));
        addTriangle(addVertex(a, n, color), addVertex(b, n, color), addVertex(c, n, color));
    }
    // a-b-c-d must wind counter-clockwise when viewed from the face's outward
    // side (this engine's view/projection is a conventional right-handed,
    // Y-up, CCW-front setup — lookAt() is textbook right-handed and both
    // perspective/orthographic projections negate Y to correct for Vulkan's
    // flipped NDC, so no extra handedness quirk to account for here).
    void addQuad(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d, const UiColor& color) {
        addFlatTriangle(a, d, c, color);
        addFlatTriangle(a, c, b, color);
    }
};

// Six-face box, corners wound counter-clockwise when viewed from outside
// (mirrors the settlement/unit marker boxes in strategy_map_mesh.cc).
void addBox(CityMeshBuilder& builder, float minX, float minZ, float maxX, float maxZ,
            float minY, float maxY, const UiColor& color) {
    const Vector3 corners[8] = {
        {minX, minY, minZ}, {maxX, minY, minZ}, {maxX, minY, maxZ}, {minX, minY, maxZ},
        {minX, maxY, minZ}, {maxX, maxY, minZ}, {maxX, maxY, maxZ}, {minX, maxY, maxZ},
    };
    builder.addQuad(corners[4], corners[5], corners[6], corners[7], color);  // top
    builder.addQuad(corners[3], corners[2], corners[1], corners[0], color);  // bottom
    builder.addQuad(corners[0], corners[1], corners[5], corners[4], color);  // -Z
    builder.addQuad(corners[2], corners[3], corners[7], corners[6], color);  // +Z
    builder.addQuad(corners[3], corners[0], corners[4], corners[7], color);  // -X
    builder.addQuad(corners[1], corners[2], corners[6], corners[5], color);  // +X
}

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

// Same _dupenv_s-on-Windows pattern as core/log.cc's getEnvironmentVariable.
std::string readEnv(const char* name) {
#if defined(_WIN32)
    char* value = nullptr;
    std::size_t valueLength = 0;
    if (_dupenv_s(&value, &valueLength, name) != 0 || value == nullptr) {
        return {};
    }
    std::string result(value);
    std::free(value);
    return result;
#else
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::string();
#endif
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

    // Sunlight + shadow maps + AO carry the visual read here (there's no
    // texture detail on these flat-shaded boxes to fall back on). Ray tracing
    // is explicitly off — this scene never uses RT shadows/reflections/GI (see
    // ShadowMode::ShadowMaps below), so building and tearing down a BLAS/TLAS
    // on every scene re-upload was pure wasted GPU work — and, until a
    // synchronization bug in that teardown path was fixed at the engine level,
    // the actual source of an intermittent hang on quit. GameApp::init()
    // already disables this as an accidental side effect of setStrategyMapMode
    // (borrowed for its SSAO tuning, see below), but that's the wrong flag to
    // depend on for intent — opt out directly so it can't silently come back
    // if that side effect ever changes.
    m_renderer.setRayTracingEnabled(false);
    m_renderer.setSsaoEnabled(true);
    m_renderer.setVertexAoEnabled(true);
    m_renderer.setShadowSettings(render::ShadowSettings{render::ShadowMode::ShadowMaps});
    // Pixar-Up-style lighting direction: a warm mid-morning sun (the physical
    // sky model turns golden at lower elevations while the SH ambient stays
    // sky-blue, giving the warm-key / cool-fill contrast that look leans on)
    // at 38 deg elevation so buildings cast readable shadows ~1.3x their
    // height, angled across the iso view rather than straight down it.
    m_renderer.setSunAngles(50.0f, -38.0f);
    // GameApp::init() always calls setStrategyMapMode(true), which tunes SSAO
    // radius/bias for the hex strategy map's much larger, flatter scale
    // (radius=7, bias=6). At this world's kTileWorldSize=1 with ~1-3 unit
    // building heights, that bias is larger than any real depth step in the
    // scene, so occlusion can never trigger. Re-tune for this scale: a
    // ~1.5-tile radius reads as soft contact shading between buildings.
    m_renderer.setAmbientOcclusionTuning(1.6f, 0.06f, 0.95f);

    m_season = seasonForMonth(m_month);
    generateTerrain();
    seedCity();
    // ODAI_CITY_DEMO=1: force the seeded zone bands to development levels
    // 1/2/3 (south rows denser) so all three architectural eras — 1890s brick,
    // 1930s deco setbacks, 1960s curtain-wall — are on screen immediately.
    // Purely a dev/visual-verification aid; normal play grows into the same
    // levels over simulated months.
    if (const std::string demo = readEnv("ODAI_CITY_DEMO"); !demo.empty() && demo != "0") {
        for (int r = 25; r <= 31; ++r) {
            const float dev = r <= 26 ? 0.5f : (r <= 28 ? 1.5f : 2.5f);
            for (int c = 16; c <= 31; ++c) {
                Tile& t = tile(c, r);
                if (t.zone != Zone::None) t.develop = dev;
            }
        }
        // Start the demo in October: autumn foliage and pumpkins on screen
        // immediately (press N to skip months and tour the other seasons).
        m_month = 9;
        m_season = seasonForMonth(m_month);
    }
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
    for (Tile& t : m_tiles) { t.powered = false; t.poweredRoad = false; t.nearRoad = false; }

    m_numRoad = m_numPolice = m_numFire = m_numClinic = m_numSchool = m_numPark = m_numPower = 0;
    m_numLibrary = m_numAmphitheater = 0;
    float residents = 0.0f, comJobs = 0.0f, indJobs = 0.0f;

    constexpr int kRoadReach = 3;   // Chebyshev tiles a road services

    std::vector<std::pair<int, int>> powerPlants;

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
                    case Building::Library: ++m_numLibrary; break;
                    case Building::Amphitheater: ++m_numAmphitheater; break;
                    case Building::Power:  ++m_numPower;  break;
                    default: break;
                }
                if (t.building == Building::Power) {
                    powerPlants.emplace_back(c, r);
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

    // Pass 2.5: power grid. A plant's direct glow (above) only reaches a fixed
    // radius, which stranded any zone built further out even when it was
    // properly road-connected back to the plant — the "No power" complaint on
    // a perfectly reasonable layout. Real power distribution follows wires
    // along the street grid, so flood outward from each plant along connected
    // road tiles (4-directional, matching how road segments actually join)
    // and light up a short halo around every tile the grid reaches. This is
    // additive on top of the direct radius, never subtractive.
    constexpr int kPlantFeedReach = 2;   // plant's own substation reach onto nearby roads
    constexpr int kPowerLineReach = 2;   // how far power reaches off a powered road tile
    const auto stampPowerAround = [&](int cc, int rr) {
        for (int dr = -kPowerLineReach; dr <= kPowerLineReach; ++dr)
            for (int dc = -kPowerLineReach; dc <= kPowerLineReach; ++dc)
                if (inBounds(cc + dc, rr + dr)) tile(cc + dc, rr + dr).powered = true;
    };
    std::vector<int> frontier;  // encodes tile index as r * kGridW + c
    for (const auto& [pc, pr] : powerPlants) {
        for (int dr = -kPlantFeedReach; dr <= kPlantFeedReach; ++dr) {
            for (int dc = -kPlantFeedReach; dc <= kPlantFeedReach; ++dc) {
                const int cc = pc + dc, rr = pr + dr;
                if (!inBounds(cc, rr)) continue;
                Tile& rt = tile(cc, rr);
                if (rt.road && !rt.poweredRoad) {
                    rt.poweredRoad = true;
                    stampPowerAround(cc, rr);
                    frontier.push_back(rr * kGridW + cc);
                }
            }
        }
    }
    for (std::size_t qi = 0; qi < frontier.size(); ++qi) {
        const int idx = frontier[qi];
        const int rr = idx / kGridW, cc = idx % kGridW;
        static constexpr int kDc[4] = {1, -1, 0, 0};
        static constexpr int kDr[4] = {0, 0, 1, -1};
        for (int k = 0; k < 4; ++k) {
            const int nc = cc + kDc[k], nr = rr + kDr[k];
            if (!inBounds(nc, nr)) continue;
            Tile& nt = tile(nc, nr);
            if (nt.road && !nt.poweredRoad) {
                nt.poweredRoad = true;
                stampPowerAround(nc, nr);
                frontier.push_back(nr * kGridW + nc);
            }
        }
    }

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
    const float eduTarget    = clamp01((m_numSchool * 700.0f + m_numLibrary * 380.0f) / pop) * 100.0f;
    const float healthTarget = clamp01(m_numClinic * 900.0f / pop) * 100.0f;
    const float safety       = clamp01((m_numPolice + m_numFire) * 650.0f / pop);
    const float parkCov      = clamp01(m_numPark * 450.0f / pop);
    const float cultureCov   = clamp01(m_numAmphitheater * 600.0f / pop);

    m_education = lerpf(m_education, eduTarget, m_statEase);
    m_health    = lerpf(m_health, healthTarget, m_statEase);

    float happyTarget = 46.0f + 0.15f * m_education + 0.15f * m_health +
                        20.0f * parkCov + 12.0f * safety + 14.0f * cultureCov +
                        (m_powerCoverage - 1.0f) * 35.0f - 4.0f;
    // Active fires and standing rubble weigh on the city's mood.
    happyTarget -= std::min(12.0f, static_cast<float>(m_burningTiles) * 1.5f +
                                       static_cast<float>(m_charredTiles) * 0.4f);
    happyTarget = std::clamp(happyTarget, 0.0f, 100.0f);
    m_happiness = lerpf(m_happiness, happyTarget, m_statEase);

    computeDesirability();
}

// Per-tile land value: scatter "amenity" (nice) and "nuisance" (bad) influence
// from things already modelled here — parks, service coverage and waterfront lift
// desirability; power plants and developed industry drag it down. The growth step
// reads this so WHERE you zone finally matters, and the overlay makes it readable.
// This is the SimCity move: turn an invisible spatial pressure into a field the
// player can see, reason about, and set their own goals against.
void CityBuilderApp::computeDesirability() {
    std::array<float, static_cast<std::size_t>(kGridW) * kGridH> amenity{};
    std::array<float, static_cast<std::size_t>(kGridW) * kGridH> nuisance{};

    auto splat = [&](std::array<float, static_cast<std::size_t>(kGridW) * kGridH>& field,
                     int c, int r, int radius, float peak) {
        for (int dr = -radius; dr <= radius; ++dr) {
            for (int dc = -radius; dc <= radius; ++dc) {
                if (!inBounds(c + dc, r + dr)) continue;
                const float d = std::sqrt(static_cast<float>(dc * dc + dr * dr));
                if (d > radius) continue;
                const float w = 1.0f - d / static_cast<float>(radius);  // linear falloff
                field[static_cast<std::size_t>(r + dr) * kGridW + (c + dc)] += peak * w;
            }
        }
    };

    // Scatter influence from every source into the two fields.
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            const Tile& t = tile(c, r);
            if (t.terrain == Terrain::Water) splat(amenity, c, r, 3, 0.10f);  // scenic waterfront
            if (t.bldgOrigin) {
                const InfluenceSpec inf = buildingInfluence(t.building);
                if (inf.radius > 0) splat(inf.nuisance ? nuisance : amenity, c, r, inf.radius, inf.peak);
            }
            if (t.zone == Zone::Industrial && t.develop > kDevEps) {
                splat(nuisance, c, r, 4, 0.16f * t.develop);  // pollution / heavy traffic
            }
            if (t.charred) {
                splat(nuisance, c, r, kCharNuisanceRadius, kCharNuisancePeak);  // burnt-out blight
            }
            if (t.road && t.trafficLoad > kCongestionStart) {
                // Jammed arterials hurt the lots that front them — growth begets
                // traffic begets falling land value, the classic feedback loop.
                const float over = std::min(1.0f, (t.trafficLoad - kCongestionStart) / 2.5f);
                splat(nuisance, c, r, kCongestionNuisanceRadius, kCongestionNuisancePeak * over);
            }
        }
    }

    // Fold the fields into a per-tile score. Homes and shops crave amenities and
    // flee nuisances; industry mostly ignores scenery and tolerates its own kind,
    // so its land value stays flatter (it just needs power + roads elsewhere).
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            Tile& t = tile(c, r);
            const std::size_t i = static_cast<std::size_t>(r) * kGridW + c;
            float score;
            if (t.zone == Zone::Industrial) {
                score = 0.55f + 0.25f * amenity[i] - 0.10f * nuisance[i];
            } else {  // residential, commercial, and open land shown as res-potential
                score = 0.42f + amenity[i] - 0.85f * nuisance[i];
            }
            t.desirability = clamp01(score);
        }
    }
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
        if (t.fireTicks > 0) {  // burning down, not growing
            t.develop = std::max(0.0f, t.develop - 0.25f);
            continue;
        }
        if (t.charred) { t.develop = 0.0f; continue; }
        const float dem = t.zone == Zone::Residential ? m_resDemand
                          : t.zone == Zone::Commercial ? m_comDemand
                                                       : m_indDemand;
        if (t.powered && t.nearRoad) {
            // Citywide demand sets the ceiling; per-tile desirability decides how
            // much of it each parcel actually captures. A 0.5 land value is neutral
            // (multiplier 1.0), prime land overshoots (clamped to full build-out),
            // and poor land stagnates even in a hot market.
            const float desMul = 0.4f + 1.2f * t.desirability;
            const float target = std::min(3.0f, dem * 3.0f * desMul);
            t.develop += (target - t.develop) * 0.16f;
        } else {
            t.develop += (0.0f - t.develop) * 0.10f;  // decay toward abandonment
        }
        t.develop = std::clamp(t.develop, 0.0f, 3.0f);
    }

    stepFire();

    // Post-growth census, easing the city quality stats toward their targets.
    m_statEase = 0.12f;
    recomputeStats();

    // Monthly budget.
    const double income = m_population * 0.10 + m_jobs * 0.07;
    const double upkeep = m_numRoad * 0.6 + m_numPolice * 45.0 + m_numFire * 45.0 +
                          m_numClinic * 38.0 + m_numSchool * 48.0 + m_numPark * 9.0 +
                          m_numLibrary * 32.0 + m_numAmphitheater * 40.0 +
                          m_numPower * 75.0;
    m_lastNet = income - upkeep;
    m_money += m_lastNet;

    if (++m_month >= 12) { m_month = 0; ++m_year; }
    // Season boundary: repaint the whole scene right away (4x/year, so no need
    // for the growth cooldown) — foliage, ground tint, and autumn decorations
    // all change with it.
    const procgen::Season season = seasonForMonth(m_month);
    if (season != m_season) {
        m_season = season;
        m_sceneDirty = true;
    }
    pushHistory();
    m_growthDirty = true;  // develop levels changed; re-extrude on the next cooldown tick
}

// Fire is the mechanic that couples the whole board: ignition odds read the
// building era, zone, season, and weather; spread reads density; suppression
// reads fire-dept coverage; and the charred aftermath feeds back into land
// value (computeDesirability). Runs once per simulated month from stepMonth.
void CityBuilderApp::stepFire() {
    // Fire-dept coverage field, same linear-falloff splat as computeDesirability.
    std::array<float, static_cast<std::size_t>(kGridW) * kGridH> cov{};
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            const Tile& t = tile(c, r);
            if (!t.bldgOrigin || t.building != Building::Fire) continue;
            for (int dr = -kFireProtectRadius; dr <= kFireProtectRadius; ++dr) {
                for (int dc = -kFireProtectRadius; dc <= kFireProtectRadius; ++dc) {
                    if (!inBounds(c + dc, r + dr)) continue;
                    const float d = std::sqrt(static_cast<float>(dc * dc + dr * dr));
                    if (d > kFireProtectRadius) continue;
                    cov[static_cast<std::size_t>(r + dr) * kGridW + (c + dc)] +=
                        1.0f - d / static_cast<float>(kFireProtectRadius);
                }
            }
        }
    }
    const auto covAt = [&](int c, int r) {
        return std::min(1.0f, cov[static_cast<std::size_t>(r) * kGridW + c]);
    };
    const auto rnd01 = [&]() -> float {
        m_rng = m_rng * 1664525u + 1013904223u;
        return static_cast<float>((m_rng >> 8) & 0xFFFFFFu) / 16777216.0f;
    };
    const auto ignite = [&](int c, int r) {
        Tile& t = tile(c, r);
        t.fireTicks = static_cast<std::uint8_t>(covAt(c, r) > 0.45f ? kFireBurnMonthsCovered
                                                                    : kFireBurnMonths);
    };
    const float wet = m_weatherIntensity;  // 0 clear .. 1 full rain/snow

    // Snapshot the currently burning tiles first so this month's new ignitions
    // don't spread or burn down in the same month they start.
    std::vector<std::pair<int, int>> burning;
    for (int r = 0; r < kGridH; ++r)
        for (int c = 0; c < kGridW; ++c)
            if (tile(c, r).fireTicks > 0) burning.emplace_back(c, r);

    // Spread pass: each burning tile rolls against its 4-neighbours.
    static constexpr int kDc[4] = {1, -1, 0, 0};
    static constexpr int kDr[4] = {0, 0, 1, -1};
    for (const auto& [bc, br] : burning) {
        for (int k = 0; k < 4; ++k) {
            const int nc = bc + kDc[k], nr = br + kDr[k];
            if (!inBounds(nc, nr)) continue;
            const Tile& nt = tile(nc, nr);
            if (nt.zone == Zone::None || nt.develop <= kDevEps || nt.fireTicks > 0 || nt.charred)
                continue;
            const float chance = kFireSpreadChance * (1.0f - kFireSpreadWetCut * wet) *
                                 (1.0f - kFireSpreadCoverCut * covAt(nc, nr));
            if (rnd01() < chance) ignite(nc, nr);
        }
    }

    // Burn down the snapshot; a tile that runs out becomes charred rubble.
    for (const auto& [bc, br] : burning) {
        Tile& t = tile(bc, br);
        if (--t.fireTicks == 0) {
            t.charred = true;
            t.charTicks = 0;
            t.develop = 0.0f;
        }
    }

    // Fresh ignitions across the developed city.
    const bool drySummer = m_season == procgen::Season::Summer && m_weather == Weather::Clear;
    int newIgnitions = 0;
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            const Tile& t = tile(c, r);
            if (t.zone == Zone::None || t.develop <= kDevEps || t.fireTicks > 0 || t.charred)
                continue;
            float chance = kFireBaseChance;
            if (t.zone == Zone::Industrial) chance *= kFireIndustrialMul;
            if (t.develop < 1.5f) chance *= kFireOldEraMul;  // 1890s-era wood/brick
            if (drySummer) chance *= kFireDrySummerMul;
            chance *= 1.0f - 0.8f * wet;
            chance *= 1.0f - kFireCoverageCut * covAt(c, r);
            if (rnd01() < chance) {
                ignite(c, r);
                ++newIgnitions;
            }
        }
    }

    // Rubble slowly clears itself (hash-jittered so a burnt block doesn't
    // vanish in one frame), and the fire census refreshes for the HUD/mood.
    int burningNow = 0, charredNow = 0;
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            Tile& t = tile(c, r);
            if (t.charred) {
                if (++t.charTicks > kCharClearMonths + static_cast<int>(tileHash(c, r, 0xA5Bu) % 8u)) {
                    t.charred = false;
                    t.charTicks = 0;
                }
            }
            if (t.fireTicks > 0) ++burningNow;
            if (t.charred) ++charredNow;
        }
    }
    if (newIgnitions > 0 && m_burningTiles == 0) flash("Fire reported!");
    m_burningTiles = burningNow;
    m_charredTiles = charredNow;
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
            if (charge(cost)) {
                t.zone = z; t.develop = 0.0f;
                t.fireTicks = 0; t.charred = false; t.charTicks = 0;  // re-zoning clears the lot
                m_sceneDirty = true;
            }
            break;
        }
        case Tool::Road:
            if (t.terrain == Terrain::Water) { flash("Can't pave water"); break; }
            if (t.building != Building::None) { flash("Bulldoze first"); break; }
            if (t.road) break;
            if (charge(cost)) { t.road = true; t.zone = Zone::None; t.develop = 0.0f; m_sceneDirty = true; }
            break;
        default:
            placeBuilding(c, r, toolBuilding(m_tool));
            break;
    }
}

void CityBuilderApp::bulldoze(int c, int r) {
    m_sceneDirty = true;
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
    // Bulldozing extinguishes and clears the lot — dozing a lane of parcels
    // ahead of a spreading fire is a deliberate firebreak play.
    t.fireTicks = 0;
    t.charred = false;
    t.charTicks = 0;
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
        case Building::Library: cost = 550.0; break;
        case Building::Amphitheater: cost = 800.0; break;
        case Building::Power:  cost = 1200.0; break;
        default: break;
    }
    if (!charge(cost)) return false;
    m_sceneDirty = true;

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
    if (m_sceneRebuildCooldown > 0.0f) m_sceneRebuildCooldown -= dt;

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
    if (edgeDown(GLFW_KEY_9)) m_tool = Tool::Library;
    if (edgeDown(GLFW_KEY_0)) m_tool = Tool::Amphitheater;
    if (edgeDown(GLFW_KEY_MINUS)) m_tool = Tool::Power;

    if (edgeDown(GLFW_KEY_SPACE)) m_paused = !m_paused;
    if (edgeDown(GLFW_KEY_G)) m_reportsOpen = !m_reportsOpen;
    if (edgeDown(GLFW_KEY_L)) { m_showLandValue = !m_showLandValue; m_sceneDirty = true; }
    // Debug: N skips a month — handy for eyeballing season transitions.
    if (edgeDown(GLFW_KEY_N)) stepMonth();

    // Rotate the isometric view in 90 deg steps, SimCity/Cities-style.
    if (edgeDown(GLFW_KEY_Q)) m_camYawDeg -= 90.0f;
    if (edgeDown(GLFW_KEY_E)) m_camYawDeg += 90.0f;

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
        // Ambient traffic runs on real time (not sim speed) so cars cruise at
        // a believable pace at every game speed.
        updateVehicles(dt);
    }
    // Weather is pure atmosphere — it keeps falling even while paused.
    updateWeather(dt);
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

    const float ctlW = 566.0f * s, ctlH = 46.0f * s;
    lo.controls = UiRect::fromXYWH(lo.map.minX + 14.0f * s, lo.fh - ctlH - 14.0f * s, ctlW, ctlH);

    const float mmS = 190.0f * s;
    lo.minimap = UiRect::fromXYWH(lo.fw - mmS - 14.0f * s, lo.fh - mmS - 14.0f * s, mmS, mmS);

    const float rW = 470.0f * s, rH = 544.0f * s;
    lo.reports = UiRect::fromXYWH(lo.map.minX + (lo.map.width() - rW) * 0.5f,
                                  lo.map.minY + 22.0f * s, rW, rH);
    return lo;
}

void CityBuilderApp::clampCameraFocus() {
    // Keep the look-at point roughly over the grid (plus a zoom-scaled margin)
    // so panning/zooming can't lose the city entirely off-camera.
    const float margin = m_camZoom * 0.5f;
    const float loX = -margin, hiX = kGridW * kTileWorldSize + margin;
    const float loZ = -margin, hiZ = kGridH * kTileWorldSize + margin;
    m_camFocusX = std::clamp(m_camFocusX, loX, hiX);
    m_camFocusZ = std::clamp(m_camFocusZ, loZ, hiZ);
}

void CityBuilderApp::handleCamera(const Layout& lo) {
    const bool overWorld = !m_mouseOverUi;
    m_camera = computeCameraPose();

    // Zoom toward the cursor: keep whatever ground point is under the mouse
    // fixed on screen as orthoHalfHeight changes.
    if (overWorld && m_uiInput.scrollDelta != 0.0f) {
        float oldGX = 0.0f, oldGZ = 0.0f;
        const bool hadGround = screenToGroundXZ(m_uiInput.mousePx, lo, oldGX, oldGZ);
        const float newZoom = std::clamp(m_camZoom * (1.0f - 0.12f * m_uiInput.scrollDelta),
                                         kCamMinZoom, kCamMaxZoom);
        if (newZoom != m_camZoom) {
            m_camZoom = newZoom;
            m_camera = computeCameraPose();
            float newGX = 0.0f, newGZ = 0.0f;
            if (hadGround && screenToGroundXZ(m_uiInput.mousePx, lo, newGX, newGZ)) {
                m_camFocusX += (oldGX - newGX);
                m_camFocusZ += (oldGZ - newGZ);
                m_camera = computeCameraPose();
            }
        }
    }

    // Right-drag pan: whatever ground point was under the cursor last frame
    // tracks the cursor this frame (robust under any yaw/pitch/zoom).
    if (overWorld && m_uiInput.button(UiMouseButton::Right).down) {
        const UiVec2 newPos = m_uiInput.mousePx;
        const UiVec2 oldPos{newPos.x - m_uiInput.mouseDeltaPx.x, newPos.y - m_uiInput.mouseDeltaPx.y};
        float oldGX = 0.0f, oldGZ = 0.0f, newGX = 0.0f, newGZ = 0.0f;
        if (screenToGroundXZ(oldPos, lo, oldGX, oldGZ) && screenToGroundXZ(newPos, lo, newGX, newGZ)) {
            m_camFocusX -= (newGX - oldGX);
            m_camFocusZ -= (newGZ - oldGZ);
            m_camera = computeCameraPose();
        }
    }

    clampCameraFocus();
    m_camera = computeCameraPose();
}

void CityBuilderApp::handleMapPaint(const Layout& /*lo*/) {
    if (m_mouseOverUi || m_hoverC < 0) return;
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
        m_camFocusX = kGridW * kTileWorldSize * 0.5f;
        m_camFocusZ = kGridH * kTileWorldSize * 0.5f;
        m_camZoom   = kGridW * kTileWorldSize * 0.95f;  // conservative: fits the whole grid
        m_camYawDeg = 45.0f;                             // classic diagonal city-builder view
        m_camInit = true;
    }

    m_mouseOverUi = lo.topBar.contains(m_uiInput.mousePx) ||
                    lo.palette.contains(m_uiInput.mousePx) ||
                    lo.controls.contains(m_uiInput.mousePx) ||
                    lo.minimap.contains(m_uiInput.mousePx) ||
                    (m_reportsOpen && lo.reports.contains(m_uiInput.mousePx));

    handleCamera(lo);

    // Hover tile: cast the mouse pixel onto the ground plane (y=0) and floor
    // into grid coordinates — the 3-D analogue of the old pixel/tilePx divide.
    m_hoverC = m_hoverR = -1;
    if (!m_mouseOverUi) {
        float gx = 0.0f, gz = 0.0f;
        if (screenToGroundXZ(m_uiInput.mousePx, lo, gx, gz)) {
            const int c = static_cast<int>(std::floor(gx / kTileWorldSize));
            const int r = static_cast<int>(std::floor(gz / kTileWorldSize));
            if (inBounds(c, r)) { m_hoverC = c; m_hoverR = r; }
        }
    }

    handleMapPaint(lo);

    // Growth-driven rebuilds are cadence-limited (see m_sceneRebuildCooldown);
    // player edits set m_sceneDirty directly and skip the wait.
    if (m_growthDirty && m_sceneRebuildCooldown <= 0.0f) {
        m_sceneDirty = true;
        m_growthDirty = false;
        m_sceneRebuildCooldown = 1.2f;
    }

    // The 3-D scene is a GPU buffer upload, not a per-frame draw call: only
    // rebuild it when a tile actually changed (see the m_sceneDirty sites).
    if (m_sceneDirty) {
        m_renderer.uploadImportedScene(buildCityScene());
        m_sceneDirty = false;
    }

    beginFrameDraw();
    drawTopBar(lo);
    drawPalette(lo);
    drawControls(lo);
    drawMinimap(lo);
    if (m_reportsOpen) drawReports(lo);
    drawWorldOverlay(lo);
    drawFlash(lo);

    // Tilt-shift depth of field completes the diorama read: the ground at the
    // focus point stays sharp and the frame's near/far edges melt away like a
    // macro photo of a model. Focus tracks the camera pull-back distance.
    const float focusDist = m_camZoom / std::tan(odai::math::radians(kDioramaFovDeg) * 0.5f);
    m_renderer.setDepthOfField(true, focusDist, std::max(8.0f, m_camZoom * 0.85f), 6.0f);

    // Cars and weather particles are per-frame streamed geometry (FrameArena),
    // not part of the uploaded scene — animation never triggers a scene
    // re-upload.
    const render::ImportedActorFrameData actorData = buildActorFrameData();
    submitFrame(m_camera, 0.0f, actorData.vertices.empty() ? nullptr : &actorData);
}

ImportedScene CityBuilderApp::buildCityScene() const {
    ImportedScene scene{};
    scene.sourceTag = "citybuilder";
    scene.boundsMin[0] = scene.boundsMin[1] = scene.boundsMin[2] = std::numeric_limits<float>::max();
    scene.boundsMax[0] = scene.boundsMax[1] = scene.boundsMax[2] = std::numeric_limits<float>::lowest();
    CityMeshBuilder builder(scene);
    const float ts = kTileWorldSize;

    // Seasonal ground palette: winter buries grass under snow and skims the
    // water with ice; autumn browns off; spring reads fresher.
    auto seasonalGrass = [&](const UiColor& g) {
        switch (m_season) {
            case procgen::Season::Winter: return mix(g, UiColor::fromRgbHex(0xDCE2E6), 0.78f);
            case procgen::Season::Autumn: return mix(g, UiColor::fromRgbHex(0x9A7A38), 0.32f);
            case procgen::Season::Spring: return mix(g, UiColor::fromRgbHex(0x74A93F), 0.28f);
            default:                      return g;
        }
    };
    const bool winter = m_season == procgen::Season::Winter;
    const bool autumn = m_season == procgen::Season::Autumn;

    // ── Parceling: group contiguous same-zone tiles into rectangular plots
    // (1x1 up to 3x2, weighted per zone — industry runs biggest) so blocks
    // read as varied city lots instead of a stamp of identical squares.
    // Deterministic greedy row-major scan re-derived from the zone map on
    // every rebuild; repainting zones naturally re-parcels. The sim itself
    // stays per-tile — a plot renders one building sized to the merged lot,
    // using the member tiles' average development / land value.
    struct Plot {
        short c = 0, r = 0;
        std::uint8_t w = 1, d = 1;
        float develop = 0.0f;
        float desirability = 0.5f;
        bool powered = false;
    };
    std::vector<Plot> plots;
    std::vector<short> plotIndex(static_cast<std::size_t>(kGridW) * kGridH, -1);
    const auto parcelable = [&](int c, int r, Zone z) {
        if (!inBounds(c, r)) return false;
        const Tile& pt = tile(c, r);
        return pt.terrain == Terrain::Grass && pt.zone == z && !pt.road &&
               pt.building == Building::None && pt.fireTicks == 0 && !pt.charred &&
               plotIndex[static_cast<std::size_t>(r) * kGridW + c] < 0;
    };
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            const Tile& t = tile(c, r);
            if (!parcelable(c, r, t.zone) || t.zone == Zone::None) continue;
            const std::uint32_t roll = tileHash(c, r, 0x9107C0DEu) % 100u;
            int pw = 1, pd = 1;
            if (t.zone == Zone::Industrial) {
                if (roll >= 15 && roll < 40) { pw = 2; pd = 1; }
                else if (roll < 60) { pw = 1; pd = 2; }
                else if (roll < 80) { pw = 2; pd = 2; }
                else if (roll < 92) { pw = 3; pd = 2; }
                else if (roll >= 92) { pw = 2; pd = 3; }
            } else if (t.zone == Zone::Commercial) {
                if (roll >= 30 && roll < 55) { pw = 2; pd = 1; }
                else if (roll < 75) { pw = 1; pd = 2; }
                else if (roll < 92) { pw = 2; pd = 2; }
                else if (roll >= 92) { pw = 3; pd = 2; }
            } else {
                if (roll >= 45 && roll < 68) { pw = 2; pd = 1; }
                else if (roll < 88) { pw = 1; pd = 2; }
                else if (roll >= 88) { pw = 2; pd = 2; }
            }
            const auto rectOk = [&](int w, int d) {
                for (int dr = 0; dr < d; ++dr)
                    for (int dc = 0; dc < w; ++dc)
                        if ((dr != 0 || dc != 0) && !parcelable(c + dc, r + dr, t.zone)) return false;
                return true;
            };
            while (!rectOk(pw, pd)) {
                if (pw >= pd && pw > 1) --pw;
                else if (pd > 1) --pd;
                else break;  // 1x1 always fits (only the origin, already checked)
            }
            Plot plot;
            plot.c = static_cast<short>(c);
            plot.r = static_cast<short>(r);
            plot.w = static_cast<std::uint8_t>(pw);
            plot.d = static_cast<std::uint8_t>(pd);
            int poweredCount = 0;
            float devSum = 0.0f, desSum = 0.0f;
            for (int dr = 0; dr < pd; ++dr) {
                for (int dc = 0; dc < pw; ++dc) {
                    const Tile& member = tile(c + dc, r + dr);
                    plotIndex[static_cast<std::size_t>(r + dr) * kGridW + (c + dc)] =
                        static_cast<short>(plots.size());
                    devSum += member.develop;
                    desSum += member.desirability;
                    if (member.powered) ++poweredCount;
                }
            }
            const int count = pw * pd;
            plot.develop = devSum / static_cast<float>(count);
            plot.desirability = desSum / static_cast<float>(count);
            plot.powered = poweredCount * 2 >= count;
            plots.push_back(plot);
        }
    }

    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            const Tile& t = tile(c, r);
            const float x0 = c * ts, z0 = r * ts, x1 = x0 + ts, z1 = z0 + ts;

            if (t.terrain == Terrain::Water) {
                UiColor wc = ((c + r) & 1) ? kWater : kWaterAlt;
                if (winter) wc = mix(wc, UiColor::fromRgbHex(0xA8C4D4), 0.55f);
                builder.addQuad({x0, 0.0f, z0}, {x1, 0.0f, z0}, {x1, 0.0f, z1}, {x0, 0.0f, z1}, wc);
                continue;
            }

            // Ground: land-value heat map when toggled, otherwise grass tinted
            // by the baked scenicPhase jitter so a block reads as organic.
            if (m_showLandValue) {
                builder.addQuad({x0, 0.0f, z0}, {x1, 0.0f, z0}, {x1, 0.0f, z1}, {x0, 0.0f, z1},
                                heat(t.desirability));
            } else {
                builder.addQuad({x0, 0.0f, z0}, {x1, 0.0f, z0}, {x1, 0.0f, z1}, {x0, 0.0f, z1},
                                seasonalGrass(mix(kGrassAlt, kGrass, t.scenicPhase)));
            }

            if (t.road) {
                const bool nN = inBounds(c, r - 1) && tile(c, r - 1).road;
                const bool nS = inBounds(c, r + 1) && tile(c, r + 1).road;
                const bool nW = inBounds(c - 1, r) && tile(c - 1, r).road;
                const bool nE = inBounds(c + 1, r) && tile(c + 1, r).road;
                const float side = ts * 0.16f;   // sidewalk band width from the tile edge
                const float ry = 0.02f;          // asphalt surface height
                const float sy = 0.045f;         // sidewalk top (raised curb above asphalt)
                const float cx = (x0 + x1) * 0.5f, cz = (z0 + z1) * 0.5f;

                // Asphalt: runs to the tile edge on connected sides so
                // neighbouring road tiles pave seamlessly; stops at the
                // sidewalk band elsewhere.
                const float ax0 = nW ? x0 : x0 + side, ax1 = nE ? x1 : x1 - side;
                const float az0 = nN ? z0 : z0 + side, az1 = nS ? z1 : z1 - side;
                builder.addQuad({ax0, ry, az0}, {ax1, ry, az0}, {ax1, ry, az1}, {ax0, ry, az1},
                                kAsphalt);

                // Raised sidewalk slabs along the unconnected edges. Boxes (not
                // flat quads) so the curb face catches shadow/AO. The N/S
                // strips span the full tile width; E/W strips are clipped to
                // avoid double-covering the corners.
                auto walk = [&](float wx0, float wz0, float wx1, float wz1) {
                    addBox(builder, wx0, wz0, wx1, wz1, 0.0f, sy, kSidewalk);
                };
                if (!nN) walk(x0, z0, x1, z0 + side);
                if (!nS) walk(x0, z1 - side, x1, z1);
                if (!nW) walk(x0, nN ? z0 : z0 + side, x0 + side, nS ? z1 : z1 - side);
                if (!nE) walk(x1 - side, nN ? z0 : z0 + side, x1, nS ? z1 : z1 - side);

                const bool ns = nN || nS, ew = nW || nE;
                const float ly = 0.03f;  // markings float just above the asphalt
                if (ns && ew) {
                    // Intersection: zebra crosswalk bands across each entrance.
                    auto zebra = [&](bool alongX, float edge) {
                        for (int i = 0; i < 4; ++i) {
                            const float o = -0.21f + 0.14f * static_cast<float>(i) + 0.02f;
                            if (alongX) {
                                builder.addQuad({cx + o, ly, edge}, {cx + o + 0.10f, ly, edge},
                                                {cx + o + 0.10f, ly, edge + 0.05f},
                                                {cx + o, ly, edge + 0.05f}, kCrosswalk);
                            } else {
                                builder.addQuad({edge, ly, cz + o}, {edge + 0.05f, ly, cz + o},
                                                {edge + 0.05f, ly, cz + o + 0.10f},
                                                {edge, ly, cz + o + 0.10f}, kCrosswalk);
                            }
                        }
                    };
                    if (nN) zebra(true, z0 + 0.04f);
                    if (nS) zebra(true, z1 - 0.09f);
                    if (nW) zebra(false, x0 + 0.04f);
                    if (nE) zebra(false, x1 - 0.09f);
                } else if (ns) {
                    // Straight N-S run: dashed yellow centre line.
                    const float lw = ts * 0.032f;
                    for (int i = 0; i < 3; ++i) {
                        const float dz0 = z0 + ts * (0.10f + 0.34f * static_cast<float>(i));
                        const float dz1 = dz0 + ts * 0.15f;
                        builder.addQuad({cx - lw, ly, dz0}, {cx + lw, ly, dz0},
                                        {cx + lw, ly, dz1}, {cx - lw, ly, dz1}, kLaneDash);
                    }
                } else if (ew) {
                    const float lw = ts * 0.032f;
                    for (int i = 0; i < 3; ++i) {
                        const float dx0 = x0 + ts * (0.10f + 0.34f * static_cast<float>(i));
                        const float dx1 = dx0 + ts * 0.15f;
                        builder.addQuad({dx0, ly, cz - lw}, {dx1, ly, cz - lw},
                                        {dx1, ly, cz + lw}, {dx0, ly, cz + lw}, kLaneDash);
                    }
                }

                // Utility corridor: poles + wire spans follow the north/west
                // sidewalk band of every road tile the power grid actually
                // reaches (t.poweredRoad, flood-filled from each plant in
                // recomputeStats). Rail positions are fixed fractions of the
                // tile edge, so segments in the same row/column join up into
                // one continuous line rather than looking tile-stamped.
                if (t.poweredRoad) {
                    const bool pnN = inBounds(c, r - 1) && tile(c, r - 1).road && tile(c, r - 1).poweredRoad;
                    const bool pnS = inBounds(c, r + 1) && tile(c, r + 1).road && tile(c, r + 1).poweredRoad;
                    const bool pnW = inBounds(c - 1, r) && tile(c - 1, r).road && tile(c - 1, r).poweredRoad;
                    const bool pnE = inBounds(c + 1, r) && tile(c + 1, r).road && tile(c + 1, r).poweredRoad;
                    const float rail = ts * 0.10f;   // inset from the tile edge, inside the sidewalk band
                    const float railX = x0 + rail;   // constant per column: vertical spans align tile-to-tile
                    const float railZ = z0 + rail;   // constant per row: horizontal spans align tile-to-tile
                    const float wireY = 0.30f, wireT = 0.010f, wireHalf = ts * 0.010f;
                    if (pnN || pnS) {
                        const float zA = pnN ? z0 : cz, zB = pnS ? z1 : cz;
                        addBox(builder, railX - wireHalf, zA, railX + wireHalf, zB, wireY, wireY + wireT,
                              kWire);
                    }
                    if (pnW || pnE) {
                        const float xA = pnW ? x0 : cx, xB = pnE ? x1 : cx;
                        addBox(builder, xA, railZ - wireHalf, xB, railZ + wireHalf, wireY, wireY + wireT,
                              kWire);
                    }
                    if ((pnN || pnS || pnW || pnE) && (c + r) % 2 == 0) {
                        const std::uint32_t pv = tileHash(c, r, 0x901EDu) % kPoleVariants;
                        procgen::appendTriMesh(cachedPowerPole(pv), {railX, sy, railZ},
                                               {1.0f, 1.0f, 1.0f}, scene);
                    }
                }

                // Streetlamps along the opposite (south/east) sidewalk —
                // ambience only, present on any road regardless of power.
                if (tileHash(c, r, 0x1A4Fu) % 1000u < 340u) {
                    const std::uint32_t lv = tileHash(c, r, 0x7A4Fu) % kLampVariants;
                    procgen::appendTriMesh(cachedStreetlamp(lv), {x1 - ts * 0.10f, sy, z1 - ts * 0.10f},
                                           {1.0f, 1.0f, 1.0f}, scene);
                }
                continue;
            }

            if (t.building != Building::None) {
                if (!t.bldgOrigin) continue;  // drawn by the origin cell
                const int fp = std::max<int>(1, t.footprint);
                const float bx1 = x0 + fp * ts, bz1 = z0 + fp * ts;
                const float pad = ts * 0.10f;
                const float height = buildingHeight(t.building);
                const UiColor roofCol = mix(buildingRoof(t.building), UiColor(0, 0, 0, 1), 0.12f);
                addBox(builder, x0 + pad, z0 + pad, bx1 - pad, bz1 - pad, 0.0f, height, roofCol);
                if (t.building == Building::Park) {
                    // Proper procgen trees so the lone-tile park doesn't read as a slab.
                    for (int i = 0; i < 4; ++i) {
                        const float px = x0 + (0.28f + 0.45f * (i & 1)) * (bx1 - x0);
                        const float pz = z0 + (0.28f + 0.45f * (i >> 1)) * (bz1 - z0);
                        const std::uint32_t tv = tileHash(c, r, 0x9A7Cu + static_cast<std::uint32_t>(i));
                        procgen::appendTriMesh(cachedTree(tv % kTreeVariants), {px, height, pz},
                                               {1.0f, 1.0f, 1.0f}, scene);
                    }
                }
                continue;
            }

            if (t.zone != Zone::None) {
                if (t.charred) {
                    // Burnt-out lot: a low ash slab plus a couple of debris
                    // chunks, hash-jittered so a burned block reads as ruin,
                    // not a tidy grid of grey tiles.
                    const UiColor ash  = UiColor::fromRgbHex(0x2E2A26);
                    const UiColor soot = UiColor::fromRgbHex(0x453D34);
                    const float pad = ts * 0.10f;
                    addBox(builder, x0 + pad, z0 + pad, x1 - pad, z1 - pad, 0.0f, 0.09f, ash);
                    const std::uint32_t h = tileHash(c, r, 0xC1DE7u);
                    for (int i = 0; i < 2; ++i) {
                        const std::uint32_t hi = h ^ (0x9E3779B9u * static_cast<std::uint32_t>(i + 1));
                        const float dx = 0.22f + 0.42f * static_cast<float>(hi & 0xffu) / 255.0f;
                        const float dz = 0.22f + 0.42f * static_cast<float>((hi >> 8) & 0xffu) / 255.0f;
                        const float dh = 0.16f + 0.16f * static_cast<float>((hi >> 16) & 0xffu) / 255.0f;
                        addBox(builder, x0 + dx * ts - 0.08f, z0 + dz * ts - 0.08f,
                               x0 + dx * ts + 0.08f, z0 + dz * ts + 0.08f, 0.0f, dh, soot);
                    }
                    continue;
                }
                const UiColor zc = t.zone == Zone::Residential ? kZoneR
                                   : t.zone == Zone::Commercial ? kZoneC
                                                                : kZoneI;
                const short pi = plotIndex[static_cast<std::size_t>(r) * kGridW + c];
                const Plot* plot = pi >= 0 ? &plots[static_cast<std::size_t>(pi)] : nullptr;
                const float dev = plot ? plot->develop : t.develop;
                if (dev > kDevEps) {
                    // Era-styled CSG building, one per plot, drawn by the
                    // plot's origin tile. The development level picks the
                    // architectural era (the city visibly modernizes as
                    // parcels densify), land value picks the residential
                    // wealth tier, and a stable per-tile hash picks the
                    // variant — so a plot keeps its building identity across
                    // rebuilds with no extra stored state.
                    if (plot && (plot->c != c || plot->r != r)) continue;  // member tile: covered by origin
                    const int pw = plot ? plot->w : 1, pd = plot ? plot->d : 1;
                    const int level = 1 + std::min(2, static_cast<int>(dev));
                    const auto kind = t.zone == Zone::Residential ? procgen::BuildingKind::Residential
                                      : t.zone == Zone::Commercial ? procgen::BuildingKind::Commercial
                                                                   : procgen::BuildingKind::Industrial;
                    const float desirability = plot ? plot->desirability : t.desirability;
                    const int tier = t.zone == Zone::Residential ? residentialTier(desirability) : 1;
                    const std::uint32_t variant = tileHash(c, r, 0xB17D5EEDu) % 8u;
                    procgen::Color3 tint{1.0f, 1.0f, 1.0f};
                    const bool powered = plot ? plot->powered : t.powered;
                    if (!powered) tint = procgen::Color3{0.55f, 0.45f, 0.40f};  // brown-out tint
                    if (t.fireTicks > 0) tint = procgen::Color3{0.42f, 0.28f, 0.20f};  // scorched

                    // Face the door toward a road on the plot perimeter
                    // (hash-picked when it fronts several). Turn k rotates
                    // local -Z to: 0 -> -Z (north), 1 -> -X (west),
                    // 2 -> +Z (south), 3 -> +X (east).
                    const auto anyRoad = [&](int cc0, int rr0, int cc1, int rr1) {
                        for (int rr = rr0; rr <= rr1; ++rr)
                            for (int cc = cc0; cc <= cc1; ++cc)
                                if (inBounds(cc, rr) && tile(cc, rr).road) return true;
                        return false;
                    };
                    int facings[4];
                    int numFacings = 0;
                    if (anyRoad(c, r - 1, c + pw - 1, r - 1)) facings[numFacings++] = 0;
                    if (anyRoad(c - 1, r, c - 1, r + pd - 1)) facings[numFacings++] = 1;
                    if (anyRoad(c, r + pd, c + pw - 1, r + pd)) facings[numFacings++] = 2;
                    if (anyRoad(c + pw, r, c + pw, r + pd - 1)) facings[numFacings++] = 3;
                    const std::uint32_t fh = tileHash(c, r, 0xFACE5u);
                    const int turns = numFacings > 0
                                          ? facings[fh % static_cast<std::uint32_t>(numFacings)]
                                          : static_cast<int>(fh & 3u);

                    // Odd turns need the building generated with swapped lot
                    // extents so the quarter-turn lands it back on the plot;
                    // the per-turn offsets rebase the origin-rotated mesh
                    // onto the lot rectangle.
                    const bool swapDims = (turns & 1) != 0;
                    const procgen::TriMesh& bm =
                        cachedBuilding(kind, level, tier, variant, pw, pd, swapDims);
                    const float pad = ts * 0.10f;
                    const float lotX = x0 + pad, lotZ = z0 + pad;
                    const float lotW = pw * ts - 2.0f * pad, lotD = pd * ts - 2.0f * pad;
                    const float genW = swapDims ? lotD : lotW;
                    const float genD = swapDims ? lotW : lotD;
                    switch (turns) {
                        case 1:
                            procgen::appendTriMeshRotated(bm, {lotX, 0.0f, lotZ + genW}, 1,
                                                          {0.0f, 0.0f, 0.0f}, tint, scene);
                            break;
                        case 2:
                            procgen::appendTriMeshRotated(bm, {lotX + genW, 0.0f, lotZ + genD}, 2,
                                                          {0.0f, 0.0f, 0.0f}, tint, scene);
                            break;
                        case 3:
                            procgen::appendTriMeshRotated(bm, {lotX + genD, 0.0f, lotZ}, 3,
                                                          {0.0f, 0.0f, 0.0f}, tint, scene);
                            break;
                        default:
                            procgen::appendTriMesh(bm, {lotX, 0.0f, lotZ}, tint, scene);
                            break;
                    }
                } else {
                    // Zoned but undeveloped: a faint tinted marker flush with the ground.
                    const UiColor tint = mix(seasonalGrass(mix(kGrassAlt, kGrass, t.scenicPhase)), zc, 0.35f);
                    const float pad = ts * 0.08f;
                    builder.addQuad({x0 + pad, 0.015f, z0 + pad}, {x1 - pad, 0.015f, z0 + pad},
                                    {x1 - pad, 0.015f, z1 - pad}, {x0 + pad, 0.015f, z1 - pad}, tint);
                }
            } else {
                // Pure grass parcel: deterministic ambient trees. Parcels that
                // face a road get street trees at a high rate; open meadow
                // gets a sparse scatter. Hash-driven, so placement is stable
                // across rebuilds without stored state.
                const bool byRoad = (inBounds(c, r - 1) && tile(c, r - 1).road) ||
                                    (inBounds(c, r + 1) && tile(c, r + 1).road) ||
                                    (inBounds(c - 1, r) && tile(c - 1, r).road) ||
                                    (inBounds(c + 1, r) && tile(c + 1, r).road);
                const std::uint32_t h = tileHash(c, r, 0x7EE0F00Du);
                if (h % 1000u < (byRoad ? 450u : 90u)) {
                    const std::uint32_t variant = (h >> 10) % kTreeVariants;
                    const float jx = 0.30f + 0.40f * static_cast<float>((h >> 13) & 0xffu) / 255.0f;
                    const float jz = 0.30f + 0.40f * static_cast<float>((h >> 21) & 0xffu) / 255.0f;
                    procgen::appendTriMesh(cachedTree(variant), {x0 + jx * ts, 0.0f, z0 + jz * ts},
                                           {1.0f, 1.0f, 1.0f}, scene);
                    // A second tree on some road-facing parcels reads as a planted row.
                    if (byRoad && (h & 1u)) {
                        procgen::appendTriMesh(cachedTree((variant + 3u) % kTreeVariants),
                                               {x0 + (1.0f - jx) * ts, 0.0f, z0 + (1.0f - jz) * ts},
                                               {1.0f, 1.0f, 1.0f}, scene);
                    }
                }
                // Autumn: pumpkins appear in yards along the streets — a
                // little cluster per lucky parcel.
                const std::uint32_t hp = tileHash(c, r, 0xF00D5EEDu);
                if (autumn && hp % 1000u < (byRoad ? 280u : 50u)) {
                    const int count = 1 + static_cast<int>((hp >> 12) % 3u);
                    for (int i = 0; i < count; ++i) {
                        const std::uint32_t hi = hp ^ (0x9E3779B9u * static_cast<std::uint32_t>(i + 1));
                        const float px = 0.18f + 0.64f * static_cast<float>(hi & 0xffu) / 255.0f;
                        const float pz = 0.18f + 0.64f * static_cast<float>((hi >> 8) & 0xffu) / 255.0f;
                        procgen::appendTriMesh(cachedPumpkin(hi >> 16),
                                               {x0 + px * ts, 0.0f, z0 + pz * ts},
                                               {1.0f, 1.0f, 1.0f}, scene);
                    }
                }
            }
        }
    }

    if (!scene.packedIndices.empty()) {
        scene.packedDraws.push_back(
            ImportedScenePackedDraw{0, static_cast<std::uint32_t>(scene.packedIndices.size())});
    }
    return scene;
}

const procgen::TriMesh& CityBuilderApp::cachedBuilding(procgen::BuildingKind kind, int level,
                                                       int tier, std::uint32_t variant, int plotW,
                                                       int plotD, bool swapDims) const {
    const std::uint32_t key = ((static_cast<std::uint32_t>(kind) & 0xfu) << 24) |
                              ((static_cast<std::uint32_t>(level) & 0xfu) << 20) |
                              ((static_cast<std::uint32_t>(tier) & 0xfu) << 16) |
                              ((variant & 0xfu) << 12) |
                              ((static_cast<std::uint32_t>(plotW) & 0xfu) << 8) |
                              ((static_cast<std::uint32_t>(plotD) & 0xfu) << 4) |
                              (swapDims ? 1u : 0u);
    const auto it = m_buildingCache.find(key);
    if (it != m_buildingCache.end()) {
        return it->second;
    }
    procgen::BuildingDesc desc;
    // Development level doubles as the architectural era: parcels start as
    // 1890s brick, densify into 1930s deco setbacks, and top out as 1960s
    // curtain-wall modernism.
    desc.era = level <= 1   ? procgen::Era::E1890s
               : level == 2 ? procgen::Era::E1930s
                            : procgen::Era::E1960s;
    desc.kind = kind;
    desc.level = level;
    desc.wealthTier = tier;
    const float pad = kTileWorldSize * 0.10f;
    const float lotW = static_cast<float>(plotW) * kTileWorldSize - 2.0f * pad;
    const float lotD = static_cast<float>(plotD) * kTileWorldSize - 2.0f * pad;
    desc.lotWidth = swapDims ? lotD : lotW;
    desc.lotDepth = swapDims ? lotW : lotD;
    desc.seed = key * 0x9E3779B9u;
    return m_buildingCache.emplace(key, procgen::generateBuilding(desc)).first->second;
}

const procgen::TriMesh& CityBuilderApp::cachedTree(std::uint32_t variant) const {
    variant %= kTreeVariants;
    const std::uint32_t key = (static_cast<std::uint32_t>(m_season) << 8) | variant;
    const auto it = m_treeCache.find(key);
    if (it != m_treeCache.end()) {
        return it->second;
    }
    return m_treeCache
        .emplace(key, procgen::generateTree(variant, 0xA11CE5u + variant * 977u, m_season))
        .first->second;
}

const procgen::TriMesh& CityBuilderApp::cachedPumpkin(std::uint32_t variant) const {
    constexpr std::uint32_t kPumpkinVariants = 4;
    if (m_pumpkinMeshes.empty()) {
        m_pumpkinMeshes.reserve(kPumpkinVariants);
        for (std::uint32_t i = 0; i < kPumpkinVariants; ++i) {
            m_pumpkinMeshes.push_back(procgen::generatePumpkin(0xF00Du + i * 131u));
        }
    }
    return m_pumpkinMeshes[variant % kPumpkinVariants];
}

const procgen::TriMesh& CityBuilderApp::cachedPowerPole(std::uint32_t variant) const {
    if (m_poleMeshes.empty()) {
        m_poleMeshes.reserve(kPoleVariants);
        for (std::uint32_t i = 0; i < kPoleVariants; ++i) {
            m_poleMeshes.push_back(procgen::generatePowerPole(0xB01Eu + i * 197u));
        }
    }
    return m_poleMeshes[variant % kPoleVariants];
}

const procgen::TriMesh& CityBuilderApp::cachedStreetlamp(std::uint32_t variant) const {
    if (m_lampMeshes.empty()) {
        m_lampMeshes.reserve(kLampVariants);
        for (std::uint32_t i = 0; i < kLampVariants; ++i) {
            m_lampMeshes.push_back(procgen::generateStreetlamp(0x7A4Fu + i * 211u));
        }
    }
    return m_lampMeshes[variant % kLampVariants];
}

// ─────────────────────────────────────────────────────────────────────────────
// Ambient traffic
// ─────────────────────────────────────────────────────────────────────────────
namespace {
// Right-hand lane offset from the road centre, in tile units. Asphalt spans
// 0.16..0.84 across a tile, so lane centres sit at ±0.17 from the middle.
constexpr float kLaneOffset = 0.17f;
// Fleet size tracks population (not road count) so traffic is a truthful
// signal of city activity — an empty grid of roads stays quiet.
constexpr int kMaxCars = 72;
constexpr int kCarsBase = 4;
constexpr int kPopPerCar = 55;
}  // namespace

void CityBuilderApp::respawnVehicle(Vehicle& v) {
    // Weighted reservoir-sample over road tiles (no stored road list): a road's
    // weight is the development around it, so downtown streets are busy and a
    // dead industrial cul-de-sac stays quiet — traffic as a truthful heat map.
    float totalW = 0.0f;
    short pickC = -1, pickR = -1;
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            if (!tile(c, r).road) continue;
            float dev = 0.0f;
            for (int dr = -1; dr <= 1; ++dr)
                for (int dc = -1; dc <= 1; ++dc)
                    if (inBounds(c + dc, r + dr)) dev += tile(c + dc, r + dr).develop;
            const float w = 0.3f + dev;
            totalW += w;
            m_trafficRng = m_trafficRng * 1664525u + 1013904223u;
            const float roll = static_cast<float>((m_trafficRng >> 8) & 0xFFFFFFu) / 16777216.0f;
            if (roll <= w / totalW) {
                pickC = static_cast<short>(c);
                pickR = static_cast<short>(r);
            }
        }
    }
    v.cx = pickC;
    v.cr = pickR;
    v.t = 0.0f;
    m_trafficRng = m_trafficRng * 1664525u + 1013904223u;
    v.variant = static_cast<std::uint8_t>((m_trafficRng >> 8) % kCarVariants);
    v.speed = 1.0f + 0.5f * static_cast<float>((m_trafficRng >> 16) & 0xffu) / 255.0f;
    v.inX = 1;
    v.inZ = 0;
    if (pickC >= 0) {
        pickExit(v);
        v.inX = v.outX;
        v.inZ = v.outZ;
    }
}

bool CityBuilderApp::pickExit(Vehicle& v) {
    struct Dir {
        signed char x, z;
    };
    constexpr Dir kDirs[4] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    Dir options[4];
    int optionCount = 0;
    for (const Dir& d : kDirs) {
        if (d.x == -v.inX && d.z == -v.inZ) continue;  // no U-turns unless dead end
        if (!inBounds(v.cx + d.x, v.cr + d.z) || !tile(v.cx + d.x, v.cr + d.z).road) continue;
        options[optionCount++] = d;
    }
    if (optionCount == 0) {
        // Dead end: turn back if the road behind still exists.
        if (inBounds(v.cx - v.inX, v.cr - v.inZ) && tile(v.cx - v.inX, v.cr - v.inZ).road) {
            v.outX = static_cast<signed char>(-v.inX);
            v.outZ = static_cast<signed char>(-v.inZ);
            return true;
        }
        return false;
    }
    // Commute-ish routing: weight each exit by the development around its
    // destination tile (cars drift toward busy districts) with a straight-on
    // preference so traffic still flows instead of jittering at every corner.
    float weights[4];
    float totalW = 0.0f;
    for (int i = 0; i < optionCount; ++i) {
        const Dir d = options[i];
        const int tc = v.cx + d.x, tr = v.cr + d.z;
        float dev = 0.0f;
        static constexpr int kDc[4] = {1, -1, 0, 0};
        static constexpr int kDr[4] = {0, 0, 1, -1};
        for (int k = 0; k < 4; ++k)
            if (inBounds(tc + kDc[k], tr + kDr[k])) dev += tile(tc + kDc[k], tr + kDr[k]).develop;
        float w = 0.4f + 0.3f * dev;
        if (d.x == v.inX && d.z == v.inZ) w *= 2.0f;  // keep-straight bias
        weights[i] = w;
        totalW += w;
    }
    m_trafficRng = m_trafficRng * 1664525u + 1013904223u;
    float roll = static_cast<float>((m_trafficRng >> 8) & 0xFFFFFFu) / 16777216.0f * totalW;
    int chosen = optionCount - 1;
    for (int i = 0; i < optionCount; ++i) {
        if (roll < weights[i]) { chosen = i; break; }
        roll -= weights[i];
    }
    v.outX = options[chosen].x;
    v.outZ = options[chosen].z;
    return true;
}

void CityBuilderApp::updateVehicles(float dt) {
    // Fleet size tracks population, capped by the road network, so traffic
    // density is a readout of how much city there actually is.
    const int target = std::min({kMaxCars, m_numRoad / 2,
                                 kCarsBase + m_population / kPopPerCar});
    while (static_cast<int>(m_vehicles.size()) < target) {
        Vehicle v;
        respawnVehicle(v);
        if (v.cx < 0) break;  // no roads at all
        m_vehicles.push_back(v);
    }
    if (static_cast<int>(m_vehicles.size()) > target) {
        m_vehicles.resize(static_cast<std::size_t>(target));
    }

    // trafficLoad is an exponential moving average of car-seconds spent on
    // each road tile: cars deposit dt below, the whole field decays here.
    // computeDesirability reads it as a nuisance source on jammed roads.
    const float decay = 1.0f - kCongestionDecay * dt;
    for (Tile& t : m_tiles) t.trafficLoad *= decay;

    for (Vehicle& v : m_vehicles) {
        // Road bulldozed underneath: find a new home.
        if (!inBounds(v.cx, v.cr) || !tile(v.cx, v.cr).road) {
            respawnVehicle(v);
            if (v.cx < 0) continue;
        }
        Tile& here = tile(v.cx, v.cr);
        here.trafficLoad += dt;
        // Congestion is visible: cars slow down and queue on loaded tiles.
        const float jam = std::max(0.0f, here.trafficLoad - kCongestionStart);
        const float mul = std::max(0.4f, 1.0f / (1.0f + 0.35f * jam));
        v.t += v.speed * mul * dt;
        int guard = 0;
        while (v.t >= 1.0f && guard++ < 4) {
            v.t -= 1.0f;
            v.cx = static_cast<short>(v.cx + v.outX);
            v.cr = static_cast<short>(v.cr + v.outZ);
            v.inX = v.outX;
            v.inZ = v.outZ;
            if (!inBounds(v.cx, v.cr) || !tile(v.cx, v.cr).road || !pickExit(v)) {
                respawnVehicle(v);
                break;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Weather
// ─────────────────────────────────────────────────────────────────────────────
namespace {
constexpr int kMaxRainDrops = 520;
constexpr int kMaxSnowDrops = 380;
}  // namespace

void CityBuilderApp::respawnDrop(WeatherDrop& d, bool atTop) {
    m_weatherRng = m_weatherRng * 1664525u + 1013904223u;
    const std::uint32_t h = m_weatherRng >> 8;
    // Spawn box tracks the camera focus so precipitation always fills the view.
    const float span = m_camZoom * 1.25f + 4.0f;
    d.x = m_camFocusX + span * (static_cast<float>(h & 0x3ffu) / 511.5f - 1.0f);
    d.z = m_camFocusZ + span * (static_cast<float>((h >> 10) & 0x3ffu) / 511.5f - 1.0f);
    d.phase = static_cast<float>((h >> 20) & 0xffu) * 0.0246f;
    d.speed = 0.8f + 0.4f * static_cast<float>((h >> 4) & 0xffu) / 255.0f;
    d.y = atTop ? 5.5f + 2.5f * static_cast<float>((h >> 14) & 0xffu) / 255.0f
                : 8.0f * static_cast<float>((h >> 6) & 0x3ffu) / 1023.0f;
}

void CityBuilderApp::updateWeather(float dt) {
    m_weatherTimer -= dt;
    if (m_weatherTimer <= 0.0f) {
        m_weatherRng = m_weatherRng * 1664525u + 1013904223u;
        const std::uint32_t roll = (m_weatherRng >> 8) % 100u;
        std::uint32_t wetChance = 25u;
        switch (m_season) {
            case procgen::Season::Spring: wetChance = 40u; break;
            case procgen::Season::Summer: wetChance = 22u; break;
            case procgen::Season::Autumn: wetChance = 38u; break;
            case procgen::Season::Winter: wetChance = 45u; break;
        }
        const bool wet = roll < wetChance;
        m_weatherTarget = !wet ? Weather::Clear
                               : (m_season == procgen::Season::Winter ? Weather::Snow : Weather::Rain);
        m_weatherTimer = 18.0f + static_cast<float>((m_weatherRng >> 16) % 22u);
    }

    // Intensity ramps so storms roll in and clear out instead of popping.
    if (m_weatherTarget != Weather::Clear) m_weather = m_weatherTarget;
    const float target = m_weatherTarget == Weather::Clear ? 0.0f : 1.0f;
    const float step = 0.5f * dt;
    m_weatherIntensity += std::clamp(target - m_weatherIntensity, -step, step);
    if (m_weatherTarget == Weather::Clear && m_weatherIntensity <= 0.01f) {
        m_weather = Weather::Clear;
    }

    const bool snow = m_weather == Weather::Snow;
    const int maxDrops = snow ? kMaxSnowDrops : kMaxRainDrops;
    const int active = static_cast<int>(m_weatherIntensity * static_cast<float>(maxDrops));
    while (static_cast<int>(m_drops.size()) < active) {
        WeatherDrop d;
        respawnDrop(d, false);
        m_drops.push_back(d);
    }
    if (static_cast<int>(m_drops.size()) > active) {
        m_drops.resize(static_cast<std::size_t>(active));
    }

    const float fall = snow ? 1.1f : 7.5f;
    for (WeatherDrop& d : m_drops) {
        d.y -= fall * d.speed * dt;
        if (snow) {
            // Lazy sinusoidal sway so flakes drift instead of plummeting.
            d.x += std::sin(m_time * 1.7f + d.phase) * 0.35f * dt;
            d.z += std::cos(m_time * 1.3f + d.phase) * 0.25f * dt;
        }
        if (d.y < 0.0f) respawnDrop(d, true);
    }
}

render::ImportedActorFrameData CityBuilderApp::buildActorFrameData() {
    m_actorVertices.clear();
    m_actorIndices.clear();
    m_actorDraws.clear();
    if (m_carMeshes.empty()) {
        m_carMeshes.reserve(kCarVariants);
        for (std::uint32_t i = 0; i < kCarVariants; ++i) {
            m_carMeshes.push_back(procgen::generateVehicle(0xCA5133Du + i * 7919u));
        }
    }

    const float ts = kTileWorldSize;
    for (const Vehicle& v : m_vehicles) {
        if (v.cx < 0) continue;
        // Quadratic bezier across the tile: entry/exit points sit on the
        // right-hand lane of the incoming/outgoing directions; the control
        // point is the intersection of the two lane lines, which folds turns
        // into smooth arcs and leaves straights linear.
        const float cx = (static_cast<float>(v.cx) + 0.5f) * ts;
        const float cz = (static_cast<float>(v.cr) + 0.5f) * ts;
        const float inX = static_cast<float>(v.inX), inZ = static_cast<float>(v.inZ);
        const float outX = static_cast<float>(v.outX), outZ = static_cast<float>(v.outZ);
        // Right-hand perpendicular of a direction (x,z) is (-z, x).
        const float p0x = cx - 0.5f * inX * ts + (-inZ) * kLaneOffset * ts;
        const float p0z = cz - 0.5f * inZ * ts + (inX)*kLaneOffset * ts;
        const float p2x = cx + 0.5f * outX * ts + (-outZ) * kLaneOffset * ts;
        const float p2z = cz + 0.5f * outZ * ts + (outX)*kLaneOffset * ts;
        float p1x, p1z;
        if (v.inX == v.outX && v.inZ == v.outZ) {
            p1x = 0.5f * (p0x + p2x);
            p1z = 0.5f * (p0z + p2z);
        } else {
            p1x = cx + (-inZ) * kLaneOffset * ts + (-outZ) * kLaneOffset * ts;
            p1z = cz + (inX)*kLaneOffset * ts + (outX)*kLaneOffset * ts;
        }
        const float t = v.t, u = 1.0f - t;
        const float px = u * u * p0x + 2.0f * u * t * p1x + t * t * p2x;
        const float pz = u * u * p0z + 2.0f * u * t * p1z + t * t * p2z;
        float vx = 2.0f * u * (p1x - p0x) + 2.0f * t * (p2x - p1x);
        float vz = 2.0f * u * (p1z - p0z) + 2.0f * t * (p2z - p1z);
        const float vlen = std::sqrt(vx * vx + vz * vz);
        if (vlen > 1e-5f) {
            vx /= vlen;
            vz /= vlen;
        } else {
            vx = inX;
            vz = inZ;
        }

        // Rotate the +X-facing car mesh onto the velocity heading and place it.
        const procgen::TriMesh& mesh = m_carMeshes[v.variant % m_carMeshes.size()];
        const std::uint32_t base = static_cast<std::uint32_t>(m_actorVertices.size());
        for (const ImportedScenePackedVertex& src : mesh.vertices) {
            ImportedScenePackedVertex dst = src;
            const float lx = src.position[0], lz = src.position[2];
            dst.position[0] = px + lx * vx - lz * vz;
            dst.position[2] = pz + lx * vz + lz * vx;
            dst.position[1] = src.position[1] + 0.02f;  // ride on the asphalt surface
            const float nx = src.normal[0], nz = src.normal[2];
            dst.normal[0] = nx * vx - nz * vz;
            dst.normal[2] = nx * vz + nz * vx;
            m_actorVertices.push_back(dst);
        }
        for (const std::uint32_t index : mesh.indices) {
            m_actorIndices.push_back(base + index);
        }
    }

    // Shared crossed-quad particle emitter: two vertical quads at right angles,
    // both windings so they survive backface culling from any yaw, lit as
    // upward-facing so they stay uniformly bright. Used by precipitation,
    // flames, and smoke — all stateless, driven by position + m_time.
    const auto pushCross = [&](float x, float y, float z, float halfW, float len, float cr,
                               float cg, float cb) {
        const auto pushQuad = [&](const Vector3& a, const Vector3& b, const Vector3& cVert,
                                  const Vector3& dVert) {
            const std::uint32_t base = static_cast<std::uint32_t>(m_actorVertices.size());
            for (const Vector3& p : {a, b, cVert, dVert}) {
                ImportedScenePackedVertex v{};
                v.position[0] = p.x;
                v.position[1] = p.y;
                v.position[2] = p.z;
                v.normal[1] = 1.0f;
                v.color[0] = cr;
                v.color[1] = cg;
                v.color[2] = cb;
                m_actorVertices.push_back(v);
            }
            for (const std::uint32_t i :
                 {base, base + 1u, base + 2u, base, base + 2u, base + 3u,
                  base, base + 2u, base + 1u, base, base + 3u, base + 2u}) {
                m_actorIndices.push_back(i);
            }
        };
        pushQuad({x - halfW, y, z}, {x + halfW, y, z}, {x + halfW, y + len, z},
                 {x - halfW, y + len, z});
        pushQuad({x, y, z - halfW}, {x, y, z + halfW}, {x, y + len, z + halfW},
                 {x, y + len, z - halfW});
    };
    const auto fract = [](float x) { return x - std::floor(x); };

    // Fires and working smoke: stateless per-tile particles keyed off the grid
    // each frame (no particle state to store — position, phase, and size all
    // derive from the tile hash and m_time).
    const float ts2 = kTileWorldSize;
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            const Tile& t = tile(c, r);
            const float x0 = c * ts2, z0 = r * ts2;
            if (t.fireTicks > 0) {
                // Flickering flame tongues plus dark smoke climbing off the roof.
                const std::uint32_t h0 = tileHash(c, r, 0xF1AEu);
                for (int i = 0; i < 4; ++i) {
                    const std::uint32_t hi = h0 ^ (0x9E3779B9u * static_cast<std::uint32_t>(i + 1));
                    const float fx = x0 + (0.20f + 0.60f * static_cast<float>(hi & 0xffu) / 255.0f) * ts2;
                    const float fz = z0 + (0.20f + 0.60f * static_cast<float>((hi >> 8) & 0xffu) / 255.0f) * ts2;
                    const float flick = 0.55f + 0.45f * std::sin(m_time * (5.0f + static_cast<float>(hi & 7u)) +
                                                                 static_cast<float>((hi >> 3) & 0xffu) * 0.1f);
                    const float fh = (0.45f + 0.35f * static_cast<float>((hi >> 16) & 0xffu) / 255.0f) *
                                     (0.6f + 0.4f * flick);
                    if ((i & 1) != 0) pushCross(fx, 0.06f, fz, 0.09f, fh, 1.0f, 0.72f, 0.14f);
                    else              pushCross(fx, 0.06f, fz, 0.11f, fh, 0.95f, 0.36f, 0.08f);
                }
                for (int i = 0; i < 2; ++i) {
                    const std::uint32_t hi = h0 ^ (0x5851F42Du * static_cast<std::uint32_t>(i + 1));
                    const float phase = static_cast<float>(hi & 0xffu) / 255.0f;
                    const float f = fract(m_time * 0.18f + phase);
                    const float sx = x0 + 0.5f * ts2 + std::sin(m_time * 0.6f + phase * 6.28f) * 0.15f * f;
                    const float sz = z0 + 0.5f * ts2;
                    const float g = 0.20f + 0.12f * f;  // dark soot brightening as it thins
                    pushCross(sx, 0.8f + f * 1.8f, sz, 0.10f + 0.14f * f, 0.16f + 0.10f * f, g, g, g);
                }
            } else if (t.zone == Zone::Industrial && t.develop > 1.1f && t.powered &&
                       tileHash(c, r, 0x580CEu) % 100u < 30u) {
                // Where there's smoke there's economic activity: powered,
                // developed industry puffs grey working smoke.
                const std::uint32_t h0 = tileHash(c, r, 0x5A0CEu);
                for (int i = 0; i < 3; ++i) {
                    const std::uint32_t hi = h0 ^ (0x9E3779B9u * static_cast<std::uint32_t>(i + 1));
                    const float phase = static_cast<float>(hi & 0xffu) / 255.0f + static_cast<float>(i) / 3.0f;
                    const float f = fract(m_time * 0.12f + phase);
                    const float sx = x0 + (0.35f + 0.30f * static_cast<float>((hi >> 8) & 0xffu) / 255.0f) * ts2;
                    const float sz = z0 + (0.35f + 0.30f * static_cast<float>((hi >> 16) & 0xffu) / 255.0f) * ts2;
                    const float baseY = 0.9f + 0.5f * std::min(2.0f, t.develop);
                    const float g = 0.48f + 0.18f * f;
                    pushCross(sx, baseY + f * 1.5f, sz, 0.05f + 0.09f * f, 0.09f + 0.07f * f, g, g, g);
                }
            } else if (t.bldgOrigin && t.building == Building::Power) {
                // The plant's steam column — a lighter, taller plume.
                const std::uint32_t h0 = tileHash(c, r, 0x57EAAu);
                for (int i = 0; i < 3; ++i) {
                    const std::uint32_t hi = h0 ^ (0x9E3779B9u * static_cast<std::uint32_t>(i + 1));
                    const float phase = static_cast<float>(hi & 0xffu) / 255.0f + static_cast<float>(i) / 3.0f;
                    const float f = fract(m_time * 0.10f + phase);
                    const float sx = x0 + 0.9f * ts2 + std::sin(m_time * 0.4f + phase * 6.28f) * 0.12f * f;
                    const float sz = z0 + 0.9f * ts2;
                    const float g = 0.72f + 0.14f * f;
                    pushCross(sx, 2.6f + f * 2.2f, sz, 0.10f + 0.20f * f, 0.14f + 0.12f * f, g, g, g);
                }
            }
        }
    }

    // Precipitation: thin tall streaks for rain, small flakes for snow.
    if (!m_drops.empty()) {
        const bool snow = m_weather == Weather::Snow;
        const float w = snow ? 0.014f : 0.0045f;
        const float len = snow ? 0.016f : 0.17f;
        const float cr = snow ? 0.93f : 0.60f;
        const float cg = snow ? 0.95f : 0.68f;
        const float cb = snow ? 0.98f : 0.78f;
        for (const WeatherDrop& d : m_drops) {
            pushCross(d.x, d.y, d.z, w, len, cr, cg, cb);
        }
    }

    if (!m_actorIndices.empty()) {
        m_actorDraws.push_back(
            ImportedScenePackedDraw{0, static_cast<std::uint32_t>(m_actorIndices.size())});
    }
    render::ImportedActorFrameData data;
    data.vertices = m_actorVertices;
    data.indices = m_actorIndices;
    data.draws = m_actorDraws;
    return data;
}

render::CameraPose CityBuilderApp::computeCameraPose() const {
    render::CameraPose cam{};
    cam.orthographic = false;
    cam.orthoHalfHeight = m_camZoom;  // kept meaningful for any ortho fallback math
    cam.yawDegrees = m_camYawDeg;
    cam.pitchDegrees = kCamPitchDeg;
    cam.fovDegrees = kDioramaFovDeg;

    const float yawR = odai::math::radians(cam.yawDegrees);
    const float pitchR = odai::math::radians(cam.pitchDegrees);
    const float cp = std::cos(pitchR);
    const Vector3 fwd{std::cos(yawR) * cp, std::sin(pitchR), std::sin(yawR) * cp};
    // Pull back until the narrow frustum's half-height at the focus point
    // equals m_camZoom, so the zoom control keeps its orthographic meaning.
    const float distance = m_camZoom / std::tan(odai::math::radians(kDioramaFovDeg) * 0.5f);
    cam.x = m_camFocusX - fwd.x * distance;
    cam.y = -fwd.y * distance;  // focus point sits on the y=0 ground plane
    cam.z = m_camFocusZ - fwd.z * distance;
    return cam;
}

bool CityBuilderApp::screenToGroundXZ(UiVec2 screenPx, const Layout& lo, float& outX, float& outZ) const {
    if (lo.fw <= 0.0f || lo.fh <= 0.0f) return false;
    const float yaw = odai::math::radians(m_camera.yawDegrees);
    const float pitch = odai::math::radians(m_camera.pitchDegrees);
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float cy = std::cos(yaw), sy = std::sin(yaw);
    const float fwdX = cy * cp, fwdY = sp, fwdZ = sy * cp;
    const float rgtX = -sy, rgtZ = cy;                     // rgtY is always 0
    const float upX = -cy * sp, upY = cp, upZ = -sy * sp;

    const float ndcX = screenPx.x / lo.fw * 2.0f - 1.0f;
    const float ndcY = 1.0f - screenPx.y / lo.fh * 2.0f;
    const float aspect = lo.fw / lo.fh;

    float roX, roY, roZ, rdX, rdY, rdZ;
    if (m_camera.orthographic) {
        const float halfH = m_camera.orthoHalfHeight;
        const float halfW = halfH * aspect;
        roX = m_camera.x + rgtX * ndcX * halfW + upX * ndcY * halfH;
        roY = m_camera.y + upY * ndcY * halfH;
        roZ = m_camera.z + rgtZ * ndcX * halfW + upZ * ndcY * halfH;
        rdX = fwdX;
        rdY = fwdY;
        rdZ = fwdZ;
    } else {
        // Perspective: rays fan out from the eye through the image plane,
        // matching perspectiveVulkan(fovY, aspect) in the renderer.
        const float tanHalf = std::tan(odai::math::radians(m_camera.fovDegrees) * 0.5f);
        roX = m_camera.x;
        roY = m_camera.y;
        roZ = m_camera.z;
        rdX = fwdX + rgtX * ndcX * tanHalf * aspect + upX * ndcY * tanHalf;
        rdY = fwdY + upY * ndcY * tanHalf;
        rdZ = fwdZ + rgtZ * ndcX * tanHalf * aspect + upZ * ndcY * tanHalf;
    }
    if (std::abs(rdY) < 1e-5f) return false;
    const float t = -roY / rdY;
    if (t < 0.0f) return false;
    outX = roX + t * rdX;
    outZ = roZ + t * rdZ;
    return true;
}

UiVec2 CityBuilderApp::worldToScreen(float wx, float wy, float wz, const Layout& lo) const {
    const float yaw = odai::math::radians(m_camera.yawDegrees);
    const float pitch = odai::math::radians(m_camera.pitchDegrees);
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float cy = std::cos(yaw), sy = std::sin(yaw);
    const float fwdX = cy * cp, fwdY = sp, fwdZ = sy * cp;
    const float rgtX = -sy, rgtZ = cy;                     // rgtY is always 0
    const float upX = -cy * sp, upY = cp, upZ = -sy * sp;

    const float dx = wx - m_camera.x, dy = wy - m_camera.y, dz = wz - m_camera.z;
    const float camX = dx * rgtX + dz * rgtZ;
    const float camY = dx * upX + dy * upY + dz * upZ;

    const float aspect = lo.fw / lo.fh;
    float ndcX = 0.0f, ndcY = 0.0f;
    if (m_camera.orthographic) {
        const float halfH = m_camera.orthoHalfHeight;
        const float halfW = halfH * aspect;
        ndcX = halfW > 0.0f ? camX / halfW : 0.0f;
        ndcY = halfH > 0.0f ? camY / halfH : 0.0f;
    } else {
        const float camFwd = dx * fwdX + dy * fwdY + dz * fwdZ;  // view depth
        if (camFwd > 1e-4f) {
            const float tanHalf = std::tan(odai::math::radians(m_camera.fovDegrees) * 0.5f);
            ndcX = camX / (camFwd * tanHalf * aspect);
            ndcY = camY / (camFwd * tanHalf);
        }
    }
    return UiVec2{(ndcX * 0.5f + 0.5f) * lo.fw, (1.0f - (ndcY * 0.5f + 0.5f)) * lo.fh};
}

void CityBuilderApp::drawWorldOverlay(const Layout& lo) {
    const float s = lo.s;

    // Influence preview: with a building tool selected, show the exact radius
    // its desirability splat will cover (and, for the power plant, its direct
    // powered radius) as rings projected onto the ground — the invisible
    // spatial rule becomes a visible promise before money is spent.
    const auto strokeGroundRing = [&](float cx, float cz, float radiusWorld, const UiColor& col,
                                      float widthPx) {
        constexpr int kSeg = 48;
        UiVec2 pts[kSeg];
        for (int i = 0; i < kSeg; ++i) {
            const float a = static_cast<float>(i) / kSeg * 6.2831853f;
            pts[i] = worldToScreen(cx + std::cos(a) * radiusWorld, 0.03f,
                                   cz + std::sin(a) * radiusWorld, lo);
        }
        m_uiDrawList.addPolylineAA(pts, kSeg, col, widthPx, true);
    };
    if (!m_mouseOverUi && m_hoverC >= 0) {
        const Building tb = toolBuilding(m_tool);
        if (tb != Building::None) {
            const int fp = footprintOf(tb);
            const float ccx = (static_cast<float>(m_hoverC) + fp * 0.5f) * kTileWorldSize;
            const float ccz = (static_cast<float>(m_hoverR) + fp * 0.5f) * kTileWorldSize;
            const InfluenceSpec inf = buildingInfluence(tb);
            if (inf.radius > 0) {
                const UiColor rc = inf.nuisance ? kBad : kTools[static_cast<int>(m_tool)].color;
                strokeGroundRing(ccx, ccz, inf.radius * kTileWorldSize, withA(rc, 0.85f), 2.0f * s);
            }
            if (tb == Building::Power) {
                strokeGroundRing(ccx, ccz, kPowerRadius * kTileWorldSize, withA(kGold, 0.9f),
                                 2.0f * s);
            }
        }
    }

    // Hover footprint preview: project the ground-plane quad corners to screen
    // and stroke them, since the tile is a parallelogram under the iso camera.
    if (!m_mouseOverUi && m_hoverC >= 0) {
        const Building b = toolBuilding(m_tool);
        const int fp = b != Building::None ? footprintOf(b) : 1;
        bool valid = true;
        for (int dy = 0; dy < fp && valid; ++dy)
            for (int dx = 0; dx < fp && valid; ++dx)
                valid = inBounds(m_hoverC + dx, m_hoverR + dy) &&
                        !(tile(m_hoverC + dx, m_hoverR + dy).terrain == Terrain::Water &&
                          m_tool != Tool::Bulldoze);
        const UiColor hc = valid ? kTools[static_cast<int>(m_tool)].color : kBad;
        const float x0 = m_hoverC * kTileWorldSize, z0 = m_hoverR * kTileWorldSize;
        const float x1 = x0 + fp * kTileWorldSize, z1 = z0 + fp * kTileWorldSize;
        const UiVec2 poly[4] = {
            worldToScreen(x0, 0.05f, z0, lo), worldToScreen(x1, 0.05f, z0, lo),
            worldToScreen(x1, 0.05f, z1, lo), worldToScreen(x0, 0.05f, z1, lo),
        };
        m_uiDrawList.addPolylineAA(poly, 4, withA(hc, 0.95f), 2.5f * s, true);
    }

    // Stalled-tile tooltip: name the one blocking cause so a parcel that refuses
    // to grow reads as "no road access" instead of an unexplained dead lot. A
    // parcel that isn't stalled instead names its character — the shop or
    // neighbourhood class living on that tile — so hovering a built lot tells
    // you what it actually is.
    if (!m_mouseOverUi && m_hoverC >= 0) {
        const Tile& ht = tile(m_hoverC, m_hoverR);
        if (ht.zone != Zone::None) {
            const char* cause = nullptr;
            if (ht.fireTicks > 0) cause = "ON FIRE!";
            else if (ht.charred) cause = "Burnt-out ruins";
            else if (!ht.nearRoad) cause = "No road access";
            else if (!ht.powered) cause = "No power";
            else if (ht.develop < 2.4f) {
                const float dem = ht.zone == Zone::Residential ? m_resDemand
                                  : ht.zone == Zone::Commercial ? m_comDemand
                                                                 : m_indDemand;
                if (dem < 0.18f) cause = "Low demand";
                else if (ht.desirability < 0.4f) cause = "Low land value";
            }
            std::string label;
            UiColor labelColor = kBad;
            if (cause) {
                label = cause;
            } else if (ht.develop > kDevEps) {
                labelColor = kText;
                if (ht.zone == Zone::Residential) {
                    label = residentialFlavorName(m_hoverC, m_hoverR, ht.desirability);
                } else if (ht.zone == Zone::Commercial) {
                    label = pickFlavor(kCommercialFlavors, kNumCommercialFlavors,
                                       m_hoverC, m_hoverR, 0xC0FFEE1Cu).name;
                } else {
                    label = pickFlavor(kIndustrialFlavors, kNumIndustrialFlavors,
                                       m_hoverC, m_hoverR, 0x57EE1u).name;
                }
            }
            if (!label.empty()) {
                const UiVec2 mp = m_uiInput.mousePx;
                const float tw = m_uiFont.measureText(label);
                const UiRect tip = UiRect::fromXYWH(mp.x + 16.0f * s, mp.y + 16.0f * s,
                                                    tw + 16.0f * s, 24.0f * s);
                m_uiDrawList.addRoundRectFilled(tip, withA(UiColor::fromRgbHex(0x14181F), 0.92f),
                                                5.0f * s);
                m_uiDrawList.addRoundRect(tip, kEdge, 5.0f * s, s);
                textCenter(m_uiFont, label, (tip.minX + tip.maxX) * 0.5f,
                           (tip.minY + tip.maxY) * 0.5f, labelColor);
            }
        }
    }

    // Floating building tags, projected from each building's roof-top centre —
    // the 3-D analogue of the scrim-backed tag drawn directly on the old 2-D roof.
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            const Tile& t = tile(c, r);
            if (t.building == Building::None || !t.bldgOrigin) continue;
            const char* tag = buildingTag(t.building);
            if (!tag || !*tag) continue;
            const int fp = std::max<int>(1, t.footprint);
            const float cx = (c + fp * 0.5f) * kTileWorldSize;
            const float cz = (r + fp * 0.5f) * kTileWorldSize;
            const UiVec2 sp = worldToScreen(cx, buildingHeight(t.building) + 0.12f, cz, lo);
            if (sp.x < 0.0f || sp.y < 0.0f || sp.x > lo.fw || sp.y > lo.fh) continue;
            const float tw = m_uiFontBold.measureText(tag);
            const float th = m_uiFontBold.lineHeightPx();
            const UiRect scrim = UiRect::fromXYWH(sp.x - tw * 0.5f - 4.0f * s, sp.y - th * 0.5f - 2.0f * s,
                                                  tw + 8.0f * s, th + 4.0f * s);
            m_uiDrawList.addRoundRectFilled(scrim, withA(UiColor(0, 0, 0, 1), 0.45f), 3.0f * s);
            textCenter(m_uiFontBold, tag, sp.x, sp.y, kText);
        }
    }

    // Land-value legend: a red→green gradient key so the overlay reads at a glance.
    if (m_showLandValue) {
        const float lw = 168.0f * s, lh = 12.0f * s;
        const float lx = lo.map.minX + 14.0f * s;
        const float ly = lo.map.minY + 32.0f * s;
        const UiRect chip = UiRect::fromXYWH(lx - 8.0f * s, ly - 24.0f * s, lw + 16.0f * s,
                                             lh + 44.0f * s);
        m_uiDrawList.addRoundRectFilled(chip, withA(kPanel, 0.92f), 7.0f * s);
        m_uiDrawList.addRoundRect(chip, kEdge, 7.0f * s, s);
        textLeft(m_uiFontBold, "Land Value", lx, ly - 12.0f * s, kText);
        const int seg = 24;
        for (int i = 0; i < seg; ++i) {
            const float f0 = static_cast<float>(i) / seg;
            m_uiDrawList.addRectFilled(
                UiRect::fromXYWH(lx + f0 * lw, ly, lw / seg + 1.0f, lh), heat(f0));
        }
        m_uiDrawList.addRect(UiRect::fromXYWH(lx, ly, lw, lh), kEdge, s);
        textLeft(m_uiFont, "poor", lx, ly + lh + 9.0f * s, kTextDim);
        textRight(m_uiFont, "prime", lx + lw, ly + lh + 9.0f * s, kTextDim);
    }
}

void CityBuilderApp::drawTopBar(const Layout& lo) {
    const float s = lo.s;
    m_uiDrawList.addRectFilled(lo.topBar, kPanel);
    m_uiDrawList.addRectFilled(UiRect{lo.topBar.minX, lo.topBar.maxY - 1.0f, lo.topBar.maxX,
                                      lo.topBar.maxY}, kEdge);
    const float cy = lo.topBar.minY + lo.topBar.height() * 0.5f;

    textLeft(m_uiFontBold, "OdaiCity", 16.0f * s, cy - 9.0f * s, kText);
    std::string date = std::string(kMonths[m_month]) + " · Year " + std::to_string(m_year) +
                       " · " + seasonName(m_season);
    if (m_weather == Weather::Rain) date += " · Rain";
    else if (m_weather == Weather::Snow) date += " · Snow";
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
    toolRow(Tool::Library);
    toolRow(Tool::Amphitheater);
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

    if (uiButton(UiRect::fromXYWH(r.maxX - 96.0f * s - 116.0f * s, by, 108.0f * s, bh),
                 "Land Value", m_showLandValue, kAccent)) {
        m_showLandValue = !m_showLandValue;
        m_sceneDirty = true;
    }

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
            else if (t.fireTicks > 0) col = UiColor::fromRgbHex(0xFF7A2E);  // burning: it pops
            else if (t.charred) col = UiColor::fromRgbHex(0x3A3430);
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

    // Camera viewport: unproject the four screen corners onto the ground plane
    // and draw the resulting quadrilateral — under the tilted iso camera it's
    // a parallelogram, not an axis-aligned rect. Deliberately NOT clamped per
    // corner to the grid range: the true view often extends past the map
    // edges, and clamping each corner independently warps the shape into
    // something that no longer matches the real frustum. The active clip
    // rect (pushClip above) already crops the outline to the minimap panel.
    const UiVec2 screenCorners[4] = {{0.0f, 0.0f}, {lo.fw, 0.0f}, {lo.fw, lo.fh}, {0.0f, lo.fh}};
    UiVec2 viewCorners[4];
    bool viewHit = true;
    for (int i = 0; i < 4; ++i) {
        float wx = 0.0f, wz = 0.0f;
        viewHit = viewHit && screenToGroundXZ(screenCorners[i], lo, wx, wz);
        viewCorners[i] = {ox + (wx / kTileWorldSize) * cell, oy + (wz / kTileWorldSize) * cell};
    }
    if (viewHit) {
        m_uiDrawList.addPolylineAA(viewCorners, 4, withA(UiColor(1, 1, 1, 1), 0.85f),
                                   std::max(1.0f, 1.5f * s), true);
    }
    m_uiDrawList.popClip();

    // Click to recentre the camera on the corresponding ground point.
    const UiRect inner = UiRect::fromXYWH(ox, oy, mapW, mapH);
    if (inner.contains(m_uiInput.mousePx) && m_uiInput.button(UiMouseButton::Left).down) {
        m_camFocusX = (m_uiInput.mousePx.x - ox) / cell * kTileWorldSize;
        m_camFocusZ = (m_uiInput.mousePx.y - oy) / cell * kTileWorldSize;
        clampCameraFocus();
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
    if (pct) {
        // Percent metrics move slowly month to month; pinning the axis to a fixed
        // 0-100% box makes a healthy, gently-trending city read as a dead flat
        // line. Zoom to the data with a floor on the span so real movement is
        // still visible, while a metric that genuinely swings keeps its full range.
        const float dataMn = mn, dataMx = mx;
        constexpr float kMinSpan = 20.0f;
        const float span = std::clamp((dataMx - dataMn) * 1.35f, kMinSpan, 100.0f);
        const float center = (dataMn + dataMx) * 0.5f;
        mn = std::clamp(center - span * 0.5f, 0.0f, 100.0f - span);
        mx = mn + span;
        if (mx > 100.0f) { mx = 100.0f; mn = std::max(0.0f, mx - span); }
    } else {
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
                                          area.maxY}, withA(mc, 0.20f));
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
