#include "games/citybuilder/citybuilder_app.h"

#include "math/math.h"
#include "procgen/city_terrain.h"
#include "procgen/civic_generator.h"
#include "procgen/props.h"
#include "procgen/rng.h"
#include "ui/font.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
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
// Tree species indices into procgen::generateTree: 0 broadleaf, 1 conifer,
// 2 birch, 3 poplar, 4 willow, 5 blossom, 6 oak, 7 yard shrub.
constexpr std::uint32_t kTreeVariants = 8;
constexpr std::uint32_t kCarVariants = 8;
constexpr std::uint32_t kPoleVariants = 3;
constexpr std::uint32_t kLampVariants = 3;

// ── Ambient agent tuning ─────────────────────────────────────────────────────
// Right-hand lane offset from the road centre, in tile units. Asphalt spans
// 0.16..0.84 across a tile, so lane centres sit at ±0.17 from the middle.
constexpr float kLaneOffset = 0.17f;
constexpr float kCarsPerRoadTile = 0.18f;
constexpr int kMaxCars = 44;
// Pedestrians keep to the sidewalk band (tile edge is at ±0.5).
constexpr float kWalkOffset = 0.42f;
constexpr int kMaxPedestrians = 60;
constexpr int kMaxBoats = 6;
constexpr std::uint32_t kPedVariants = 8;
constexpr std::uint32_t kBoatVariants = 4;
constexpr UiColor kAsphalt   = UiColor::fromRgbHex(0x303237);
constexpr UiColor kSidewalk  = UiColor::fromRgbHex(0x8E9092);
constexpr UiColor kBridgeStone = UiColor::fromRgbHex(0x9A968C);  // deck slab + pilings
constexpr UiColor kBoardwalk = UiColor::fromRgbHex(0xA8865A);    // seawall promenade planks
constexpr UiColor kSeawallStone = UiColor::fromRgbHex(0x8E8A80); // wall lip over the water
constexpr UiColor kRailIron  = UiColor::fromRgbHex(0x2A2C30);    // promenade / bridge railings
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

// Real seconds per simulated month at 1x speed. A slow, ambient pace: one
// month (a "day" in feel) takes about a minute, so a season (3 months) takes
// a few minutes rather than flashing by in a second.
constexpr float kMonthInterval = 60.0f;
constexpr float kDevEps        = 0.06f;
constexpr int   kHistMax       = 180;
// A freshly zoned lot sits "on the market" for a short while before a buyer
// bites and ground actually breaks — reads as the parcel getting sold rather
// than construction starting the instant you paint the zone. Measured in real
// seconds (accumulated every frame in onTick, scaled by game speed) rather
// than simulated months, so it stays a brief, fixed wait no matter how slow
// or fast a simulated month is tuned to run — the once-a-month economic
// heartbeat (kMonthInterval) is a separate knob from "how long until this
// specific lot finds a buyer."
constexpr float kZoneListingSeconds = 12.0f;

// The civic day/week clock. Deliberately separate from the economic month:
// one theatrical day lasts a real minute at 1x, so the whole weekly routine
// (commute waves, school bus, Mon/Thu trash day, Saturday soccer) cycles in
// about seven minutes of play.
constexpr float kDayLengthSeconds = 60.0f;
const char* const kWeekdays[7] = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
bool isTrashDay(int weekday) { return weekday == 0 || weekday == 3; }  // Mon & Thu

// Fixed isometric camera tilt (yaw rotates in 90 deg steps via Q/E).
constexpr float kCamPitchDeg = -52.0f;
// Diorama camera: a long lens from far away keeps the isometric composition
// but adds the subtle vanishing-line convergence that makes model cities read
// as miniatures. 20 deg is the renderer's minimum FOV clamp.
constexpr float kDioramaFovDeg = 20.0f;
constexpr float kCamMinZoom  = 6.0f;
constexpr float kCamMaxZoom  = 70.0f;

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

// Residential parcels don't get individual "shop names" — instead the whole
// block reads as a class of neighbourhood, driven by the existing land-value
// (desirability) field: poor land stays a trailer park / RV court even after
// it develops, prime land becomes an estate. tier 0 = low, 1 = mid, 2 = high.
// Names themselves come from the Lua namegen (see businessNameAt/blockNameAt).
int residentialTier(float desirability) {
    return desirability < 0.35f ? 0 : (desirability < 0.65f ? 1 : 2);
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

// Zoning and bulldozing default to a drag-a-rectangle-then-release gesture
// (apply once to the whole box) rather than painting every tile the cursor
// crosses — the natural way to lay out a block. Roads stay freeform paint
// since a road is a path you trace, not an area you fill.
bool isBoxSelectTool(Tool t) {
    return t == Tool::Bulldoze || t == Tool::ZoneR || t == Tool::ZoneC || t == Tool::ZoneI;
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

// Civic buildings are generated CSG meshes (see cachedCivic); the park keeps a
// flat green slab under its centerpiece and trees.
constexpr float kParkSlabHeight = 0.12f;

procgen::CivicKind civicKindOf(Building b) {
    switch (b) {
        case Building::Police: return procgen::CivicKind::Police;
        case Building::Fire:   return procgen::CivicKind::Fire;
        case Building::Clinic: return procgen::CivicKind::Clinic;
        case Building::School: return procgen::CivicKind::School;
        case Building::Park:   return procgen::CivicKind::Park;
        case Building::Library: return procgen::CivicKind::Library;
        case Building::Amphitheater: return procgen::CivicKind::Amphitheater;
        default:               return procgen::CivicKind::PowerPlant;
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

    // Lua content host (names, story templates, need schedules, tuning config).
    // Loads mods/citybuilder/scripts; a broken or missing script falls back to
    // compiled-in defaults, so this can't stop the game from booting.
    m_script = odai::citybuilder::createCityScriptHost();

    // World seed: ODAI_CITY_SEED for reproducible maps, else wall clock. All
    // generated content derives from this + position hashes.
    std::uint32_t seed = 0;
    if (const std::string seedEnv = readEnv("ODAI_CITY_SEED"); !seedEnv.empty()) {
        seed = static_cast<std::uint32_t>(std::strtoul(seedEnv.c_str(), nullptr, 10));
    }
    if (seed == 0) {
        seed = static_cast<std::uint32_t>(
            std::chrono::system_clock::now().time_since_epoch().count());
    }
    m_worldSeed = seed ? seed : 1u;
    m_rng = m_worldSeed;
    m_script->seedRng(m_worldSeed);
    m_cityName = m_script->cityName(m_worldSeed);
    std::printf("[citybuilder] world seed = %u (%s)\n", m_worldSeed, m_cityName.c_str());

    // ODAI_CITY_STORY=1: crank citizen event and trip rates for eyeball QA of
    // the ticker and routed traffic.
    if (const std::string story = readEnv("ODAI_CITY_STORY"); !story.empty() && story != "0") {
        m_storyBoost = 10.0f;
    }
    m_citizens.configure(m_script.get(), m_worldSeed, m_storyBoost);

    // Start the civic clock on a seeded weekday just before the morning rush,
    // so the first thing a new mayor sees is the town waking up.
    m_weekday = static_cast<int>(m_worldSeed % 7u);
    m_dayClock = kDayLengthSeconds * (6.8f / 24.0f);

    m_season = seasonForMonth(m_month);
    generateTerrain();
    seedCity();
    // ODAI_CITY_DEMO=1: force the seeded zone bands to development levels
    // 1/2/3 (south rows denser) so all three architectural eras — 1890s brick,
    // 1930s deco setbacks, 1960s curtain-wall — are on screen immediately.
    // Purely a dev/visual-verification aid; normal play grows into the same
    // levels over simulated months.
    if (const std::string demo = readEnv("ODAI_CITY_DEMO"); !demo.empty() && demo != "0") {
        for (int r = m_siteR - 3; r <= m_siteR + 3; ++r) {
            const float dev = r <= m_siteR - 2 ? 0.5f : (r <= m_siteR ? 1.5f : 2.5f);
            for (int c = m_siteC - 9; c <= m_siteC + 6; ++c) {
                if (!inBounds(c, r)) continue;
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

    procgen::CityTerrainDesc desc;
    desc.width = kGridW;
    desc.height = kGridH;
    desc.seed = m_worldSeed;
    if (m_script) {
        procgen::CityTerrainParams& p = desc.params;
        p.landMin = static_cast<float>(m_script->configNumber("terrain.land_min", p.landMin));
        p.riverWidthMin = static_cast<int>(
            m_script->configNumber("terrain.river_width_min", p.riverWidthMin));
        p.riverWidthMax = static_cast<int>(
            m_script->configNumber("terrain.river_width_max", p.riverWidthMax));
        p.lakeMax = static_cast<int>(m_script->configNumber("terrain.lake_max", p.lakeMax));
        p.coastChance =
            static_cast<float>(m_script->configNumber("terrain.coast_chance", p.coastChance));
        p.forestFreq =
            static_cast<float>(m_script->configNumber("terrain.forest_freq", p.forestFreq));
    }

    const procgen::CityTerrain terrain = procgen::generateCityTerrain(desc);
    if (!terrain.valid) {
        std::fprintf(stderr,
                     "[citybuilder] terrain invariants failed for seed %u; using best attempt\n",
                     m_worldSeed);
    }

    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            const std::size_t idx = static_cast<std::size_t>(r) * kGridW + c;
            Tile& t = tile(c, r);
            t = Tile{};
            t.scenicPhase = rnd();
            t.terrain = terrain.water[idx] != 0u ? Terrain::Water : Terrain::Grass;
            m_forest[idx] = terrain.forest[idx];
        }
    }
    m_riverPath = terrain.riverPath;
    m_siteC = terrain.siteC;
    m_siteR = terrain.siteR;
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
    // placeBuilding validates water/bounds/overlap and returns false; spiral
    // outward until a spot takes, so an awkward site still gets its civics.
    auto placeBuildingNear = [&](int c0, int r0, Building b) {
        if (placeBuilding(c0, r0, b)) return;
        for (int radius = 1; radius <= 6; ++radius) {
            for (int dr = -radius; dr <= radius; ++dr) {
                for (int dc = -radius; dc <= radius; ++dc) {
                    if (std::max(std::abs(dc), std::abs(dr)) != radius) continue;
                    if (placeBuilding(c0 + dc, r0 + dr, b)) return;
                }
            }
        }
    };

    // The classic starter layout, anchored on the terrain generator's scored
    // city site instead of fixed coordinates.
    const int bc = m_siteC, br = m_siteR;

    // A simple road grid around the city centre.
    for (int c = bc - 9; c <= bc + 9; ++c) { landRoad(c, br - 4); landRoad(c, br + 4); }
    for (int r = br - 4; r <= br + 4; ++r) {
        landRoad(bc - 5, r);
        landRoad(bc + 1, r);
        landRoad(bc + 7, r);
    }

    // Residential to the west, commercial in the middle, industry to the east.
    for (int r = br - 3; r <= br + 3; ++r) {
        for (int c = bc - 9; c <= bc - 6; ++c) landZone(c, r, Zone::Residential, 1.2f);
        for (int c = bc - 4; c <= bc; ++c) landZone(c, r, Zone::Commercial, 0.9f);
        for (int c = bc + 2; c <= bc + 6; ++c) landZone(c, r, Zone::Industrial, 0.8f);
    }

    placeBuildingNear(bc - 3, br + 6, Building::Power);
    placeBuildingNear(bc + 4, br + 6, Building::School);
    m_money = 50000.0;  // placeBuilding charged; restore the starting grant
}

// ─────────────────────────────────────────────────────────────────────────────
// Naming (Lua namegen, cached by seed)
// ─────────────────────────────────────────────────────────────────────────────
void CityBuilderApp::nameAnchor(int c, int r, int& outC, int& outR) const {
    outC = c;
    outR = r;
    if (!inBounds(c, r)) return;
    const PlotInfo& p = m_tilePlots[static_cast<std::size_t>(r) * kGridW + c];
    if (p.c < 0 || !inBounds(p.c, p.r)) return;  // no plot record: name the tile itself
    // A strip mall is a linear commercial plot (a row of storefronts): each tile
    // is its own shop. Blocky plots and all residential/industrial buildings get
    // one name for the whole building, keyed off the plot origin.
    const bool stripMall = tile(p.c, p.r).zone == Zone::Commercial &&
                           std::min(p.w, p.d) == 1 && std::max(p.w, p.d) >= 2;
    if (!stripMall) {
        outC = p.c;
        outR = p.r;
    }
}

const odai::citybuilder::BusinessName& CityBuilderApp::businessNameAt(int c, int r, const Tile&) {
    int ac = c, ar = r;
    nameAnchor(c, r, ac, ar);
    const Tile& a = tile(ac, ar);
    const bool industrial = a.zone == Zone::Industrial;
    const int level = 1 + std::min(2, static_cast<int>(a.develop));
    const int era = level - 1;  // 0=1890s, 1=1930s, 2=1960s — same mapping the mesher uses
    const int tier = residentialTier(a.desirability);
    const std::uint32_t seed = tileHash(ac, ar, 0xC0FFEE1Cu) ^ m_worldSeed;
    const std::uint32_t key =
        procgen::hash2d(static_cast<int>(seed & 0x7fffffffu),
                        (era << 4) | (tier << 2) | (industrial ? 1 : 0), 0xB1213Bu);
    auto it = m_businessNames.find(key);
    if (it == m_businessNames.end()) {
        it = m_businessNames.emplace(key, m_script->businessName(industrial, tier, era, seed)).first;
    }
    return it->second;
}

const std::string& CityBuilderApp::blockNameAt(int c, int r, const Tile&) {
    int ac = c, ar = r;
    nameAnchor(c, r, ac, ar);  // residential never strips, so this is the plot origin
    const Tile& a = tile(ac, ar);
    const int tier = residentialTier(a.desirability);
    const std::uint32_t seed = tileHash(ac, ar, 0x51DE17u) ^ m_worldSeed;
    const std::uint32_t key = seed ^ (static_cast<std::uint32_t>(tier) * 0x9E3779B9u);
    auto it = m_blockNames.find(key);
    if (it == m_blockNames.end()) {
        it = m_blockNames.emplace(key, m_script->blockName(tier, seed)).first;
    }
    return it->second;
}

const std::string& CityBuilderApp::streetNameAt(int c, int r) {
    // A road tile belongs to the horizontal run starting at its westernmost
    // contiguous road tile and the vertical run starting at its northernmost;
    // display the longer run's name so an avenue reads as one street.
    int wc = c;
    while (inBounds(wc - 1, r) && tile(wc - 1, r).road) --wc;
    int ec = c;
    while (inBounds(ec + 1, r) && tile(ec + 1, r).road) ++ec;
    int nr = r;
    while (inBounds(c, nr - 1) && tile(c, nr - 1).road) --nr;
    int sr = r;
    while (inBounds(c, sr + 1) && tile(c, sr + 1).road) ++sr;
    const std::uint32_t id = (ec - wc) >= (sr - nr) ? procgen::hash2d(wc, r, 0x57A337u)
                                                    : procgen::hash2d(c, nr, 0x57A338u);
    auto it = m_streetNames.find(id);
    if (it == m_streetNames.end()) {
        it = m_streetNames.emplace(id, m_script->streetName(id ^ m_worldSeed)).first;
    }
    return it->second;
}

// ─────────────────────────────────────────────────────────────────────────────
// Citizen sim glue
// ─────────────────────────────────────────────────────────────────────────────
void CityBuilderApp::rebuildDestinations() {
    m_destinations.clear();
    m_homeSites.clear();
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            Tile& t = tile(c, r);
            if (t.zone == Zone::Commercial && t.develop > kDevEps) {
                // Same seed the hover tooltip uses, so a citizen's yoga studio
                // is the storefront the player can actually find.
                const odai::citybuilder::BusinessName& biz = businessNameAt(c, r, t);
                m_destinations.push_back(
                    {static_cast<short>(c), static_cast<short>(r), biz.category, biz.name});
            } else if (t.zone == Zone::Residential && t.develop > 0.5f) {
                m_homeSites.push_back({static_cast<short>(c), static_cast<short>(r), t.develop});
            } else if (t.bldgOrigin && t.building == Building::Park) {
                m_destinations.push_back(
                    {static_cast<short>(c), static_cast<short>(r), "park", "the park"});
            } else if (t.bldgOrigin && t.building == Building::School) {
                m_destinations.push_back(
                    {static_cast<short>(c), static_cast<short>(r), "school", "the school"});
            }
        }
    }
}

void CityBuilderApp::reconcileCitizens() {
    ReconcileInput in;
    in.population = m_population;
    in.homes = &m_homeSites;
    in.destinations = &m_destinations;
    in.streetName = [this](short c, short r) -> std::string {
        short rc = 0, rr = 0;
        if (nearestRoad(c, r, rc, rr)) return streetNameAt(rc, rr);
        return {};
    };
    m_citizens.reconcileMonthly(in);
}

bool CityBuilderApp::nearestRoad(short c, short r, short& outC, short& outR) const {
    for (int radius = 0; radius <= 3; ++radius) {
        for (int dr = -radius; dr <= radius; ++dr) {
            for (int dc = -radius; dc <= radius; ++dc) {
                if (std::max(std::abs(dc), std::abs(dr)) != radius) continue;
                if (inBounds(c + dc, r + dr) && tile(c + dc, r + dr).road) {
                    outC = static_cast<short>(c + dc);
                    outR = static_cast<short>(r + dr);
                    return true;
                }
            }
        }
    }
    return false;
}

bool CityBuilderApp::routeRoad(short fromC, short fromR, short toC, short toR,
                               std::vector<std::uint16_t>& outRoute) {
    outRoute.clear();
    if (!inBounds(fromC, fromR) || !inBounds(toC, toR)) return false;
    if (!tile(fromC, fromR).road || !tile(toC, toR).road) return false;
    const auto pack = [](int c, int r) {
        return static_cast<std::uint16_t>(r * kGridW + c);
    };
    const std::uint16_t start = pack(fromC, fromR);
    const std::uint16_t goal = pack(toC, toR);
    if (start == goal) return false;

    // Plain BFS over <= 3136 road tiles — microseconds, and trips spawn at
    // well under 1 Hz, so no need for anything fancier.
    std::vector<std::int16_t> parent(static_cast<std::size_t>(kGridW) * kGridH, -2);
    std::vector<std::uint16_t> queue;
    queue.reserve(256);
    queue.push_back(start);
    parent[start] = -1;
    bool found = false;
    for (std::size_t head = 0; head < queue.size() && !found; ++head) {
        const std::uint16_t cur = queue[head];
        const int cc = cur % kGridW, cr = cur / kGridW;
        const int nc[4] = {cc - 1, cc + 1, cc, cc};
        const int nr[4] = {cr, cr, cr - 1, cr + 1};
        for (int k = 0; k < 4; ++k) {
            if (!inBounds(nc[k], nr[k]) || !tile(nc[k], nr[k]).road) continue;
            const std::uint16_t next = pack(nc[k], nr[k]);
            if (parent[next] != -2) continue;
            parent[next] = static_cast<std::int16_t>(cur);
            if (next == goal) {
                found = true;
                break;
            }
            queue.push_back(next);
        }
    }
    if (!found) return false;
    for (std::uint16_t cur = goal;;) {
        outRoute.push_back(cur);
        const std::int16_t p = parent[cur];
        if (p < 0) break;
        cur = static_cast<std::uint16_t>(p);
    }
    std::reverse(outRoute.begin(), outRoute.end());
    return true;
}

void CityBuilderApp::spawnCitizenTrip() {
    constexpr int kMaxRoutedCars = 16;  // rides above the ambient kMaxCars budget
    if (static_cast<int>(m_routedVehicles.size()) >= kMaxRoutedCars) return;
    CitizenSim::TripContext ctx;
    ctx.weekday = m_weekday;
    ctx.hour = dayHour();
    CitizenSim::Trip trip;
    if (!m_citizens.rollTrip(m_destinations, ctx, trip)) return;
    short fc = 0, fr = 0, tc = 0, tr = 0;
    if (!nearestRoad(trip.fromC, trip.fromR, fc, fr)) return;
    if (!nearestRoad(trip.toC, trip.toR, tc, tr)) return;
    if (fc == tc && fr == tr) return;

    Vehicle v;
    if (!routeRoad(fc, fr, tc, tr, v.route) || v.route.size() < 3) return;
    v.cx = fc;
    v.cr = fr;
    v.routeIdx = 0;
    const std::uint16_t next = v.route[1];
    v.outX = static_cast<signed char>(static_cast<int>(next % kGridW) - fc);
    v.outZ = static_cast<signed char>(static_cast<int>(next / kGridW) - fr);
    v.inX = v.outX;
    v.inZ = v.outZ;
    m_trafficRng = m_trafficRng * 1664525u + 1013904223u;
    v.variant = static_cast<std::uint8_t>((m_trafficRng >> 8) % kCarVariants);
    v.speed = 1.1f + 0.3f * static_cast<float>((m_trafficRng >> 16) & 0xffu) / 255.0f;
    m_routedVehicles.push_back(std::move(v));
}

void CityBuilderApp::advanceRoutedFleet(std::vector<Vehicle>& fleet, float dt,
                                        bool arrivalPedestrian) {
    for (std::size_t i = 0; i < fleet.size();) {
        Vehicle& v = fleet[i];
        bool drop = !inBounds(v.cx, v.cr) || !tile(v.cx, v.cr).road;
        if (!drop) {
            v.t += v.speed * dt;
            int guard = 0;
            while (!drop && v.t >= 1.0f && guard++ < 4) {
                v.t -= 1.0f;
                // Advance onto the next route tile.
                ++v.routeIdx;
                if (v.routeIdx >= v.route.size()) {
                    drop = true;
                    break;
                }
                const std::uint16_t cur = v.route[v.routeIdx];
                v.cx = static_cast<short>(cur % kGridW);
                v.cr = static_cast<short>(cur / kGridW);
                v.inX = v.outX;
                v.inZ = v.outZ;
                if (!inBounds(v.cx, v.cr) || !tile(v.cx, v.cr).road) {
                    drop = true;  // road bulldozed under the route
                    break;
                }
                if (v.routeIdx + 1 < v.route.size()) {
                    const std::uint16_t next = v.route[v.routeIdx + 1];
                    const int nc = next % kGridW, nr = next / kGridW;
                    v.outX = static_cast<signed char>(nc - v.cx);
                    v.outZ = static_cast<signed char>(nr - v.cr);
                    if (std::abs(v.outX) + std::abs(v.outZ) != 1) {
                        drop = true;
                    }
                } else {
                    // Arrived: the driver hops out as a pedestrian for a while
                    // (citizen cars only; service vehicles just end their run).
                    if (arrivalPedestrian &&
                        static_cast<int>(m_pedestrians.size()) < kMaxPedestrians + 8) {
                        Pedestrian p;
                        p.cx = v.cx;
                        p.cr = v.cr;
                        p.t = 0.4f;
                        p.speed = 0.18f;
                        p.variant = v.variant;
                        p.inX = v.inX;
                        p.inZ = v.inZ;
                        p.outX = v.outX;
                        p.outZ = v.outZ;
                        m_pedestrians.push_back(p);
                    }
                    drop = true;
                }
            }
        }
        if (drop) {
            fleet[i] = std::move(fleet.back());
            fleet.pop_back();
        } else {
            ++i;
        }
    }
}

void CityBuilderApp::updateRoutedVehicles(float dt) {
    advanceRoutedFleet(m_routedVehicles, dt, true);
}

// ─────────────────────────────────────────────────────────────────────────────
// Day/week schedule
// ─────────────────────────────────────────────────────────────────────────────
float CityBuilderApp::dayHour() const { return m_dayClock / kDayLengthSeconds * 24.0f; }

bool CityBuilderApp::buildServiceRoute(const std::vector<std::pair<short, short>>& waypoints,
                                       std::vector<std::uint16_t>& outRoute) {
    outRoute.clear();
    if (waypoints.size() < 2) return false;
    std::vector<std::uint16_t> leg;
    for (std::size_t i = 0; i + 1 < waypoints.size(); ++i) {
        if (!routeRoad(waypoints[i].first, waypoints[i].second, waypoints[i + 1].first,
                       waypoints[i + 1].second, leg)) {
            return false;
        }
        // Skip the leg's first node after the first leg — it's the previous
        // leg's last node.
        outRoute.insert(outRoute.end(), leg.begin() + (outRoute.empty() ? 0 : 1), leg.end());
    }
    return outRoute.size() >= 3;
}

void CityBuilderApp::spawnSchoolBusRun() {
    // Home base: the road outside a school. No school, no bus.
    short schoolC = -1, schoolR = -1;
    for (int r = 0; r < kGridH && schoolC < 0; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            if (tile(c, r).building == Building::School && tile(c, r).bldgOrigin) {
                if (nearestRoad(static_cast<short>(c), static_cast<short>(r), schoolC, schoolR)) break;
                schoolC = -1;
            }
        }
    }
    if (schoolC < 0) return;

    // Pickup stops: road tiles fronting developed residential, spread out by
    // reservoir-sampling three of them.
    std::vector<std::pair<short, short>> stops(3, {-1, -1});
    int found = 0;
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            if (!tile(c, r).road) continue;
            bool residential = false;
            for (int k = 0; k < 4 && !residential; ++k) {
                const int nc = c + (k == 0) - (k == 1);
                const int nr = r + (k == 2) - (k == 3);
                residential = inBounds(nc, nr) && tile(nc, nr).zone == Zone::Residential &&
                              tile(nc, nr).develop > 0.5f;
            }
            if (!residential) continue;
            ++found;
            m_trafficRng = m_trafficRng * 1664525u + 1013904223u;
            const std::size_t slot = (m_trafficRng >> 8) % 3u;
            if (static_cast<int>((m_trafficRng >> 12) % static_cast<std::uint32_t>(found)) <= 2) {
                stops[slot] = {static_cast<short>(c), static_cast<short>(r)};
            }
        }
    }
    std::vector<std::pair<short, short>> waypoints;
    waypoints.push_back({schoolC, schoolR});
    for (const auto& s : stops) {
        if (s.first >= 0 && s != waypoints.back()) waypoints.push_back(s);
    }
    waypoints.push_back({schoolC, schoolR});
    if (waypoints.size() < 3) return;

    Vehicle bus;
    if (!buildServiceRoute(waypoints, bus.route)) return;
    bus.cx = schoolC;
    bus.cr = schoolR;
    bus.routeIdx = 0;
    const std::uint16_t next = bus.route[1];
    bus.outX = static_cast<signed char>(static_cast<int>(next % kGridW) - schoolC);
    bus.outZ = static_cast<signed char>(static_cast<int>(next / kGridW) - schoolR);
    bus.inX = bus.outX;
    bus.inZ = bus.outZ;
    bus.speed = 0.85f;  // the bus is never in a hurry
    bus.variant = 0;
    m_busFleet.push_back(std::move(bus));
}

void CityBuilderApp::spawnGarbageRun() {
    // The truck rolls out from the power plant (industrial edge of town) and
    // loops through residential streets before heading back.
    short depotC = -1, depotR = -1;
    for (int r = 0; r < kGridH && depotC < 0; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            const Tile& t = tile(c, r);
            const bool depotish = (t.building == Building::Power && t.bldgOrigin) ||
                                  (t.zone == Zone::Industrial && t.develop > kDevEps);
            if (depotish) {
                if (nearestRoad(static_cast<short>(c), static_cast<short>(r), depotC, depotR)) break;
                depotC = -1;
            }
        }
    }
    if (depotC < 0) return;

    std::vector<std::pair<short, short>> stops(4, {-1, -1});
    int found = 0;
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            if (!tile(c, r).road) continue;
            bool residential = false;
            for (int k = 0; k < 4 && !residential; ++k) {
                const int nc = c + (k == 0) - (k == 1);
                const int nr = r + (k == 2) - (k == 3);
                residential = inBounds(nc, nr) && tile(nc, nr).zone == Zone::Residential &&
                              tile(nc, nr).develop > kDevEps;
            }
            if (!residential) continue;
            ++found;
            m_trafficRng = m_trafficRng * 1664525u + 1013904223u;
            const std::size_t slot = (m_trafficRng >> 8) % 4u;
            if (static_cast<int>((m_trafficRng >> 12) % static_cast<std::uint32_t>(found)) <= 3) {
                stops[slot] = {static_cast<short>(c), static_cast<short>(r)};
            }
        }
    }
    std::vector<std::pair<short, short>> waypoints;
    waypoints.push_back({depotC, depotR});
    for (const auto& s : stops) {
        if (s.first >= 0 && s != waypoints.back()) waypoints.push_back(s);
    }
    waypoints.push_back({depotC, depotR});
    if (waypoints.size() < 3) return;

    Vehicle truck;
    if (!buildServiceRoute(waypoints, truck.route)) return;
    truck.cx = depotC;
    truck.cr = depotR;
    truck.routeIdx = 0;
    const std::uint16_t next = truck.route[1];
    truck.outX = static_cast<signed char>(static_cast<int>(next % kGridW) - depotC);
    truck.outZ = static_cast<signed char>(static_cast<int>(next / kGridW) - depotR);
    truck.inX = truck.outX;
    truck.inZ = truck.outZ;
    truck.speed = 0.65f;  // trundles, pausing in spirit at every can
    truck.variant = 0;
    m_trashFleet.push_back(std::move(truck));
}

void CityBuilderApp::updateSchedule(float dt) {
    m_dayClock += dt * static_cast<float>(m_speed);
    if (m_dayClock >= kDayLengthSeconds) {
        m_dayClock -= kDayLengthSeconds;
        m_weekday = (m_weekday + 1) % 7;
        m_busMorningDone = m_busAfternoonDone = false;
        m_trashRunDone = false;
        m_soccerStoryDone = false;
        if (m_weekday == 5) m_citizens.endWorkWeek();  // Friday night: everyone clocks out
    }
    const float hour = dayHour();
    const bool schoolDay = m_weekday < 5;

    // School bus: morning pickup loop and the 3 o'clock run.
    if (schoolDay && !m_busMorningDone && hour >= 7.4f) {
        m_busMorningDone = true;
        spawnSchoolBusRun();
    }
    if (schoolDay && !m_busAfternoonDone && hour >= 15.0f) {
        m_busAfternoonDone = true;
        spawnSchoolBusRun();
    }

    // Trash day: cans hit the curb early, the truck rolls mid-morning, and
    // the cans disappear by evening. Toggling the flag re-extrudes the scene
    // through the usual growth cooldown.
    const bool cansOut = isTrashDay(m_weekday) && hour >= 6.5f && hour < 18.0f;
    if (cansOut != m_trashDayActive) {
        m_trashDayActive = cansOut;
        m_growthDirty = true;
    }
    if (isTrashDay(m_weekday) && !m_trashRunDone && hour >= 9.0f) {
        m_trashRunDone = true;
        spawnGarbageRun();
    }

    // Saturday soccer: one ticker beat when practice kicks off at a park.
    if (m_weekday == 5 && !m_soccerStoryDone && hour >= 8.5f) {
        m_soccerStoryDone = true;
        for (const Destination& d : m_destinations) {
            if (d.category != "park") continue;
            short rc = 0, rr = 0;
            std::string street;
            if (nearestRoad(d.c, d.r, rc, rr)) street = streetNameAt(rc, rr);
            m_citizens.emitWeekendStory(d, street);
            break;
        }
    }

    advanceRoutedFleet(m_busFleet, dt, false);
    advanceRoutedFleet(m_trashFleet, dt, false);
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

    constexpr int kRoadReach   = 3;   // Chebyshev tiles a road services
    constexpr int kPowerRadius = 9;   // a plant's own direct glow, roads or not

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
                switch (t.building) {
                    case Building::Park:   splat(amenity, c, r, 5, 0.55f); break;
                    case Building::Police: splat(amenity, c, r, 6, 0.28f); break;
                    case Building::Fire:   splat(amenity, c, r, 6, 0.24f); break;
                    case Building::Clinic: splat(amenity, c, r, 6, 0.24f); break;
                    case Building::School: splat(amenity, c, r, 6, 0.30f); break;
                    case Building::Library: splat(amenity, c, r, 5, 0.26f); break;
                    case Building::Amphitheater: splat(amenity, c, r, 7, 0.50f); break;  // culture draw
                    case Building::Power:  splat(nuisance, c, r, 5, 0.45f); break;  // dirty neighbour
                    default: break;
                }
            }
            if (t.zone == Zone::Industrial && t.develop > kDevEps) {
                splat(nuisance, c, r, 4, 0.16f * t.develop);  // pollution / heavy traffic
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

    // Grow / abandon each zoned parcel. Commercial lots that develop for the
    // first time this month become ticker "openings".
    std::vector<std::pair<short, short>> opened;
    for (Tile& t : m_tiles) {
        if (t.zone == Zone::None) continue;
        const float before = t.develop;
        const float dem = t.zone == Zone::Residential ? m_resDemand
                          : t.zone == Zone::Commercial ? m_comDemand
                                                       : m_indDemand;
        if (t.powered && t.nearRoad) {
            // A vacant lot sits on the market for a short real-time while
            // before anything breaks ground, no matter how hot demand is —
            // reads as the parcel getting sold rather than construction
            // starting the instant it's painted. The listing clock itself is
            // ticked every frame in onTick (real time, not simulated months);
            // this just gates growth on it having run out.
            if (t.zoneAge >= kZoneListingSeconds) {
                // Citywide demand sets the ceiling; per-tile desirability decides
                // how much of it each parcel actually captures. A 0.5 land value
                // is neutral (multiplier 1.0), prime land overshoots (clamped to
                // full build-out), and poor land stagnates even in a hot market.
                const float desMul = 0.4f + 1.2f * t.desirability;
                const float target = std::min(3.0f, dem * 3.0f * desMul);
                t.develop += (target - t.develop) * 0.16f;
            }
        } else {
            t.develop += (0.0f - t.develop) * 0.10f;  // decay toward abandonment
        }
        t.develop = std::clamp(t.develop, 0.0f, 3.0f);
        if (t.zone == Zone::Commercial && before <= kDevEps && t.develop > kDevEps) {
            const std::size_t idx = static_cast<std::size_t>(&t - m_tiles.data());
            opened.emplace_back(static_cast<short>(idx % kGridW), static_cast<short>(idx / kGridW));
        }
    }

    // Post-growth census, easing the city quality stats toward their targets.
    m_statEase = 0.12f;
    recomputeStats();

    // Citizen layer: refresh named destinations, announce openings (max 2 a
    // month so the ticker never firehoses), churn the roster, fire the Lua
    // month hook.
    rebuildDestinations();
    int openingsEmitted = 0;
    for (const auto& [oc, orr] : opened) {
        if (openingsEmitted >= 2) break;
        for (const Destination& d : m_destinations) {
            if (d.c == oc && d.r == orr) {
                short rc = 0, rr = 0;
                std::string street;
                if (nearestRoad(oc, orr, rc, rr)) street = streetNameAt(rc, rr);
                m_citizens.emitOpening(d, street);
                ++openingsEmitted;
                break;
            }
        }
    }
    reconcileCitizens();
    if (m_script) {
        odai::citybuilder::CityScriptStats stats;
        stats.population = m_population;
        stats.money = m_money;
        stats.month = m_month + 1;
        stats.year = m_year;
        m_script->fireMonthStep(stats);
    }

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

bool CityBuilderApp::charge(double cost) {
    if (m_money >= cost) { m_money -= cost; return true; }
    flash("Insufficient funds");
    return false;
}

void CityBuilderApp::flash(std::string msg) {
    m_flashMsg = std::move(msg);
    m_flashTimer = 1.8f;
}

void CityBuilderApp::setTool(Tool t) {
    if (t == m_tool) return;
    // Switching tools mid-drag abandons the box rather than applying the new
    // tool to a rectangle the player started under a different one.
    m_boxSelecting = false;
    m_boxStartC = m_boxStartR = m_boxEndC = m_boxEndR = -1;
    m_tool = t;
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
            if (charge(cost)) { t.zone = z; t.develop = 0.0f; t.zoneAge = 0.0f; m_sceneDirty = true; }
            break;
        }
        case Tool::Road:
            if (t.building != Building::None) { flash("Bulldoze first"); break; }
            if (t.road) break;
            // Roads over water become bridges — pricier per segment.
            if (charge(t.terrain == Terrain::Water ? cost * 4.0 : cost)) {
                t.road = true;
                t.zone = Zone::None;
                t.develop = 0.0f;
                m_sceneDirty = true;
            }
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
        t.zoneAge = 0.0f;
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
    if (m_script) {
        odai::citybuilder::CityScriptStats stats;
        stats.population = m_population;
        stats.money = m_money;
        stats.month = m_month + 1;
        stats.year = m_year;
        m_script->fireBuildingPlaced(c, r, buildingTag(b), stats);
    }
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

    if (edgeDown(GLFW_KEY_X)) setTool(Tool::Bulldoze);
    if (edgeDown(GLFW_KEY_1)) setTool(Tool::ZoneR);
    if (edgeDown(GLFW_KEY_2)) setTool(Tool::ZoneC);
    if (edgeDown(GLFW_KEY_3)) setTool(Tool::ZoneI);
    if (edgeDown(GLFW_KEY_R)) setTool(Tool::Road);
    if (edgeDown(GLFW_KEY_4)) setTool(Tool::Police);
    if (edgeDown(GLFW_KEY_5)) setTool(Tool::Fire);
    if (edgeDown(GLFW_KEY_6)) setTool(Tool::Clinic);
    if (edgeDown(GLFW_KEY_7)) setTool(Tool::School);
    if (edgeDown(GLFW_KEY_8)) setTool(Tool::Park);
    if (edgeDown(GLFW_KEY_9)) setTool(Tool::Library);
    if (edgeDown(GLFW_KEY_0)) setTool(Tool::Amphitheater);
    if (edgeDown(GLFW_KEY_MINUS)) setTool(Tool::Power);

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
        // Zone "listing" clock: real time, not simulated months, so it stays a
        // short wait regardless of how slow kMonthInterval is tuned. Only lots
        // that are zoned, vacant, and actually connected (powered + road)
        // count down; anything else resets so it relists once connected.
        const float zoneDt = dt * static_cast<float>(m_speed);
        for (Tile& t : m_tiles) {
            if (t.zone == Zone::None) continue;
            if (t.develop > kDevEps) {
                t.zoneAge = kZoneListingSeconds;  // already broke ground; gate no longer applies
            } else if (t.powered && t.nearRoad) {
                t.zoneAge += zoneDt;
            } else {
                t.zoneAge = 0.0f;
            }
        }

        m_simAccum += dt * static_cast<float>(m_speed);
        int guard = 0;
        while (m_simAccum >= kMonthInterval && guard++ < 8) {
            m_simAccum -= kMonthInterval;
            stepMonth();
        }
        // Ambient traffic runs on real time (not sim speed) so cars cruise at
        // a believable pace at every game speed.
        updateVehicles(dt);
        updatePedestrians(dt);
        updateBoats(dt);

        // The civic day/week clock: commute waves, school bus, trash day,
        // Saturday soccer.
        updateSchedule(dt);

        // Citizen trips: routed cars head to actual named destinations. The
        // cadence breathes with the clock — rush hours surge, nights go
        // quiet, weekends stroll (ODAI_CITY_STORY speeds everything up 5x).
        updateRoutedVehicles(dt);
        const float hour = dayHour();
        float cadence = 1.0f;
        const bool rush = m_weekday < 5 &&
                          ((hour >= 7.0f && hour < 9.5f) || (hour >= 17.0f && hour < 19.5f));
        if (rush) cadence = 2.6f;                          // the waves
        else if (hour >= 21.0f || hour < 5.5f) cadence = 0.35f;  // sleepy town
        m_tripTimer -= dt * cadence * (m_storyBoost > 1.0f ? 5.0f : 1.0f);
        if (m_tripTimer <= 0.0f) {
            m_trafficRng = m_trafficRng * 1664525u + 1013904223u;
            m_tripTimer = 1.5f + 1.5f * static_cast<float>((m_trafficRng >> 8) & 0xffu) / 255.0f;
            spawnCitizenTrip();
        }
    }
    // Weather is pure atmosphere — it keeps falling even while paused.
    updateWeather(dt);
    // Ticker chips fade on real time too (they're chrome, not simulation).
    for (TickerItem& item : m_citizens.ticker()) item.age += dt;
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

    const float ctlW = 718.0f * s, ctlH = 46.0f * s;
    lo.controls = UiRect::fromXYWH(lo.map.minX + 14.0f * s, lo.fh - ctlH - 14.0f * s, ctlW, ctlH);

    const float mmS = 190.0f * s;
    lo.minimap = UiRect::fromXYWH(lo.fw - mmS - 14.0f * s, lo.fh - mmS - 14.0f * s, mmS, mmS);

    // Citizen-story ticker: a chip lane docked just above the controls bar, in
    // the otherwise-empty strip left of the minimap.
    lo.ticker = UiRect::fromXYWH(lo.controls.minX, lo.controls.minY - 72.0f * s,
                                 std::min(560.0f * s, lo.fw - mmS - lo.controls.minX - 28.0f * s),
                                 64.0f * s);

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
    if (isBoxSelectTool(m_tool)) {
        const auto& mouse = m_uiInput.button(UiMouseButton::Left);
        if (!m_boxSelecting) {
            if (m_mouseOverUi || m_hoverC < 0 || !mouse.pressed) return;
            m_boxSelecting = true;
            m_boxStartC = m_boxEndC = m_hoverC;
            m_boxStartR = m_boxEndR = m_hoverR;
            return;
        }
        // Once a drag has started, keep tracking it even if the cursor drifts
        // over a UI panel or off the grid edge — the box just stops growing
        // in that direction until the cursor comes back over the map.
        if (m_hoverC >= 0) {
            m_boxEndC = m_hoverC;
            m_boxEndR = m_hoverR;
        }
        if (mouse.released || !mouse.down) {
            const int c0 = std::min(m_boxStartC, m_boxEndC), c1 = std::max(m_boxStartC, m_boxEndC);
            const int r0 = std::min(m_boxStartR, m_boxEndR), r1 = std::max(m_boxStartR, m_boxEndR);
            for (int r = r0; r <= r1; ++r) {
                for (int c = c0; c <= c1; ++c) applyTool(c, r);
            }
            m_boxSelecting = false;
            m_boxStartC = m_boxStartR = m_boxEndC = m_boxEndR = -1;
        }
        return;
    }

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
        // Open on the seeded city, not the grid centre — the terrain generator
        // may have anchored the starter town anywhere on the map.
        m_camFocusX = (static_cast<float>(m_siteC) + 0.5f) * kTileWorldSize;
        m_camFocusZ = (static_cast<float>(m_siteR) + 0.5f) * kTileWorldSize;
        m_camZoom   = kGridW * kTileWorldSize * 0.95f;  // conservative: fits the whole grid
        m_camYawDeg = 45.0f;                             // classic diagonal city-builder view
        m_camInit = true;
    }

    const bool tickerVisible =
        !m_citizens.ticker().empty() && m_citizens.ticker().back().age < 12.0f;
    m_mouseOverUi = lo.topBar.contains(m_uiInput.mousePx) ||
                    lo.palette.contains(m_uiInput.mousePx) ||
                    lo.controls.contains(m_uiInput.mousePx) ||
                    lo.minimap.contains(m_uiInput.mousePx) ||
                    (tickerVisible && lo.ticker.contains(m_uiInput.mousePx)) ||
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
    drawTicker(lo);
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

    // Prop scatter rates (per-mille per eligible tile), tunable from the Lua
    // Config.scatter table; read once per rebuild, never per tile.
    const auto scatterRate = [&](const char* key, double fallback) {
        return static_cast<std::uint32_t>(m_script ? m_script->configNumber(key, fallback)
                                                   : fallback);
    };
    const std::uint32_t hydrantRate = scatterRate("scatter.hydrant_per_mille", 120.0);
    const std::uint32_t benchRate = scatterRate("scatter.bench_per_mille", 260.0);
    const std::uint32_t billboardRate = scatterRate("scatter.billboard_per_mille", 180.0);
    const std::uint32_t busStopRate = scatterRate("scatter.bus_stop_per_mille", 140.0);

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
    m_tilePlots.fill(PlotInfo{});  // persistent per-tile plot record for the tooltip
    const auto parcelable = [&](int c, int r, Zone z) {
        if (!inBounds(c, r)) return false;
        const Tile& pt = tile(c, r);
        return pt.terrain == Terrain::Grass && pt.zone == z && !pt.road &&
               pt.building == Building::None &&
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
                    const std::size_t mi = static_cast<std::size_t>(r + dr) * kGridW + (c + dc);
                    plotIndex[mi] = static_cast<short>(plots.size());
                    m_tilePlots[mi] = PlotInfo{static_cast<short>(c), static_cast<short>(r),
                                               static_cast<std::uint8_t>(pw),
                                               static_cast<std::uint8_t>(pd)};
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

            const bool bridge = t.terrain == Terrain::Water && t.road;
            if (t.terrain == Terrain::Water) {
                UiColor wc = ((c + r) & 1) ? kWater : kWaterAlt;
                if (winter) wc = mix(wc, UiColor::fromRgbHex(0xA8C4D4), 0.55f);
                builder.addQuad({x0, 0.0f, z0}, {x1, 0.0f, z0}, {x1, 0.0f, z1}, {x0, 0.0f, z1}, wc);
                if (!bridge) continue;  // a road on water carries on into the road branch
            } else {
                // Ground: land-value heat map when toggled, otherwise grass
                // tinted by the baked scenicPhase jitter so a block reads as
                // organic.
                if (m_showLandValue) {
                    builder.addQuad({x0, 0.0f, z0}, {x1, 0.0f, z0}, {x1, 0.0f, z1}, {x0, 0.0f, z1},
                                    heat(t.desirability));
                } else {
                    builder.addQuad({x0, 0.0f, z0}, {x1, 0.0f, z0}, {x1, 0.0f, z1}, {x0, 0.0f, z1},
                                    seasonalGrass(mix(kGrassAlt, kGrass, t.scenicPhase)));
                }

                // Seawall promenade: urban shoreline (within road reach) gets a
                // boardwalk strip, a stone lip over the water, and an iron
                // railing along every water-facing edge — with the occasional
                // bench looking out and a lamp at the corner. Wilderness
                // shoreline stays natural.
                if (!t.road && t.nearRoad) {
                    const auto waterAt = [&](int nc, int nr) {
                        return inBounds(nc, nr) && tile(nc, nr).terrain == Terrain::Water &&
                               !tile(nc, nr).road;
                    };
                    const float walkD = ts * 0.14f;   // boardwalk depth from the edge
                    const float walkH = 0.035f;
                    const float railH = 0.105f;
                    const std::uint32_t ph = tileHash(c, r, 0x5EA9A11u);
                    // dir 0=N 1=W 2=S 3=E; bench turns face the water.
                    for (int dir = 0; dir < 4; ++dir) {
                        const int dc = dir == 1 ? -1 : (dir == 3 ? 1 : 0);
                        const int dr = dir == 0 ? -1 : (dir == 2 ? 1 : 0);
                        if (!waterAt(c + dc, r + dr)) continue;
                        const bool alongX = (dr != 0);  // edge runs east-west
                        if (alongX) {
                            const float ez = dr < 0 ? z0 : z1;               // water edge z
                            const float wz0 = dr < 0 ? z0 : z1 - walkD;
                            const float wz1 = dr < 0 ? z0 + walkD : z1;
                            addBox(builder, x0, wz0, x1, wz1, 0.0f, walkH, kBoardwalk);
                            addBox(builder, x0, ez - 0.02f, x1, ez + 0.02f, -0.02f, walkH + 0.008f,
                                   kSeawallStone);
                            const float railZ = dr < 0 ? z0 + 0.028f : z1 - 0.028f;
                            addBox(builder, x0, railZ - 0.008f, x1, railZ + 0.008f, railH - 0.012f,
                                   railH, kRailIron);
                            for (int p = 0; p < 3; ++p) {
                                const float px = x0 + ts * (0.16f + 0.34f * static_cast<float>(p));
                                addBox(builder, px - 0.008f, railZ - 0.008f, px + 0.008f,
                                       railZ + 0.008f, walkH, railH, kRailIron);
                            }
                            if (ph % 100u < 30u) {
                                const float bz = dr < 0 ? z0 + 0.085f : z1 - 0.085f;
                                procgen::appendTriMeshRotated(
                                    cachedBench(ph >> 8), {(x0 + x1) * 0.5f, walkH, bz},
                                    dr < 0 ? 0 : 2, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, scene);
                            }
                            if ((ph >> 4) % 100u < 22u) {
                                procgen::appendTriMesh(cachedStreetlamp(ph >> 10),
                                                       {x0 + ts * 0.08f, walkH,
                                                        dr < 0 ? z0 + 0.07f : z1 - 0.07f},
                                                       {1.0f, 1.0f, 1.0f}, scene);
                            }
                        } else {
                            const float ex = dc < 0 ? x0 : x1;               // water edge x
                            const float wx0 = dc < 0 ? x0 : x1 - walkD;
                            const float wx1 = dc < 0 ? x0 + walkD : x1;
                            addBox(builder, wx0, z0, wx1, z1, 0.0f, walkH, kBoardwalk);
                            addBox(builder, ex - 0.02f, z0, ex + 0.02f, z1, -0.02f, walkH + 0.008f,
                                   kSeawallStone);
                            const float railX = dc < 0 ? x0 + 0.028f : x1 - 0.028f;
                            addBox(builder, railX - 0.008f, z0, railX + 0.008f, z1, railH - 0.012f,
                                   railH, kRailIron);
                            for (int p = 0; p < 3; ++p) {
                                const float pz = z0 + ts * (0.16f + 0.34f * static_cast<float>(p));
                                addBox(builder, railX - 0.008f, pz - 0.008f, railX + 0.008f,
                                       pz + 0.008f, walkH, railH, kRailIron);
                            }
                            if (ph % 100u < 30u) {
                                const float bx = dc < 0 ? x0 + 0.085f : x1 - 0.085f;
                                procgen::appendTriMeshRotated(
                                    cachedBench(ph >> 8), {bx, walkH, (z0 + z1) * 0.5f},
                                    dc < 0 ? 1 : 3, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, scene);
                            }
                            if ((ph >> 4) % 100u < 22u) {
                                procgen::appendTriMesh(cachedStreetlamp(ph >> 10),
                                                       {dc < 0 ? x0 + 0.07f : x1 - 0.07f, walkH,
                                                        z0 + ts * 0.08f},
                                                       {1.0f, 1.0f, 1.0f}, scene);
                            }
                        }
                    }
                }
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

                if (bridge) {
                    // Bridge structure: a stone deck slab under the roadway,
                    // chunky corner pilings sunk into the water, and iron
                    // railings riding the parapet walk strips. The deck stays
                    // flush with land asphalt so cars, pedestrians, and the
                    // power flood-fill cross without any special casing.
                    addBox(builder, x0, z0, x1, z1, -0.015f, ry, kBridgeStone);
                    const float pier = ts * 0.07f;
                    addBox(builder, x0, z0, x0 + pier, z0 + pier, -0.045f, 0.0f, kBridgeStone);
                    addBox(builder, x1 - pier, z0, x1, z0 + pier, -0.045f, 0.0f, kBridgeStone);
                    addBox(builder, x0, z1 - pier, x0 + pier, z1, -0.045f, 0.0f, kBridgeStone);
                    addBox(builder, x1 - pier, z1 - pier, x1, z1, -0.045f, 0.0f, kBridgeStone);
                    const float railH = 0.11f;
                    const auto railRun = [&](float rx0, float rz0, float rx1, float rz1) {
                        addBox(builder, rx0, rz0, rx1, rz1, railH - 0.012f, railH, kRailIron);
                        // Three posts spaced along the run.
                        for (int p = 0; p < 3; ++p) {
                            const float f = 0.16f + 0.34f * static_cast<float>(p);
                            const float px = rx0 + (rx1 - rx0) * f;
                            const float pz = rz0 + (rz1 - rz0) * f;
                            addBox(builder, px - 0.008f, pz - 0.008f, px + 0.008f, pz + 0.008f,
                                   sy, railH, kRailIron);
                        }
                    };
                    if (!nN) railRun(x0, z0 + 0.055f, x1, z0 + 0.071f);
                    if (!nS) railRun(x0, z1 - 0.071f, x1, z1 - 0.055f);
                    if (!nW) railRun(x0 + 0.055f, nN ? z0 : z0 + side, x0 + 0.071f,
                                     nS ? z1 : z1 - side);
                    if (!nE) railRun(x1 - 0.071f, nN ? z0 : z0 + side, x1 - 0.055f,
                                     nS ? z1 : z1 - side);
                }

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
                if (t.poweredRoad && !bridge) {
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
                // Bridges carry only their railings; no lamps/hydrants/stops.
                if (!bridge && tileHash(c, r, 0x1A4Fu) % 1000u < 340u) {
                    const std::uint32_t lv = tileHash(c, r, 0x7A4Fu) % kLampVariants;
                    procgen::appendTriMesh(cachedStreetlamp(lv), {x1 - ts * 0.10f, sy, z1 - ts * 0.10f},
                                           {1.0f, 1.0f, 1.0f}, scene);
                }

                // Trash day: cans line the curb of residential-fronting
                // streets from breakfast until the evening (m_trashDayActive
                // toggles a rebuild at both ends of the window).
                if (!bridge && m_trashDayActive) {
                    bool residential = false;
                    for (int k = 0; k < 4 && !residential; ++k) {
                        const int nc = c + (k == 0) - (k == 1);
                        const int nr = r + (k == 2) - (k == 3);
                        residential = inBounds(nc, nr) && tile(nc, nr).zone == Zone::Residential &&
                                      tile(nc, nr).develop > kDevEps;
                    }
                    const std::uint32_t th = tileHash(c, r, 0x7245C4u);
                    if (residential && th % 100u < 45u) {
                        procgen::appendTriMesh(cachedTrashCan(th >> 8),
                                               {x0 + ts * 0.24f, sy, z1 - ts * 0.075f},
                                               {1.0f, 1.0f, 1.0f}, scene);
                        if (th & 1u) {
                            procgen::appendTriMesh(cachedTrashCan(th >> 9),
                                                   {x0 + ts * 0.30f, sy, z1 - ts * 0.075f},
                                                   {1.0f, 1.0f, 1.0f}, scene);
                        }
                    }
                }

                // Hydrants take the southwest sidewalk corner (lamps hold SE,
                // power poles NW) so the street furniture never stacks.
                if (!bridge && tileHash(c, r, 0x94D64u) % 1000u < hydrantRate) {
                    procgen::appendTriMesh(cachedHydrant(tileHash(c, r, 0x94D65u)),
                                           {x0 + ts * 0.10f, sy, z1 - ts * 0.10f},
                                           {1.0f, 1.0f, 1.0f}, scene);
                }

                // Bus stops appear where the road fronts a dense commercial
                // strip: at least 3 developed commercial neighbours.
                int devCom = 0;
                for (int k = 0; k < 4; ++k) {
                    const int nc = c + (k == 0) - (k == 1);
                    const int nr = r + (k == 2) - (k == 3);
                    if (inBounds(nc, nr) && tile(nc, nr).zone == Zone::Commercial &&
                        tile(nc, nr).develop > kDevEps) {
                        ++devCom;
                    }
                }
                if (!bridge && devCom >= 3 && tileHash(c, r, 0xB0557u) % 1000u < busStopRate) {
                    procgen::appendTriMesh(cachedBusStop(tileHash(c, r, 0xB0558u)),
                                           {cx, sy, z1 - ts * 0.085f}, {1.0f, 1.0f, 1.0f}, scene);
                }
                continue;
            }

            if (t.building != Building::None) {
                if (!t.bldgOrigin) continue;  // drawn by the origin cell
                const int fp = std::max<int>(1, t.footprint);
                const float bx1 = x0 + fp * ts, bz1 = z0 + fp * ts;
                const float pad = ts * 0.10f;
                const std::uint32_t variant = tileHash(c, r, 0xC171C5u) % 4u;
                if (t.building == Building::Park) {
                    // Green slab, a generated gazebo/fountain centerpiece, and
                    // proper procgen trees so the park reads as a garden.
                    const UiColor slab = mix(buildingRoof(t.building), UiColor(0, 0, 0, 1), 0.12f);
                    addBox(builder, x0 + pad, z0 + pad, bx1 - pad, bz1 - pad, 0.0f, kParkSlabHeight,
                           slab);
                    procgen::appendTriMesh(cachedCivic(t.building, variant),
                                           {x0 + pad, kParkSlabHeight, z0 + pad},
                                           {1.0f, 1.0f, 1.0f}, scene);
                    // Park planting: a stately mix — oak patriarch, blossom
                    // ornamental, broadleaf, birch — one per quadrant corner.
                    constexpr std::uint32_t kParkSpecies[4] = {6u, 5u, 0u, 2u};
                    const std::uint32_t rot = tileHash(c, r, 0x9A7C0u) & 3u;
                    for (int i = 0; i < 4; ++i) {
                        const float px = x0 + (0.20f + 0.60f * (i & 1)) * (bx1 - x0);
                        const float pz = z0 + (0.20f + 0.60f * (i >> 1)) * (bz1 - z0);
                        procgen::appendTriMesh(
                            cachedTree(kParkSpecies[(static_cast<std::uint32_t>(i) + rot) & 3u]),
                            {px, kParkSlabHeight, pz}, {1.0f, 1.0f, 1.0f}, scene);
                    }
                    continue;
                }
                // Face the entrance toward an adjacent road, exactly like the
                // zoned plots do. Civic lots are square, so the rotation spins
                // the cached mesh about the lot centre with no extent swap.
                const auto anyRoad = [&](int cc0, int rr0, int cc1, int rr1) {
                    for (int rr = rr0; rr <= rr1; ++rr)
                        for (int cc = cc0; cc <= cc1; ++cc)
                            if (inBounds(cc, rr) && tile(cc, rr).road) return true;
                    return false;
                };
                int facings[4];
                int numFacings = 0;
                if (anyRoad(c, r - 1, c + fp - 1, r - 1)) facings[numFacings++] = 0;
                if (anyRoad(c - 1, r, c - 1, r + fp - 1)) facings[numFacings++] = 1;
                if (anyRoad(c, r + fp, c + fp - 1, r + fp)) facings[numFacings++] = 2;
                if (anyRoad(c + fp, r, c + fp, r + fp - 1)) facings[numFacings++] = 3;
                const std::uint32_t fh = tileHash(c, r, 0xFACE5u);
                const int turns = numFacings > 0
                                      ? facings[fh % static_cast<std::uint32_t>(numFacings)]
                                      : static_cast<int>(fh & 3u);
                const procgen::TriMesh& bm = cachedCivic(t.building, variant);
                const float lotHalf = (fp * ts - 2.0f * pad) * 0.5f;
                procgen::appendTriMeshRotated(bm, {x0 + pad, 0.0f, z0 + pad}, turns,
                                              {lotHalf, 0.0f, lotHalf}, {1.0f, 1.0f, 1.0f}, scene);
                continue;
            }

            if (t.zone != Zone::None) {
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

                    // Yard greenery: residential lots get a shrub or a small
                    // ornamental tree tucked into the pad ring at a seeded
                    // corner — front gardens sell the neighbourhood read.
                    if (kind == procgen::BuildingKind::Residential) {
                        const std::uint32_t yh = tileHash(c, r, 0x9A2D8Bu);
                        if (yh % 100u < 55u) {
                            const std::uint32_t yv = ((yh >> 8) & 3u) == 0u ? 5u : 7u;
                            const int corner = (yh >> 4) & 3;
                            const float yx = (corner & 1) ? x0 + pw * ts - 0.06f * ts
                                                          : x0 + 0.06f * ts;
                            const float yz = (corner & 2) ? z0 + pd * ts - 0.06f * ts
                                                          : z0 + 0.06f * ts;
                            procgen::appendTriMesh(cachedTree(yv), {yx, 0.0f, yz},
                                                   {1.0f, 1.0f, 1.0f}, scene);
                        }
                    }

                    // Billboards on the rear corner of busy commercial and
                    // industrial plots, panel spun toward the same street the
                    // building faces (the post sits at the mesh origin).
                    if (kind != procgen::BuildingKind::Residential && level >= 2 &&
                        tileHash(c, r, 0xB111Bu) % 1000u < billboardRate) {
                        const signed char kFaceX[4] = {0, -1, 0, 1};
                        const signed char kFaceZ[4] = {-1, 0, 1, 0};
                        const float plotCx = x0 + pw * ts * 0.5f;
                        const float plotCz = z0 + pd * ts * 0.5f;
                        const float bbx = plotCx - kFaceX[turns] * 0.36f * pw * ts -
                                          kFaceZ[turns] * 0.26f * pw * ts;
                        const float bbz = plotCz - kFaceZ[turns] * 0.36f * pd * ts +
                                          kFaceX[turns] * 0.26f * pd * ts;
                        procgen::appendTriMeshRotated(cachedBillboard(tileHash(c, r, 0xB111Cu)),
                                                      {bbx, 0.0f, bbz}, turns, {0.0f, 0.0f, 0.0f},
                                                      tint, scene);
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
                // The terrain generator's fbm forest mask clumps trees into
                // readable groves instead of a uniform speckle, and the
                // context picks the species: willows own the waterline, mixed
                // broadleaf/conifer/birch stands fill the forest, ornamentals
                // and poplars line the streets.
                const bool byWater = (inBounds(c, r - 1) && tile(c, r - 1).terrain == Terrain::Water) ||
                                     (inBounds(c, r + 1) && tile(c, r + 1).terrain == Terrain::Water) ||
                                     (inBounds(c - 1, r) && tile(c - 1, r).terrain == Terrain::Water) ||
                                     (inBounds(c + 1, r) && tile(c + 1, r).terrain == Terrain::Water);
                const float forest = m_forest[static_cast<std::size_t>(r) * kGridW + c];
                std::uint32_t rate = static_cast<std::uint32_t>(
                    static_cast<float>(byRoad ? 450u : 90u) * (0.4f + 1.6f * forest));
                if (byWater) rate = std::max(rate, 300u);  // banks stay leafy
                if (h % 1000u < rate) {
                    const std::uint32_t sp = (h >> 10) % 100u;
                    std::uint32_t variant;
                    if (byWater) {
                        variant = sp < 55u ? 4u : (sp < 78u ? 2u : 0u);
                    } else if (forest > 0.55f) {
                        variant = sp < 35u ? 0u : sp < 60u ? 1u : sp < 82u ? 2u : 6u;
                    } else if (byRoad) {
                        variant = sp < 38u ? 5u : sp < 62u ? 0u : sp < 84u ? 3u : 2u;
                    } else {
                        variant = sp < 30u ? 0u : sp < 50u ? 1u : sp < 68u ? 2u : sp < 86u ? 5u : 6u;
                    }
                    const float jx = 0.30f + 0.40f * static_cast<float>((h >> 13) & 0xffu) / 255.0f;
                    const float jz = 0.30f + 0.40f * static_cast<float>((h >> 21) & 0xffu) / 255.0f;
                    procgen::appendTriMesh(cachedTree(variant), {x0 + jx * ts, 0.0f, z0 + jz * ts},
                                           {1.0f, 1.0f, 1.0f}, scene);
                    // A second tree on some road-facing or forested parcels
                    // reads as a planted row / thicker stand.
                    if ((byRoad || forest > 0.55f) && (h & 1u)) {
                        procgen::appendTriMesh(cachedTree((variant + 2u) % 7u),
                                               {x0 + (1.0f - jx) * ts, 0.0f, z0 + (1.0f - jz) * ts},
                                               {1.0f, 1.0f, 1.0f}, scene);
                    }
                }
                // Benches face the street next to parks and along pleasant
                // road-fronting parcels (desirability-gated).
                bool nearPark = false;
                for (int k = 0; k < 4 && !nearPark; ++k) {
                    const int nc = c + (k == 0) - (k == 1);
                    const int nr = r + (k == 2) - (k == 3);
                    nearPark = inBounds(nc, nr) && tile(nc, nr).building == Building::Park;
                }
                const std::uint32_t benchGate =
                    nearPark ? benchRate : (byRoad && t.desirability > 0.62f ? benchRate / 3u : 0u);
                if (benchGate > 0u && tileHash(c, r, 0xBE7C4u) % 1000u < benchGate) {
                    procgen::appendTriMesh(cachedBench(tileHash(c, r, 0xBE7C5u)),
                                           {x0 + ts * 0.5f, 0.0f, z0 + ts * 0.18f},
                                           {1.0f, 1.0f, 1.0f}, scene);
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

const procgen::TriMesh& CityBuilderApp::cachedCivic(Building b, std::uint32_t variant) const {
    const std::uint32_t key = (static_cast<std::uint32_t>(b) << 4) | (variant & 0xfu);
    const auto it = m_civicCache.find(key);
    if (it != m_civicCache.end()) {
        return it->second;
    }
    procgen::CivicDesc desc;
    desc.kind = civicKindOf(b);
    const int fp = footprintOf(b);
    const float pad = kTileWorldSize * 0.10f;
    desc.lotWidth = static_cast<float>(fp) * kTileWorldSize - 2.0f * pad;
    desc.lotDepth = desc.lotWidth;
    desc.seed = key * 0x9E3779B9u ^ m_worldSeed;
    return m_civicCache.emplace(key, procgen::generateCivicBuilding(desc)).first->second;
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

const procgen::TriMesh& CityBuilderApp::cachedBench(std::uint32_t variant) const {
    constexpr std::uint32_t kBenchVariants = 3;
    if (m_benchMeshes.empty()) {
        m_benchMeshes.reserve(kBenchVariants);
        for (std::uint32_t i = 0; i < kBenchVariants; ++i) {
            m_benchMeshes.push_back(procgen::generateBench(0xBE7C4u + i * 401u));
        }
    }
    return m_benchMeshes[variant % kBenchVariants];
}

const procgen::TriMesh& CityBuilderApp::cachedHydrant(std::uint32_t variant) const {
    constexpr std::uint32_t kHydrantVariants = 3;
    if (m_hydrantMeshes.empty()) {
        m_hydrantMeshes.reserve(kHydrantVariants);
        for (std::uint32_t i = 0; i < kHydrantVariants; ++i) {
            m_hydrantMeshes.push_back(procgen::generateHydrant(0x94D64u + i * 613u));
        }
    }
    return m_hydrantMeshes[variant % kHydrantVariants];
}

const procgen::TriMesh& CityBuilderApp::cachedBillboard(std::uint32_t variant) const {
    constexpr std::uint32_t kBillboardVariants = 4;
    if (m_billboardMeshes.empty()) {
        m_billboardMeshes.reserve(kBillboardVariants);
        for (std::uint32_t i = 0; i < kBillboardVariants; ++i) {
            m_billboardMeshes.push_back(procgen::generateBillboard(0xB111Bu + i * 761u));
        }
    }
    return m_billboardMeshes[variant % kBillboardVariants];
}

const procgen::TriMesh& CityBuilderApp::cachedBusStop(std::uint32_t variant) const {
    constexpr std::uint32_t kBusStopVariants = 2;
    if (m_busStopMeshes.empty()) {
        m_busStopMeshes.reserve(kBusStopVariants);
        for (std::uint32_t i = 0; i < kBusStopVariants; ++i) {
            m_busStopMeshes.push_back(procgen::generateBusStop(0xB0557u + i * 883u));
        }
    }
    return m_busStopMeshes[variant % kBusStopVariants];
}

const procgen::TriMesh& CityBuilderApp::cachedTrashCan(std::uint32_t variant) const {
    constexpr std::uint32_t kTrashCanVariants = 3;
    if (m_trashCanMeshes.empty()) {
        m_trashCanMeshes.reserve(kTrashCanVariants);
        for (std::uint32_t i = 0; i < kTrashCanVariants; ++i) {
            m_trashCanMeshes.push_back(procgen::generateTrashCan(0x7245Cu + i * 449u));
        }
    }
    return m_trashCanMeshes[variant % kTrashCanVariants];
}

// ─────────────────────────────────────────────────────────────────────────────
// Ambient traffic
// ─────────────────────────────────────────────────────────────────────────────
namespace {
// Shared tile-to-tile direction picker for every ambient agent (cars,
// pedestrians, boats — their in/out/cx fields match): no U-turns unless dead
// end, prefer straight at straightPct. `passable` owns the bounds check.
template <typename Agent, typename Passable>
bool pickAgentExit(Agent& a, std::uint32_t& rngState, std::uint32_t straightPct,
                   Passable&& passable) {
    struct Dir {
        signed char x, z;
    };
    constexpr Dir kDirs[4] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    Dir options[4];
    int optionCount = 0;
    bool straightAvailable = false;
    for (const Dir& d : kDirs) {
        if (d.x == -a.inX && d.z == -a.inZ) continue;  // no U-turns unless dead end
        if (!passable(a.cx + d.x, a.cr + d.z)) continue;
        options[optionCount++] = d;
        if (d.x == a.inX && d.z == a.inZ) straightAvailable = true;
    }
    if (optionCount == 0) {
        // Dead end: turn back if the path behind still exists.
        if (passable(a.cx - a.inX, a.cr - a.inZ)) {
            a.outX = static_cast<signed char>(-a.inX);
            a.outZ = static_cast<signed char>(-a.inZ);
            return true;
        }
        return false;
    }
    rngState = rngState * 1664525u + 1013904223u;
    const std::uint32_t roll = rngState >> 8;
    if (straightAvailable && roll % 100u < straightPct) {
        a.outX = a.inX;
        a.outZ = a.inZ;
    } else {
        const Dir d = options[roll % static_cast<std::uint32_t>(optionCount)];
        a.outX = d.x;
        a.outZ = d.z;
    }
    return true;
}
}  // namespace

void CityBuilderApp::respawnVehicle(Vehicle& v) {
    // Reservoir-sample a random road tile so we don't need a stored road list.
    int found = 0;
    short pickC = -1, pickR = -1;
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            if (!tile(c, r).road) continue;
            ++found;
            m_trafficRng = m_trafficRng * 1664525u + 1013904223u;
            if (static_cast<int>((m_trafficRng >> 8) % static_cast<std::uint32_t>(found)) == 0) {
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
    return pickAgentExit(v, m_trafficRng, 62u, [this](int c, int r) {
        return inBounds(c, r) && tile(c, r).road;
    });
}

void CityBuilderApp::updateVehicles(float dt) {
    // Fleet size tracks the road network; cars fade in as the city grows.
    const int target = std::min(kMaxCars, static_cast<int>(kCarsPerRoadTile * static_cast<float>(m_numRoad)));
    while (static_cast<int>(m_vehicles.size()) < target) {
        Vehicle v;
        respawnVehicle(v);
        if (v.cx < 0) break;  // no roads at all
        m_vehicles.push_back(v);
    }
    if (static_cast<int>(m_vehicles.size()) > target) {
        m_vehicles.resize(static_cast<std::size_t>(target));
    }

    for (Vehicle& v : m_vehicles) {
        // Road bulldozed underneath: find a new home.
        if (!inBounds(v.cx, v.cr) || !tile(v.cx, v.cr).road) {
            respawnVehicle(v);
            if (v.cx < 0) continue;
        }
        v.t += v.speed * dt;
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
// Ambient pedestrians & boats
// ─────────────────────────────────────────────────────────────────────────────
void CityBuilderApp::respawnPedestrian(Pedestrian& p) {
    // Reservoir-sample a road tile with developed non-industrial frontage —
    // people stroll where the shops and homes are, not along empty highways.
    const auto walkable = [this](int c, int r) {
        if (!inBounds(c, r) || !tile(c, r).road) return false;
        for (int k = 0; k < 4; ++k) {
            const int nc = c + (k == 0) - (k == 1);
            const int nr = r + (k == 2) - (k == 3);
            if (!inBounds(nc, nr)) continue;
            const Tile& n = tile(nc, nr);
            if (n.zone != Zone::None && n.zone != Zone::Industrial && n.develop > 0.2f) return true;
            if (n.building != Building::None && n.building != Building::Power) return true;
        }
        return false;
    };
    int found = 0;
    short pickC = -1, pickR = -1;
    for (int r = 0; r < kGridH; ++r) {
        for (int c = 0; c < kGridW; ++c) {
            if (!walkable(c, r)) continue;
            ++found;
            m_trafficRng = m_trafficRng * 1664525u + 1013904223u;
            if (static_cast<int>((m_trafficRng >> 8) % static_cast<std::uint32_t>(found)) == 0) {
                pickC = static_cast<short>(c);
                pickR = static_cast<short>(r);
            }
        }
    }
    p.cx = pickC;
    p.cr = pickR;
    p.t = 0.0f;
    m_trafficRng = m_trafficRng * 1664525u + 1013904223u;
    p.variant = static_cast<std::uint8_t>((m_trafficRng >> 8) % kPedVariants);
    p.speed = 0.15f + 0.10f * static_cast<float>((m_trafficRng >> 16) & 0xffu) / 255.0f;
    p.inX = 1;
    p.inZ = 0;
    if (pickC >= 0) {
        const auto onRoad = [this](int c, int r) { return inBounds(c, r) && tile(c, r).road; };
        if (pickAgentExit(p, m_trafficRng, 55u, onRoad)) {
            p.inX = p.outX;
            p.inZ = p.outZ;
        } else {
            p.cx = -1;  // isolated road stub; try again next tick
        }
    }
}

void CityBuilderApp::updatePedestrians(float dt) {
    const int target = std::min(kMaxPedestrians, m_population / 120);
    while (static_cast<int>(m_pedestrians.size()) < target) {
        Pedestrian p;
        respawnPedestrian(p);
        if (p.cx < 0) break;
        m_pedestrians.push_back(p);
    }
    if (static_cast<int>(m_pedestrians.size()) > target) {
        m_pedestrians.resize(static_cast<std::size_t>(target));
    }

    const auto onRoad = [this](int c, int r) { return inBounds(c, r) && tile(c, r).road; };
    for (Pedestrian& p : m_pedestrians) {
        if (!inBounds(p.cx, p.cr) || !tile(p.cx, p.cr).road) {
            respawnPedestrian(p);
            if (p.cx < 0) continue;
        }
        p.t += p.speed * dt;
        int guard = 0;
        while (p.t >= 1.0f && guard++ < 2) {
            p.t -= 1.0f;
            p.cx = static_cast<short>(p.cx + p.outX);
            p.cr = static_cast<short>(p.cr + p.outZ);
            p.inX = p.outX;
            p.inZ = p.outZ;
            if (!inBounds(p.cx, p.cr) || !tile(p.cx, p.cr).road ||
                !pickAgentExit(p, m_trafficRng, 55u, onRoad)) {
                respawnPedestrian(p);
                break;
            }
        }
    }
}

void CityBuilderApp::respawnBoat(Boat& b) {
    // Spawn on the river centerline (guaranteed connected water); lakes get
    // traffic only when a river meander clips them.
    b.cx = -1;
    if (m_riverPath.empty()) return;
    m_trafficRng = m_trafficRng * 1664525u + 1013904223u;
    const auto& [pc, pr] = m_riverPath[(m_trafficRng >> 8) % m_riverPath.size()];
    if (!inBounds(pc, pr) || tile(pc, pr).terrain != Terrain::Water || tile(pc, pr).road) return;
    b.cx = pc;
    b.cr = pr;
    b.t = 0.0f;
    m_trafficRng = m_trafficRng * 1664525u + 1013904223u;
    b.variant = static_cast<std::uint8_t>((m_trafficRng >> 8) % kBoatVariants);
    b.speed = 0.22f + 0.16f * static_cast<float>((m_trafficRng >> 16) & 0xffu) / 255.0f;
    b.bobPhase = static_cast<float>((m_trafficRng >> 4) & 0xffu) * 0.0246f;
    b.inX = 0;
    b.inZ = 1;
    // Low bridges block boat traffic; boats treat them as banks and turn back.
    const auto onWater = [this](int c, int r) {
        return inBounds(c, r) && tile(c, r).terrain == Terrain::Water && !tile(c, r).road;
    };
    if (pickAgentExit(b, m_trafficRng, 70u, onWater)) {
        b.inX = b.outX;
        b.inZ = b.outZ;
    } else {
        b.cx = -1;
    }
}

void CityBuilderApp::updateBoats(float dt) {
    const int target = m_riverPath.empty() ? 0 : kMaxBoats;
    while (static_cast<int>(m_boats.size()) < target) {
        Boat b;
        respawnBoat(b);
        if (b.cx < 0) break;
        m_boats.push_back(b);
    }

    const auto onWater = [this](int c, int r) {
        return inBounds(c, r) && tile(c, r).terrain == Terrain::Water && !tile(c, r).road;
    };
    for (Boat& b : m_boats) {
        if (b.cx < 0 || !onWater(b.cx, b.cr)) {
            respawnBoat(b);
            if (b.cx < 0) continue;
        }
        b.t += b.speed * dt;
        int guard = 0;
        while (b.t >= 1.0f && guard++ < 2) {
            b.t -= 1.0f;
            b.cx = static_cast<short>(b.cx + b.outX);
            b.cr = static_cast<short>(b.cr + b.outZ);
            b.inX = b.outX;
            b.inZ = b.outZ;
            if (!onWater(b.cx, b.cr) || !pickAgentExit(b, m_trafficRng, 70u, onWater)) {
                respawnBoat(b);
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
    if (m_pedMeshes.empty()) {
        m_pedMeshes.reserve(kPedVariants);
        for (std::uint32_t i = 0; i < kPedVariants; ++i) {
            m_pedMeshes.push_back(procgen::generatePedestrian(0x9ED0u + i * 331u));
        }
    }
    if (m_boatMeshes.empty()) {
        m_boatMeshes.reserve(kBoatVariants);
        for (std::uint32_t i = 0; i < kBoatVariants; ++i) {
            m_boatMeshes.push_back(procgen::generateBoat(i, 0xB0A7u + i * 4409u));
        }
    }
    if (m_busMeshes.empty()) {
        m_busMeshes.push_back(procgen::generateSchoolBus(0x5CB005u));
    }
    if (m_trashTruckMeshes.empty()) {
        m_trashTruckMeshes.push_back(procgen::generateGarbageTruck(0x6A3BA6Eu));
    }

    const float ts = kTileWorldSize;

    // Rotate a +X-facing mesh onto the (vx, vz) heading and place it at
    // (px, yLift, pz) — shared by every ambient agent.
    const auto pushOriented = [&](const procgen::TriMesh& mesh, float px, float pz, float vx,
                                  float vz, float yLift) {
        const std::uint32_t base = static_cast<std::uint32_t>(m_actorVertices.size());
        for (const ImportedScenePackedVertex& src : mesh.vertices) {
            ImportedScenePackedVertex dst = src;
            const float lx = src.position[0], lz = src.position[2];
            dst.position[0] = px + lx * vx - lz * vz;
            dst.position[2] = pz + lx * vz + lz * vx;
            dst.position[1] = src.position[1] + yLift;
            const float nx = src.normal[0], nz = src.normal[2];
            dst.normal[0] = nx * vx - nz * vz;
            dst.normal[2] = nx * vz + nz * vx;
            m_actorVertices.push_back(dst);
        }
        for (const std::uint32_t index : mesh.indices) {
            m_actorIndices.push_back(base + index);
        }
    };

    // Quadratic bezier across a tile on a rail offset from the centre line:
    // entry/exit points sit on the rail of the incoming/outgoing directions;
    // the control point is the rail-line intersection, which folds turns into
    // smooth arcs and leaves straights linear. Returns position + heading.
    const auto railPoint = [&](short tcx, short tcr, signed char tinX, signed char tinZ,
                               signed char toutX, signed char toutZ, float t, float rail,
                               float& outPx, float& outPz, float& outVx, float& outVz) {
        const float cx = (static_cast<float>(tcx) + 0.5f) * ts;
        const float cz = (static_cast<float>(tcr) + 0.5f) * ts;
        const float inX = static_cast<float>(tinX), inZ = static_cast<float>(tinZ);
        const float outX = static_cast<float>(toutX), outZ = static_cast<float>(toutZ);
        // Right-hand perpendicular of a direction (x,z) is (-z, x).
        const float p0x = cx - 0.5f * inX * ts + (-inZ) * rail * ts;
        const float p0z = cz - 0.5f * inZ * ts + (inX)*rail * ts;
        const float p2x = cx + 0.5f * outX * ts + (-outZ) * rail * ts;
        const float p2z = cz + 0.5f * outZ * ts + (outX)*rail * ts;
        float p1x, p1z;
        if (tinX == toutX && tinZ == toutZ) {
            p1x = 0.5f * (p0x + p2x);
            p1z = 0.5f * (p0z + p2z);
        } else {
            p1x = cx + (-inZ) * rail * ts + (-outZ) * rail * ts;
            p1z = cz + (inX)*rail * ts + (outX)*rail * ts;
        }
        const float u = 1.0f - t;
        outPx = u * u * p0x + 2.0f * u * t * p1x + t * t * p2x;
        outPz = u * u * p0z + 2.0f * u * t * p1z + t * t * p2z;
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
        outVx = vx;
        outVz = vz;
    };

    for (const Vehicle& v : m_vehicles) {
        if (v.cx < 0) continue;
        float px, pz, vx, vz;
        railPoint(v.cx, v.cr, v.inX, v.inZ, v.outX, v.outZ, v.t, kLaneOffset, px, pz, vx, vz);
        pushOriented(m_carMeshes[v.variant % m_carMeshes.size()], px, pz, vx, vz,
                     0.02f);  // ride on the asphalt surface
    }
    for (const Vehicle& v : m_routedVehicles) {
        if (v.cx < 0) continue;
        float px, pz, vx, vz;
        railPoint(v.cx, v.cr, v.inX, v.inZ, v.outX, v.outZ, v.t, kLaneOffset, px, pz, vx, vz);
        pushOriented(m_carMeshes[v.variant % m_carMeshes.size()], px, pz, vx, vz, 0.02f);
    }
    for (const Vehicle& v : m_busFleet) {
        if (v.cx < 0) continue;
        float px, pz, vx, vz;
        railPoint(v.cx, v.cr, v.inX, v.inZ, v.outX, v.outZ, v.t, kLaneOffset, px, pz, vx, vz);
        pushOriented(m_busMeshes[0], px, pz, vx, vz, 0.02f);
    }
    for (const Vehicle& v : m_trashFleet) {
        if (v.cx < 0) continue;
        float px, pz, vx, vz;
        railPoint(v.cx, v.cr, v.inX, v.inZ, v.outX, v.outZ, v.t, kLaneOffset, px, pz, vx, vz);
        pushOriented(m_trashTruckMeshes[0], px, pz, vx, vz, 0.02f);
    }

    for (const Pedestrian& p : m_pedestrians) {
        if (p.cx < 0) continue;
        float px, pz, vx, vz;
        railPoint(p.cx, p.cr, p.inX, p.inZ, p.outX, p.outZ, p.t, kWalkOffset, px, pz, vx, vz);
        pushOriented(m_pedMeshes[p.variant % m_pedMeshes.size()], px, pz, vx, vz,
                     0.045f);  // on the sidewalk top
    }

    for (const Boat& b : m_boats) {
        if (b.cx < 0) continue;
        float px, pz, vx, vz;
        railPoint(b.cx, b.cr, b.inX, b.inZ, b.outX, b.outZ, b.t, 0.0f, px, pz, vx, vz);
        const float bob = 0.012f * std::sin(m_time * 1.6f + b.bobPhase);
        pushOriented(m_boatMeshes[b.variant % m_boatMeshes.size()], px, pz, vx, vz, 0.018f + bob);
    }

    // Precipitation: crossed double-sided quads per drop (thin tall streaks
    // for rain, small flakes for snow), lit as upward-facing so they stay
    // uniformly bright at any camera yaw.
    if (!m_drops.empty()) {
        const bool snow = m_weather == Weather::Snow;
        const float w = snow ? 0.014f : 0.0045f;
        const float len = snow ? 0.016f : 0.17f;
        const float cr = snow ? 0.93f : 0.60f;
        const float cg = snow ? 0.95f : 0.68f;
        const float cb = snow ? 0.98f : 0.78f;
        auto pushQuad = [&](const Vector3& a, const Vector3& b, const Vector3& cVert,
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
            // Both windings so the quad survives backface culling from any yaw.
            for (const std::uint32_t i :
                 {base, base + 1u, base + 2u, base, base + 2u, base + 3u,
                  base, base + 2u, base + 1u, base, base + 3u, base + 2u}) {
                m_actorIndices.push_back(i);
            }
        };
        for (const WeatherDrop& d : m_drops) {
            pushQuad({d.x - w, d.y, d.z}, {d.x + w, d.y, d.z}, {d.x + w, d.y + len, d.z},
                     {d.x - w, d.y + len, d.z});
            pushQuad({d.x, d.y, d.z - w}, {d.x, d.y, d.z + w}, {d.x, d.y + len, d.z + w},
                     {d.x, d.y + len, d.z - w});
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

    if (m_boxSelecting && m_boxStartC >= 0 && m_boxEndC >= 0) {
        // Active box-select drag: outline the whole rectangle being dragged,
        // not just the tile under the cursor.
        const UiColor hc = kTools[static_cast<int>(m_tool)].color;
        const int c0 = std::min(m_boxStartC, m_boxEndC), c1 = std::max(m_boxStartC, m_boxEndC);
        const int r0 = std::min(m_boxStartR, m_boxEndR), r1 = std::max(m_boxStartR, m_boxEndR);
        const float x0 = c0 * kTileWorldSize, z0 = r0 * kTileWorldSize;
        const float x1 = (c1 + 1) * kTileWorldSize, z1 = (r1 + 1) * kTileWorldSize;
        const UiVec2 poly[4] = {
            worldToScreen(x0, 0.05f, z0, lo), worldToScreen(x1, 0.05f, z0, lo),
            worldToScreen(x1, 0.05f, z1, lo), worldToScreen(x0, 0.05f, z1, lo),
        };
        m_uiDrawList.addPolylineAA(poly, 4, withA(hc, 0.95f), 2.5f * s, true);
    } else if (!m_mouseOverUi && m_hoverC >= 0) {
        // Hover footprint preview: project the ground-plane quad corners to
        // screen and stroke them, since the tile is a parallelogram under the
        // iso camera.
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
        // Always lead with the place's NAME; any blocker ("No power", "Low
        // demand", …) rides underneath as a second line so the tooltip reads
        // consistently whether or not the lot is thriving.
        std::string title;
        std::string note;
        UiColor noteColor = kBad;
        if (ht.zone != Zone::None) {
            const bool developed = ht.develop > kDevEps;
            if (developed) {
                title = (ht.zone == Zone::Residential) ? blockNameAt(m_hoverC, m_hoverR, ht)
                                                       : businessNameAt(m_hoverC, m_hoverR, ht).name;
            } else {
                title = ht.zone == Zone::Residential ? "Residential lot"
                        : ht.zone == Zone::Commercial ? "Commercial lot"
                                                       : "Industrial lot";
            }
            const float dem = ht.zone == Zone::Residential ? m_resDemand
                              : ht.zone == Zone::Commercial ? m_comDemand
                                                            : m_indDemand;
            if (!ht.nearRoad) note = "No road access";
            else if (!ht.powered) note = "No power";
            else if (!developed) {
                if (dem < 0.18f) { note = "Low demand"; noteColor = kGold; }
                else if (ht.desirability < 0.4f) { note = "Low land value"; noteColor = kGold; }
                else { note = "For sale — awaiting a buyer"; noteColor = kTextDim; }
            } else if (ht.develop < 2.4f) {
                if (dem < 0.18f) { note = "Low demand"; noteColor = kGold; }
                else if (ht.desirability < 0.4f) { note = "Low land value"; noteColor = kGold; }
            }
        } else if (ht.road) {
            title = streetNameAt(m_hoverC, m_hoverR);
        }

        if (!title.empty()) {
            const UiVec2 mp = m_uiInput.mousePx;
            const float titleW = m_uiFontBold.measureText(title);
            const float noteW = note.empty() ? 0.0f : m_uiFont.measureText(note);
            const float boxW = std::max(titleW, noteW) + 18.0f * s;
            const float boxH = (note.empty() ? 24.0f : 42.0f) * s;
            const UiRect tip = UiRect::fromXYWH(mp.x + 16.0f * s, mp.y + 16.0f * s, boxW, boxH);
            m_uiDrawList.addRoundRectFilled(tip, withA(UiColor::fromRgbHex(0x14181F), 0.92f),
                                            5.0f * s);
            m_uiDrawList.addRoundRect(tip, kEdge, 5.0f * s, s);
            if (note.empty()) {
                textLeft(m_uiFontBold, title, tip.minX + 9.0f * s,
                         (tip.minY + tip.maxY) * 0.5f, kText);
            } else {
                textLeft(m_uiFontBold, title, tip.minX + 9.0f * s, tip.minY + 14.0f * s, kText);
                textLeft(m_uiFont, note, tip.minX + 9.0f * s, tip.maxY - 13.0f * s, noteColor);
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
            // Tags float just above the generated mesh's actual roofline (the
            // cache lookup is free; the variant hash matches the mesher's).
            const std::uint32_t tagVariant = tileHash(c, r, 0xC171C5u) % 4u;
            float tagY = cachedCivic(t.building, tagVariant).boundsMax.y + 0.12f;
            if (t.building == Building::Park) tagY += kParkSlabHeight;
            const UiVec2 sp = worldToScreen(cx, tagY, cz, lo);
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

    textLeft(m_uiFontBold, m_cityName, 16.0f * s, cy - 9.0f * s, kText);
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

        if (hover && m_uiInput.button(UiMouseButton::Left).pressed) setTool(tool);
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
    textLeft(m_uiFontBold, date, x, r.minY + r.height() * 0.5f - 8.0f * s, kText);
    // The civic clock underneath: weekday + time drives the visible routines
    // (rush hour, school bus, trash day), so the mayor can see why the
    // streets just filled up.
    {
        const float hour = dayHour();
        const int hh24 = static_cast<int>(hour) % 24;
        const int mm = static_cast<int>((hour - std::floor(hour)) * 60.0f);
        const int hh12 = hh24 % 12 == 0 ? 12 : hh24 % 12;
        char clock[24];
        std::snprintf(clock, sizeof(clock), "%s %d:%02d %s", kWeekdays[m_weekday], hh12, mm,
                      hh24 < 12 ? "AM" : "PM");
        textLeft(m_uiFont, clock, x, r.minY + r.height() * 0.5f + 9.0f * s, kTextDim);
    }
    x += 108.0f * s;

    // Live R/C/I demand snapshot — the quick-glance readout that used to sit
    // in the top bar, moved down here now that Reports (which has the actual
    // historical charts) no longer opens automatically.
    {
        const float cy = r.minY + r.height() * 0.5f;
        const float dem[3] = {m_resDemand, m_comDemand, m_indDemand};
        const UiColor demc[3] = {kZoneR, kZoneC, kZoneI};
        const char* deml[3] = {"R", "C", "I"};
        for (int i = 0; i < 3; ++i) {
            const UiRect sw = UiRect::fromXYWH(x, cy - 5.0f * s, 10.0f * s, 10.0f * s);
            m_uiDrawList.addRoundRectFilled(sw, demc[i], 2.0f * s);
            x += 14.0f * s;
            char buf[8];
            std::snprintf(buf, sizeof(buf), "%s %d%%", deml[i],
                          static_cast<int>(std::lround(clamp01(dem[i]) * 100.0f)));
            textLeft(m_uiFont, buf, x, cy, kTextDim);
            x += m_uiFont.measureText(buf) + 14.0f * s;
        }
    }

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
            if (t.building != Building::None) col = buildingRoof(t.building);
            else if (t.road) col = UiColor::fromRgbHex(0x55595F);  // bridges read as road
            else if (t.terrain == Terrain::Water) col = kWater;
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

void CityBuilderApp::drawTicker(const Layout& lo) {
    std::deque<TickerItem>& items = m_citizens.ticker();
    if (items.empty()) return;
    const float s = lo.s;
    constexpr float kShowSecs = 12.0f;  // chips linger, then fade over the last 3 s

    int shown = 0;
    float y = lo.ticker.maxY;
    for (auto it = items.rbegin(); it != items.rend() && shown < 2; ++it) {
        TickerItem& item = *it;
        if (item.age >= kShowSecs) break;  // older chips are older still
        const float alpha = std::clamp((kShowSecs - item.age) / 3.0f, 0.0f, 1.0f);

        UiColor accent = kZoneC;
        switch (item.kind) {
            case TickerKind::Opening:   accent = kGold; break;
            case TickerKind::Drama:     accent = UiColor::fromRgbHex(0xE06AB0); break;
            case TickerKind::Arrival:   accent = kGood; break;
            case TickerKind::Departure: accent = kBad; break;
            default: break;
        }

        const float th = 26.0f * s;
        const float tw = std::min(m_uiFont.measureText(item.text) + 34.0f * s, lo.ticker.width());
        const UiRect chip = UiRect::fromXYWH(lo.ticker.minX, y - th, tw, th);
        m_uiDrawList.addRoundRectFilled(chip, withA(kPanel, 0.90f * alpha), 6.0f * s);
        m_uiDrawList.addRoundRect(chip, withA(accent, 0.55f * alpha), 6.0f * s, s);
        m_uiDrawList.addRoundRectFilled(
            UiRect::fromXYWH(chip.minX + 9.0f * s, (chip.minY + chip.maxY) * 0.5f - 3.0f * s,
                             6.0f * s, 6.0f * s),
            withA(accent, alpha), 3.0f * s);
        textLeft(m_uiFont, item.text, chip.minX + 22.0f * s,
                 (chip.minY + chip.maxY) * 0.5f - 8.0f * s, withA(kText, alpha));

        // Clicking a chip pans the camera to where the story happened.
        if (item.c >= 0 && chip.contains(m_uiInput.mousePx) &&
            m_uiInput.button(UiMouseButton::Left).pressed) {
            m_camFocusX = (static_cast<float>(item.c) + 0.5f) * kTileWorldSize;
            m_camFocusZ = (static_cast<float>(item.r) + 0.5f) * kTileWorldSize;
        }
        y -= th + 6.0f * s;
        ++shown;
    }
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
