#pragma once

#include "engine/game_app.h"
#include "import/imported_scene.h"
#include "ui/ui_types.h"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// A compact SimCity-2013-style city builder. Zone tiles (residential /
// commercial / industrial), lay roads, drop municipal buildings (police, fire,
// clinic, school, park, library, amphitheater, power plant), watch a money /
// population / power HUD,
// a live minimap, and floating report windows that plot Education, Health,
// Happiness, Population and Treasury over time.
//
// The tile grid, economy and history series are simulated purely on the CPU
// (see stepMonth()). Each time the grid changes, buildCityScene() extrudes it
// into a packed vertex-color ImportedScene (tiles/buildings as flat-shaded
// boxes) uploaded once to the real 3-D renderer via an isometric orthographic
// camera — the same "CPU state -> ImportedScene -> Renderer::uploadImportedScene"
// path the hex strategy map uses, so no new Vulkan code is needed. Chrome
// (top bar, palette, controls, minimap, reports) and thin per-frame overlays
// (hover outline, stalled-tile tooltip, land-value legend) stay in the 2-D
// UI draw list, composited over the 3-D frame in the UI pass.
namespace odai::games::citybuilder {

enum class Terrain : std::uint8_t { Grass, Water };
enum class Zone : std::uint8_t { None, Residential, Commercial, Industrial };
enum class Building : std::uint8_t { None, Police, Fire, Clinic, School, Park, Library, Amphitheater, Power };

struct Tile {
    Terrain terrain = Terrain::Grass;
    Zone    zone     = Zone::None;
    Building building = Building::None;
    bool  road        = false;
    bool  bldgOrigin  = false;   // top-left tile of a multi-tile building footprint
    std::uint8_t footprint = 0;  // footprint side length, stored on the origin tile
    short bOriginC = -1;         // origin tile of the building this cell belongs to
    short bOriginR = -1;
    float develop  = 0.0f;       // 0..3 growth level for zoned tiles
    bool  powered  = false;      // within range of a power plant
    bool  nearRoad = false;      // within reach of a road (required to develop)
    float desirability = 0.5f;   // 0..1 spatial land value; modulates growth target
    float scenicPhase = 0.0f;    // per-tile jitter so a block doesn't look uniform
};

class CityBuilderApp : public engine::GameApp {
public:
    static constexpr int kGridW = 56;
    static constexpr int kGridH = 56;

    enum class Tool : int {
        Bulldoze,
        ZoneR, ZoneC, ZoneI,
        Road,
        Police, Fire, Clinic, School, Park, Library, Amphitheater, Power,
        Count
    };

    enum class Metric : int { Population, Treasury, Education, Health, Happiness, Count };

protected:
    bool onInit() override;
    void onTick(float dt) override;
    void onRender(float dt) override;

private:
    // ── Grid access ──────────────────────────────────────────────────────────
    [[nodiscard]] bool inBounds(int c, int r) const {
        return c >= 0 && c < kGridW && r >= 0 && r < kGridH;
    }
    Tile& tile(int c, int r) { return m_tiles[static_cast<std::size_t>(r) * kGridW + c]; }
    [[nodiscard]] const Tile& tile(int c, int r) const {
        return m_tiles[static_cast<std::size_t>(r) * kGridW + c];
    }

    // ── World / economy ──────────────────────────────────────────────────────
    void generateTerrain();
    void seedCity();
    void recomputeStats();   // power/road coverage, population, jobs, eased city stats
    void computeDesirability();  // per-tile spatial land value (amenities vs. nuisances)
    void pushHistory();      // append a sample to each metric series
    void stepMonth();
    void applyTool(int c, int r);
    void bulldoze(int c, int r);
    bool placeBuilding(int c, int r, Building b);
    bool charge(double cost);
    void flash(std::string msg);

    // ── Per-frame layout & input ─────────────────────────────────────────────
    struct Layout {
        float s = 1.0f, fw = 0.0f, fh = 0.0f;
        ui::UiRect topBar{}, palette{}, map{}, controls{}, minimap{}, reports{};
    };
    [[nodiscard]] Layout computeLayout() const;
    void clampCameraFocus();
    void handleCamera(const Layout& lo);
    void handleMapPaint(const Layout& lo);
    bool edgeDown(int key);

    // ── 3-D scene / camera ───────────────────────────────────────────────────
    static constexpr float kTileWorldSize = 1.0f;  // world units per grid tile

    [[nodiscard]] odai::importer::ImportedScene buildCityScene() const;
    [[nodiscard]] render::CameraPose computeCameraPose() const;
    // Orthographic ray from a screen pixel to the world y=0 ground plane.
    bool screenToGroundXZ(ui::UiVec2 screenPx, const Layout& lo, float& outX, float& outZ) const;
    [[nodiscard]] ui::UiVec2 worldToScreen(float wx, float wy, float wz, const Layout& lo) const;

    // ── Drawing ──────────────────────────────────────────────────────────────
    void drawWorldOverlay(const Layout& lo);  // hover outline, tooltip, land-value legend
    void drawTopBar(const Layout& lo);
    void drawPalette(const Layout& lo);
    void drawControls(const Layout& lo);
    void drawMinimap(const Layout& lo);
    void drawReports(const Layout& lo);
    void drawFlash(const Layout& lo);

    // ── Small immediate-mode helpers ─────────────────────────────────────────
    bool uiButton(const ui::UiRect& r, std::string_view label, bool active,
                  const ui::UiColor& accent, const ui::Font* font = nullptr,
                  bool enabled = true);
    void strokeLine(ui::UiVec2 a, ui::UiVec2 b, float widthPx, const ui::UiColor& c);
    void textLeft(const ui::Font& f, std::string_view s, float x, float y, const ui::UiColor& c);
    void textCenter(const ui::Font& f, std::string_view s, float cx, float cy, const ui::UiColor& c);
    void textRight(const ui::Font& f, std::string_view s, float rx, float cy, const ui::UiColor& c);

    [[nodiscard]] const std::vector<float>& history(Metric m) const;

    // ── State ────────────────────────────────────────────────────────────────
    std::array<Tile, static_cast<std::size_t>(kGridW) * kGridH> m_tiles{};

    Tool   m_tool = Tool::ZoneR;
    double m_money = 50000.0;
    int    m_population = 0;
    int    m_jobs = 0;
    float  m_education = 8.0f;
    float  m_health = 12.0f;
    float  m_happiness = 55.0f;
    float  m_powerCoverage = 1.0f;        // fraction of developed tiles powered
    float  m_resDemand = 0.45f, m_comDemand = 0.40f, m_indDemand = 0.35f;
    double m_lastNet = 0.0;
    int    m_year = 1;
    int    m_month = 0;                    // 0..11

    std::vector<float> m_histPop, m_histMoney, m_histEdu, m_histHealth, m_histHappy;

    // Building / network counts, refreshed by recomputeStats().
    int m_numRoad = 0, m_numPolice = 0, m_numFire = 0, m_numClinic = 0;
    int m_numSchool = 0, m_numPark = 0, m_numPower = 0;
    int m_numLibrary = 0, m_numAmphitheater = 0;
    float m_statEase = 1.0f;               // easing rate applied to city quality stats

    int   m_speed = 1;                     // 1 / 2 / 3
    bool  m_paused = false;
    float m_simAccum = 0.0f;

    float m_camFocusX = 0.0f, m_camFocusZ = 0.0f;  // world-space (tile units) look-at point
    float m_camZoom = 18.0f;                        // orthoHalfHeight, world units
    float m_camYawDeg = 45.0f;                       // rotates in 90 deg steps (Q/E)
    bool  m_camInit = false;
    bool  m_sceneDirty = true;                       // rebuild+upload buildCityScene() next frame
    // Growth (stepMonth) marks m_growthDirty instead of m_sceneDirty directly and
    // waits out m_sceneRebuildCooldown before promoting it — uploadImportedScene()
    // rebuilds GPU ray-tracing geometry on every call, so re-uploading on every
    // simulated month (up to ~2x/sec at 3x speed) causes visible stutter. Direct
    // player edits (bulldoze/zone/road/building) still set m_sceneDirty instantly.
    bool  m_growthDirty = false;
    float m_sceneRebuildCooldown = 0.0f;

    int m_hoverC = -1, m_hoverR = -1;
    bool m_mouseOverUi = false;
    bool m_dragPainting = false;

    bool   m_reportsOpen = true;
    bool   m_showLandValue = false;        // toggle the desirability data overlay
    Metric m_metric = Metric::Population;

    float       m_flashTimer = 0.0f;
    std::string m_flashMsg;
    float       m_time = 0.0f;

    std::unordered_map<int, bool> m_keyPrev;
    std::uint32_t m_rng = 0x1234567u;

    render::CameraPose m_camera{};
};

}  // namespace odai::games::citybuilder
