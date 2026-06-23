#pragma once

#include "engine/game_app.h"
#include "ui/ui_types.h"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// A compact SimCity-2013-style city builder built entirely on the headless UI
// draw list. Zone tiles (residential / commercial / industrial), lay roads, drop
// municipal buildings (police, fire, clinic, school, park, power plant), watch a
// money / population / power HUD, a live minimap, and floating report windows that
// plot Education, Health, Happiness, Population and Treasury over time.
//
// Everything is immediate-mode: the map, the chrome and the charts are all
// re-emitted to m_uiDrawList every frame. The only persistent state is the tile
// grid, the economy, and the per-month history series feeding the report plots.
namespace odai::games::citybuilder {

enum class Terrain : std::uint8_t { Grass, Water };
enum class Zone : std::uint8_t { None, Residential, Commercial, Industrial };
enum class Building : std::uint8_t { None, Police, Fire, Clinic, School, Park, Power };

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
        Police, Fire, Clinic, School, Park, Power,
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
    void clampCamera(const ui::UiRect& map);
    void handleCamera(const Layout& lo);
    void handleMapPaint(const Layout& lo);
    bool edgeDown(int key);

    // ── Drawing ──────────────────────────────────────────────────────────────
    void drawMap(const Layout& lo);
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
    float m_statEase = 1.0f;               // easing rate applied to city quality stats

    int   m_speed = 1;                     // 1 / 2 / 3
    bool  m_paused = false;
    float m_simAccum = 0.0f;

    float m_camX = 0.0f, m_camY = 0.0f, m_tilePx = 0.0f;
    bool  m_camInit = false;

    int m_hoverC = -1, m_hoverR = -1;
    bool m_mouseOverUi = false;
    bool m_dragPainting = false;

    bool   m_reportsOpen = true;
    Metric m_metric = Metric::Population;

    float       m_flashTimer = 0.0f;
    std::string m_flashMsg;
    float       m_time = 0.0f;

    std::unordered_map<int, bool> m_keyPrev;
    std::uint32_t m_rng = 0x1234567u;

    render::CameraPose m_camera{};
};

}  // namespace odai::games::citybuilder
