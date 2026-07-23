#pragma once

#include "engine/game_app.h"
#include "games/citybuilder/citybuilder_citizens.h"
#include "games/citybuilder/script/city_script.h"
#include "import/imported_scene.h"
#include "procgen/building_generator.h"
#include "procgen/props.h"
#include "ui/ui_types.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
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
    bool  powered  = false;      // within range of a power plant (direct or via the grid)
    bool  poweredRoad = false;   // road tile carrying power from a plant (drives pole/wire art)
    bool  nearRoad = false;      // within reach of a road (required to develop)
    float desirability = 0.5f;   // 0..1 spatial land value; modulates growth target
    float scenicPhase = 0.0f;    // per-tile jitter so a block doesn't look uniform
    float zoneAge = 0.0f;        // real seconds connected+vacant since zoned (listing period)
    float trafficLoad = 0.0f;    // EMA of car occupancy on this road tile (congestion)
    std::uint8_t fireTicks = 0;  // months of burning left; 0 = not on fire
    std::uint8_t charTicks = 0;  // months since burning out (charred rubble ages away)
    bool  charred = false;       // burnt-out ruin: develop = 0, drags neighbours down
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
        Match,   // arson-on-demand: set one developed parcel alight, on purpose
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
    // Lua-namegen results, cached by seed so hover tooltips never re-enter Lua.
    // The business category is load-bearing for the citizen sim: sims route to
    // the same named storefront the tooltip shows. A whole building shares one
    // name (keyed off its plot origin) rather than a different one per tile;
    // strip-mall commercial plots (a linear row of storefronts) are the sole
    // exception and keep a distinct name per tile.
    const odai::citybuilder::BusinessName& businessNameAt(int c, int r, const Tile& t);
    const std::string& blockNameAt(int c, int r, const Tile& t);
    // Tile whose seed names the building (c,r) belongs to (its plot origin, or
    // the tile itself for a strip-mall storefront / when no plot is known).
    void nameAnchor(int c, int r, int& outC, int& outR) const;
    // Street names key off the colinear road run the tile belongs to, so a
    // whole avenue shares one name with no stored per-tile state.
    const std::string& streetNameAt(int c, int r);
    void recomputeStats();   // power/road coverage, population, jobs, eased city stats
    void computeDesirability();  // per-tile spatial land value (amenities vs. nuisances)
    void pushHistory();      // append a sample to each metric series
    void stepMonth();
    void stepFire();         // ignition, spread, burn-out, and rubble aging
    void applyTool(int c, int r);
    // Switches the active tool, cancelling (without applying) any box-select
    // drag in progress so a hotkey or palette click mid-drag can't apply the
    // wrong tool to a stale rectangle.
    void setTool(Tool t);
    void bulldoze(int c, int r);
    bool placeBuilding(int c, int r, Building b);
    bool charge(double cost);
    void flash(std::string msg);

    // ── Per-frame layout & input ─────────────────────────────────────────────
    struct Layout {
        float s = 1.0f, fw = 0.0f, fh = 0.0f;
        ui::UiRect topBar{}, palette{}, map{}, controls{}, minimap{}, reports{}, ticker{};
    };
    [[nodiscard]] Layout computeLayout() const;
    void clampCameraFocus();
    void handleCamera(const Layout& lo);
    void handleMapPaint(const Layout& lo);
    bool edgeDown(int key);

    // ── 3-D scene / camera ───────────────────────────────────────────────────
    static constexpr float kTileWorldSize = 1.0f;  // world units per grid tile

    [[nodiscard]] odai::importer::ImportedScene buildCityScene() const;
    // Memoized era-styled zone buildings keyed by (kind, level, tier, variant,
    // plot size, orientation parity) — generated lazily on first use. swapDims
    // generates the building with lot extents exchanged so an odd quarter-turn
    // rotation lands it back onto the plot footprint. mutable because
    // buildCityScene() is const.
    const procgen::TriMesh& cachedBuilding(procgen::BuildingKind kind, int level, int tier,
                                           std::uint32_t variant, int plotW, int plotD,
                                           bool swapDims) const;
    // Civic buildings get recognizable CSG silhouettes (hose tower, colonnade,
    // smokestacks, ...) — 4 seeded variants per service. Lots are square, so no
    // swapped-extents variant is needed for rotation.
    const procgen::TriMesh& cachedCivic(Building b, std::uint32_t variant) const;
    // Trees are cached per (season, variant) so the whole canopy repaints on a
    // season change without regenerating on every scene rebuild.
    const procgen::TriMesh& cachedTree(std::uint32_t variant) const;
    const procgen::TriMesh& cachedPumpkin(std::uint32_t variant) const;
    const procgen::TriMesh& cachedPowerPole(std::uint32_t variant) const;
    const procgen::TriMesh& cachedStreetlamp(std::uint32_t variant) const;
    const procgen::TriMesh& cachedBench(std::uint32_t variant) const;
    const procgen::TriMesh& cachedHydrant(std::uint32_t variant) const;
    const procgen::TriMesh& cachedBillboard(std::uint32_t variant) const;
    const procgen::TriMesh& cachedBusStop(std::uint32_t variant) const;
    const procgen::TriMesh& cachedTrashCan(std::uint32_t variant) const;
    const procgen::TriMesh& cachedSnowman(std::uint32_t variant) const;

    // ── Rising buildings ─────────────────────────────────────────────────────
    // When a plot's building first appears (construction completes, or an era
    // upgrade swaps the mesh), it isn't popped into the static scene — it
    // rises out of the ground over ~1.5s with a dirt burst, drawn through the
    // per-frame actor stream. buildCityScene() detects the appearance (it is
    // the only place that knows when a plot renders a building), skips the
    // static emit while the rise is live, and hands the exact placement
    // parameters to buildActorFrameData(). All bookkeeping is mutable because
    // buildCityScene() is const.
    struct RisingBuilding {
        short c = 0, r = 0;            // plot origin tile
        std::uint8_t pw = 1, pd = 1;   // plot extent, tiles
        std::uint8_t level = 1, tier = 1, turns = 0;
        bool swapDims = false;
        std::uint32_t variant = 0;
        procgen::BuildingKind kind = procgen::BuildingKind::Residential;
        procgen::Color3 tint{1.0f, 1.0f, 1.0f};
        float t0 = 0.0f;               // rise start (m_time)
        bool burst = false;            // dirt-burst fx fired yet?
    };

    // ── Ambient traffic ──────────────────────────────────────────────────────
    // Cars follow the road graph tile-to-tile on the right-hand lane; their
    // geometry is rebuilt each frame from cached car meshes and streamed to
    // the renderer via ImportedActorFrameData (a FrameArena upload), so the
    // static city scene is never re-uploaded for animation.
    struct Vehicle {
        short cx = 0, cr = 0;          // tile currently being traversed
        signed char inX = 1, inZ = 0;  // direction the car entered with
        signed char outX = 1, outZ = 0;  // direction it will leave with
        float t = 0.0f;                // 0..1 progress across the tile
        float speed = 1.2f;            // tiles per second
        std::uint8_t variant = 0;      // index into m_carMeshes
        // Citizen-trip route (BFS over the road graph, packed r*kGridW+c).
        // Ambient cars leave this empty; routed cars despawn on arrival.
        std::vector<std::uint16_t> route;
        std::uint16_t routeIdx = 0;
    };
    void updateVehicles(float dt);
    void respawnVehicle(Vehicle& v);
    bool pickExit(Vehicle& v);         // choose outX/outZ at the current tile

    // ── Citizen trips (destination-routed traffic) ───────────────────────────
    void rebuildDestinations();        // named businesses + civic destinations
    void reconcileCitizens();          // monthly roster churn + story rolls
    void spawnCitizenTrip();           // roll a schedule-appropriate trip into a routed car
    bool routeRoad(short fromC, short fromR, short toC, short toR,
                   std::vector<std::uint16_t>& outRoute);   // BFS on road tiles
    bool nearestRoad(short c, short r, short& outC, short& outR) const;
    void updateRoutedVehicles(float dt);
    void drawTicker(const Layout& lo);

    // ── Day/week schedule (civic clock) ──────────────────────────────────────
    // A theatrical clock, separate from the economic month: one day lasts
    // kDayLengthSeconds at 1x and drives everything routine — commute waves,
    // the school bus loop, trash day, Saturday soccer. See updateSchedule().
    [[nodiscard]] float dayHour() const;            // 0..24
    void updateSchedule(float dt);                  // clock + scheduled spawns
    // Chain BFS legs through waypoints into one long route (school bus /
    // garbage truck loops). Returns false if any leg is unroutable.
    bool buildServiceRoute(const std::vector<std::pair<short, short>>& waypoints,
                           std::vector<std::uint16_t>& outRoute);
    void spawnSchoolBusRun();
    void spawnGarbageRun();
    // Advance a fleet of route-following vehicles; shared by citizen cars
    // (which drop a pedestrian on arrival) and the service fleets (which
    // simply despawn at the end of their loop).
    void advanceRoutedFleet(std::vector<Vehicle>& fleet, float dt, bool arrivalPedestrian);

    // ── Sims ─────────────────────────────────────────────────────────────────
    // Little box-people going about their day on the sidewalks: they spawn
    // where the city is actually alive (developed homes and shops, parks) and
    // wander the road graph on the sidewalk band, bobbing as they walk. Same
    // per-frame actor stream as the cars — never a scene upload. Distinct from
    // the Pedestrian below: Sims are ambient population density, not tied to a
    // citizen's routed trip.
    struct Sim {
        short cx = 0, cr = 0;
        signed char inX = 1, inZ = 0;
        signed char outX = 1, outZ = 0;
        float t = 0.0f;
        float speed = 0.3f;            // tiles per second (walking pace)
        float phase = 0.0f;            // bob/stride offset so crowds don't sync
        std::uint8_t variant = 0;      // index into m_simMeshes
    };
    void updateSims(float dt);
    void respawnSim(Sim& s);
    bool pickSimExit(Sim& s);          // wander-y routing: parks and shops attract

    // ── Ambient pedestrians & boats ──────────────────────────────────────────
    // Structurally slow vehicles: pedestrians walk the sidewalk rail of road
    // tiles near developed frontage (dropped off by an arriving citizen trip —
    // see advanceRoutedFleet); boats drift the connected river/lake water (the
    // terrain generator guarantees edge-to-edge connectivity, so they never
    // strand). Both stream through the same actor path as cars.
    struct Pedestrian {
        short cx = 0, cr = 0;
        signed char inX = 1, inZ = 0;
        signed char outX = 1, outZ = 0;
        float t = 0.0f;
        float speed = 0.2f;            // tiles per second (strolling)
        std::uint8_t variant = 0;      // index into m_pedMeshes
    };
    void updatePedestrians(float dt);
    void respawnPedestrian(Pedestrian& p);

    struct Boat {
        short cx = 0, cr = 0;
        signed char inX = 1, inZ = 0;
        signed char outX = 1, outZ = 0;
        float t = 0.0f;
        float speed = 0.3f;
        float bobPhase = 0.0f;         // per-boat offset into the bob sinusoid
        std::uint8_t variant = 0;      // index into m_boatMeshes
    };
    void updateBoats(float dt);
    void respawnBoat(Boat& b);

    // ── Fire trucks ──────────────────────────────────────────────────────────
    // When something burns and the city has a Fire Dept, red trucks roll out
    // from the station, navigate the road graph toward the nearest fire, park
    // beside it, and hose it down (stepFire treats a parked truck as maximum
    // coverage on the tiles around it). Light bar and water arc are stateless
    // per-frame particles.
    struct FireTruck {
        short cx = 0, cr = 0;
        signed char inX = 1, inZ = 0;
        signed char outX = 1, outZ = 0;
        float t = 0.0f;
        float speed = 1.9f;            // sirens on — faster than traffic, no jams
        bool  parked = false;
        bool  returning = false;       // fires are out; driving home to despawn
        short tgtC = -1, tgtR = -1;    // burning tile being fought (or home, returning)
        short homeC = -1, homeR = -1;  // staging road tile it rolled out from
        // Livelock guard: the exit picker is a memoryless greedy descent, so a
        // road graph that requires temporarily moving AWAY from the target
        // (e.g. around the river to the one bridge) can trap a truck pacing
        // between two tiles forever. Track the best distance reached; too many
        // moves without improving it means the target is unreachable — give up.
        short bestDist = 32767;
        std::uint8_t stallMoves = 0;
    };
    void updateFireTrucks(float dt);
    bool pickTruckExit(FireTruck& tk);  // goal-directed: descend distance to target
    [[nodiscard]] bool truckSuppressed(int c, int r) const;  // parked truck hosing this tile?

    // ── Celebration FX ───────────────────────────────────────────────────────
    // Short-lived stateless particle bursts (zone puffs, build bursts, level-up
    // confetti) rendered from (position, birth time, color) each frame.
    struct Fx {
        float x = 0.0f, z = 0.0f;
        float t0 = 0.0f;
        float r = 1.0f, g = 1.0f, b = 1.0f;
        std::uint8_t kind = 0;         // 0 = small puff, 1 = build burst, 2 = confetti
    };
    void addFx(float worldX, float worldZ, const ui::UiColor& color, std::uint8_t kind);

    // ── Weather ──────────────────────────────────────────────────────────────
    // A simple state machine rolls the sky every ~half minute with seasonal
    // odds (winter precipitates as snow). Precipitation is a particle field
    // around the camera focus, streamed with the cars — never a scene upload.
    enum class Weather : std::uint8_t { Clear, Rain, Snow };
    struct WeatherDrop {
        float x = 0.0f, y = 0.0f, z = 0.0f;
        float phase = 0.0f;  // per-drop drift phase (snow sway)
        float speed = 1.0f;  // fall speed multiplier
    };
    void updateWeather(float dt);
    void respawnDrop(WeatherDrop& d, bool atTop);

    // ── Severe weather ───────────────────────────────────────────────────────
    // Two continuous atmosphere variables — surface heat (season + the city's
    // own industrial heat island, cooled by rain) and convective instability
    // (charges in hot clear spells, discharges as rain) — place each incoming
    // front on a continuum: drizzle, thunderstorm, or a tornado-bearing storm.
    // The funnel is the release valve of energy the simulation (and partly the
    // player's zoning) accumulated, never a scripted event: spawning one
    // consumes the stored instability, so the atmosphere must recharge before
    // another is possible.
    struct Tornado {
        float x = 0.0f, z = 0.0f;   // world position
        float heading = 0.0f;       // radians; wanders, follows wind + warm ground
        float intensity = 1.0f;     // decays; faster over water/parks/open land
    };
    void updateSevereWeather(float dt);  // funnel spawn/motion — sim-time, respects pause
    void stepTornadoDamage();            // monthly damage, feeds the charred/fire loops

    // Appends this frame's transformed car geometry and weather particles into
    // the actor scratch buffers and returns the frame data for submitFrame.
    render::ImportedActorFrameData buildActorFrameData();
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

    // Per-tile plot membership, re-derived from the zone map every scene rebuild
    // (see buildCityScene's parceling). Lets the hover tooltip and citizen
    // destinations name a whole building consistently instead of per tile.
    struct PlotInfo { short c = -1, r = -1; std::uint8_t w = 1, d = 1; };
    mutable std::array<PlotInfo, static_cast<std::size_t>(kGridW) * kGridH> m_tilePlots{};

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
    int m_burningTiles = 0, m_charredTiles = 0;  // refreshed by stepFire()
    float m_statEase = 1.0f;               // easing rate applied to city quality stats

    int   m_speed = 1;                     // 1 / 2 / 3
    bool  m_paused = false;
    float m_simAccum = 0.0f;

    float m_camFocusX = 0.0f, m_camFocusZ = 0.0f;  // world-space (tile units) look-at point
    float m_camZoom = 18.0f;                        // orthoHalfHeight, world units
    float m_camYawDeg = 45.0f;                       // rotates in 90 deg steps (Q/E)
    bool  m_camInit = false;
    // Zoom LOD tier for the era buildings: 1 adds the window pass when the
    // camera is close enough to read facades; crossing the (hysteretic)
    // threshold re-extrudes the scene through the normal rebuild path.
    int   m_lodDetail = 0;
    bool  m_sceneDirty = true;                       // rebuild+upload buildCityScene() next frame
    // Growth (stepMonth) marks m_growthDirty instead of m_sceneDirty directly and
    // waits out m_sceneRebuildCooldown before promoting it — uploadImportedScene()
    // rebuilds GPU ray-tracing geometry on every call, so re-uploading on every
    // simulated month (up to ~2x/sec at 3x speed) causes visible stutter. Direct
    // player edits (bulldoze/zone/road/building) still set m_sceneDirty instantly.
    bool  m_growthDirty = false;
    float m_sceneRebuildCooldown = 0.0f;
    mutable std::unordered_map<std::uint32_t, procgen::TriMesh> m_buildingCache;
    mutable std::unordered_map<std::uint32_t, procgen::TriMesh> m_civicCache;
    mutable std::unordered_map<std::uint32_t, procgen::TriMesh> m_treeCache;
    mutable std::vector<procgen::TriMesh> m_pumpkinMeshes;
    mutable std::vector<procgen::TriMesh> m_poleMeshes;
    mutable std::vector<procgen::TriMesh> m_lampMeshes;
    // Rise-animation bookkeeping (see RisingBuilding). m_builtSeen remembers
    // the highest building level each plot origin has shown, so appearances
    // and era upgrades trigger a rise exactly once; entries are dropped when a
    // plot stops rendering a building (charred / decayed / re-zoned) so a
    // rebuilt lot rises again. m_riseScratch is a reused ImportedScene the
    // actor pass appends the rotated mesh into, so placement math is shared
    // with the static path rather than duplicated.
    mutable std::vector<RisingBuilding> m_rising;
    mutable std::unordered_map<std::uint32_t, std::uint8_t> m_builtSeen;
    mutable odai::importer::ImportedScene m_riseScratch;
    mutable std::vector<procgen::TriMesh> m_snowmanMeshes;

    procgen::Season m_season = procgen::Season::Winter;  // recomputed in onInit
    Weather m_weather = Weather::Clear;
    Weather m_weatherTarget = Weather::Clear;
    float m_weatherIntensity = 0.0f;   // 0..1 ramp so precipitation fades in/out
    float m_weatherTimer = 14.0f;      // seconds until the next sky roll
    std::uint32_t m_weatherRng = 0xBAD5EEDu;
    std::vector<WeatherDrop> m_drops;

    float m_atmoHeat = 0.3f;           // surface heat: season + city heat island - rain
    float m_atmoInstability = 0.2f;    // convective energy: charges clear, spends as storms
    float m_stormSeverity = 0.0f;      // heat x instability, sampled when a front rolls in
    float m_cityHeat = 0.0f;           // industrial develop + power plants (recomputeStats)
    float m_windX = 1.0f, m_windZ = 0.0f;  // prevailing wind, rolled per front
    bool  m_debugForceStorm = false;   // ODAI_CITY_STORM=1: prime the atmosphere for testing
    std::vector<Tornado> m_tornadoes;

    std::vector<Vehicle> m_vehicles;
    std::vector<procgen::TriMesh> m_carMeshes;             // lazily filled variants
    std::vector<Pedestrian> m_pedestrians;
    std::vector<procgen::TriMesh> m_pedMeshes;
    std::vector<Boat> m_boats;
    std::vector<procgen::TriMesh> m_boatMeshes;
    // Static scatter props, lazily generated like pumpkins/poles/lamps.
    mutable std::vector<procgen::TriMesh> m_benchMeshes;
    mutable std::vector<procgen::TriMesh> m_hydrantMeshes;
    mutable std::vector<procgen::TriMesh> m_billboardMeshes;
    mutable std::vector<procgen::TriMesh> m_busStopMeshes;
    std::vector<Sim> m_sims;
    std::vector<procgen::TriMesh> m_simMeshes;             // lazily filled variants
    std::vector<FireTruck> m_trucks;
    procgen::TriMesh m_truckMesh;                          // lazily built
    std::vector<Fx> m_fx;
    std::uint32_t m_trafficRng = 0x51CA7B1u;
    // Per-frame actor stream scratch, reused to avoid per-frame allocation.
    std::vector<odai::importer::ImportedScenePackedVertex> m_actorVertices;
    std::vector<std::uint32_t> m_actorIndices;
    std::vector<odai::importer::ImportedScenePackedDraw> m_actorDraws;

    int m_hoverC = -1, m_hoverR = -1;
    bool m_mouseOverUi = false;
    bool m_dragPainting = false;

    // Box-select zoning/bulldozing: drag from a corner tile to an opposite
    // corner, release to apply the tool to every tile in the rectangle in one
    // action (rather than painting whatever the cursor happens to cross).
    bool m_boxSelecting = false;
    int  m_boxStartC = -1, m_boxStartR = -1;
    int  m_boxEndC = -1, m_boxEndR = -1;

    // Reports start closed: the first thing a new mayor should see is their
    // city, not a chart floating over it (G or the Reports button opens it).
    bool   m_reportsOpen = false;
    bool   m_showLandValue = false;        // toggle the desirability data overlay
    Metric m_metric = Metric::Population;

    float       m_flashTimer = 0.0f;
    std::string m_flashMsg;
    float       m_moneyFlashTimer = 0.0f;  // pulses the treasury red when a charge fails
    // Camera-controls hint chip: each line disappears once the player has
    // actually performed that action, so the chip teaches then gets out of
    // the way (and never nags someone who already knows).
    bool m_usedPan = false, m_usedZoom = false, m_usedRotate = false;
    float       m_time = 0.0f;

    std::unordered_map<int, bool> m_keyPrev;
    std::uint32_t m_rng = 0x1234567u;

    // ── World generation & Lua content ───────────────────────────────────────
    // All generated content (terrain, names, citizen stories) derives from
    // m_worldSeed + position hashes, so it is stable within a session. If a
    // save system is ever added, m_worldSeed must be serialized first.
    std::uint32_t m_worldSeed = 1u;
    short m_siteC = kGridW / 2;            // scored city-site anchor from terrain gen
    short m_siteR = kGridH / 2;
    std::array<float, static_cast<std::size_t>(kGridW) * kGridH> m_forest{};  // 0..1 tree density
    std::vector<std::pair<short, short>> m_riverPath;  // ordered river centerline
    // Lua content host: name generators, story templates, need schedules,
    // tuning config (mods/citybuilder/scripts). Never called per-frame.
    std::unique_ptr<odai::citybuilder::CityScriptHost> m_script;
    std::string m_cityName = "OdaiCity";
    std::unordered_map<std::uint32_t, odai::citybuilder::BusinessName> m_businessNames;
    std::unordered_map<std::uint32_t, std::string> m_blockNames;
    std::unordered_map<std::uint32_t, std::string> m_streetNames;

    // ── Citizen sim ──────────────────────────────────────────────────────────
    CitizenSim m_citizens;
    std::vector<Destination> m_destinations;   // rebuilt each month
    std::vector<HomeSite> m_homeSites;
    std::vector<Vehicle> m_routedVehicles;     // citizen trips, above the ambient cap
    float m_tripTimer = 3.0f;
    float m_storyBoost = 1.0f;                 // ODAI_CITY_STORY=1 -> 10x events

    // ── Day/week clock & scheduled services ──────────────────────────────────
    float m_dayClock = 0.0f;                   // seconds into the current day
    int m_weekday = 0;                         // 0=Mon .. 6=Sun
    bool m_busMorningDone = false;             // one-shot flags, reset at day rollover
    bool m_busAfternoonDone = false;
    bool m_trashRunDone = false;
    bool m_soccerStoryDone = false;
    bool m_trashDayActive = false;             // curbside cans in the scene today
    std::vector<Vehicle> m_busFleet;           // school bus runs in flight
    std::vector<Vehicle> m_trashFleet;         // garbage truck runs in flight
    mutable std::vector<procgen::TriMesh> m_busMeshes;
    mutable std::vector<procgen::TriMesh> m_trashTruckMeshes;
    mutable std::vector<procgen::TriMesh> m_trashCanMeshes;

    render::CameraPose m_camera{};
};

}  // namespace odai::games::citybuilder
