#pragma once

// Free-roam viewer for cooked Fallout: New Vegas content.
//
// This exists because the only other thing that could display a cooked scene
// was odai_material_editor, which runs with wantsMinimalRendering() and so
// cannot show a lit surface, a sky, or a shadow. A real GameApp gets the full
// pass stack, which is the whole point: sun, shadow maps, ambient occlusion,
// atmosphere and the time-of-day that drives them.
//
// Scene selection, in order: --scene <path>, then $ODAI_FNV_SCENE.
//
// Controls:
//   W/A/S/D + mouse   walk (fly, in fly mode)
//   Space             jump (walk mode) / ascend (fly mode)
//   Ctrl              descend (fly mode)
//   Shift             sprint
//   E                 enter/exit through the door you are facing
//   F                 toggle walk / fly
//   [ / ]             step time of day back / forward
//   P                 pause the day cycle
//   Tab               release the mouse
//   F3                CPU timing overlay (GameApp reserves this)

#include "core/job_system.h"
#include "engine/game_app.h"
#include "games/newvegas/newvegas_collision.h"
#include "import/fnv/cell_streamer.h"
#include "import/imported_scene.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace odai::games::newvegas {

class NewVegasApp : public engine::GameApp {
public:
    void setScenePath(std::string path) { m_scenePath = std::move(path); }
    // Render this many frames, write a PPM capture, then quit. Lets a visual
    // change be checked without a human at the monitor -- the reason it exists
    // is that a Wayland desktop refuses external screenshot capture, so bugs
    // that are obvious on screen were otherwise being diagnosed blind.
    void setScreenshotRequest(std::string path, int warmupFrames) {
        m_screenshotPath = std::move(path);
        m_screenshotWarmupFrames = warmupFrames;
    }

    // Stream directly from the game's own data directory (the one holding
    // FalloutNV.esm and the .bsa archives) instead of loading a cooked scene.
    // Mutually exclusive with setScenePath(): the streamer owns renderer
    // residency and a full-scene upload would clear its chunks.
    void setStreamDataPath(std::string path) { m_streamDirectory = std::move(path); }
    void setStreamPlugin(std::string plugin) { m_streamPlugin = std::move(plugin); }
    void setStreamWorldspace(std::string worldspace) { m_streamWorldspace = std::move(worldspace); }
    // Spawn on the doorstep of this interior cell. Empty means "centre of the
    // worldspace" instead.
    void setStreamSpawnInterior(std::string editorId) { m_streamSpawnInterior = std::move(editorId); }
    // On-disk cache for built cells. Empty string disables it.
    void setStreamCacheDirectory(std::string path) { m_streamCacheDirectory = std::move(path); }
    void setStreamCacheEnabled(bool enabled) { m_streamCacheEnabled = enabled; }

protected:
    // Fallout's world is ~70 units per metre; the strategy-map preset's AO
    // radius and forced ray-tracing-off are both wrong at that scale. See the
    // base declaration for what opting out actually changes.
    bool wantsStrategyMapTuning() const override { return false; }

    bool onInit() override;
    bool initStreaming();
    void updateStreaming(float deltaSeconds);
    void runCollisionSelfTest();
    void onTick(float deltaSeconds) override;
    void onRender(float deltaSeconds) override;

private:
    void applyTimeOfDay();
    void updateCamera(float deltaSeconds);
    void drawHud();
    // Flattens the cooked terrain mesh into a regular height lattice so the
    // camera can be ground-clamped without a per-frame ray cast.
    void buildGroundHeightField(const importer::ImportedScene& scene);
    // Loads a cooked scene and takes the camera to `arrival` when given, or to
    // the Goodsprings placements otherwise. Re-entrant: this is what walking
    // through a door calls.
    bool loadScene(const std::filesystem::path& path, const float* arrivalPosition, const float* arrivalYawDegrees);
    // Nearest door the camera is close to and facing, or -1. Also what the HUD
    // prompt reads.
    [[nodiscard]] int findUsableDoor() const;
    void useDoor(const importer::ImportedSceneDoor& door);
    // Bilinear ground height at a world XZ. false outside the cooked terrain or
    // over a lattice hole, in which case outHeight is untouched.
    [[nodiscard]] bool groundHeightAt(float x, float z, float& outHeight) const;

    std::string m_scenePath;
    // Doors in the loaded scene, plus what is needed to find their targets:
    // interiors are cooked beside the exterior as "<stem>_<CellEditorID>.bin"
    // (importedSceneInteriorFileName), and a door with an empty target cell is
    // the way back to "<stem>.bin".
    std::vector<importer::ImportedSceneDoor> m_doors;
    std::filesystem::path m_sceneDirectory;
    std::string m_exteriorStem;
    bool m_doorKeyLatch = false;
    // Screenshot-and-quit mode; empty path means normal interactive running.
    std::string m_screenshotPath;
    int m_screenshotWarmupFrames = 8;
    int m_framesRendered = 0;

    // Terrain height lattice: row-major, m_groundCols * m_groundRows samples at
    // kGroundGridSpacing, with m_groundOriginX/Z the world position of sample 0.
    std::vector<float> m_groundHeights;
    float m_groundOriginX = 0.0f;
    float m_groundOriginZ = 0.0f;
    int m_groundCols = 0;
    int m_groundRows = 0;

    // Camera. Bethesda units are ~1.43 cm, so these speeds are large numbers
    // that correspond to ordinary human-scale movement: 400 u/s is about
    // 5.7 m/s, a fast jog.
    float m_cameraX = 0.0f;
    float m_cameraY = 0.0f;
    float m_cameraZ = 0.0f;
    float m_yawDegrees = 0.0f;
    // Level at spawn: onInit stands the camera on the ground in Goodsprings, and
    // from eye height the horizon (and therefore the sky) belongs on screen.
    float m_pitchDegrees = 0.0f;
    double m_lastCursorX = 0.0;
    double m_lastCursorY = 0.0;
    bool m_hasCursorSample = false;
    bool m_mouseCaptured = true;

    // Time of day in hours [0, 24). Drives the sun angle, and through it the
    // shadow direction and the atmosphere's sky colour.
    float m_timeOfDayHours = 9.5f;
    bool m_dayCyclePaused = true;
    float m_dayCycleHoursPerSecond = 0.15f;
    bool m_bracketLeftLatch = false;
    bool m_bracketRightLatch = false;
    bool m_pauseLatch = false;
    bool m_tabLatch = false;
    // Ground-clamped FPS movement; F drops back to the old free-fly camera.
    bool m_walkMode = true;
    bool m_walkModeLatch = false;
    // Jump state, walk mode only. Not latched: holding Space bunny-hops, which
    // is what Fallout does too.
    bool m_airborne = false;
    float m_verticalVelocity = 0.0f;

    // Cell streaming. Null unless --stream was given. The job system is owned
    // here rather than by the streamer so its thread count is visible at the
    // call site and so it outlives every in-flight load.
    std::string m_streamDirectory;
    std::string m_streamPlugin = "FalloutNV.esm";
    std::string m_streamWorldspace = "WastelandNV";
    // Where the game itself starts you: the doorstep of Doc Mitchell's house in
    // Goodsprings.
    std::string m_streamSpawnInterior = "GSDocMitchellHouse";
    std::string m_streamCacheDirectory;
    bool m_streamCacheEnabled = true;
    std::unique_ptr<core::JobSystem> m_streamJobs;
    std::unique_ptr<importer::fnv::CellStreamer> m_streamer;
    // Collision for the streamed world. The single-scene (--scene) path keeps
    // using the older height field built from the whole scene.
    CollisionWorld m_collision;
    // Previous frame's camera position, differenced to get the velocity the
    // planner predicts with. Cheaper and more honest than plumbing the movement
    // code's own velocity, which is reset by collision and jumping.
    float m_previousCameraX = 0.0f;
    float m_previousCameraY = 0.0f;
    float m_previousCameraZ = 0.0f;
    bool m_hasPreviousCameraPosition = false;
    float m_streamStatsLogTimer = 0.0f;
    bool m_collisionSelfTestDone = false;
};

}  // namespace odai::games::newvegas
