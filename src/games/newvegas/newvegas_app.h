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

#include "anim/skeleton.h"
#include "core/job_system.h"
#include "engine/game_app.h"
#include "import/fnv/character_builder.h"
#include "games/newvegas/newvegas_collision.h"
#include "import/fnv/cell_streamer.h"
#include "ui/nav_focus.h"
#include "ui/nav_input.h"
#include "ui/toast_host.h"
#include "import/imported_scene.h"

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_set>
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
    // Stand one GPU-skinned character in bind pose against the sky, instead of
    // loading a world at all.
    //
    // This is a rig check, not a game mode: it isolates "does the skeleton bind
    // and skin correctly" from every other thing that can make a character look
    // wrong (placement, animation, streaming, lighting). Assets still come from
    // the installed game's archives, so it needs a Data directory found the
    // same way streaming finds one -- it just does not stream anything from it.
    //
    // Empty partPaths means the default male body.
    void setCharacterMode(std::string skeletonPath, std::vector<std::string> partPaths) {
        m_characterMode = true;
        if (!skeletonPath.empty()) {
            m_characterSkeletonPath = std::move(skeletonPath);
        }
        if (!partPaths.empty()) {
            m_characterPartPaths = std::move(partPaths);
        }
    }

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
    // Loads the skeleton and body parts, binds them, and uploads the result to
    // skinned instance slot 0. Also frames the camera on the bind-pose bounds,
    // because the character's own extent is the only sensible thing to point at
    // when there is no world.
    bool initCharacter(const std::filesystem::path& dataFilesPath);
    // Re-submits the bind pose. Called every frame, not once: the backend
    // consumes the pose during the frame it was set for and does not retain it.
    void updateCharacterPose();
    void updateStreaming(float deltaSeconds);
    void runCollisionSelfTest();
    void onTick(float deltaSeconds) override;
    void onRender(float deltaSeconds) override;

private:
    void applyTimeOfDay();
    // Reads keyboard AND gamepad into one device-agnostic nav snapshot. Both
    // always run: a player can have a controller plugged in and still reach for
    // Escape, and making them exclusive means whichever the game guessed wrong
    // about simply stops working.
    void pollNavInput(float deltaSeconds);
    // Checks the regions covering the camera and toasts any not seen before.
    void updateRegionDiscovery();
    void drawPipBoyHud();
    // The pause menu. Controller-navigable; returns nothing because every entry
    // acts on app state directly.
    void drawPauseMenu();
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

    // Bind-pose character view (--character). See setCharacterMode.
    bool m_characterMode = false;
    std::string m_characterSkeletonPath = "characters\\_male\\skeleton.nif";
    std::vector<std::string> m_characterPartPaths = {"characters\\_male\\upperbody.nif"};
    importer::fnv::FalloutCharacter m_character;
    // The draw list handed to the renderer, one entry per body part. Held as a
    // member because ImportedSkinnedMeshTemplate takes spans -- a local would
    // dangle the moment initCharacter returned.
    std::vector<importer::ImportedScenePackedDraw> m_characterDraws;
    std::vector<odai::math::Matrix4> m_characterBindPose;
    // Bind pose with the actor's world placement folded in, which is how the
    // skinning pass wants it: there is no separate instance transform for a
    // skinned actor, so world placement rides on the bone matrices.
    std::vector<odai::math::Matrix4> m_characterPoseScratch;
    // Where the character stands, in world units. Folded into every bone matrix
    // each frame rather than being an instance transform, because a skinned
    // actor has no instance transform.
    float m_characterWorldX = 0.0f;
    float m_characterWorldY = 0.0f;
    float m_characterWorldZ = 0.0f;

    // ---- UI ----------------------------------------------------------
    // Two notification hosts, because they are two different idioms (see
    // ToastPlacement): m_banner is the big centred "Goodsprings" announcement,
    // m_toasts is the corner stack for everything else. One host cannot be both
    // at once -- placement is a property of the host's style, and a discovery
    // arriving while a corner toast is up must not restyle it mid-fade.
    ui::ToastHost m_banner;
    ui::ToastHost m_toasts;
    ui::UiNavInput m_nav;
    ui::UiNavRepeater m_navRepeat;
    ui::UiNavStickMapper m_navStick;
    ui::NavFocusRing m_menuFocus;
    bool m_menuOpen = false;
    // True once a controller (or the d-pad keys) last drove the UI, false once
    // the mouse moves. Drives whether the focus highlight is drawn at all --
    // an always-on highlight looks broken with a mouse, and a never-on one
    // makes the menu unusable from a couch.
    bool m_navDriving = false;
    // Region names already announced. Discovery is once per session per region;
    // walking back into Nipton does not re-announce it.
    std::unordered_set<std::string> m_discoveredRegions;
    float m_regionPollSeconds = 0.0f;

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
