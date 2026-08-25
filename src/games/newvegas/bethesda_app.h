#pragma once

#include "games/newvegas/bethesda_actors.h"
#include "audio/wav_writer.h"
#include "render/video_writer.h"
#include "games/newvegas/newvegas_victor.h"

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
#include "games/newvegas/bethesda_collision.h"
#include "bethesda/navigation_world.h"
#include "games/newvegas/newvegas_traversal_state.h"
#include "import/fnv/cell_streamer.h"
#include "import/fnv/content_profile.h"
#include "import/fnv/weather_records.h"
#include "ui/nav_focus.h"
#include "ui/nav_input.h"
#include "ui/toast_host.h"
#include "ui/widgets/tes3_journal_panel.h"
#include "import/imported_scene.h"
#include "bethesda/bethesda_session.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace odai::games::newvegas {

// Victor, the Goodsprings Securitron: placed, talked to with E, and speaking
// his own imported dialogue. See newvegas_victor.h.

// Loads flythrough waypoints from a text file, returning how many were read (0
// on failure). Free rather than a member because the waypoint list is tour
// DATA, not app state, and because framing a flythrough is iterative -- a
// rebuild per waypoint guess makes that loop useless. It is also what lets one
// binary tour three different games.
int loadTourFile(const std::string& path);

class BethesdaApp : public engine::GameApp {
public:
    // Normal exploration FOV, and the narrowed one a conversation eases to.
    // 75 -> 55 is a 1.36x magnification: enough that Victor reads as the
    // subject of the shot, gentle enough not to feel like a cutscene.
    //
    // These are HORIZONTAL, which is what Gamebryo's own fDefaultFOV means and
    // where the 75 came from. CameraPose::fovDegrees is VERTICAL, so they are
    // converted against the live aspect ratio before being handed over -- see
    // verticalFovDegreesFor. Feeding 75 straight in as a vertical FOV, which is
    // what this used to do, renders ~107 degrees horizontal at 16:9: a good 32
    // degrees wider than New Vegas itself, which is why building facades
    // stretched as the camera passed them.
    //
    // Converting rather than hardcoding a vertical number keeps it correct on
    // ultrawide, where a fixed vertical FOV silently widens instead.
    static constexpr float kDefaultHorizontalFovDegrees = 75.0f;
    static constexpr float kConversationHorizontalFovDegrees = 55.0f;
    // fovY = 2 * atan(tan(fovX / 2) / aspect), aspect = width / height.
    [[nodiscard]] static float verticalFovDegreesFor(
        float horizontalFovDegrees, float aspectRatio);

    void setScenePath(std::string path) { m_scenePath = std::move(path); }
    // Render this many frames, write a PPM capture, then quit. Lets a visual
    // change be checked without a human at the monitor -- the reason it exists
    // is that a Wayland desktop refuses external screenshot capture, so bugs
    // that are obvious on screen were otherwise being diagnosed blind.
    void setScreenshotRequest(std::string path, int warmupFrames) {
        m_screenshotPath = std::move(path);
        m_screenshotWarmupFrames = warmupFrames;
    }

    // Fly a fixed cinematic path through Goodsprings instead of reading input.
    // Non-zero seconds is the whole tour; the path itself is kGoodspringsTour.
    void setFlythroughSeconds(float seconds) { m_flythroughSeconds = seconds; }

    // Record every frame as <directory>/frame_%05d.ppm and quit after `frames`.
    //
    // The world advances by a FIXED 1/fps step while this is on, not by real
    // elapsed time. A recording has to play at the speed it was authored for,
    // and a 28 ms frame that took 60 ms to render would otherwise stretch that
    // moment of the tour to twice its length -- the camera slows down exactly
    // where the renderer is busiest, which is where the interesting geometry is.
    void setCaptureSequence(std::string directory, int frames, float fps) {
        m_captureDirectory = std::move(directory);
        m_captureFrames = frames;
        m_captureFixedDt = (fps > 0.0f) ? (1.0f / fps) : (1.0f / 30.0f);
    }

    // The same capture, encoded on the fly instead of written out as stills.
    // Prefer this: at the sizes the swapchain actually opens a PPM is ~7.7 MB,
    // and three locations at 60 fps is over 30 GB of files that exist only to
    // be read once and deleted.
    void setCaptureVideo(std::string outputPath, int frames, float fps) {
        m_captureVideoPath = std::move(outputPath);
        m_captureFrames = frames;
        m_captureVideoFps = (fps > 0.0f) ? fps : 30.0f;
        m_captureFixedDt = 1.0f / m_captureVideoFps;
    }
    void setCaptureAudio(bool enabled) { m_captureAudioRequested = enabled; }
    void setCaptureSeed(std::uint32_t seed) { m_captureSeed = seed; }
    void setTimeOfDayHours(float hours) {
        m_timeOfDayHours = hours;
        m_timeOfDayExplicit = true;
    }

    // Stream directly from the game's own data directory (the one holding
    // FalloutNV.esm and the .bsa archives) instead of loading a cooked scene.
    // Mutually exclusive with setScenePath(): the streamer owns renderer
    // residency and a full-scene upload would clear its chunks.
    void setStreamDataPath(std::string path) { m_streamDirectory = std::move(path); }
    void setStreamPlugin(std::string plugin) { m_streamPlugin = std::move(plugin); }
    void setLoadOrderPath(std::string path) { m_loadOrderPath = std::move(path); }
    void setContentProfilePath(std::string path) { m_contentProfilePath = std::move(path); }
    void setContentProfileModsRoot(std::string path) { m_contentProfileModsRoot = std::move(path); }
    void setCompatibilityReportPath(std::string path) { m_compatibilityReportPath = std::move(path); }
    void setForceContentReindex(bool enabled) { m_forceContentReindex = enabled; }
    // An additional plugin loaded after the main one, repeatable, in load order.
    // Its masters are pulled in automatically, so naming only the mod is enough.
    // Records are read for weather; cell contents still come from the main
    // plugin alone.
    void addPlugin(std::string plugin) { m_extraPlugins.push_back(std::move(plugin)); }
    // Force a specific weather by editor ID (e.g. "WEAVarNV01"). Empty picks
    // one from the worldspace's climate.
    void setWeather(std::string editorId) { m_requestedWeatherEditorId = std::move(editorId); }
    void setUpscalerSettings(const render::UpscalerSettings& settings) { m_upscalerSettings = settings; }
    [[nodiscard]] render::UpscalerSettings upscalerSettings() const { return m_upscalerSettings; }
    // GameApp consumes this before renderer init, which is the only point at
    // which the quality preset can still choose the render resolution.
    render::UpscalerSettings requestedUpscalerSettings() const override { return m_upscalerSettings; }
    void setStreamWorldspace(std::string worldspace) {
        m_streamWorldspace = std::move(worldspace);
        m_streamWorldspaceExplicit = true;
        m_explicitStart = true;
    }
    // Spawn on the doorstep of this interior cell. Empty means "centre of the
    // worldspace" instead.
    // Start inside a named interior cell -- the room is built and uploaded at
    // startup and the player stands in it, instead of spawning on its doorstep
    // out in the worldspace.
    void startInsideInterior(std::string editorId) {
        m_startInsideInterior = std::move(editorId);
        m_explicitStart = true;
    }

    void setStreamSpawnInterior(std::string editorId) {
        m_streamSpawnInterior = std::move(editorId);
        m_streamSpawnInteriorExplicit = true;
        m_explicitStart = true;
    }
    void setTraversalStatePath(std::string path) { m_traversalStatePath = std::move(path); }
    void setResumeEnabled(bool enabled) { m_resumeEnabled = enabled; }
    void setScenario(std::string id);
    void setGameplaySavePath(std::string path) { m_gameplaySavePath = std::move(path); }
    void setGameplayLoadPath(std::string path) { m_gameplayLoadPath = std::move(path); }
    void setTes3StartQuest(std::string id, std::int32_t index) {
        m_tes3StartQuest = std::move(id);
        m_tes3StartQuestIndex = index;
    }
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
    // Repeatable, in load order; later directories win. Only meaningful in
    // streaming mode -- a cooked scene already has its textures baked in.
    void addModDirectory(std::string path) { m_modDirectories.push_back(std::move(path)); }
    void setStreamCacheDirectory(std::string path) { m_streamCacheDirectory = std::move(path); }
    void setStreamCacheEnabled(bool enabled) { m_streamCacheEnabled = enabled; }

protected:
    // Fallout's world is ~70 units per metre; the strategy-map preset's AO
    // radius and forced ray-tracing-off are both wrong at that scale. See the
    // base declaration for what opting out actually changes.

    bool onInit() override;
    audio::AudioConfig audioConfig() const override;
    bool initStreaming();
    bool resolveConfiguredContentProfile();
    bool initBethesdaSession();
    bool loadScenarioQuestDefinitions(const bethesda::ScenarioDefinition& scenario);
    bool saveGameplayState();
    bool loadGameplayState();
    bool registerBethesdaPlayerController();
    bool recoverBethesdaPlayerControllerFromIntersectingFloor();
    void pullBethesdaPlayerControllerState();
    void relocateBethesdaPlayerControllerToCamera();
    void unregisterBethesdaActorControllers();
    void pullBethesdaActorControllerStates();
    void stepBethesdaActorControllers(float fixedDeltaSeconds);
    void submitBethesdaActorControllerIntents();
    [[nodiscard]] std::optional<bethesda::ObjectId> runtimeObjectIdForActor(
        const SkinnedActor& actor) const;
    void syncBethesdaPlayerState(bool applyNow);
    void syncBethesdaActors(bool addMissing, bool applyNow);
    void restoreBethesdaActorsFromSession();
    bool ensureSkyrimActorCatalog();
    bool bindAndMaterializeScenarioReferences(std::string& outError);
    [[nodiscard]] bool runtimeOriginSpaceForReference(
        std::uint32_t referenceFormId,
        bethesda::RuntimeSpaceState& outSpace) const;
    [[nodiscard]] bool runtimeSpaceForPosition(
        const float enginePosition[3],
        bethesda::RuntimeSpaceState& outSpace) const;
    [[nodiscard]] bool runtimeSpaceIsResident(
        const bethesda::RuntimeSpaceState& space) const;
    void cacheBethesdaCollisionCell(
        const importer::CellCoord& cell, const importer::ImportedScene& scene);
    void removeBethesdaCollisionCell(const importer::CellCoord& cell);
    void registerBethesdaCollisionCell(const importer::CellCoord& cell);
    void registerCachedBethesdaCollision();
    // Loads the skeleton and body parts, binds them, and uploads the result to
    // skinned instance slot 0. Also frames the camera on the bind-pose bounds,
    // because the character's own extent is the only sensible thing to point at
    // when there is no world.
    bool initCharacter(const std::filesystem::path& dataFilesPath);
    // Re-submits the bind pose. Called every frame, not once: the backend
    // consumes the pose during the frame it was set for and does not retain it.
    void updateCharacterPose();
    void updateStreaming(float deltaSeconds);
    // Skyrim stores generated distant buildings in four-cell .bto tiles.
    // Keep a small window around the camera so child-worldspace cities (most
    // visibly Whiterun) retain their authored skyline from the parent world.
    void updateSkyrimObjectLod(const float bethesdaPosition[3]);
    // Skyrim's matching .btr terrain tiles form the distant ground underneath
    // those object shells. Keep a ring outside detailed LAND residency.
    void updateSkyrimTerrainLod(const float bethesdaPosition[3]);
    void runCollisionSelfTest();
    void onTick(float deltaSeconds) override;
    void onRender(float deltaSeconds) override;
    void onShutdown() override;

private:
    void applyTimeOfDay();
    // Reads WTHR/CLMT across the load order and picks the active weather. No-op
    // unless a plugin beyond the base game is loaded or a weather was named.
    void initWeather();
    // Pushes the active weather's colours for the current hour at the renderer.
    // Called from applyTimeOfDay, so moving time also moves the sky.
    void applyWeather();
    // Makes `weatherFormId` the active weather and re-does everything that
    // depends on it: cloud layers, sky gradient, fog, audio, tonemap.
    void selectWeather(std::uint32_t weatherFormId);
    // Post-processing curve selection. Called unconditionally from onInit, NOT
    // from the weather path -- see the definition.
    void applyTonemapSettings();
    // Distant landscape from the game's own LOD pyramid; see the definition.
    void loadDistantLandLod();
    std::size_t m_distantLodChunk = static_cast<std::size_t>(-1);
    std::int32_t m_skyrimTerrainLodTileX = 0;
    std::int32_t m_skyrimTerrainLodTileZ = 0;
    bool m_skyrimTerrainLodTileValid = false;
    std::size_t m_skyrimObjectLodChunk = static_cast<std::size_t>(-1);
    std::int32_t m_skyrimObjectLodTileX = 0;
    std::int32_t m_skyrimObjectLodTileZ = 0;
    bool m_skyrimObjectLodTileValid = false;
    // Fills m_weatherChoices, once. Prefers the weathers this worldspace's
    // climate actually runs -- with Nevada Skies loaded that IS the mod's
    // weather set, and it is a far more useful list than every WTHR in the load
    // order. Falls back to all of them when the climate names fewer than two.
    void buildWeatherChoices();
    // Opens the picker scrolled to the weather currently in effect.
    void openWeatherPicker();
    // The weather sub-page of the pause menu. Returns true when it drew, which
    // is what tells drawPauseMenu to skip the menu proper.
    bool drawWeatherPicker(const ui::UiRect& panelArea, float scale);
    bool drawCompatibilityPanel(const ui::UiRect& panelArea, float scale);
    // Rain/wind loops and a music bed, pulled from the installed game's own
    // audio. No-op when the weather is dry or the assets are missing.
    void initWeatherAudio();
    audio::SoundHandle loadAmbientDescriptor(std::uint32_t descriptorFormId);
    void updateSkyrimAmbience(float deltaSeconds);
    void clearSkyrimAmbience();
    // Reads keyboard AND gamepad into one device-agnostic nav snapshot. Both
    // always run: a player can have a controller plugged in and still reach for
    // Escape, and making them exclusive means whichever the game guessed wrong
    // about simply stops working.
    void pollNavInput(float deltaSeconds);
    void updateGiftMenu();
    // Checks the regions covering the camera and toasts any not seen before.
    void updateRegionDiscovery();
    void saveTraversalState(bool force);
    // Fills the renderer's stats panel with this game's own readouts --
    // streaming residency and timings, camera, weather. No-op while the panel
    // is closed.
    void updateDebugStats();
    void drawPipBoyHud();
    // The pause menu. Controller-navigable; returns nothing because every entry
    // acts on app state directly.
    void drawPauseMenu();
    void drawGiftMenu();
    void updateTes3JournalInput();
    void syncTes3JournalPanel();
    void drawTes3Journal();
    void updateCamera(float deltaSeconds);
    // Advances the scripted tour and points the camera. Returns false once the
    // path has run out, which hands the camera back to the player.
    bool updateFlythrough(float deltaSeconds);
    void drawHud();
    // The conversation, as a centred modal card rather than text in the corner.
    // Split out of drawHud because it is the one piece of this HUD with real
    // layout in it -- wrapping, measured row heights, a selection highlight --
    // and inlining that left drawHud unreadable.
    void drawDialoguePanel(
        const dialogue::DialogueNode& node, int screenWidth, int screenHeight, float scale);
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
    [[nodiscard]] int findLootableActorInReach() const;
    bool lootActor(int actorIndex);
    bool configureGoldenClawPuzzleForCurrentSpace(std::string& outError);
    [[nodiscard]] bool goldenClawPuzzleInReach() const;
    bool rotateGoldenClawRing(std::size_t ringIndex);
    bool useGoldenClawPuzzle();
    bool refreshGoldenClawPresentation(std::string& outError);
    void useDoor(const importer::ImportedSceneDoor& door);
    bool completeDoorTransition(const importer::ImportedSceneDoor& door, std::string& outError);
    // Diagnostic/demo placement shared by startup and Skyrim space reloads.
    // Keeping it here prevents Skyrim's deferred population pass from silently
    // bypassing ODAI_FNV_ACTORS_PARADE.
    void arrangeActorParadeIfRequested();
    void reloadActorsForCurrentSpace();
    // Realize CPU-built actor meshes on the GPU over multiple frames. A whole
    // town can contain dozens of unique bodies and textures; uploading all of
    // them in one onTick starves GLFW event polling and makes close/input look
    // frozen even though the process is still doing useful work.
    void queueActorUploads();
    void updateDoorTransition(float deltaSeconds);
    void rebuildStreamDoors();
    // Bilinear ground height at a world XZ. false outside the cooked terrain or
    // over a lattice hole, in which case outHeight is untouched.
    [[nodiscard]] bool groundHeightAt(float x, float z, float& outHeight) const;

    std::string m_scenePath;
    // Doors in the loaded scene, plus what is needed to find their targets:
    // interiors are cooked beside the exterior as "<stem>_<CellEditorID>.bin"
    // (importedSceneInteriorFileName), and a door with an empty target cell is
    // the way back to "<stem>.bin".
    std::vector<importer::ImportedSceneDoor> m_doors;
    std::unordered_map<
        importer::CellCoord, std::vector<importer::ImportedSceneDoor>, importer::CellCoordHash>
        m_streamDoorsByCell;
    enum class DoorTransitionPhase : std::uint8_t { None, FadeOut, FadeIn };
    DoorTransitionPhase m_doorTransitionPhase = DoorTransitionPhase::None;
    std::optional<importer::ImportedSceneDoor> m_pendingDoor;
    float m_doorTransitionAlpha = 0.0f;
    struct GoldenClawPuzzleBinding {
        bethesda::ObjectId door;
        bethesda::RecordKey requiredItem;
        bethesda::RecordKey quest;
        std::int32_t successStage = 0;
        std::uint32_t keyholeReferenceFormId = 0u;
        std::vector<std::uint32_t> collisionReferenceFormIds;
        float position[3] = {};
    };
    std::optional<GoldenClawPuzzleBinding> m_goldenClawPuzzle;
    bool m_goldenClawRingKeyLatch[3] = {};
    std::size_t m_interiorChunk = render::Renderer::kInvalidImportedChunkIndex;
    // Full immutable imported payload retained while a presentation-only copy
    // filters persistent references such as the opened claw mechanism.
    std::optional<importer::ImportedScene> m_currentInteriorSourceScene;
    std::string m_currentInteriorEditorId;
    std::filesystem::path m_sceneDirectory;
    std::string m_exteriorStem;
    bool m_choiceKeyLatch[9] = {};
    bool m_doorKeyLatch = false;

    // Conversation type, one step above the HUD's body face. The dialogue is
    // the only thing in this app a player READS at length rather than glances
    // at, and it is read from a couch -- so it gets its own scale steps rather
    // than borrowing the 28 px body size the status strip uses. Both are
    // optional: a failed bake falls back to the body face, which loses the
    // hierarchy but never the text.
    ui::Font m_dialogueFont;        // what Victor says
    ui::Font m_dialogueChoiceFont;  // what the player can say back
    // Morrowind's journal is a reading surface, not a HUD card. A dedicated
    // serif family gives its long authored entries the book-like cadence of
    // the original game without changing the sans-serif combat/HUD language.
    ui::Font m_tes3JournalFont;
    ui::Font m_tes3JournalBoldFont;
    ui::Font m_tes3JournalItalicFont;
    // Which reply is highlighted. Reset whenever the conversation moves to a
    // new node, tracked by id rather than by a "node changed" flag because the
    // runtime offers no such signal.
    int m_dialogueChoice = 0;
    std::string m_dialogueChoiceNodeId;
    // Top edge of the conversation card in framebuffer pixels, published by
    // drawDialoguePanel and consumed by updateCamera's aim.
    //
    // The camera has to know how tall the card actually is: it frames Victor's
    // face just ABOVE it, and the card grows with the number of replies and
    // with how far the text wraps. Framing against a fixed fraction instead
    // works for a four-reply node and puts the card over his face on a taller
    // one. One frame stale (drawing happens after the camera update), which is
    // invisible against a 0.12 s eased turn.
    float m_dialoguePanelTopPx = 0.0f;
    // Live field of view, eased. A conversation narrows it, which magnifies the
    // speaker without moving the player -- the same read as Skyrim's dolly, but
    // it cannot walk the camera into geometry or fight collision the way an
    // actual position push-in would.
    //
    // The aim's pitch offset is derived from this same value rather than from a
    // constant, because the offset that puts his face above the card is a
    // function of FOV: hold one fixed while the other eases and the framing
    // slides during the zoom.
    // The LIVE VERTICAL fov, eased toward whichever horizontal constant above
    // applies. Vertical because that is what the renderer takes and what the
    // dialogue framing maths is written against (it works in half-heights).
    // Seeded at the 16:9 conversion of the default so the very first frame,
    // before a framebuffer size is known, is already close.
    float m_cameraFovDegrees = 45.0f;
    // The streaming load order, kept so actor discovery and dialogue can use the
    // same one the cell streamer does. Empty when no extra plugins were loaded.
    importer::fnv::FalloutLoadOrder m_streamLoadOrder;
    // Immutable winning-record catalog. Authored ACHR ownership is provenance,
    // not live residency: packages can move an interior-owned actor outside.
    // It is scanned once, then filtered by BethesdaWorld::currentSpace on every
    // settled streaming-ring refresh.
    importer::fnv::FalloutActorScan m_skyrimActorCatalog;
    std::unordered_map<std::uint32_t, std::string> m_skyrimActorVoiceFolderPlugin;
    bool m_skyrimActorCatalogReady = false;
    bool m_streamIsMorrowind = false;
    bool m_streamIsSkyrim = false;
    // Conversation depth of field, eased 0..1 alongside the dolly. A long lens
    // does not only magnify, it throws the background out — the two arriving
    // together is what makes the shot read as a lens rather than as a crop.
    float m_dialogueDofBlend = 0.0f;
    // Whether the renderer's DoF is currently ours. Outside a conversation this
    // game does not touch DoF at all, so the debug sliders keep working; this
    // flag is what lets it hand the setting back exactly once on the way out
    // instead of overwriting it every frame forever.
    bool m_dialogueDofActive = false;
    // Screenshot-and-quit mode; empty path means normal interactive running.
    std::string m_screenshotPath;
    int m_screenshotWarmupFrames = 8;
    int m_framesRendered = 0;

    // Scripted tour and frame-sequence recording. See the setters.
    float m_flythroughSeconds = 0.0f;
    float m_flythroughTime = 0.0f;
    // Tour actor tracking: the actor the aim latched onto, and the smoothed aim
    // point. Both exist so the last stretch of the flythrough does not inherit
    // the per-frame ground-settle steps in an actor's position.
    int m_tourTrackedActor = -1;
    float m_tourAim[3] = {0.0f, 0.0f, 0.0f};
    bool m_tourAimValid = false;
    // Critically damped angle filter state for the tour camera. The velocities
    // are what make it C1; without them it is a plain exponential and arrives
    // at every turn with a corner.
    float m_tourYawVelocity = 0.0f;
    float m_tourPitchVelocity = 0.0f;
    bool m_tourAnglesValid = false;
    std::string m_captureDirectory;
    std::string m_captureVideoPath;
    render::VideoWriter m_captureVideo;
    audio::WavWriter m_captureAudio;
    std::vector<std::uint8_t> m_captureRgb;
    std::vector<float> m_capturePcm;
    float m_captureVideoFps = 30.0f;
    float m_visualTimeSeconds = 0.0f;
    int m_captureFrames = 0;
    int m_captureWritten = 0;
    bool m_captureStarted = false;
    bool m_captureAudioRequested = false;
    bool m_captureAudioPrimed = false;
    std::uint64_t m_captureAudioFramesWritten = 0u;
    std::uint32_t m_captureSeed = 0u;
    std::filesystem::path m_captureTemporaryVideoPath;
    std::filesystem::path m_captureTemporaryAudioPath;
    float m_captureFixedDt = 0.0f;
    // Frames to render before the first is kept. Streaming, auto-exposure and
    // TAA all need a few: recording from frame 0 opens on a half-loaded town
    // under a mid-adaptation exposure.
    int m_captureWarmupFrames = 60;
    // Ceiling on waiting for streaming to settle. A worldspace whose residency
    // set never stops churning must not stall the capture forever -- record a
    // slightly unfinished frame rather than nothing at all.
    int m_captureWarmupFrameCeiling = 900;
    // A video capture pins the authored tour corridor before recording. Unlike
    // ordinary streaming, this must settle completely: fixed-rate simulation
    // can cross several cells while worker IO is still building the first one.
    bool m_captureRoutePreloadActive = false;
    std::unordered_set<importer::CellCoord, importer::CellCoordHash> m_capturePinnedCells;
    bool m_captureSkyrimLodBoundsValid = false;
    std::int32_t m_captureSkyrimLodMinTileX = 0;
    std::int32_t m_captureSkyrimLodMinTileZ = 0;
    std::int32_t m_captureSkyrimLodMaxTileX = 0;
    std::int32_t m_captureSkyrimLodMaxTileZ = 0;
    bool m_captureSkyrimTerrainLodFrozen = false;
    bool m_captureSkyrimObjectLodFrozen = false;
    bool m_captureUploadsReady = false;
    // Frames rendered but not kept: at least m_captureWarmupFrames AND, while
    // streaming, until the streamer goes idle. Both, because auto-exposure and
    // TAA need frames while cell loading needs wall time, and neither is a
    // proxy for the other.
    bool captureWarmupComplete();
    bool beginCaptureAudio();
    bool writeCaptureAudioFrame();
    bool finishCaptureVideo();

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
    // Which skinned instance slot Victor owns.
    //
    // NOT slot 0: --character's isolation harness uploads there, and the two
    // modes are NOT mutually exclusive -- character mode streams the world, so
    // Victor loads alongside it. Sharing a slot meant Victor's template
    // overwrote the harness's and both pushed a pose to it every frame, each
    // clobbering the other. They happen to have the same bone count (both are
    // Securitrons in the obvious test), so the pose-size guard never caught it.
    static constexpr std::uint32_t kVictorSkinnedInstance = 1u;
    // The rest of the town starts above Victor. Slot 0 stays with --character's
    // isolation harness.
    static constexpr std::uint32_t kFirstCrowdSkinnedInstance = 2u;
    // How far around the player to populate, in Bethesda units (~70/metre), so
    // ~170 m. Wide enough to cover Goodsprings from the spawn, tight enough
    // that the slot budget is spent on actors the player can actually see.
    static constexpr float kActorLoadRadius = 12000.0f;

    // EVERYONE, Victor included. He used to be his own member of his own type,
    // which meant the upload, the pose, the conversation and the camera framing
    // each existed twice by the time the town arrived. Held for the process
    // lifetime because a skinned template is uploaded from spans into these.
    std::vector<SkinnedActor> m_actors;
    bool m_actorsUploadPending = false;
    std::size_t m_nextActorUploadIndex = 0u;
    std::size_t m_actorUploadSuccessCount = 0u;
    std::size_t m_actorUploadedTextureCount = 0u;
    std::size_t m_actorTotalTextureCount = 0u;
    // Victor's index in m_actors, or -1. He is an ordinary actor now; this
    // exists only for his spawn-side placement and the log lines that name him.
    int m_victorIndex = -1;
    // The actor the player is in conversation with, or -1. An index rather than
    // a pointer because m_actors outlives any one frame but is not stable
    // across a reload.
    int m_talkingActor = -1;

    [[nodiscard]] SkinnedActor* talkingActor() {
        return (m_talkingActor >= 0 && m_talkingActor < static_cast<int>(m_actors.size()))
            ? &m_actors[static_cast<std::size_t>(m_talkingActor)]
            : nullptr;
    }
    [[nodiscard]] const SkinnedActor* talkingActor() const {
        return (m_talkingActor >= 0 && m_talkingActor < static_cast<int>(m_actors.size()))
            ? &m_actors[static_cast<std::size_t>(m_talkingActor)]
            : nullptr;
    }
    void beginConversation(int actorIndex);
    [[nodiscard]] bool beginTes3Conversation(int actorIndex);
    [[nodiscard]] bool beginBethesdaConversation(int actorIndex);
    void rebuildTes3ConversationTree(
        SkinnedActor& actor, const bethesda::Tes3DialogueResponse& response);
    void rebuildBethesdaConversationTree(
        SkinnedActor& actor, std::vector<bethesda::SkyrimDialogueChoice> choices);
    [[nodiscard]] int findBethesdaDialogueActorInReach(
        const float cameraPosition[3], float cameraYawRadians);
    [[nodiscard]] int findTes3DialogueActorInReach(
        const float cameraPosition[3], float cameraYawRadians) const;
    void chooseConversationChoice(std::size_t index);
    void endConversation();
    bool m_bethesdaDialogueActive = false;
    enum class Tes3DialogueActionKind : std::uint8_t { Topic, Choice, Goodbye };
    struct Tes3DialogueAction {
        Tes3DialogueActionKind kind = Tes3DialogueActionKind::Goodbye;
        std::string topic;
        std::int32_t choice = 0;
    };
    bool m_tes3DialogueActive = false;
    std::vector<Tes3DialogueAction> m_tes3DialogueActions;
    bethesda::ObjectId m_bethesdaDialogueSpeaker;
    bethesda::ObjectId m_bethesdaDialoguePlayer;
    std::vector<bethesda::SkyrimDialogueChoice> m_bethesdaDialogueChoices;
    bethesda::RecordKey m_bethesdaDialoguePendingEndInfo;
    std::vector<bethesda::RecordKey> m_bethesdaDialogueNextTopics;
    // Set by main-thread cell callbacks and consumed once the asynchronous
    // ring settles. This keeps Skyrim actor/quest residency aligned with the
    // streamer's planner without rescanning the plugin once per arriving cell.
    bool m_skyrimActorResidencyDirty = false;
    // Engine space; y == 0 means "use his authored ACRE position".
    float m_victorSpawnPosition[3] = {};
    // The actor "press E to talk" is currently offering, or -1. Resolved once
    // per tick by findActorInReach so the prompt and the keypress can never
    // disagree about who is being addressed.
    int m_activationActor = -1;
    int m_activationLootActor = -1;
    bool m_mouseCaptured = true;

    // Time of day in hours [0, 24). Drives the sun angle, and through it the
    // shadow direction and the atmosphere's sky colour.
    float m_timeOfDayHours = 9.5f;
    bool m_timeOfDayExplicit = false;
    // Weather, from WTHR records across the load order. Empty tables mean the
    // procedural sky, which is what an unmodded run gets.
    // Upscaler request. What actually runs may differ -- see
    // Renderer::upscalerStatus() and the note at the parse site.
    render::UpscalerSettings m_upscalerSettings{};
    importer::fnv::FalloutWeatherTables m_weatherTables;
    std::vector<std::string> m_extraPlugins;  // beyond m_streamPlugin, in load order
    std::string m_loadOrderPath;
    std::string m_loadOrderFingerprint;
    std::string m_contentProfilePath;
    std::string m_contentProfileModsRoot;
    std::string m_compatibilityReportPath;
    bool m_forceContentReindex = false;
    std::optional<importer::fnv::ResolvedContentProfile> m_contentProfile;
    std::filesystem::path m_traversalStatePath;
    std::optional<TraversalState> m_resumeState;
    bool m_resumeEnabled = true;
    bool m_explicitStart = false;
    float m_stateSaveSeconds = 0.0f;
    std::unordered_set<std::uint32_t> m_discoveredMarkerIds;
    std::vector<TraversalDiscovery> m_discoveredLocations;
    std::string m_requestedWeatherEditorId;
    std::uint32_t m_activeWeatherFormId = 0;
    // When the weather's colour slots peak, from the active climate's TNAM.
    // 6 and 19 are the samplers' own rough Fallout defaults and stand in when
    // no climate resolves -- SkyrimClimate authors 7.75 and 18.25, so leaving
    // these pinned samples the wrong two slots for over an hour either side of
    // dawn and dusk. Read once per climate rather than per frame.
    float m_sunriseHour = 6.0f;
    float m_sunsetHour = 19.0f;
    // Which of the active weather's cloud layers each of the renderer's four
    // slots is drawing, as an index into FalloutWeatherRecord::cloudLayers;
    // -1 for a slot with nothing in it.
    //
    // A SLOT NUMBER IS NOT A LAYER NUMBER and storing only "is this slot in
    // use" is what made the Skyrim sky solid black: the tints were then read
    // from PNAM rows 0..3 while the textures came from layers 8, 16 and 28, and
    // Skyrim authors a black daytime tint on exactly the layers it disables.
    int m_cloudLayerSource[4] = {-1, -1, -1, -1};
    audio::SoundHandle m_rainLoop{};
    audio::SoundHandle m_windLoop{};
    audio::AmbientHandle m_rainAmbient{};
    audio::AmbientHandle m_windAmbient{};
    audio::MusicHandle m_musicTrack{};
    struct ActivePlacedAmbient {
        std::uint32_t descriptorFormId = 0;
        audio::AmbientHandle handle{};
    };
    std::unordered_map<
        importer::CellCoord,
        std::vector<importer::fnv::FalloutSoundEmitterRecord>,
        importer::CellCoordHash> m_streamAmbientEmittersByCell;
    std::unordered_map<std::uint32_t, ActivePlacedAmbient> m_activePlacedAmbients;
    std::unordered_map<std::uint32_t, audio::AmbientHandle> m_activeRegionAmbients;
    std::unordered_map<std::uint32_t, audio::SoundHandle> m_ambientSounds;
    float m_regionAmbiencePollSeconds = 0.0f;
    std::uint32_t m_ambienceRandomState = 1u;
    bool m_dayCyclePaused = true;
    float m_dayCycleHoursPerSecond = 0.15f;
    bool m_bracketLeftLatch = false;
    bool m_bracketRightLatch = false;
    bool m_pauseLatch = false;
    bool m_quitKeyLatch = false;
    bool m_escapeLatch = false;
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
    // Slower than the library default (0.40 s / 0.11 s), because every list in
    // this app is SHORT. 0.11 s is nine rows a second, which is right for
    // scrolling an inventory and far too quick for a four-reply conversation or
    // a four-row menu -- a held key laps the list twice a second and overshoots
    // whatever you were aiming at. A tap still moves exactly one either way.
    ui::UiNavRepeater m_navRepeat{ui::UiNavRepeatConfig{0.45f, 0.18f}};
    ui::UiNavStickMapper m_navStick;
    ui::NavFocusRing m_menuFocus;
    bool m_menuOpen = false;
    std::unique_ptr<ui::Tes3JournalPanel> m_tes3JournalPanel;
    bool m_tes3JournalOpen = false;
    bool m_tes3JournalKeyLatch = false;
    std::string m_tes3StartQuest;
    std::int32_t m_tes3StartQuestIndex = 0;
    std::string m_tes3PinnedQuest;
    std::size_t m_tes3JournalSyncedVisits = 0u;
    int m_giftMenuChoice = 0;
    std::uint64_t m_presentedGiftMenuSequence = 0u;
    // Weather picker, a sub-page of the pause menu. Choices are remapped
    // formIDs into m_weatherTables, built once on first open.
    ui::NavFocusRing m_weatherFocus;
    bool m_weatherPickerOpen = false;
    bool m_compatibilityPanelOpen = false;
    std::vector<std::uint32_t> m_weatherChoices;
    // First row of the visible window into m_weatherChoices. The list is far
    // longer than the panel (473 weathers with Nevada Skies installed), so it
    // scrolls rather than being drawn in full.
    int m_weatherScrollTop = 0;
    // True once a controller (or the d-pad keys) last drove the UI, false once
    // the mouse moves. Drives whether the focus highlight is drawn at all --
    // an always-on highlight looks broken with a mouse, and a never-on one
    // makes the menu unusable from a couch.
    bool m_navDriving = false;
    // Region names already announced. Discovery is once per session per region;
    // walking back into Nipton does not re-announce it.
    std::unordered_set<std::string> m_discoveredRegions;
    float m_regionPollSeconds = 0.0f;
    // ODAI_FNV_BENCH_HEADING is applied once, not every frame, or the walk
    // would never turn away from it.
    bool m_benchHeadingApplied = false;

    // Cell streaming. Null unless --stream was given. The job system is owned
    // here rather than by the streamer so its thread count is visible at the
    // call site and so it outlives every in-flight load.
    std::string m_streamDirectory;
    std::string m_streamPlugin = "FalloutNV.esm";
    std::string m_streamWorldspace = "WastelandNV";
    bool m_streamWorldspaceExplicit = false;
    // Where the game itself starts you: the doorstep of Doc Mitchell's house in
    // Goodsprings.
    // Doc Mitchell's house, which is where New Vegas starts you. A DEFAULT, not
    // a constant: streaming Fallout 3 through the same path looked for a cell
    // that game has never heard of, and warned about it every launch.
    std::string m_streamSpawnInterior = "GSDocMitchellHouse";
    bool m_streamSpawnInteriorExplicit = false;
    // Start INSIDE this interior rather than on its doorstep. Empty means the
    // doorstep, which is what the viewer has always done.
    std::string m_startInsideInterior;
    bool m_interiorStarted = false;
    std::string m_streamCacheDirectory;
    // Asset override roots, in load order. See addModDirectory.
    std::vector<std::string> m_modDirectories;
    bool m_streamCacheEnabled = true;
    std::unique_ptr<core::JobSystem> m_streamJobs;
    std::unique_ptr<importer::fnv::CellStreamer> m_streamer;
    // Collision for the streamed world. The single-scene (--scene) path keeps
    // using the older height field built from the whole scene.
    CollisionWorld m_collision;
    ActorNavigationWorld m_actorNavigation;
    struct BethesdaCollisionMesh {
        std::vector<odai::math::Vector3> vertices;
        std::vector<std::uint32_t> indices;
        // One entry per triangle, parallel to indices in groups of three.
        std::vector<std::uint32_t> triangleSourceReferenceFormIds;
    };
    std::unordered_map<importer::CellCoord, BethesdaCollisionMesh,
        importer::CellCoordHash> m_bethesdaCollisionByCell;
    std::unordered_set<std::uint32_t> m_disabledBethesdaCollisionReferences;
    bool m_bethesdaCollisionBroadPhaseDirty = false;
    // Previous frame's camera position, differenced to get the velocity the
    // planner predicts with. Cheaper and more honest than plumbing the movement
    // code's own velocity, which is reset by collision and jumping.
    float m_previousCameraX = 0.0f;
    float m_previousCameraY = 0.0f;
    float m_previousCameraZ = 0.0f;
    bool m_hasPreviousCameraPosition = false;
    float m_streamStatsLogTimer = 0.0f;
    bool m_collisionSelfTestDone = false;
    bethesda::BethesdaSession m_bethesdaSession;
    bool m_bethesdaSessionConfigured = false;
    bool m_bethesdaPlayerControllerRegistered = false;
    bool m_bethesdaControllerOwnsCamera = false;
    std::string m_scenarioId;
    std::string m_scenarioStartMarker;
    std::filesystem::path m_gameplaySavePath;
    std::filesystem::path m_gameplayLoadPath;
    bool m_gameplaySaveKeyLatch = false;
    bool m_gameplayLoadKeyLatch = false;
    bool m_meleeAttackPending = false;
    bool m_meleeAttackButtonLatch = false;
};

// Source compatibility for integrations built against the original
// Fallout-New-Vegas-only runtime name.
using NewVegasApp = BethesdaApp;

}  // namespace odai::games::newvegas
