#pragma once

#include "audio/audio.h"
#include "core/input.h"
#include "game/advisor.h"
#include "game/economy.h"
#include "game/game_sim.h"
#include "game/strategy_map.h"
#include "game/units.h"
#include "import/gpu_scene.h"
#include "script/script_engine.h"
#include "ui/animation.h"
#include "render/renderer.h"
#include "sim/simulation.h"
#include "ui/font.h"
#include "ui/ui_context.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_input.h"
#include "world/clipmap_index.h"
#include "world/world.h"

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct GLFWwindow;

namespace odai::ui {
class AdvisorsPanel;
class Button;
class CommandViewPanel;
class DonutChart;
class GreatPeoplePanel;
class IconButton;
class Image;
class Label;
class LineChart;
class Panel;
class ProductionPanel;
class RichTextView;
class StatBadgeRow;
class TechTreePanel;
class ToastManager;
class Toolbar;
class Window;
class WorldTrackerPanel;
}

// App subsystem
// Responsible for: coordinating startup, per-frame update flow, and shutdown.
// Should NOT do: contain gameplay rules, low-level rendering internals, or thread management.
namespace odai::app {

class App {
public:
    bool init();
    void run();
    void update(float dt, float simulationAlpha);
    void shutdown();

private:
    struct AppConfig {
        odai::render::ShadowMode shadowMode = odai::render::ShadowMode::Auto;
        bool enableSsao = true;
        float masterVolume = 1.0f;
        float musicVolume = 0.6f;
        float ambientVolume = 0.5f;
        float uiVolume = 0.8f;
        bool audioMuted = false;
    };

    struct CameraRaycastResult {
        bool hitSolid = false;
        int solidX = 0;
        int solidY = 0;
        int solidZ = 0;
        float hitDistance = 0.0f;
        bool hasHitFaceNormal = false;
        int hitFaceNormalX = 0;
        int hitFaceNormalY = 0;
        int hitFaceNormalZ = 0;
        bool hasAdjacentEmpty = false;
        int adjacentEmptyX = 0;
        int adjacentEmptyY = 0;
        int adjacentEmptyZ = 0;
    };

    struct InteractionRaycastResult {
        bool hit = false;
        bool hitPipe = false;
        bool hitBelt = false;
        bool hitTrack = false;
        bool hitSolidVoxel = false;
        int x = 0;
        int y = 0;
        int z = 0;
        float hitDistance = 0.0f;
        bool hasHitFaceNormal = false;
        int hitFaceNormalX = 0;
        int hitFaceNormalY = 0;
        int hitFaceNormalZ = 0;
    };

    struct ImportedSceneInspectHit {
        bool hit = false;
        float distance = 0.0f;
        odai::math::Vector3 position{};
        std::uint32_t drawIndex = 0;
        std::uint32_t triangleIndex = 0;
        std::uint32_t textureIndex = 0xffffffffu;
        std::uint32_t flags = 0u;
    };

    void pollInput();
    void updateCamera(float dt);
    void setupDemoUi(float viewW, float viewH);
    void updateUiOverlay(float dt);
    void drawStrategyMapLabels(float fbW, float fbH);
    bool pickHexFromMouse(double mouseX, double mouseY, int fbW, int fbH,
                          int& outCol, int& outRow) const;
    // World XZ where the ray through the cursor pixel meets the y=0 ground plane.
    // Picks the orthographic (2D) or perspective (3D) basis from the current mode.
    bool rayToGroundPlane(double mouseX, double mouseY, int fbW, int fbH,
                          float& outX, float& outZ) const;
    // Switch the strategy map between 2D (flat ortho) and 3D (extruded perspective):
    // re-mesh + re-upload the scene, toggle SSAO, and reframe the camera.
    void setStrategyMap3D(bool enabled);
    // Re-mesh the strategy map (terrain + live unit tokens) for the current relief
    // mode and re-upload to the renderer. Called after unit moves and end-of-turn.
    void rebuildStrategyMapScene();
    // Seed a few playable units on friendly settlements so the supply mechanic has
    // something to drive in the prototype.
    void spawnDemoUnits();
    // Left click on hex (col,row): select the unit standing there, or clear the
    // current selection when clicking empty ground.
    void selectUnitAtHex(int col, int row);
    // While the right button is held: resolve what a release at (col,row) would do
    // for the selected unit (march along an A* path, or attack an enemy in reach)
    // and stash it for the preview overlay.
    void updateMoveAttackPreview(int targetCol, int targetRow);
    // On right-button release: carry out the previewed move order or attack.
    void commitMoveAttackPreview();
    // Drop any pending move/attack preview.
    void clearMoveAttackPreview();
    // Open (or refresh) the city production panel for the city on (col,row).
    void openProductionPanelForCity(int col, int row);
    // Rebuild the tech-tree window rows from current research state. Safe to call
    // anywhere except inside a tech row's own click callback.
    void refreshTechTree();
    // Rebuild the Great People window from the player's roster (settled figures +
    // any awaiting placement). Safe to call anywhere except inside a rail row's own
    // click callback (use m_greatPeopleDirty to defer in that case).
    void refreshGreatPeople();
    // The city a newly-settled great person should join: the city whose production
    // panel is currently open, else the player's largest city. Null if none.
    [[nodiscard]] odai::game::City* greatPersonHostCity();
    // Set `id` as the active research target (resets banked science if it changes)
    // and update the window in place. Safe to call from a row's click callback.
    void selectResearch(const std::string& id);
    // Accrue science for one turn and complete the current research if funded.
    void advanceResearch();
    // True if a Locked-gate tech's accomplishment is satisfied by current app
    // state. Conditions the app's game model can't observe are treated as met so
    // those techs gate on prerequisites alone.
    [[nodiscard]] bool techGateSatisfiedInApp(const odai::game::TechGate& gate) const;
    // Concise rich-text description (era, cost, unlocks, gate) for a technology.
    [[nodiscard]] std::string techDescription(const odai::game::TechDef& tech) const;
    // Number of idle cities (no production queued) + unvisited idle units.
    // Zero = End Turn is safe to press.
    [[nodiscard]] int idleCount() const;
    // Pan the camera to a hex tile.
    void focusOnHex(std::uint32_t col, std::uint32_t row);
    // Jump to the next idle city (opens production panel) or idle unit (selects it).
    void cycleToNextIdle();
    // Fire the End Turn / Next Idle action — same logic as the button callback,
    // also called from the Enter key handler.
    void fireEndTurn();
    // Seed the economy World (m_gameWorld) from the loaded strategy map's
    // settlements: one Empire per owner, one City per settlement, starting
    // territory claimed. The player's empire is marked aiManaged=false.
    void seedGameWorldFromSettlements();
    // Player's empire in m_gameWorld (the one with id == m_playerOwner), or null.
    [[nodiscard]] odai::game::Empire* playerEmpire();
    // The player-owned economy city on (col,row) in m_gameWorld, or null.
    [[nodiscard]] odai::game::City* playerCityAt(int col, int row);
    // Push live player-empire yields/treasury/turn into the top toolbar badges.
    void refreshToolbar();
    // Rebuild the left World Tracker rail (research / production / society / next
    // action) from the live game world. Cheap; called on turn + selection changes.
    void refreshWorldTracker();
    // Rebuild the right Command View rail from the hovered hex + selected unit.
    // Gated on a change in (hex, unit, turn) to avoid per-frame rebuilds.
    void refreshCommandView();
    // Rebuild the event-feed text from the tail of m_gameWorld.events.
    void refreshEventFeed();
    // After a turn step: raise eureka/unlock toasts and era-transition banners for
    // any new player events, and advance m_lastEventCount / m_lastEraIndex.
    void fireTurnBanners();
    // Set/clear TileFlag_Border on map tiles that lie on a territory frontier (a
    // neighbour with a different owner), so the renderer draws empire boundaries.
    void recomputeBorderFlags(odai::game::StrategyMap& map) const;
    // Append settlement markers for any economy city that lacks one (e.g. cities the
    // AI founds with settlers) so the on-map labels stay consistent with the world.
    void syncSettlementsWithCities();
    void syncGameplayUiState();
    void assignInventoryItemToSelectedHotbar(odai::render::InventoryItemId itemId);
    void handleInventoryClick(float mouseX, float mouseY, float displayWidth, float displayHeight);
    [[nodiscard]] bool isAnyUiVisible() const;
    [[nodiscard]] bool isSolidWorldVoxel(int worldX, int worldY, int worldZ) const;
    [[nodiscard]] bool worldToChunkLocal(
        int worldX,
        int worldY,
        int worldZ,
        std::size_t& outChunkIndex,
        int& outLocalX,
        int& outLocalY,
        int& outLocalZ
    ) const;
    [[nodiscard]] bool worldToChunkLocalConst(
        int worldX,
        int worldY,
        int worldZ,
        const odai::world::Chunk*& outChunk,
        int& outLocalX,
        int& outLocalY,
        int& outLocalZ
    ) const;
    [[nodiscard]] bool findGroundSupportY(float eyeX, float eyeY, float eyeZ, int& outSupportY) const;
    [[nodiscard]] bool doesPlayerOverlapSolid(float eyeX, float eyeY, float eyeZ) const;
    [[nodiscard]] bool doesPlayerOverlapConveyorBelt(float eyeX, float eyeY, float eyeZ) const;
    void resolvePlayerCollisions(float dt);
    [[nodiscard]] CameraRaycastResult raycastFromCamera() const;
    [[nodiscard]] InteractionRaycastResult raycastInteractionFromCamera(bool includePipes) const;
    [[nodiscard]] ImportedSceneInspectHit raycastImportedSceneFromCamera() const;
    void inspectImportedSceneFromCamera() const;
    [[nodiscard]] bool computePipePlacementFromInteractionRaycast(
        const InteractionRaycastResult& raycast,
        int& outX,
        int& outY,
        int& outZ,
        int& outAxisX,
        int& outAxisY,
        int& outAxisZ
    ) const;
    [[nodiscard]] bool computeBeltPlacementFromInteractionRaycast(
        const InteractionRaycastResult& raycast,
        int& outX,
        int& outY,
        int& outZ,
        int& outAxisX,
        int& outAxisY,
        int& outAxisZ
    ) const;
    [[nodiscard]] bool computeTrackPlacementFromInteractionRaycast(
        const InteractionRaycastResult& raycast,
        int& outX,
        int& outY,
        int& outZ,
        int& outAxisX,
        int& outAxisY,
        int& outAxisZ
    ) const;
    [[nodiscard]] bool isWorldVoxelInBounds(int x, int y, int z) const;
    void cycleSelectedHotbar(int direction);
    void selectHotbarSlot(int hotbarIndex);
    void toggleDebugUi();
    [[nodiscard]] odai::world::Voxel selectedPlaceVoxel() const;
    [[nodiscard]] bool computePlacementVoxelFromRaycast(const CameraRaycastResult& raycast, int& outX, int& outY, int& outZ) const;
    [[nodiscard]] bool applyVoxelEdit(
        int targetX,
        int targetY,
        int targetZ,
        odai::world::Voxel voxel,
        std::vector<std::size_t>& outDirtyChunkIndices
    );
    [[nodiscard]] bool isPipeAtWorld(int worldX, int worldY, int worldZ, std::size_t* outPipeIndex) const;
    [[nodiscard]] bool isBeltAtWorld(int worldX, int worldY, int worldZ, std::size_t* outBeltIndex) const;
    [[nodiscard]] bool isTrackAtWorld(int worldX, int worldY, int worldZ, std::size_t* outTrackIndex) const;
    bool loadConfig(const std::filesystem::path& configPath);
    bool saveConfig(const std::filesystem::path& configPath) const;
    void regenerateWorld();
    void refreshStreamingWindow(bool forceRendererUpload);
    [[nodiscard]] bool tryPlaceVoxelFromCameraRay(std::vector<std::size_t>& outDirtyChunkIndices);
    [[nodiscard]] bool tryRemoveVoxelFromCameraRay(std::vector<std::size_t>& outDirtyChunkIndices);
    [[nodiscard]] bool tryPlacePipeFromCameraRay();
    [[nodiscard]] bool tryRemovePipeFromCameraRay();
    [[nodiscard]] bool tryPlaceBeltFromCameraRay();
    [[nodiscard]] bool tryRemoveBeltFromCameraRay();
    [[nodiscard]] bool tryPlaceTrackFromCameraRay();
    [[nodiscard]] bool tryRemoveTrackFromCameraRay();
    void resetVoxelBreakProgress();

    struct CameraState {
        float x = 0.0f;
        float y = 10.0f;
        float z = 10.0f;
        float yawDegrees = -90.0f;
        float pitchDegrees = 0.0f;
        float velocityX = 0.0f;
        float velocityY = 0.0f;
        float velocityZ = 0.0f;
        float smoothedMouseDeltaX = 0.0f;
        float smoothedMouseDeltaY = 0.0f;
        float fovDegrees = 90.0f;
        bool onGround = false;
    };

    GLFWwindow* m_window = nullptr;
    odai::core::InputState m_input{};
    CameraState m_camera{};
    CameraState m_cameraPrevious{};
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
    float m_pendingMouseDeltaX = 0.0f;
    float m_pendingMouseDeltaY = 0.0f;
    bool m_hasMouseSample = false;
    bool m_wasPlaceBlockDown = false;
    bool m_wasRemoveBlockDown = false;
    bool m_debugUiVisible = false;
    bool m_inventoryVisible = false;
    bool m_wasToggleConfigUiDown = false;
    bool m_wasToggleFrameStatsDown = false;
    bool m_dayCycleEnabled = false;
    bool m_wasToggleDayCycleDown = false;
    float m_dayCyclePhase = 0.0f;
    bool m_hoverEnabled = false;
    bool m_wasToggleHoverDown = false;
    bool m_wasInventoryKeyDown = false;
    bool m_wasEscapeKeyDown = false;
    bool m_wasRegenerateWorldDown = false;
    bool m_wasToggleImportedTerrainDown = false;
    bool m_wasToggleImportedStaticsDown = false;
    bool m_wasToggleImportedTexturesDown = false;
    bool m_wasToggleImportedFlatShadingDown = false;
    bool m_wasToggleImportedWaterDebugDown = false;
    bool m_wasInspectImportedSceneDown = false;
    bool m_wasPrevBlockDown = false;
    bool m_wasNextBlockDown = false;
    bool m_gamepadConnected = false;
    bool m_worldDirty = false;
    float m_worldAutosaveElapsedSeconds = 0.0f;
    bool m_voxelBreakTargetValid = false;
    int m_voxelBreakTargetX = 0;
    int m_voxelBreakTargetY = 0;
    int m_voxelBreakTargetZ = 0;
    int m_voxelBreakClicks = 0;
    AppConfig m_config{};
    odai::render::GameplayUiState m_gameplayUiState{};
    std::vector<std::size_t> m_visibleChunkIndices;
    std::vector<std::uint8_t> m_visibleChunkGraceFrames;
    std::vector<std::uint8_t> m_previousVisibleChunkMask;
    std::vector<std::uint8_t> m_currentVisibleChunkMask;
    std::vector<std::uint8_t> m_directlyVisibleChunkMask;
    odai::world::ClipmapConfig m_appliedClipmapConfig{};
    bool m_hasAppliedClipmapConfig = false;

    odai::ui::Font m_uiFont;
    odai::ui::Font m_uiFontBold;
    odai::ui::Font m_uiFontItalic;
    odai::ui::Font m_uiFontTitle;  // Larger size for window title bars.
    odai::ui::FontSet m_uiFonts{};
    odai::ui::UiContext m_uiContext;
    odai::ui::UiDrawList m_uiDrawList;
    odai::ui::UiInput m_uiInput;
    odai::ui::Label* m_hudStatsLabel = nullptr;
    odai::ui::Label* m_hudTileInfoLabel = nullptr;
    odai::ui::Widget* m_hudTileInfoWindow = nullptr;
    odai::ui::RichTextView* m_civpediaView = nullptr;
    odai::ui::Widget* m_civpediaWindow = nullptr;
    odai::ui::Image* m_civpediaPortrait = nullptr;   // Unit/building portrait in CivPedia header.
    odai::ui::Label* m_civpediaNameLabel = nullptr;  // Name + class label in CivPedia header.
    odai::ui::Panel*  m_civPanel = nullptr;
    odai::ui::Button* m_civCollapseBtn = nullptr;
    bool              m_civExpanded = true;
    odai::ui::Tween   m_civExpandTween{};
    float             m_civFullH = 0.0f;
    float             m_civHeaderH = 0.0f;
    odai::ui::Image* m_civPortraitImage = nullptr;
    odai::ui::Label* m_civLeaderNameLabel = nullptr;
    odai::ui::ProductionPanel* m_productionPanel = nullptr;
    std::string m_cityProductionId;  // City's currently selected build (legacy mock).

    // Tech-tree research window.
    odai::ui::Window*        m_techTreeWindow = nullptr;
    odai::ui::TechTreePanel* m_techTreePanel  = nullptr;

    // Great People window: the player's roster of famous figures + the settle flow.
    odai::ui::Window*           m_greatPeopleWindow = nullptr;
    odai::ui::GreatPeoplePanel* m_greatPeoplePanel  = nullptr;
    std::string m_selectedGreatPersonId;  // rail selection in the Great People window
    bool m_greatPeopleDirty = false;      // rebuild requested (deferred out of a click callback)
    std::vector<std::string> m_researchedTechs;  // completed tech ids
    std::string m_researchTechId;                // current research target ("" = none)
    int m_scienceAccumulated = 0;                // science banked toward m_researchTechId
    int m_sciencePerTurn     = 14;               // mock science output (matches toolbar)
    // Per-city production control: which city the panel currently shows.
    odai::ui::Button* m_endTurnBtn = nullptr;
    int m_selectedCityCol = -1;
    int m_selectedCityRow = -1;
    // The faction the human player controls. Only this owner's cities/units count
    // toward the idle gate and may have production assigned; AI factions share the
    // map (and the CityState list) but are never the player's to manage.
    std::uint8_t m_playerOwner = 1;
    float m_setupViewW = 0.0f;
    float m_setupViewH = 0.0f;
    // Unit ids the player has "visited" this turn (via Next Idle); visited units
    // leave the idle count so they don't block End Turn without explicit orders.
    std::unordered_set<std::uint32_t> m_visitedUnitIds;
    // Unit selection and action bar.
    std::string m_selectedUnitType;      // e.g. "warrior", "" = nothing selected
    std::string m_prevSelectedUnitType;  // for change detection in updateUiOverlay
    odai::ui::Widget* m_unitActionPanel = nullptr;
    odai::ui::Image*  m_unitActionPortrait = nullptr;
    odai::ui::Label*  m_unitActionName = nullptr;
    static constexpr int kMaxActionSlots = 6;
    odai::ui::IconButton* m_actionSlotBtns[kMaxActionSlots]{};
    // Left "World Tracker" rail and right "Command View" rail (ornate HUD panels).
    odai::ui::WorldTrackerPanel* m_worldTracker = nullptr;
    odai::ui::CommandViewPanel*  m_commandView  = nullptr;
    odai::ui::UiRect m_worldTrackerRect{};  // Stored so refresh* can rebuild in place.
    odai::ui::UiRect m_commandViewRect{};
    // Change-detection keys so refreshCommandView only rebuilds when the selection
    // actually changes (col, row, unit id, turn).
    int m_cvHexCol = -2;
    int m_cvHexRow = -2;
    std::uint32_t m_cvUnitId = 0xFFFFFFFFu;
    int m_cvTurn = -1;
    // Top resource toolbar + its live readouts.
    odai::ui::Toolbar* m_toolbar = nullptr;
    odai::ui::Label* m_toolbarTurnLabel = nullptr;
    std::size_t m_tbScienceItem = 0;
    std::size_t m_tbCultureItem = 0;
    std::size_t m_tbGoldItem = 0;
    std::size_t m_tbFaithItem = 0;
    std::size_t m_tbFoodItem = 0;
    std::size_t m_tbProdItem = 0;
    // Narrated event feed (scrolling history of world.events) + transient banners.
    odai::ui::RichTextView* m_eventFeedView = nullptr;
    odai::ui::ToastManager* m_toasts = nullptr;
    // Pending typed characters (filled by the GLFW char callback) and edit-key edges.
    std::vector<std::uint32_t> m_pendingTextInput;
    float m_pendingScrollDelta = 0.0f;  // Accumulated scroll wheel ticks (GLFW callback).
    bool m_backspacePrev = false;
    bool m_enterPrev = false;
    std::uint32_t m_windowFrameTexture = 0;  // odai::ui::UiTextureId for the 9-slice frame.
    std::uint32_t m_uiSheetTexture = 0;     // odai::ui::UiTextureId for the uisheet sprite atlas.
    // Tooltip fade animation + latched content so it can fade out after the hover ends.
    odai::ui::Tween m_tooltipFade{};
    std::string m_lastTooltipText;
    odai::ui::UiVec2 m_lastTooltipAnchor{};
    float m_uiScale = 1.0f;
    float m_uiBuildMsEma = 0.0f;   // Smoothed CPU cost of UI update+build (ms).
    int m_uiBuildLogCounter = 0;
    bool m_uiFontReady = false;
    // Strategy-map mode: the cursor stays visible for clicking UI and the mouse
    // never rotates the camera (no FPS-style mouselook).
    bool m_strategyMapMode = false;
    float m_mapOrthoHalfHeight = 2000.0f;
    float m_mapMinX = 0.0f;
    float m_mapMaxX = 0.0f;
    float m_mapMinZ = 0.0f;
    float m_mapMaxZ = 0.0f;
    // 3D relief view: tilted perspective camera orbiting a focus point on the
    // ground. 2D is the flat top-down orthographic default.
    bool m_strategyMap3D = false;
    bool m_wasToggleMapViewDown = false;
    float m_map3DPitchDeg = -45.0f;   // Downward tilt of the 3D camera.
    float m_map3DDistance = 3000.0f;  // Eye distance from the focus point.
    float m_map3DFocusX = 0.0f;       // Ground point the 3D camera centers on.
    float m_map3DFocusZ = 0.0f;
    float m_mapFocusVelocityX = 0.0f; // Pan inertia velocity (world units/s).
    float m_mapFocusVelocityZ = 0.0f;
    float m_panDispSmoothX = 0.0f;    // Low-passed per-frame pan displacement (world units).
    float m_panDispSmoothZ = 0.0f;
    bool m_wasStrategyMapDragging = false;
    odai::render::CameraPose m_renderCameraPose{};
    int m_uiDemoClicks = 0;
    int m_currentTurn = 1;
    int m_hoveredHexCol = -1;
    int m_hoveredHexRow = -1;

    odai::game::StrategyMap m_strategyMap;
    // Live 4X economy: cities grow/found/produce, empires research and race for
    // wonders. Its `map` is a copy of m_strategyMap whose tile owners change as
    // borders expand; the strategy-map mesh is rebuilt from m_gameWorld.map.
    odai::game::World m_gameWorld;
    std::vector<odai::game::TurnSample> m_samples;  // per-turn metrics from stepTurn
    // Lua scripting host: loaded from mods/base/scripts and installed via
    // odai::game::setModHost so stepTurn fires mod event hooks. Null until init().
    std::unique_ptr<odai::script::ScriptHost> m_scriptHost;
    std::size_t m_lastEventCount = 0;  // events already shown as banners
    int m_lastEraIndex = -1;           // player's last-seen era (for era-transition banners)
    odai::game::GameState m_gameState;       // Live units marching on the map.
    std::uint32_t m_selectedUnitId = 0;      // Currently selected unit id; 0 == none.
    // Click edge detection for unit select (left) and move orders (right). A
    // press+release with little movement is a click; a left drag stays a map pan.
    bool m_mapLeftPrevDown = false;
    bool m_mapPressOverUi = false;
    float m_mapPressX = 0.0f;
    float m_mapPressY = 0.0f;
    bool m_mapRightPrevDown = false;
    bool m_mapRightPressOverUi = false;
    float m_mapRightPressX = 0.0f;
    float m_mapRightPressY = 0.0f;
    // Pending right-drag move/attack preview for the selected unit.
    bool m_previewActive = false;
    bool m_previewIsAttack = false;
    std::uint32_t m_previewTargetCol = 0;
    std::uint32_t m_previewTargetRow = 0;
    std::vector<std::array<std::uint32_t, 2>> m_previewPath;

    odai::sim::Simulation m_simulation;
    odai::world::World m_world;
    odai::world::ChunkClipmapIndex m_chunkClipmapIndex;
    odai::render::Renderer m_renderer;

    odai::audio::Audio m_audio;
    odai::audio::SoundHandle m_uiClickSfx;
    odai::audio::SoundHandle m_endTurnSfx;
    odai::audio::SoundHandle m_ambientLoop;
    odai::audio::MusicHandle m_menuMusic;
    bool m_importedSceneDemoEnabled = false;
    bool m_importedShowTerrain = true;
    bool m_importedShowStatics = true;
    bool m_importedShowTextures = true;
    bool m_importedFlatShading = false;
    bool m_importedWaterDebug = false;
    std::filesystem::path m_importedScenePath;
    odai::importer::ImportedScene m_importedScene;
    // Terrain textures loaded once at init; copied into StrategyMapMeshOptions on each
    // re-mesh to avoid repeated disk I/O and mip-chain generation.
    std::vector<odai::importer::ImportedSceneTexture> m_terrainTextures;
    // GPU-instanced, tessellated hex land surface. Active only in 3D relief mode on a
    // tessellation-capable device; otherwise the flat imported-static land is kept.
    odai::importer::HexTerrainData m_hexTerrain;
    bool m_useHexTerrain = false;
};

} // namespace odai::app
