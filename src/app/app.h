#pragma once

#include "core/input.h"
#include "game/strategy_map.h"
#include "import/gpu_scene.h"
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
#include <string>
#include <unordered_map>
#include <vector>

struct GLFWwindow;

namespace odai::ui {
class Button;
class DonutChart;
class IconButton;
class Image;
class Label;
class LineChart;
class Panel;
class ProductionPanel;
class RichTextView;
class StatBadgeRow;
class Toolbar;
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
    bool pickHexFromMouse(double mouseX, double mouseY, int fbW, int fbH,
                          int& outCol, int& outRow) const;
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
    std::string m_cityProductionId;  // City's currently selected build.
    // Unit selection and action bar.
    std::string m_selectedUnitType;      // e.g. "warrior", "" = nothing selected
    std::string m_prevSelectedUnitType;  // for change detection in updateUiOverlay
    odai::ui::Widget* m_unitActionPanel = nullptr;
    odai::ui::Image*  m_unitActionPortrait = nullptr;
    odai::ui::Label*  m_unitActionName = nullptr;
    static constexpr int kMaxActionSlots = 6;
    odai::ui::IconButton* m_actionSlotBtns[kMaxActionSlots]{};
    // Top resource toolbar + its live readouts.
    odai::ui::Toolbar* m_toolbar = nullptr;
    odai::ui::Label* m_toolbarTurnLabel = nullptr;
    std::size_t m_tbGoldItem = 0;
    std::size_t m_tbScienceItem = 0;
    std::size_t m_tbFaithItem = 0;
    float m_tbGold = 240.0f;
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
    int m_uiDemoClicks = 0;
    int m_currentTurn = 1;
    int m_hoveredHexCol = -1;
    int m_hoveredHexRow = -1;

    odai::game::StrategyMap m_strategyMap;

    odai::sim::Simulation m_simulation;
    odai::world::World m_world;
    odai::world::ChunkClipmapIndex m_chunkClipmapIndex;
    odai::render::Renderer m_renderer;
    bool m_importedSceneDemoEnabled = false;
    bool m_importedShowTerrain = true;
    bool m_importedShowStatics = true;
    bool m_importedShowTextures = true;
    bool m_importedFlatShading = false;
    bool m_importedWaterDebug = false;
    std::filesystem::path m_importedScenePath;
    odai::importer::ImportedScene m_importedScene;
};

} // namespace odai::app
