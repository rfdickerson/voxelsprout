#pragma once

#include "audio/sound_engine.h"
#include "core/input.h"
#include "game/game_state.h"
#include "game/lua_script.h"
#include "import/gpu_scene.h"
#include "render/renderer.h"
#include "sim/simulation.h"
#include "world/clipmap_index.h"
#include "world/imported_scene_collision.h"
#include "world/navmesh.h"
#include "world/world.h"

#include <filesystem>
#include <future>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct GLFWwindow;

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
        bool enableMusic = true;
        float musicVolume = 0.35f;
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
    void resolveImportedScenePlayerCollisions(float dt);
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
    void initializeBalmoraGuards(bool reusePreparedNavmesh = false);
    void updateBalmoraGuards(float dt);
    void rebuildBalmoraGuardRenderFrame(float simulationAlpha);
    void initializeBalmoraDoorActivation();
    void initializeMorrowindGameplayScripts();
    [[nodiscard]] bool tryActivateMorrowindScript();
    [[nodiscard]] bool tryActivateBalmoraDoor();
    [[nodiscard]] bool enterMorrowindInterior(const odai::importer::MorrowindDoorReference& door);
    [[nodiscard]] bool leaveMorrowindInterior(const odai::importer::MorrowindDoorReference& door);
    [[nodiscard]] std::pair<int, int> currentMorrowindExteriorCell() const;
    [[nodiscard]] bool updateMorrowindExteriorStreaming(bool force);
    void startMorrowindExteriorStreamingPrepare(int centerCellX, int centerCellY);
    [[nodiscard]] bool pollMorrowindExteriorStreamingPrepare();

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

    struct BalmoraGuardPrototype {
        std::vector<odai::importer::ImportedScenePackedVertex> vertices;
        std::vector<std::uint32_t> indices;
        std::vector<odai::importer::ImportedScenePackedDraw> draws;
    };

    struct BalmoraGuardAgent {
        odai::math::Vector3 position{};
        odai::math::Vector3 previousPosition{};
        float yawRadians = 0.0f;
        float previousYawRadians = 0.0f;
        float walkPhase = 0.0f;
        float previousWalkPhase = 0.0f;
        float speed = 92.0f;
        std::vector<odai::math::Vector3> route;
        std::vector<odai::world::NavmeshPathPoint> path;
        std::size_t routeIndex = 0;
        std::size_t pathIndex = 0;
        float pathRetryCooldownSeconds = 0.0f;
    };

    struct MorrowindInteriorCacheEntry {
        odai::importer::ImportedScene scene;
        odai::world::ImportedSceneCollision collision;
        std::vector<odai::importer::MorrowindDoorReference> doors;
        bool sceneLoaded = false;
    };

    struct MorrowindExteriorPreparedRegion {
        odai::importer::ImportedScene scene;
        odai::importer::GpuSceneAsset gpuSceneAsset;
        odai::world::ImportedSceneCollision collision;
        odai::world::Navmesh navmesh;
        std::vector<std::pair<int, int>> loadedCells;
        std::uint32_t cacheHitCount = 0;
        std::uint32_t cacheMissCount = 0;
        bool navmeshCacheHit = false;
        int centerCellX = 0;
        int centerCellY = 0;
        bool success = false;
        std::string error;
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
    bool m_wasActivateDoorDown = false;
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

    odai::sim::Simulation m_simulation;
    odai::world::World m_world;
    odai::world::ChunkClipmapIndex m_chunkClipmapIndex;
    odai::audio::SoundEngine m_soundEngine;
    odai::render::Renderer m_renderer;
    bool m_importedSceneDemoEnabled = false;
    bool m_importedShowTerrain = true;
    bool m_importedShowStatics = true;
    bool m_importedShowTextures = true;
    bool m_importedFlatShading = false;
    bool m_importedWaterDebug = false;
    std::filesystem::path m_importedScenePath;
    std::filesystem::path m_morrowindRuntimeDataFilesPath;
    std::filesystem::path m_morrowindRuntimeCellCacheRoot;
    bool m_morrowindRuntimeExteriorStreamingEnabled = false;
    int m_morrowindRuntimeExteriorRadius = 1;
    int m_morrowindRuntimeLoadedCenterCellX = 0;
    int m_morrowindRuntimeLoadedCenterCellY = 0;
    bool m_morrowindRuntimeLoadedCenterValid = false;
    bool m_morrowindRuntimePrepareActive = false;
    int m_morrowindRuntimePrepareCenterCellX = 0;
    int m_morrowindRuntimePrepareCenterCellY = 0;
    std::future<MorrowindExteriorPreparedRegion> m_morrowindRuntimePrepareFuture;
    std::vector<odai::importer::MorrowindDoorReference> m_balmoraDoors;
    std::unordered_map<std::string, MorrowindInteriorCacheEntry> m_balmoraInteriorCache;
    std::string m_currentMorrowindInteriorCell;
    odai::game::GameState m_gameState;
    odai::game::LuaScriptRuntime m_luaScriptRuntime;
    odai::game::DialogueResult m_activeDialogue;
    std::string m_activeDialogueActorId;
    std::string m_lastScriptMessage;
    bool m_balmoraExteriorCached = false;
    odai::importer::ImportedScene m_balmoraExteriorScene;
    odai::importer::GpuSceneAsset m_balmoraExteriorGpuSceneAsset;
    odai::world::ImportedSceneCollision m_balmoraExteriorCollision;
    odai::importer::ImportedScene m_importedScene;
    odai::importer::GpuSceneAsset m_gpuSceneAsset;
    odai::importer::GpuSceneRuntime m_gpuSceneRuntime;
    odai::world::ImportedSceneCollision m_importedSceneCollision;
    odai::world::Navmesh m_balmoraNavmesh;
    BalmoraGuardPrototype m_balmoraGuardPrototype;
    std::vector<BalmoraGuardAgent> m_balmoraGuards;
    std::vector<odai::importer::ImportedScenePackedVertex> m_balmoraGuardFrameVertices;
    std::vector<std::uint32_t> m_balmoraGuardFrameIndices;
    std::vector<odai::importer::ImportedScenePackedDraw> m_balmoraGuardFrameDraws;
};

} // namespace odai::app
