#pragma once

#include "core/input.h"
#include "render/renderer.h"
#include "sim/simulation.h"
#include "world/clipmap_index.h"
#include "world/world.h"

#include <vector>

struct GLFWwindow;

// App subsystem
// Responsible for: coordinating startup, per-frame update flow, and shutdown.
// Should NOT do: contain gameplay rules, low-level rendering internals, or thread management.
namespace voxelsprout::app {

class App {
public:
    bool init();
    void run();
    void update(float dt, float simulationAlpha);
    void shutdown();

private:
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

    void pollInput();
    void updateCamera(float dt);
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
        const voxelsprout::world::Chunk*& outChunk,
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
    void selectPlaceableBlock(int blockIndex);
    [[nodiscard]] bool isPipeHotbarSelected() const;
    [[nodiscard]] bool isConveyorHotbarSelected() const;
    [[nodiscard]] bool isTrackHotbarSelected() const;
    [[nodiscard]] voxelsprout::world::Voxel selectedPlaceVoxel() const;
    [[nodiscard]] bool computePlacementVoxelFromRaycast(const CameraRaycastResult& raycast, int& outX, int& outY, int& outZ) const;
    [[nodiscard]] bool applyVoxelEdit(
        int targetX,
        int targetY,
        int targetZ,
        voxelsprout::world::Voxel voxel,
        std::vector<std::size_t>& outDirtyChunkIndices
    );
    [[nodiscard]] bool isPipeAtWorld(int worldX, int worldY, int worldZ, std::size_t* outPipeIndex) const;
    [[nodiscard]] bool isBeltAtWorld(int worldX, int worldY, int worldZ, std::size_t* outBeltIndex) const;
    [[nodiscard]] bool isTrackAtWorld(int worldX, int worldY, int worldZ, std::size_t* outTrackIndex) const;
    void regenerateWorld();
    [[nodiscard]] bool tryPlaceVoxelFromCameraRay(std::vector<std::size_t>& outDirtyChunkIndices);
    [[nodiscard]] bool tryRemoveVoxelFromCameraRay(std::vector<std::size_t>& outDirtyChunkIndices);
    [[nodiscard]] bool tryPlacePipeFromCameraRay();
    [[nodiscard]] bool tryRemovePipeFromCameraRay();
    [[nodiscard]] bool tryPlaceBeltFromCameraRay();
    [[nodiscard]] bool tryRemoveBeltFromCameraRay();
    [[nodiscard]] bool tryPlaceTrackFromCameraRay();
    [[nodiscard]] bool tryRemoveTrackFromCameraRay();

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
    voxelsprout::core::InputState m_input{};
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
    bool m_wasToggleConfigUiDown = false;
    bool m_wasToggleFrameStatsDown = false;
    bool m_dayCycleEnabled = false;
    bool m_wasToggleDayCycleDown = false;
    float m_dayCyclePhase = 0.0f;
    bool m_hoverEnabled = false;
    bool m_voxelEditModeEnabled = false;
    bool m_wasToggleHoverDown = false;
    bool m_wasToggleVoxelEditModeDown = false;
    bool m_wasRegenerateWorldDown = false;
    int m_selectedHotbarIndex = 0;
    int m_selectedBlockIndex = 0;
    bool m_wasPrevBlockDown = false;
    bool m_wasNextBlockDown = false;
    bool m_gamepadConnected = false;
    bool m_worldDirty = false;
    float m_worldAutosaveElapsedSeconds = 0.0f;
    std::vector<std::size_t> m_visibleChunkIndices;
    voxelsprout::world::ClipmapConfig m_appliedClipmapConfig{};
    bool m_hasAppliedClipmapConfig = false;

    voxelsprout::sim::Simulation m_simulation;
    voxelsprout::world::World m_world;
    voxelsprout::world::ChunkClipmapIndex m_chunkClipmapIndex;
    voxelsprout::render::Renderer m_renderer;
};

} // namespace voxelsprout::app
