#pragma once

#include "core/Input.hpp"
#include "render/Renderer.hpp"
#include "sim/Simulation.hpp"
#include "world/ChunkGrid.hpp"

struct GLFWwindow;

// App subsystem
// Responsible for: coordinating startup, per-frame update flow, and shutdown.
// Should NOT do: contain gameplay rules, low-level rendering internals, or thread management.
namespace app {

class App {
public:
    bool init();
    void run();
    void update(float dt);
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
        const world::Chunk*& outChunk,
        int& outLocalX,
        int& outLocalY,
        int& outLocalZ
    ) const;
    [[nodiscard]] bool findGroundSupportY(float eyeX, float eyeY, float eyeZ, int& outSupportY) const;
    [[nodiscard]] bool doesPlayerOverlapSolid(float eyeX, float eyeY, float eyeZ) const;
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
    [[nodiscard]] bool isPipeHotbarSelected() const;
    [[nodiscard]] bool isConveyorHotbarSelected() const;
    [[nodiscard]] bool isTrackHotbarSelected() const;
    [[nodiscard]] world::Voxel selectedPlaceVoxel() const;
    [[nodiscard]] bool computePlacementVoxelFromRaycast(const CameraRaycastResult& raycast, int& outX, int& outY, int& outZ) const;
    [[nodiscard]] bool applyVoxelEdit(int targetX, int targetY, int targetZ, world::Voxel voxel, std::size_t& outEditedChunkIndex);
    [[nodiscard]] bool isPipeAtWorld(int worldX, int worldY, int worldZ, std::size_t* outPipeIndex) const;
    [[nodiscard]] bool isBeltAtWorld(int worldX, int worldY, int worldZ, std::size_t* outBeltIndex) const;
    [[nodiscard]] bool isTrackAtWorld(int worldX, int worldY, int worldZ, std::size_t* outTrackIndex) const;
    void regenerateWorld();
    [[nodiscard]] bool tryPlaceVoxelFromCameraRay(std::size_t& outEditedChunkIndex);
    [[nodiscard]] bool tryRemoveVoxelFromCameraRay(std::size_t& outEditedChunkIndex);
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
        float fovDegrees = 75.0f;
        bool onGround = false;
    };

    GLFWwindow* m_window = nullptr;
    core::InputState m_input{};
    CameraState m_camera{};
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
    bool m_hasMouseSample = false;
    bool m_wasPlaceBlockDown = false;
    bool m_wasRemoveBlockDown = false;
    bool m_debugUiVisible = false;
    bool m_wasToggleDebugUiDown = false;
    bool m_hoverEnabled = false;
    bool m_wasToggleHoverDown = false;
    bool m_wasRegenerateWorldDown = false;
    int m_selectedHotbarIndex = 0;
    int m_selectedBlockIndex = 0;
    bool m_wasPrevBlockDown = false;
    bool m_wasNextBlockDown = false;
    bool m_gamepadConnected = false;

    sim::Simulation m_simulation;
    world::ChunkGrid m_chunkGrid;
    render::Renderer m_renderer;
};

} // namespace app
