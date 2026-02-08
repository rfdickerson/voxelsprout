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
    enum class BrushSize : int {
        Size1 = 1,
        Size2 = 2,
        Size4 = 4
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

    void pollInput();
    void updateCamera(float dt);
    [[nodiscard]] bool isSolidWorldVoxel(int worldX, int worldY, int worldZ) const;
    [[nodiscard]] bool findGroundSupportY(float eyeX, float eyeY, float eyeZ, int& outSupportY) const;
    [[nodiscard]] bool doesPlayerOverlapSolid(float eyeX, float eyeY, float eyeZ) const;
    void resolvePlayerCollisions(float dt);
    [[nodiscard]] CameraRaycastResult raycastFromCamera() const;
    [[nodiscard]] bool isChunkVoxelInBounds(int x, int y, int z) const;
    [[nodiscard]] int activeBrushSize() const;
    void snapToBrushAnchor(int inX, int inY, int inZ, int& outX, int& outY, int& outZ) const;
    [[nodiscard]] bool computePlacementVoxelFromRaycast(const CameraRaycastResult& raycast, int& outX, int& outY, int& outZ) const;
    [[nodiscard]] bool applyBrushEdit(world::Chunk& chunk, int targetX, int targetY, int targetZ, world::Voxel voxel);
    [[nodiscard]] bool tryPlaceVoxelFromCameraRay();
    [[nodiscard]] bool tryRemoveVoxelFromCameraRay();

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
    BrushSize m_brushSize = BrushSize::Size4;
    bool m_wasCycleBrushDown = false;

    sim::Simulation m_simulation;
    world::ChunkGrid m_chunkGrid;
    render::Renderer m_renderer;
};

} // namespace app
