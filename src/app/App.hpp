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
        bool hasAdjacentEmpty = false;
        int adjacentEmptyX = 0;
        int adjacentEmptyY = 0;
        int adjacentEmptyZ = 0;
    };

    void pollInput();
    void updateCamera(float dt);
    [[nodiscard]] bool isSolidWorldVoxel(int worldX, int worldY, int worldZ) const;
    [[nodiscard]] bool doesPlayerOverlapSolid(float eyeX, float eyeY, float eyeZ) const;
    void resolvePlayerCollisions(float dt);
    [[nodiscard]] CameraRaycastResult raycastFromCamera() const;
    [[nodiscard]] bool tryPlaceVoxelFromCameraRay();
    [[nodiscard]] bool tryRemoveVoxelFromCameraRay();

    struct CameraState {
        float x = 0.0f;
        float y = 2.0f;
        float z = 5.0f;
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

    sim::Simulation m_simulation;
    world::ChunkGrid m_chunkGrid;
    render::Renderer m_renderer;
};

} // namespace app
