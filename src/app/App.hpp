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
    void pollInput();
    void updateCamera(float dt);

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
    };

    GLFWwindow* m_window = nullptr;
    core::InputState m_input{};
    CameraState m_camera{};
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
    bool m_hasMouseSample = false;

    sim::Simulation m_simulation;
    world::ChunkGrid m_chunkGrid;
    render::Renderer m_renderer;
};

} // namespace app
