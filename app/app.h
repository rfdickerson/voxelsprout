#pragma once

#include "render/renderer.h"

#include <cstdint>

struct GLFWwindow;

namespace voxelsprout::app {

class App {
public:
    bool init();
    void run();
    void shutdown();

private:
    void pollInput();
    void buildUi();
    void updateOrbitCamera(float dtSeconds);

    GLFWwindow* m_window = nullptr;
    voxelsprout::render::Renderer m_renderer;
    voxelsprout::render::RenderParameters m_renderParams{};

    bool m_showUi = true;
    bool m_tabWasDown = false;
    bool m_orbitToggleWasDown = false;
    bool m_uiInteracting = false;
    int m_interactionCooldownFrames = 0;
    int m_cloudUpdateIntervalUi = 2;
    int m_maxAccumulationSamplesUi = 256;
    float m_sunAzimuthDegreesUi = -45.0f;
    float m_sunElevationDegreesUi = 70.0f;

    bool m_autoOrbitCamera = true;
    float m_orbitAngle = 1.57079637f;
    float m_orbitRadius = 14.0f;
    float m_orbitHeight = 4.0f;
    float m_orbitSpeed = 0.30f;
};

} // namespace voxelsprout::app
