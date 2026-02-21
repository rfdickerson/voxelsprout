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

    GLFWwindow* m_window = nullptr;
    voxelsprout::render::Renderer m_renderer;
    voxelsprout::render::RenderParameters m_renderParams{};

    bool m_showUi = true;
    bool m_tabWasDown = false;
    int m_cloudUpdateIntervalUi = 2;
    int m_maxAccumulationSamplesUi = 256;
    int m_cloudPresetUi = 1;
};

} // namespace voxelsprout::app
