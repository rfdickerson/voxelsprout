#include "engine/game_app.h"
#include "core/log.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace odai::engine {

std::string GameApp::resolveAssetPath(const std::string& rel) {
    std::vector<std::filesystem::path> bases;
#if defined(ODAI_PROJECT_SOURCE_DIR)
    bases.emplace_back(std::filesystem::path{ODAI_PROJECT_SOURCE_DIR});
#endif
    std::error_code ec;
    std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (!ec) {
        bases.push_back(cwd);
        bases.push_back(cwd / "..");
        bases.push_back(cwd / ".." / "..");
        bases.push_back(cwd / ".." / ".." / "..");
    }
    for (const auto& base : bases) {
        const auto candidate = base / rel;
        std::error_code existsEc;
        if (std::filesystem::exists(candidate, existsEc) && !existsEc)
            return candidate.string();
    }
    return rel;
}

static void glfwErrorCb(int code, const char* msg) {
    VOX_LOGE("engine") << "GLFW error " << code << ": " << msg;
}

bool GameApp::init(const char* title) {
    glfwSetErrorCallback(glfwErrorCb);

    if (glfwInit() == GLFW_FALSE) {
        VOX_LOGE("engine") << "glfwInit failed";
        return false;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    int winW = 1920, winH = 1080;
    if (GLFWmonitor* mon = glfwGetPrimaryMonitor()) {
        float xs = 1.0f, ys = 1.0f;
        glfwGetMonitorContentScale(mon, &xs, &ys);
        if (const GLFWvidmode* mode = glfwGetVideoMode(mon)) {
            winW = static_cast<int>(std::round(mode->width  / std::max(xs, 1.0f)));
            winH = static_cast<int>(std::round(mode->height / std::max(ys, 1.0f)));
        }
    }

    m_window = glfwCreateWindow(winW, winH, title, nullptr, nullptr);
    if (!m_window) {
        VOX_LOGE("engine") << "glfwCreateWindow failed";
        glfwTerminate();
        return false;
    }

    glfwSetWindowUserPointer(m_window, this);
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    glfwSetCharCallback(m_window, [](GLFWwindow* win, unsigned int cp) {
        if (auto* self = static_cast<GameApp*>(glfwGetWindowUserPointer(win)))
            self->m_pendingTextInput.push_back(cp);
    });
    glfwSetScrollCallback(m_window, [](GLFWwindow* win, double /*x*/, double dy) {
        if (auto* self = static_cast<GameApp*>(glfwGetWindowUserPointer(win)))
            self->m_pendingScrollDelta += static_cast<float>(dy);
    });

    m_renderer.setStrategyMapMode(true);
    if (!m_renderer.init(m_window, m_emptyGrid)) {
        VOX_LOGE("engine") << "renderer init failed";
        glfwDestroyWindow(m_window);
        m_window = nullptr;
        glfwTerminate();
        return false;
    }

    if (!onInit()) {
        m_renderer.shutdown();
        glfwDestroyWindow(m_window);
        m_window = nullptr;
        glfwTerminate();
        return false;
    }

    return true;
}

void GameApp::run() {
    double prevTime = glfwGetTime();

    while (m_window && glfwWindowShouldClose(m_window) == GLFW_FALSE) {
        const double now = glfwGetTime();
        const float  dt  = static_cast<float>(std::min(now - prevTime, 0.1));
        prevTime = now;

        glfwPollEvents();
        if (!m_window || glfwWindowShouldClose(m_window)) break;

        int fbW = 0, fbH = 0;
        glfwGetFramebufferSize(m_window, &fbW, &fbH);

        double mx = 0.0, my = 0.0;
        glfwGetCursorPos(m_window, &mx, &my);

        m_uiInput.beginFrame();
        m_uiInput.mousePx = {static_cast<float>(mx), static_cast<float>(my)};
        if (m_hasMouseSample) {
            m_uiInput.mouseDeltaPx = {
                static_cast<float>(mx - m_lastMouseX),
                static_cast<float>(my - m_lastMouseY)
            };
        }
        m_lastMouseX     = mx;
        m_lastMouseY     = my;
        m_hasMouseSample = true;

        m_uiInput.setButton(ui::UiMouseButton::Left,
            glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT)  == GLFW_PRESS);
        m_uiInput.setButton(ui::UiMouseButton::Right,
            glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
        m_uiInput.scrollDelta = m_pendingScrollDelta;
        m_uiInput.textInput   = std::move(m_pendingTextInput);
        m_pendingScrollDelta  = 0.0f;
        m_pendingTextInput.clear();

        m_uiContext.setViewport({static_cast<float>(fbW), static_cast<float>(fbH)});
        m_uiContext.update(m_uiInput);

        onTick(dt);
        onRender(dt);
    }
}

void GameApp::shutdown() {
    onShutdown();
    m_renderer.shutdown();
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

bool GameApp::loadFonts(const std::string& regularPath,
                        const std::string& boldPath,
                        const std::string& italicPath,
                        const std::string& numericPath,
                        float bodySize, float numericSize) {
    if (!m_uiFont.loadFromFile(regularPath, bodySize)    ||
        !m_uiFontBold.loadFromFile(boldPath, bodySize)   ||
        !m_uiFontItalic.loadFromFile(italicPath, bodySize) ||
        !m_uiFontNumeric.loadFromFile(numericPath, numericSize)) {
        VOX_LOGE("engine") << "font load failed";
        return false;
    }

    if (!m_renderer.setUiFontAtlas(
            m_uiFont.atlasPixels().data(), m_uiFont.atlasWidth(), m_uiFont.atlasHeight())) {
        VOX_LOGE("engine") << "setUiFontAtlas failed";
        return false;
    }
    const auto boldTex   = m_renderer.registerUiFontAtlas(
        m_uiFontBold.atlasPixels().data(),    m_uiFontBold.atlasWidth(),    m_uiFontBold.atlasHeight());
    const auto italicTex = m_renderer.registerUiFontAtlas(
        m_uiFontItalic.atlasPixels().data(),  m_uiFontItalic.atlasWidth(),  m_uiFontItalic.atlasHeight());
    const auto numTex    = m_renderer.registerUiFontAtlas(
        m_uiFontNumeric.atlasPixels().data(), m_uiFontNumeric.atlasWidth(), m_uiFontNumeric.atlasHeight());

    m_uiFontBold.setTextureId(boldTex);
    m_uiFontItalic.setTextureId(italicTex);
    m_uiFontNumeric.setTextureId(numTex);

    m_uiFonts.regular = &m_uiFont;
    m_uiFonts.bold    = &m_uiFontBold;
    m_uiFonts.italic  = &m_uiFontItalic;
    m_uiFonts.numeric = &m_uiFontNumeric;
    return true;
}

void GameApp::framebufferSize(int& outW, int& outH) const {
    outW = 0; outH = 0;
    if (m_window) glfwGetFramebufferSize(m_window, &outW, &outH);
}

float GameApp::contentScale() const {
    if (!m_window) return 1.0f;
    float xs = 1.0f, ys = 1.0f;
    glfwGetWindowContentScale(m_window, &xs, &ys);
    return std::max(xs, ys);
}

void GameApp::beginFrameDraw() {
    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);
    m_uiDrawList.reset({static_cast<float>(fbW), static_cast<float>(fbH)});
}

void GameApp::submitFrame(const render::CameraPose& camera, float simulationAlpha) {
    m_uiContext.buildAppend(m_uiDrawList);
    m_renderer.setUiDrawData(m_uiDrawList.data());
    const render::VoxelPreview noPreview{};
    m_renderer.renderFrame(
        m_emptyGrid, m_emptySimulation, camera, noPreview, simulationAlpha, {});
}

} // namespace odai::engine
