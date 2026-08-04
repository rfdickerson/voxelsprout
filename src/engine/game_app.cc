#include "engine/game_app.h"
#include "core/log.h"
#include "core/win_timer_resolution.h"
#include "ui/ui_cursor.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    core::requestHighResTimer();

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
    // Hide the OS cursor; the custom-rendered one takes over (see submitFrame).
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

    glfwSetCharCallback(m_window, [](GLFWwindow* win, unsigned int cp) {
        if (auto* self = static_cast<GameApp*>(glfwGetWindowUserPointer(win)))
            self->m_pendingTextInput.push_back(cp);
    });
    glfwSetScrollCallback(m_window, [](GLFWwindow* win, double /*x*/, double dy) {
        if (auto* self = static_cast<GameApp*>(glfwGetWindowUserPointer(win)))
            self->m_pendingScrollDelta += static_cast<float>(dy);
    });

    m_renderer.setStrategyMapMode(true);
    if (wantsMinimalRendering()) {
        m_renderer.setMinimalRenderMode(true);
    }
    if (!m_renderer.init(m_window, m_emptyGrid)) {
        VOX_LOGE("engine") << "renderer init failed";
        glfwDestroyWindow(m_window);
        m_window = nullptr;
        glfwTerminate();
        return false;
    }

    m_audio.init(audioConfig());

    if (!onInit()) {
        m_audio.shutdown();
        m_renderer.shutdown();
        glfwDestroyWindow(m_window);
        m_window = nullptr;
        glfwTerminate();
        return false;
    }

    if (!m_plugins.attachAll(*this)) {
        m_plugins.detachAll(*this);
        m_audio.shutdown();
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

    if (const char* overlayEnv = std::getenv("ODAI_PERF_OVERLAY")) {
        m_perfOverlayVisible = (overlayEnv[0] != '\0' && overlayEnv[0] != '0');
    }

    core::Stopwatch frameWatch;

    while (m_window && glfwWindowShouldClose(m_window) == GLFW_FALSE) {
        const double now = glfwGetTime();
        const float  dt  = static_cast<float>(std::min(now - prevTime, 0.1));
        prevTime = now;

        m_frameProfiler.beginFrame();
        frameWatch.restart();

        {
            core::ScopedTimerMs zone(m_frameProfiler.zoneMs(GameZone::Poll));
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

            // Edge-triggered so holding F3 does not strobe the overlay.
            const bool overlayKey = glfwGetKey(m_window, GLFW_KEY_F3) == GLFW_PRESS;
            if (overlayKey && !m_perfOverlayKeyPrev) {
                m_perfOverlayVisible = !m_perfOverlayVisible;
            }
            m_perfOverlayKeyPrev = overlayKey;

            m_uiContext.setViewport({static_cast<float>(fbW), static_cast<float>(fbH)});
        }

        {
            core::ScopedTimerMs zone(m_frameProfiler.zoneMs(GameZone::UiUpdate));
            m_uiContext.update(m_uiInput);
            m_uiContext.tick(dt);
        }
        {
            core::ScopedTimerMs zone(m_frameProfiler.zoneMs(GameZone::Tick));
            onTick(dt);
        }
        {
            core::ScopedTimerMs zone(m_frameProfiler.zoneMs(GameZone::Plugins));
            m_plugins.tickAll(*this, dt);
        }
        {
            core::ScopedTimerMs zone(m_frameProfiler.zoneMs(GameZone::Audio));
            m_audio.update(dt);
        }
        {
            core::ScopedTimerMs zone(m_frameProfiler.zoneMs(GameZone::Render));
            onRender(dt);
        }

        m_frameProfiler.endFrame(frameWatch.elapsedMs());
    }
}

void GameApp::shutdown() {
    onShutdown();
    m_plugins.detachAll(*this);
    m_audio.shutdown();
    m_renderer.shutdown();
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
    core::releaseHighResTimer();
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

void GameApp::submitFrame(const render::CameraPose& camera, float simulationAlpha,
                          const render::ImportedActorFrameData* importedActors,
                          const WorldFrameContent* worldContent) {
    {
        core::ScopedTimerMs zone(m_frameProfiler.zoneMs(GameZone::UiBuild));
        m_uiContext.buildAppend(m_uiDrawList);
    }

    // Overlay sits above game widgets but below the cursor, and is drawn after
    // the UiBuild zone closes so measuring the UI does not measure the meter.
    // Its own cost still lands in the enclosing Render zone -- unavoidable, and
    // it is zero when the overlay is hidden.
    drawPerfOverlay();

    // Custom cursor: drawn last so it renders above every widget. GameApp tools
    // have no mouselook mode, so it's always shown.
    odai::ui::drawCursor(m_uiDrawList, m_uiInput.mousePx, contentScale());
    m_renderer.setUiDrawData(m_uiDrawList.data());

    const world::ChunkGrid& grid = (worldContent && worldContent->chunkGrid)
        ? *worldContent->chunkGrid : m_emptyGrid;
    const render::VoxelPreview preview = worldContent ? worldContent->voxelPreview : render::VoxelPreview{};
    const std::span<const std::size_t> visible =
        worldContent ? worldContent->visibleChunkIndices : std::span<const std::size_t>{};

    core::ScopedTimerMs zone(m_frameProfiler.zoneMs(GameZone::Submit));
    m_renderer.renderFrame(grid, m_emptySimulation, camera, preview, simulationAlpha, visible, importedActors);
}

namespace {

// Right-aligned millisecond text with no allocation: the overlay redraws every
// frame it is visible, and this codebase's performance contract rules out
// implicit allocation on a render path even for debug UI.
struct MsText {
    char chars[16];
    [[nodiscard]] const char* c_str() const { return chars; }
};

MsText formatMs(float ms) {
    MsText out{};
    std::snprintf(out.chars, sizeof(out.chars), "%6.2f", static_cast<double>(ms));
    return out;
}

}  // namespace

void GameApp::drawPerfOverlay() {
    if (!m_perfOverlayVisible) return;
    const ui::Font* font = m_uiFonts.regular;
    // A game that never called loadFonts() has no atlas to draw glyphs from;
    // silently skip rather than emitting garbage geometry.
    if (!font) return;

    const float scale = contentScale();
    const float pad = 8.0f * scale;
    const float lineH = font->lineHeightPx();
    const float barW = 90.0f * scale;
    const float labelW = 78.0f * scale;
    const float numW = 52.0f * scale;

    // Footer strings are built up front so the panel can be sized to fit them:
    // they are wider than the zone columns at most font sizes.
    const render::FramePacingStats pacing = m_renderer.framePacingStats();
    const render::UiRenderStats uiStats = m_renderer.uiRenderStats();

    char headText[96];
    std::snprintf(headText, sizeof(headText), "%.0f fps   %.2f ms   frame %llu",
                  static_cast<double>(m_frameProfiler.fps()),
                  static_cast<double>(m_frameProfiler.channel(GameZone::Frame).ewmaMs()),
                  static_cast<unsigned long long>(m_frameProfiler.frameIndex()));
    char otherText[96];
    std::snprintf(otherText, sizeof(otherText), "other         %s",
                  formatMs(m_frameProfiler.unattributedMs()).c_str());
    char waitText[128];
    std::snprintf(waitText, sizeof(waitText), "wait slot %.2f acq %.2f pres %.2f xfer %.2f",
                  static_cast<double>(pacing.cpuWaitFrameSlotMs),
                  static_cast<double>(pacing.cpuWaitAcquireMs),
                  static_cast<double>(pacing.cpuWaitPresentMs),
                  static_cast<double>(pacing.cpuWaitTransferMs));
    char drawText[128];
    std::snprintf(drawText, sizeof(drawText), "ui draws %u  cmds %u  queued %u  late %u",
                  uiStats.drawCallCount, uiStats.commandCount,
                  pacing.queuedFrames, pacing.latePresentCount);

    // Two header rows (rate, then column labels), one row per zone, the
    // unattributed row, and two renderer-counter rows.
    const std::size_t rowCount = kGameZoneCount + 5;
    const float columnsW = labelW + (numW * 3.0f) + pad + barW;
    const float panelW = std::max({columnsW,
                                   font->measureText(headText),
                                   font->measureText(waitText),
                                   font->measureText(drawText)}) + (pad * 2.0f);
    const float panelH = (static_cast<float>(rowCount) * lineH) + (pad * 2.0f);

    const ui::UiRect panel{pad, pad, pad + panelW, pad + panelH};
    m_uiDrawList.addRoundRectFilled(panel, ui::UiColor{0.04f, 0.05f, 0.07f, 0.86f}, 4.0f * scale);
    m_uiDrawList.addRoundRect(panel, ui::UiColor{0.35f, 0.40f, 0.48f, 0.9f}, 4.0f * scale, 1.0f);

    const ui::UiColor dim{0.62f, 0.66f, 0.72f, 1.0f};
    const ui::UiColor bright{0.94f, 0.96f, 0.99f, 1.0f};

    float y = pad + pad;
    const float x = pad + pad;

    // Header: smoothed rate plus the frame budget it corresponds to.
    m_uiDrawList.addText(*font, headText, {x, y}, bright);
    y += lineH;
    m_uiDrawList.addText(*font, "zone            last    avg    p99", {x, y}, dim);
    y += lineH;

    // Bars are scaled against the frame's own average, so the widest bar is
    // whatever currently dominates rather than a fixed 16.6 ms assumption.
    const float frameAvg = std::max(m_frameProfiler.channel(GameZone::Frame).ewmaMs(), 0.01f);

    for (std::size_t i = 0; i < kGameZoneCount; ++i) {
        const auto zone = static_cast<GameZone>(i);
        const auto& ch = m_frameProfiler.channel(zone);
        const bool isFrame = (zone == GameZone::Frame);
        const ui::UiColor textColor = isFrame ? bright : dim;

        m_uiDrawList.addText(*font, gameZoneName(zone), {x, y}, textColor);
        m_uiDrawList.addText(*font, formatMs(ch.lastMs()).c_str(), {x + labelW, y}, textColor);
        m_uiDrawList.addText(*font, formatMs(ch.ewmaMs()).c_str(), {x + labelW + numW, y}, textColor);
        m_uiDrawList.addText(*font, formatMs(ch.p99Ms()).c_str(), {x + labelW + (numW * 2.0f), y}, textColor);

        if (!isFrame) {
            const float frac = std::clamp(ch.ewmaMs() / frameAvg, 0.0f, 1.0f);
            const float bx = x + labelW + (numW * 3.0f) + pad;
            const ui::UiRect track{bx, y + (lineH * 0.30f), bx + barW, y + (lineH * 0.70f)};
            m_uiDrawList.addRectFilled(track, ui::UiColor{1.0f, 1.0f, 1.0f, 0.08f});
            if (frac > 0.0f) {
                const ui::UiRect fill{bx, track.minY, bx + (barW * frac), track.maxY};
                // Nested zones read cooler so they don't look like extra cost
                // on top of the Render row they are already part of.
                m_uiDrawList.addRectFilled(fill, gameZoneIsNested(zone)
                    ? ui::UiColor{0.35f, 0.62f, 0.86f, 0.85f}
                    : ui::UiColor{0.42f, 0.80f, 0.55f, 0.85f});
            }
        }
        y += lineH;
    }

    // Unattributed time closes the accounting loop: if this is large, the cost
    // is somewhere run() does not yet measure.
    m_uiDrawList.addText(*font, otherText, {x, y}, dim);
    y += lineH;

    // The renderer's own counters, so a CPU spike can be told apart from a
    // frame the CPU merely spent blocked waiting on the GPU or the presenter.
    m_uiDrawList.addText(*font, waitText, {x, y}, dim);
    y += lineH;
    m_uiDrawList.addText(*font, drawText, {x, y}, dim);
}

void GameApp::setMouseCaptured(bool captured) {
    if (!m_window) return;
    glfwSetInputMode(m_window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_HIDDEN);
}

} // namespace odai::engine
