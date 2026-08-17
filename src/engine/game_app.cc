#include "engine/game_app.h"

#include "render/upscale/upscale_policy.h"
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
#include <string_view>

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

    // Windowed at a fixed default rather than maximized to the monitor. Maximizing
    // on a 4K display gave a ~3840x2038 swapchain, and every full-resolution pass
    // (HDR scene at 4x MSAA, SSAO, sun shafts, ReSTIR surface GI) scales with it —
    // enough to blow past the driver's hang-check timeout on an integrated GPU and
    // take a VK_ERROR_DEVICE_LOST. Override with ODAI_WINDOW_SIZE=WxH.
    int winW = 1600, winH = 900;
    bool explicitWindowSize = false;
    if (const char* sizeEnv = std::getenv("ODAI_WINDOW_SIZE")) {
        int envW = 0, envH = 0;
        if (std::sscanf(sizeEnv, "%dx%d", &envW, &envH) == 2 && envW > 0 && envH > 0) {
            winW = envW;
            winH = envH;
            explicitWindowSize = true;
        } else {
            VOX_LOGW("engine") << "ignoring malformed ODAI_WINDOW_SIZE=\"" << sizeEnv
                               << "\" (expected WxH, e.g. 1600x900)";
        }
    }
#ifdef GLFW_SCALE_FRAMEBUFFER
    // An explicit size is a render-size contract, not a logical-point request.
    // On a 2x Wayland display a 1920x1080 window otherwise creates a 3840x2160
    // swapchain, quadrupling every full-resolution pass and producing a 4K
    // capture despite the caller asking for 1080p. Leave platform-native HiDPI
    // behavior alone for the default interactive window.
    if (explicitWindowSize) {
        glfwWindowHint(GLFW_SCALE_FRAMEBUFFER, GLFW_FALSE);
    }
#endif
    // Never open larger than the monitor's logical (content-scaled) size.
    if (GLFWmonitor* mon = glfwGetPrimaryMonitor()) {
        float xs = 1.0f, ys = 1.0f;
        glfwGetMonitorContentScale(mon, &xs, &ys);
        if (const GLFWvidmode* mode = glfwGetVideoMode(mon)) {
            winW = std::min(winW, static_cast<int>(std::round(mode->width  / std::max(xs, 1.0f))));
            winH = std::min(winH, static_cast<int>(std::round(mode->height / std::max(ys, 1.0f))));
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

    m_renderer.setStrategyMapMode(wantsStrategyMapTuning());
    if (wantsMinimalRendering()) {
        m_renderer.setMinimalRenderMode(true);
    }
    // MSAA before init: the sample count sizes the render targets and is baked
    // into every pipeline, so it cannot be changed afterwards. ODAI_MSAA lets a
    // fill-rate-bound machine trade edge quality for frame time without a
    // rebuild -- on an integrated GPU at a large window this is the single
    // largest lever on main-pass cost.
    if (const char* msaaEnv = std::getenv("ODAI_MSAA")) {
        const int samples = std::atoi(msaaEnv);
        if (samples == 1 || samples == 2 || samples == 4 || samples == 8) {
            m_renderer.setMsaaSamples(static_cast<std::uint32_t>(samples));
            VOX_LOGI("engine") << "ODAI_MSAA: requesting " << samples << "x MSAA";
        } else {
            VOX_LOGW("engine") << "ignoring ODAI_MSAA=\"" << msaaEnv << "\" (expected 1, 2, 4 or 8)";
        }
    }
    // Before renderer init: the preset sizes every render target. See
    // requestedUpscalerSettings().
    {
        render::UpscalerSettings upscaler = requestedUpscalerSettings();
        if (const char* backendEnv = std::getenv("ODAI_UPSCALER")) {
            if (!render::parseUpscalerBackend(backendEnv, upscaler.backend)) {
                VOX_LOGW("engine") << "ignoring ODAI_UPSCALER=\"" << backendEnv
                                   << "\" (expected off|temporal|xess|fsr|dlss)";
            }
        }
        if (const char* qualityEnv = std::getenv("ODAI_UPSCALER_QUALITY")) {
            if (!render::parseUpscalerQuality(qualityEnv, upscaler.quality)) {
                VOX_LOGW("engine") << "ignoring ODAI_UPSCALER_QUALITY=\"" << qualityEnv << "\"";
            }
        }
        m_renderer.setUpscalerSettings(upscaler);
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

    if (const char* statsEnv = std::getenv("ODAI_FRAME_STATS")) {
        m_frameStatsSeconds = std::atof(statsEnv);
        m_frameIntervalsMs.reserve(4096);
    }
    if (const char* overlayEnv = std::getenv("ODAI_PERF_OVERLAY")) {
        m_perfOverlayVisible = (overlayEnv[0] != '\0' && overlayEnv[0] != '0');
    }
    // ODAI_DEBUG_UI opens the renderer's ImGui surface at startup instead of
    // waiting for F4. The GPU Memory readout is the reason: watching device
    // memory across a long session means having it up from the first frame, and
    // remembering to press a key is exactly what a leak hunt does not need.
    //
    //   1 / stats  -- readouts only (default): frame timings, GPU stages,
    //                 memory, draw calls, plus whatever the game pushed through
    //                 setDebugStatGroups
    //   full       -- the whole tuning console as well
    //
    // Mode and visibility are separate, so F4 keeps toggling the panel without
    // changing which one it toggles.
    if (const char* debugUiEnv = std::getenv("ODAI_DEBUG_UI")) {
        const std::string_view mode{debugUiEnv};
        const bool enabled = !mode.empty() && mode != "0";
        m_renderer.setDebugUiMode(
            (mode == "full") ? render::DebugUiMode::Full : render::DebugUiMode::Stats);
        m_renderer.setDebugUiVisible(enabled);
    }

    core::Stopwatch frameWatch;

    while (m_window && glfwWindowShouldClose(m_window) == GLFW_FALSE) {
        const double now = glfwGetTime();
        // The simulation dt is clamped so one long frame cannot teleport the
        // world; the MEASURED interval must not be, or the clamp silently
        // becomes the reported maximum. A 0.72 s streaming stall showed up as
        // exactly 100 ms in the frame histogram until this was split.
        const double frameIntervalSeconds = now - prevTime;
        const float  dt  = static_cast<float>(std::min(frameIntervalSeconds, 0.1));
        prevTime = now;

        m_frameProfiler.beginFrame();
        frameWatch.restart();

        {
            core::ScopedTimerMs zone(m_frameProfiler.zoneMs(GameZone::Poll));
            glfwPollEvents();
            if (!m_window || glfwWindowShouldClose(m_window)) break;

            int fbW = 0, fbH = 0;
            glfwGetFramebufferSize(m_window, &fbW, &fbH);
            int winW = 0, winH = 0;
            glfwGetWindowSize(m_window, &winW, &winH);

            double mx = 0.0, my = 0.0;
            glfwGetCursorPos(m_window, &mx, &my);
            // glfwGetCursorPos reports window coordinates, but the UI viewport
            // set below -- and every widget rect measured against it -- is in
            // framebuffer pixels. The two are equal only at 1x scale. On a
            // fractional/HiDPI display (1.5x: 1280x720 window, 1920x1080
            // framebuffer) an unscaled cursor tops out at 2/3 of the viewport,
            // so the bottom and right of the UI cannot be reached at all.
            if (winW > 0 && winH > 0) {
                mx *= static_cast<double>(fbW) / static_cast<double>(winW);
                my *= static_cast<double>(fbH) / static_cast<double>(winH);
            }

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

            // F4 opens the renderer's own ImGui panels (frame stats, shadows/AO,
            // sun/sky/post, render debug views). F3 above is this engine's CPU
            // timing overlay and is a different thing; both are edge-triggered
            // for the same reason. ImGui installs its own GLFW callbacks at
            // device init, so no input routing is needed here.
            const bool rendererUiKey = glfwGetKey(m_window, GLFW_KEY_F4) == GLFW_PRESS;
            if (rendererUiKey && !m_rendererDebugUiKeyPrev) {
                m_renderer.setDebugUiVisible(!m_renderer.isDebugUiVisible());
            }
            m_rendererDebugUiKeyPrev = rendererUiKey;

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

        // ODAI_FRAME_STATS=<seconds>: collect wall-clock frame intervals, print
        // a distribution, and quit.
        //
        // The INTERVAL between frames is what the player feels, not the CPU
        // time inside one -- a frame that costs 4 ms of CPU and then blocks 20
        // ms on a fence is a 24 ms frame, and only the interval shows that.
        // Percentiles rather than an average for the same reason: judder is a
        // tail phenomenon, and a mean hides one 40 ms frame per second
        // completely while that single frame is the entire complaint.
        if (m_frameStatsSeconds > 0.0) {
            m_frameIntervalsMs.push_back(static_cast<float>(frameIntervalSeconds) * 1000.0f);
            m_frameStatsElapsed += dt;
            if (m_frameStatsElapsed >= m_frameStatsSeconds) {
                reportFrameStats();
                glfwSetWindowShouldClose(m_window, GLFW_TRUE);
            }
        }
    }
}

void GameApp::reportFrameStats() {
    // Drop the first samples: window creation, swapchain setup and the first
    // pipeline compiles land there and are not what anyone means by frame time.
    constexpr std::size_t kWarmupFrames = 30;
    if (m_frameIntervalsMs.size() <= kWarmupFrames) {
        VOX_LOGW("engine") << "frame stats: too few frames to report";
        return;
    }
    std::vector<float> samples(m_frameIntervalsMs.begin() + kWarmupFrames, m_frameIntervalsMs.end());
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&samples](float fraction) {
        const auto index = static_cast<std::size_t>(fraction * static_cast<float>(samples.size() - 1));
        return samples[index];
    };
    double total = 0.0;
    for (const float sample : samples) {
        total += sample;
    }
    const double mean = total / static_cast<double>(samples.size());

    // Judder metric: how far consecutive intervals move relative to each other.
    // A steady 33 ms is smooth; alternating 8/58 averages the same and is not.
    double totalJump = 0.0;
    float worstJump = 0.0f;
    for (std::size_t i = kWarmupFrames + 1; i < m_frameIntervalsMs.size(); ++i) {
        const float jump = std::abs(m_frameIntervalsMs[i] - m_frameIntervalsMs[i - 1]);
        totalJump += jump;
        worstJump = std::max(worstJump, jump);
    }
    const double meanJump = totalJump / static_cast<double>(samples.size());

    const render::FramePacingStats pacing = m_renderer.framePacingStats();
    VOX_LOGI("engine") << "frame stats over " << samples.size() << " frames:"
                       << "  mean=" << mean << "ms (" << (1000.0 / std::max(mean, 1e-6)) << " fps)"
                       << "  p50=" << percentile(0.50f)
                       << "  p95=" << percentile(0.95f)
                       << "  p99=" << percentile(0.99f)
                       << "  max=" << samples.back();
    VOX_LOGI("engine") << "  frame-to-frame jump: mean=" << meanJump << "ms worst=" << worstJump << "ms";
    // CPU zone breakdown. Without it, "the frame is 30 ms" is a fact with no
    // next step: it does not say whether the CPU is the limiter or is merely
    // waiting on the GPU, and those have opposite fixes.
    const auto zone = [this](GameZone z) { return m_frameProfiler.channel(z).ewmaMs(); };
    VOX_LOGI("engine") << "  cpu ms (ewma): poll=" << zone(GameZone::Poll)
                       << " uiUpdate=" << zone(GameZone::UiUpdate)
                       << " tick=" << zone(GameZone::Tick)
                       << " plugins=" << zone(GameZone::Plugins)
                       << " audio=" << zone(GameZone::Audio)
                       << " render=" << zone(GameZone::Render)
                       << " (uiBuild=" << zone(GameZone::UiBuild)
                       << " submit=" << zone(GameZone::Submit) << ")"
                       << " frame=" << zone(GameZone::Frame);
    VOX_LOGI("engine") << "  cpu waits: frameSlot=" << pacing.cpuWaitFrameSlotMs
                       << "ms acquire=" << pacing.cpuWaitAcquireMs
                       << "ms present=" << pacing.cpuWaitPresentMs
                       << "ms transfer=" << pacing.cpuWaitTransferMs
                       << "ms  queuedFrames=" << pacing.queuedFrames
                       << " latePresents=" << pacing.latePresentCount;
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
                        float bodySize, float numericSize, float captionSize,
                        float displaySize) {
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

    // Optional type-scale steps. A failed bake is not fatal: the game falls back
    // to body size for that step, which loses hierarchy but never the text.
    if (captionSize > 0.0f && m_uiFontCaption.loadFromFile(regularPath, captionSize)) {
        m_uiFontCaption.setTextureId(m_renderer.registerUiFontAtlas(
            m_uiFontCaption.atlasPixels().data(), m_uiFontCaption.atlasWidth(),
            m_uiFontCaption.atlasHeight()));
    }
    if (displaySize > 0.0f && m_uiFontDisplay.loadFromFile(boldPath, displaySize)) {
        m_uiFontDisplay.setTextureId(m_renderer.registerUiFontAtlas(
            m_uiFontDisplay.atlasPixels().data(), m_uiFontDisplay.atlasWidth(),
            m_uiFontDisplay.atlasHeight()));
    }

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
    // have no mouselook mode, so it's shown unless a game suppresses it.
    if (m_cursorVisible) {
        odai::ui::drawCursor(m_uiDrawList, m_uiInput.mousePx, contentScale());
    }
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
