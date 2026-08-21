#pragma once

#include "audio/audio.h"
#include "engine/game_frame_stats.h"
#include "render/renderer.h"
#include "render/renderer_types.h"
#include "ui/font.h"
#include "ui/ui_context.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_input.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct GLFWwindow;

namespace odai::engine {

// Minimal GLFW + Vulkan base for Vulkan game executables.
// Owns the window, renderer, UI context, fonts, and game loop.
// Subclasses implement onInit(), onTick(), and onRender().
class GameApp {
public:
    virtual ~GameApp() = default;

    bool init(const char* title = "odai");
    void run();
    void shutdown();

    // Per-zone CPU timings for the loop in run(). Populated every frame for
    // every game; read it from a plugin, a game's own HUD, or the built-in
    // overlay (F3). See engine/game_frame_stats.h for the zone set.
    [[nodiscard]] const GameFrameProfiler& frameProfiler() const { return m_frameProfiler; }

    // Built-in CPU timing overlay, toggled at runtime with F3. Starts visible
    // when ODAI_PERF_OVERLAY is set to something other than "0".
    void setPerfOverlayVisible(bool visible) { m_perfOverlayVisible = visible; }
    [[nodiscard]] bool isPerfOverlayVisible() const { return m_perfOverlayVisible; }

protected:
    virtual bool onInit() = 0;
    virtual void onTick(float dt) = 0;
    virtual void onRender(float dt) = 0;
    virtual void onShutdown() {}

    // Upscaler request, consumed by init() BEFORE renderer init for the same
    // reason wantsStrategyMapTuning is: the quality preset chooses the internal
    // render resolution, and that sizes every render target at swapchain build.
    // Setting it from onInit() is too late -- the targets already exist, and the
    // request is silently ignored.
    //
    // Defaults to Off (native), so every existing game is unaffected.
    // ODAI_UPSCALER / ODAI_UPSCALER_QUALITY override whatever this returns.
    virtual render::UpscalerSettings requestedUpscalerSettings() const { return {}; }

    // Draw the software mouse cursor over the frame. Off is for a capture that
    // is the product rather than a diagnostic: the cursor sits wherever the
    // desktop left it and lands in the swapchain image like any other quad, so
    // a headless screenshot run bakes a stray arrow into the corner.
    void setCursorVisible(bool visible) { m_cursorVisible = visible; }

    // Initial audio volumes/mute state, passed to Audio::init() in init() before onInit()
    // runs. Default is AudioConfig{}'s built-in defaults (see audio/audio_types.h). Override
    // if a game wants different starting volumes; there is no persisted GameApp-level config
    // file today (unlike src/app/App's odai.cfg), so this is the explicit seam for that.
    virtual audio::AudioConfig audioConfig() const { return audio::AudioConfig{}; }

    // Load four font faces from disk and register their atlases with the renderer.
    // Call from onInit() after the renderer is up.
    //
    // captionSize and displaySize are the optional ends of a game's type scale
    // (see m_uiFontCaption / m_uiFontDisplay). Each is baked only when > 0, so a
    // game pays for exactly the steps it uses -- a size is a whole packed atlas
    // here, not a free CSS number, which is why the scale is a short explicit
    // list rather than a continuum. Sizes should come from one modular ratio:
    // at ratio 1.2 off a 15px body that is 12 / 15 / 18 / 22.
    bool loadFonts(const std::string& regularPath,
                   const std::string& boldPath,
                   const std::string& italicPath,
                   const std::string& numericPath,
                   float bodySize    = 18.0f,
                   float numericSize = 16.0f,
                   float captionSize = 0.0f,
                   float displaySize = 0.0f);

    // Resolve a relative asset path (e.g. "assets/fonts/Inter.ttf") to an
    // absolute path. Searches ODAI_PROJECT_SOURCE_DIR first, then walks up from CWD.
    static std::string resolveAssetPath(const std::string& relativePath);

    // Query current framebuffer size (DPI-aware). Valid after init().
    void framebufferSize(int& outW, int& outH) const;

    // OS content-scale factor (1.0 on 96 dpi, 2.0 on Retina / 200% Windows scaling).
    // Multiply all hardcoded pixel constants by this to stay crisp on HiDPI displays.
    float contentScale() const;

    // Reset the draw list to the current framebuffer size.
    // Call at the start of onRender() before pre-drawing any background geometry.
    void beginFrameDraw();

    // Flush the UI tree onto the draw list (appending, not resetting) then submit
    // the frame to the renderer. Call at the end of onRender().
    void submitFrame(const render::CameraPose& camera);

    // Capture the mouse into relative/FPS-look mode (GLFW_CURSOR_DISABLED), or release it
    // back to the default hidden-but-free cursor (GLFW_CURSOR_HIDDEN). Games with no
    // mouselook mode never need to call this.
    void setMouseCaptured(bool captured);

    GLFWwindow*    m_window = nullptr;
    render::Renderer m_renderer;
    audio::Audio      m_audio;

    ui::Font    m_uiFont;
    ui::Font    m_uiFontBold;
    ui::Font    m_uiFontItalic;
    ui::Font    m_uiFontNumeric;
    // Optional outer steps of the type scale, baked only if loadFonts() was
    // given a non-zero size for them (valid() is false otherwise). Caption is
    // the regular face one step down -- field labels, legend ends, unit
    // suffixes; display is the bold face one or two steps up -- the one or two
    // numbers per screen the player actually scans for. They are deliberately
    // NOT in m_uiFonts: rich_text mixes its faces on a shared baseline, so an
    // off-size face there would break inline leading.
    ui::Font    m_uiFontCaption;
    ui::Font    m_uiFontDisplay;
    ui::FontSet m_uiFonts{};

    ui::UiContext  m_uiContext;
    ui::UiDrawList m_uiDrawList;
    ui::UiInput    m_uiInput;

    std::vector<std::uint32_t> m_pendingTextInput;
    float  m_pendingScrollDelta = 0.0f;
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
    bool   m_hasMouseSample = false;

private:
    // Draws the F3 overlay onto m_uiDrawList. Called from submitFrame() after
    // the widget tree is flushed so it sits above game UI.
    void drawPerfOverlay();

    void reportFrameStats();
    // ODAI_FRAME_STATS benchmark state. Zero means the benchmark is off.
    double m_frameStatsSeconds = 0.0;
    double m_frameStatsElapsed = 0.0;
    std::vector<float> m_frameIntervalsMs;

    GameFrameProfiler m_frameProfiler;
    bool m_perfOverlayVisible = false;
    bool m_perfOverlayKeyPrev = false;
    // F4's edge-trigger state. The visibility itself lives on the renderer
    // (Renderer::isDebugUiVisible), not here, so nothing can drift out of sync
    // with it.
    bool m_rendererDebugUiKeyPrev = false;
    bool m_cursorVisible = true;
};

} // namespace odai::engine
