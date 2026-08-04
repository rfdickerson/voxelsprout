#pragma once

#include "audio/audio.h"
#include "engine/game_frame_stats.h"
#include "engine/plugin.h"
#include "render/renderer.h"
#include "render/renderer_types.h"
#include "sim/simulation.h"
#include "ui/font.h"
#include "ui/ui_context.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_input.h"
#include "world/chunk_grid.h"

#include <cstddef>
#include <cstdint>
#include <span>
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

    // Opt in to UI-only rendering: skip building the 3D scene pipelines the renderer
    // otherwise creates (pipe/imported/sky-cloud/water/grass, SSAO, hex terrain).
    // Override to return true in tools that draw nothing but the 2D UI overlay.
    // Checked in init() before the renderer is initialized. Default: false.
    virtual bool wantsMinimalRendering() const { return false; }

    // Initial audio volumes/mute state, passed to Audio::init() in init() before onInit()
    // runs. Default is AudioConfig{}'s built-in defaults (see audio/audio_types.h). Override
    // if a game wants different starting volumes; there is no persisted GameApp-level config
    // file today (unlike src/app/App's odai.cfg), so this is the explicit seam for that.
    virtual audio::AudioConfig audioConfig() const { return audio::AudioConfig{}; }

    // Voxel chunk content for a game that renders real world/ terrain (as opposed to an
    // ImportedScene or pure 2D/UI content). Pass a pointer to submitFrame() when the game
    // has resident chunks to draw; leave it null (the default) to render nothing, which
    // reproduces prior behavior exactly for every non-voxel game.
    struct WorldFrameContent {
        const world::ChunkGrid* chunkGrid = nullptr;
        std::span<const std::size_t> visibleChunkIndices{};
        render::VoxelPreview voxelPreview{};
    };

    // Load four font faces from disk and register their atlases with the renderer.
    // Call from onInit() after the renderer is up.
    bool loadFonts(const std::string& regularPath,
                   const std::string& boldPath,
                   const std::string& italicPath,
                   const std::string& numericPath,
                   float bodySize    = 18.0f,
                   float numericSize = 16.0f);

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
    // importedActors, when non-null, is per-frame dynamic geometry (packed
    // vertex-color format) streamed through the FrameArena — use it for small
    // animated meshes (vehicles, units) that would be far too expensive to
    // re-upload via uploadImportedScene() every frame. The spans must stay
    // alive through the call only.
    void submitFrame(const render::CameraPose& camera, float simulationAlpha = 0.0f,
                     const render::ImportedActorFrameData* importedActors = nullptr,
                     const WorldFrameContent* worldContent = nullptr);

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
    ui::FontSet m_uiFonts{};

    ui::UiContext  m_uiContext;
    ui::UiDrawList m_uiDrawList;
    ui::UiInput    m_uiInput;

    // Register plugins from onInit(); GameApp::init() attaches them right
    // after onInit() returns, run() ticks them every frame, and shutdown()
    // detaches them right after onShutdown(). See engine/plugin.h for the
    // onRender() caveat -- it is not fanned out automatically.
    PluginRegistry m_plugins;

    std::vector<std::uint32_t> m_pendingTextInput;
    float  m_pendingScrollDelta = 0.0f;
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
    bool   m_hasMouseSample = false;

private:
    // Draws the F3 overlay onto m_uiDrawList. Called from submitFrame() after
    // the widget tree is flushed so it sits above game UI.
    void drawPerfOverlay();

    world::ChunkGrid  m_emptyGrid;
    sim::Simulation   m_emptySimulation;

    GameFrameProfiler m_frameProfiler;
    bool m_perfOverlayVisible = false;
    bool m_perfOverlayKeyPrev = false;
};

} // namespace odai::engine
