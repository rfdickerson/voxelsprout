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

    // Opt out of the renderer's strategy-map tuning preset.
    //
    // init() applies setStrategyMapMode(true) before renderer init, and that
    // preset does three things a non-strategy game may not want: it pins the
    // ambient-occlusion radius/bias/intensity to values scaled for a hex map
    // (7 world units of AO reach), it disables the ray-tracing runtime
    // outright, and it forces voxel GI onto its legacy path. Those are the
    // right defaults for the 4X/city games this class was built for, and
    // actively wrong for a game whose world is at Bethesda scale (~70 units
    // per metre), where a 7-unit AO radius is 10 cm and the estimator
    // early-outs to "unoccluded" across the whole frame.
    //
    // Defaults to true so every existing game keeps its current behavior; a
    // game that wants the untuned renderer overrides this to false and sets
    // its own AO tuning in onInit(). Checked in init() before renderer init,
    // because setStrategyMapMode's ray-tracing effect cannot be undone later:
    // the RT shader variants and acceleration structures are skipped at
    // pipeline-creation time.
    virtual bool wantsStrategyMapTuning() const { return true; }

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
