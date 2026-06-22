#pragma once

#include "render/renderer.h"
#include "render/renderer_types.h"
#include "sim/simulation.h"
#include "ui/font.h"
#include "ui/ui_context.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_input.h"
#include "world/chunk_grid.h"

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

protected:
    virtual bool onInit() = 0;
    virtual void onTick(float dt) = 0;
    virtual void onRender(float dt) = 0;
    virtual void onShutdown() {}

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
    void submitFrame(const render::CameraPose& camera, float simulationAlpha = 0.0f);

    GLFWwindow*    m_window = nullptr;
    render::Renderer m_renderer;

    ui::Font    m_uiFont;
    ui::Font    m_uiFontBold;
    ui::Font    m_uiFontItalic;
    ui::Font    m_uiFontNumeric;
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
    world::ChunkGrid  m_emptyGrid;
    sim::Simulation   m_emptySimulation;
};

} // namespace odai::engine
