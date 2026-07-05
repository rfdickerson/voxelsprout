#pragma once

// Headless (no window, no swapchain, no display required) Vulkan capture harness
// for odai_ui screens. Renders one UiDrawData frame into an off-screen
// R8G8B8A8_UNORM color image and dumps it to a PNG, so a UI screen built purely
// against odai_ui can be visually inspected without a live GLFW window.
//
// This exists so contestant implementations in the odai_ui stress test can
// spend 100% of their effort on the UI screen itself; none of them need to
// write their own Vulkan bootstrap or PNG export code.
#include "ui/font.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"

#include <cstdint>
#include <string>

namespace odai::uistress {

struct CaptureResult {
    bool success = false;
    double submitToIdleMs = 0.0;         // Wall-clock around vkQueueSubmit..vkQueueWaitIdle for this frame.
    std::uint32_t drawCallCount = 0;      // From UiRenderer::Stats after record().
    std::uint32_t commandCount = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount = 0;
};

class OffscreenCapture {
public:
    struct Config {
        std::uint32_t width = 1600;
        std::uint32_t height = 1000;
        std::uint32_t maxTextureCount = 128;
        // Directory holding ui.vert.slang.spv / ui.frag.slang.spv. Empty ->
        // resolved automatically relative to the repo root.
        std::string shaderDir;
    };

    bool init(const Config& config);
    void shutdown();

    [[nodiscard]] odai::ui::UiVec2 sizePx() const {
        return odai::ui::UiVec2{static_cast<float>(m_width), static_cast<float>(m_height)};
    }

    // Loads a TTF and uploads its atlas as the primary font atlas (kUiFontAtlas).
    // Call once; for a second face (bold/italic) use registerFontAtlas() instead.
    bool loadPrimaryFont(odai::ui::Font& font, const std::string& ttfPath, float pixelHeight);
    odai::ui::UiTextureId registerFontAtlas(const odai::ui::Font& font);
    odai::ui::UiTextureId registerTextureRgba8(const std::uint8_t* pixels, std::uint32_t w, std::uint32_t h);
    odai::ui::UiTextureId registerTextureRgba8Mipmapped(const std::uint8_t* pixels, std::uint32_t w, std::uint32_t h);

    // Renders one frame of `drawData` (already built via UiContext::build() /
    // UiDrawList) into the off-screen target and writes it to `outPngPath`.
    // Safe to call multiple times (e.g. to capture several states of the same
    // screen); each call is a fresh frame.
    CaptureResult captureToPng(const odai::ui::UiDrawData& drawData, const std::string& outPngPath);

    // Public only so offscreen_capture.cc's free functions can operate on it;
    // its definition lives entirely in the .cc, so it's still opaque to callers.
    struct Impl;

private:
    Impl* m_impl = nullptr;
    std::uint32_t m_width = 0;
    std::uint32_t m_height = 0;
};

}  // namespace odai::uistress
