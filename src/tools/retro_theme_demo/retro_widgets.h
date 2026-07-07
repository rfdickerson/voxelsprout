#pragma once

#include "ui/ui_context.h"
#include "ui/ui_input.h"
#include "ui/ui_types.h"
#include "ui/widgets/button.h"
#include "ui/widgets/slider.h"
#include "ui/widgets/tab_bar.h"
#include "ui/widgets/text_box.h"
#include "ui/widgets/toggle.h"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Immediate-mode wrapper over the retained ui:: widgets, for the retro theme
// demo. The demo is otherwise immediate-mode, so this lets it keep that call
// style — `ui.slider("audio.vol", rect)` returns a persistent, interactive
// Slider that is created on first use, repositioned + reskinned every frame, and
// hidden when not referenced (e.g. after a theme switch).
//
// Vulkan-free: depends only on odai_ui, so it builds and unit-tests headlessly.
// All input runs in framebuffer-pixel space (the same space the demo lays out
// in), sidestepping the logical/framebuffer mismatch of GameApp's shared context.
namespace odai::tools::retro_theme_demo {

// Per-era visual parameters, filled by the demo from its ThemePalette. The skin
// functions map these onto each widget's public style members; behavior is
// unchanged, so the same widget renders correctly in any retro theme.
struct WidgetSkin {
    ui::UiColor face{0.75f, 0.75f, 0.75f, 1.f};      // control / button face
    ui::UiColor faceHover{0.82f, 0.82f, 0.82f, 1.f};
    ui::UiColor facePressed{0.66f, 0.66f, 0.66f, 1.f};
    ui::UiColor text{0, 0, 0, 1};                     // label / input text
    ui::UiColor textDim{0.5f, 0.5f, 0.5f, 1.f};       // placeholder / hint
    ui::UiColor accent{0, 0, 0.5f, 1.f};              // fill / indicator / focus / on
    ui::UiColor trough{0.5f, 0.5f, 0.5f, 1.f};        // slider track / toggle-off (sunken)
    ui::UiColor field{1, 1, 1, 1};                    // text-input background
    ui::UiColor border{0, 0, 0, 1};                   // outlines
    ui::UiColor bevelLight{1, 1, 1, 1};               // raised bevel highlight
    ui::UiColor bevelDark{0.5f, 0.5f, 0.5f, 1.f};     // raised bevel shadow
    float cornerRadius = 0.f;  // 0 = sharp (Win95/Motif/Mac); >0 = Flat/Retro-OS
    bool bevel = false;        // draw a 3D bevel on buttons (Win95/Motif)
    float scale = 1.f;         // DPI factor for stroke/knob sizes
};

class RetroUi {
public:
    // Build the hosting context + a mouse-passthrough root. Call once.
    void init();

    // Start a frame: remember the skin and hide every widget; widgets touched by
    // accessor calls this frame are re-shown.
    void beginFrame(const WidgetSkin& skin);

    // Immediate-mode accessors: create-or-reuse a persistent widget under `id`,
    // place it at `rect`, skin it for the current theme, and mark it visible.
    // The returned reference lets the caller read/set widget-specific state
    // (slider.value, textBox.value(), toggle.checked, ...). State persists across
    // frames, so do NOT overwrite it every frame unless you mean to.
    ui::Slider& slider(std::string_view id, const ui::UiRect& rect);
    ui::TextBox& textBox(std::string_view id, const ui::UiRect& rect, const ui::Font* font,
                         const char* placeholder = "");
    ui::Toggle& toggle(std::string_view id, const ui::UiRect& rect);
    ui::Button& button(std::string_view id, const ui::UiRect& rect, const ui::Font* font,
                       const char* label);
    ui::TabBar& tabs(std::string_view id, const ui::UiRect& rect, const ui::Font* font,
                     const std::vector<std::string>& labels);

    // Dispatch one frame of input (framebuffer-pixel mouse, left-button state,
    // and text codepoints) to the visible widgets, then advance animations.
    void update(const ui::UiVec2& mousePx, bool leftDown, const ui::UiVec2& viewportPx,
                const std::vector<std::uint32_t>& textInput, float dt);

    // Append all visible widget geometry to the demo's draw list.
    void append(ui::UiDrawList& drawList);

    [[nodiscard]] bool wantsMouse() const { return ctx_.wantsMouse(); }

private:
    void applySkin(ui::Slider& w) const;
    void applySkin(ui::TextBox& w) const;
    void applySkin(ui::Toggle& w) const;
    void applySkin(ui::Button& w) const;
    void applySkin(ui::TabBar& w) const;

    // Look up an existing widget of type T by id, or create+adopt one via factory.
    template <typename T, typename Factory>
    T& acquire(std::string_view id, Factory&& make);

    ui::UiContext ctx_;
    ui::UiInput input_;  // persistent so button press/release edges compute correctly
    ui::Widget* root_ = nullptr;
    WidgetSkin skin_{};
    std::unordered_map<std::string, ui::Widget*> byId_;
};

}  // namespace odai::tools::retro_theme_demo
