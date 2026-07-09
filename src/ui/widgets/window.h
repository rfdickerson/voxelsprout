#pragma once

#include "ui/font.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>
#include <optional>
#include <string>

// A floating window: optional 9-slice frame, a toolbar (title bar) with a close
// button, an outer margin, and inner content padding. Children are laid out in the
// content area (below the toolbar, inset by padding). Dragging the toolbar moves it.
namespace odai::ui {

class Window : public Widget {
public:
    Window(const Font* font, std::string title, std::function<void()> onClose = {});

    // Toolbar height in pixels. Set to kDefaultTitleBarH * dpiScale.
    static constexpr float kDefaultTitleBarH = 30.0f;
    float titleBarH = kDefaultTitleBarH;

    // Space between the widget rect and the drawn frame (outer margin).
    float margin = 0.0f;
    // Content inset from the frame interior (x: left/right, y: top/bottom).
    UiVec2 padding{0.0f, 0.0f};

    // When set, the frame is drawn as a 9-slice; otherwise a solid fill + border.
    std::optional<UiNineSlice> frame;
    UiColor frameTint{1.0f, 1.0f, 1.0f, 1.0f};  // 9-slice tint (white = texture as-is).

    UiColor titleBarColor{0.22f, 0.25f, 0.30f, 0.95f};
    UiColor bodyColor{0.04f, 0.07f, 0.11f, 0.92f};       // Solid-fallback body.
    UiColor borderColor{0.75f, 0.62f, 0.34f, 0.55f};     // Solid-fallback border + separator.
    UiColor titleColor{0.91f, 0.80f, 0.48f, 1.0f};
    // When non-null, the title text is drawn with this font instead of font_.
    // Allows a larger/bolder typeface for window titles vs body content.
    const Font* titleFont = nullptr;
    // Smaller companion face used for lowercase letters in a small-cap title.
    // Uppercase source letters retain the full titleFont height.
    const Font* titleSmallCapsFont = nullptr;
    // Corner radius for the solid-fill path (no effect when a 9-slice frame is set).
    // Kept small for the squared-off Victorian look.
    float cornerRadiusPx = 2.0f;
    // Frame bevel: two-tone highlight/shadow drawn just inside the body silhouette
    // (no effect when a 9-slice frame is set). Defaults give the subtle raised
    // ledge used by the dark HUD theme; raise the alpha toward opaque and swap in
    // a stronger highlight/shadow pair (e.g. Panel::styleMotif()'s palette) for a
    // punchier CDE/Motif-style bevel.
    UiColor frameBevelHighlightColor{1.0f, 1.0f, 1.0f, 0.14f};
    UiColor frameBevelShadowColor{0.0f, 0.0f, 0.0f, 0.45f};
    float   frameBevelThicknessPx = 1.5f;
    // Title bar ledge bevel: same idea, one edge each along the top and bottom of
    // the title bar strip (no effect when a 9-slice frame is set).
    UiColor toolbarBevelHighlightColor{1.0f, 1.0f, 1.0f, 0.18f};
    UiColor toolbarBevelShadowColor{0.0f, 0.0f, 0.0f, 0.35f};
    UiColor closeColor{0.85f, 0.55f, 0.35f, 0.85f};
    UiColor closeHoverColor{0.95f, 0.45f, 0.30f, 1.0f};
    bool showCloseButton = true;
    bool draggable = true;

    bool    showShadow    = true;
    UiColor shadowColor   {0.0f, 0.0f, 0.0f, 0.20f};
    float   shadowBlurPx  = 3.0f;
    float   shadowOffsetX = 2.0f;
    float   shadowOffsetY = 3.0f;

    void setTitle(std::string t) { title_ = std::move(t); }

    // Raise this window above all other windows sharing its parent by assigning
    // it the next value from a shared monotonic counter. Called automatically
    // when the window is clicked into (see onEvent) or shown; call directly
    // when a window is made visible programmatically (e.g. a toolbar button
    // toggling `visible = true`) so it also jumps to the front.
    void bringToFront();

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent& e) override;

    // Usable content area (below the toolbar, inset by padding).
    [[nodiscard]] UiRect contentRect() const;

private:
    [[nodiscard]] UiRect frameRect() const;
    [[nodiscard]] UiRect titleBarRect() const;
    [[nodiscard]] UiRect closeBtnRect() const;

    const Font* font_ = nullptr;
    std::string title_;
    std::function<void()> onClose_;
    bool dragging_ = false;
    bool closeHovered_ = false;
    UiVec2 dragOffset_{};
};

}  // namespace odai::ui
