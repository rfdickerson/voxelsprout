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

    UiColor titleBarColor{0.10f, 0.14f, 0.20f, 0.95f};
    UiColor bodyColor{0.04f, 0.07f, 0.11f, 0.92f};       // Solid-fallback body.
    UiColor borderColor{0.75f, 0.62f, 0.34f, 0.55f};     // Solid-fallback border + separator.
    UiColor titleColor{0.91f, 0.80f, 0.48f, 1.0f};
    // When non-null, the title text is drawn with this font instead of font_.
    // Allows a larger/bolder typeface for window titles vs body content.
    const Font* titleFont = nullptr;
    // Corner radius for the solid-fill path (no effect when a 9-slice frame is set).
    float cornerRadiusPx = 3.0f;
    UiColor closeColor{0.85f, 0.55f, 0.35f, 0.85f};
    UiColor closeHoverColor{0.95f, 0.45f, 0.30f, 1.0f};
    bool showCloseButton = true;
    bool draggable = true;

    bool    showShadow    = true;
    UiColor shadowColor   {0.0f, 0.0f, 0.0f, 0.35f};
    float   shadowBlurPx  = 6.0f;
    float   shadowOffsetX = 4.0f;
    float   shadowOffsetY = 4.0f;

    void setTitle(std::string t) { title_ = std::move(t); }

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
