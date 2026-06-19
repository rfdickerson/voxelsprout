#pragma once

#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>

// A square clickable button that renders an icon image rather than text.
// Supports the same Normal / Hover / Pressed / Disabled states as Button.
// Set textureId = kUiNoTexture to draw a blank placeholder.
namespace odai::ui {

class IconButton : public Widget {
public:
    explicit IconButton(std::function<void()> onClick)
        : onClick_(std::move(onClick)) {}

    UiTextureId textureId = kUiNoTexture;
    UiRect uvRect{};  // Sub-rect in [0,1] atlas space.

    UiColor colorNormal  {0.10f, 0.10f, 0.12f, 0.70f};
    UiColor colorHover   {0.30f, 0.24f, 0.08f, 0.90f};
    UiColor colorPressed {0.18f, 0.14f, 0.04f, 1.00f};
    UiColor colorDisabled{0.10f, 0.10f, 0.10f, 0.40f};
    UiColor borderColor  {0.75f, 0.62f, 0.34f, 0.00f};  // transparent until hovered
    UiColor borderHoverColor{0.88f, 0.72f, 0.28f, 0.80f};
    UiColor glowColor    {0.95f, 0.75f, 0.30f, 0.55f};

    float cornerRadiusPx    = 6.0f;
    float borderThicknessPx = 1.5f;
    float glowSizePx        = 0.0f;  // 0 = no glow; caller sets to e.g. 12*s
    float iconPaddingPx     = 4.0f;  // inset between border and icon image

    void setEnabled(bool enabled) { enabled_ = enabled; }
    [[nodiscard]] bool enabled() const { return enabled_; }

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent& ev) override;

private:
    std::function<void()> onClick_;
    bool hovered_       = false;
    bool pressedInside_ = false;
    bool enabled_       = true;
};

}  // namespace odai::ui
