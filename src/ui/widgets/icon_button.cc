#include "ui/widgets/icon_button.h"

#include "ui/widget.h"

namespace odai::ui {

void IconButton::draw(UiDrawList& dl) const {
    const UiRect& r = rect_;
    const bool isHovered = enabled_ && hovered_;
    const bool isPressed = enabled_ && pressedInside_ && hovered_;

    const UiColor& bg = !enabled_ ? colorDisabled
                      : isPressed ? colorPressed
                      : isHovered ? colorHover
                      : colorNormal;

    // Glow behind the button while hovered/pressed.
    if (isHovered && glowSizePx > 0.0f) {
        dl.addRoundRectGlow(r, glowColor, cornerRadiusPx, glowSizePx);
    }

    // Background fill.
    dl.addRoundRectFilled(r, bg, cornerRadiusPx);

    // Border — brighter when hovered.
    const UiColor& bc = isHovered ? borderHoverColor : borderColor;
    if (bc.a > 0.0f) {
        dl.addRoundRect(r, bc, cornerRadiusPx, borderThicknessPx);
    }

    // Icon image.
    if (textureId != kUiNoTexture) {
        const float pad = iconPaddingPx;
        const UiRect iconRect{r.minX + pad, r.minY + pad, r.maxX - pad, r.maxY - pad};
        dl.addImage(iconRect, textureId, {1.0f, 1.0f, 1.0f, enabled_ ? 1.0f : 0.45f}, uvRect);
    }

    drawChildren(dl);
}

bool IconButton::onEvent(UiEvent& ev) {
    if (!enabled_ || !visible) {
        return false;
    }
    if (ev.type == UiEvent::Type::MouseMove) {
        hovered_ = rect_.contains(ev.mousePx);
        return false;
    }
    if (ev.type == UiEvent::Type::MouseDown && ev.button == UiMouseButton::Left) {
        if (rect_.contains(ev.mousePx)) {
            pressedInside_ = true;
            return true;
        }
    }
    if (ev.type == UiEvent::Type::MouseUp && ev.button == UiMouseButton::Left) {
        const bool wasPressed = pressedInside_;
        pressedInside_ = false;
        if (wasPressed && rect_.contains(ev.mousePx) && onClick_) {
            onClick_();
        }
        return wasPressed;
    }
    return false;
}

}  // namespace odai::ui
