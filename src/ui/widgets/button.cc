#include "ui/widgets/button.h"

#include <algorithm>

namespace odai::ui {

void Button::setEnabled(bool enabled) {
    enabled_ = enabled;
    if (!enabled_) {
        hovered_ = false;
        pressedInside_ = false;
        state_ = State::Disabled;
    } else if (state_ == State::Disabled) {
        state_ = State::Normal;
    }
}

UiColor Button::backgroundForState() const {
    switch (state_) {
        case State::Hover:    return colorHover;
        case State::Pressed:  return colorPressed;
        case State::Disabled: return colorDisabled;
        case State::Normal:   break;
    }
    return colorNormal;
}

void Button::draw(UiDrawList& drawList) const {
    // Mouse-over glow behind the fill (pressed glows a touch brighter). With
    // drawGlowAtRest, the glow shows even when idle (smart turn button pulse).
    if (glowSizePx > 0.0f && glowColor.a > 0.0f &&
        (drawGlowAtRest || state_ == State::Hover || state_ == State::Pressed)) {
        UiColor glow = glowColor;
        if (state_ == State::Pressed) {
            glow.a = std::min(1.0f, glow.a * 1.4f);
        }
        drawList.addRoundRectGlow(rect_, glow, cornerRadiusPx, glowSizePx);
    }
    drawList.addRoundRectFilled(rect_, backgroundForState(), cornerRadiusPx);
    if (borderThicknessPx > 0.0f && borderColor.a > 0.0f) {
        drawList.addRoundRect(rect_, borderColor, cornerRadiusPx, borderThicknessPx);
    }
    // State accent stripe along the left edge (smart turn button).
    if (accentColor.a > 0.0f && accentWidthPx > 0.0f) {
        const float inset = std::min(borderThicknessPx + 1.0f, rect_.width());
        const UiRect stripe{rect_.minX + inset, rect_.minY + inset,
                            rect_.minX + inset + accentWidthPx, rect_.maxY - inset};
        if (stripe.valid()) {
            drawList.addRoundRectFilled(stripe, accentColor, accentWidthPx * 0.5f);
        }
    }
    if (showBevel) {
        drawList.addBevel(rect_, bevelHighlightColor, bevelShadowColor,
                          cornerRadiusPx, bevelThicknessPx,
                          bevelInward || state_ == State::Pressed);
    }
    if (font_ != nullptr && !label_.empty()) {
        const float textWidth = font_->measureText(label_);
        const float textX = rect_.minX + ((rect_.width() - textWidth) * 0.5f);
        const float textY = rect_.minY + ((rect_.height() - font_->lineHeightPx()) * 0.5f);
        drawList.pushClip(rect_);
        drawList.addText(*font_, label_, UiVec2{textX, textY}, labelColor);
        drawList.popClip();
    }
}

bool Button::onEvent(UiEvent& event) {
    if (!enabled_) {
        return false;
    }
    const bool inside = rect_.contains(event.mousePx);
    switch (event.type) {
        case UiEvent::Type::MouseMove:
            hovered_ = inside;
            if (!pressedInside_) {
                state_ = inside ? State::Hover : State::Normal;
            }
            return false;
        case UiEvent::Type::MouseDown:
            if (inside && event.button == UiMouseButton::Left) {
                pressedInside_ = true;
                state_ = State::Pressed;
                event.handled = true;
                return true;
            }
            return false;
        case UiEvent::Type::MouseUp:
            if (event.button == UiMouseButton::Left && pressedInside_) {
                pressedInside_ = false;
                state_ = inside ? State::Hover : State::Normal;
                event.handled = true;
                if (inside && onClick_) {
                    onClick_();
                }
                return true;
            }
            return false;
        case UiEvent::Type::Scroll:
        case UiEvent::Type::Text:
            return false;
    }
    return false;
}

}  // namespace odai::ui
