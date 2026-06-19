#include "ui/widgets/slider.h"

#include "ui/ui_draw_list.h"

#include <algorithm>

namespace odai::ui {

UiRect Slider::trackRect() const {
    const float cy = rect_.minY + rect_.height() * 0.5f;
    const float h = 6.0f;
    return UiRect{rect_.minX + knobRadiusPx, cy - h * 0.5f,
                  rect_.maxX - knobRadiusPx, cy + h * 0.5f};
}

float Slider::thumbX() const {
    const UiRect tr = trackRect();
    return tr.minX + value * tr.width();
}

void Slider::draw(UiDrawList& dl) const {
    const UiRect tr = trackRect();
    const float tx = thumbX();
    const float cy = rect_.minY + rect_.height() * 0.5f;

    // Track background.
    dl.addRoundRectFilled(tr, trackColor, cornerRadiusPx);

    // Filled portion from left to knob.
    if (tx > tr.minX) {
        dl.addRoundRectFilled(UiRect{tr.minX, tr.minY, tx, tr.maxY}, fillColor, cornerRadiusPx);
    }

    // Glow on hover / drag.
    if (hovered_ || dragging_) {
        const UiRect knobBounds{tx - knobRadiusPx, cy - knobRadiusPx,
                                tx + knobRadiusPx, cy + knobRadiusPx};
        dl.addRoundRectGlow(knobBounds, glowColor, knobRadiusPx, knobRadiusPx * 1.4f);
    }

    // Knob.
    const UiColor& kc = (hovered_ || dragging_) ? knobHoverColor : knobColor;
    dl.addCircleFilled(UiVec2{tx, cy}, knobRadiusPx, kc);
    dl.addCircle(UiVec2{tx, cy}, knobRadiusPx, UiColor{0.0f, 0.0f, 0.0f, 0.22f}, 1.5f);

    drawChildren(dl);
}

bool Slider::onEvent(UiEvent& ev) {
    if (!visible) {
        return false;
    }
    if (ev.type == UiEvent::Type::MouseMove) {
        hovered_ = rect_.contains(ev.mousePx);
        if (dragging_) {
            const UiRect tr = trackRect();
            const float raw = tr.width() > 0.0f
                              ? (ev.mousePx.x - tr.minX) / tr.width()
                              : 0.0f;
            value = std::clamp(raw, 0.0f, 1.0f);
            if (onChange) {
                onChange(value);
            }
        }
        return false;
    }
    if (ev.type == UiEvent::Type::MouseDown && ev.button == UiMouseButton::Left
        && rect_.contains(ev.mousePx)) {
        dragging_ = true;
        const UiRect tr = trackRect();
        const float raw = tr.width() > 0.0f
                          ? (ev.mousePx.x - tr.minX) / tr.width()
                          : 0.0f;
        value = std::clamp(raw, 0.0f, 1.0f);
        if (onChange) {
            onChange(value);
        }
        ev.handled = true;
        return true;
    }
    if (ev.type == UiEvent::Type::MouseUp) {
        dragging_ = false;
    }
    return false;
}

}  // namespace odai::ui
