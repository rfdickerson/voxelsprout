#include "ui/widgets/spinner.h"

#include "ui/ui_draw_list.h"

#include <algorithm>
#include <cstdio>

namespace odai::ui {

UiRect Spinner::fieldRect() const {
    return UiRect{rect_.minX, rect_.minY, rect_.maxX - nubWidthPx, rect_.maxY};
}

UiRect Spinner::upRect() const {
    const float h = rect_.height() * 0.5f;
    return UiRect{rect_.maxX - nubWidthPx, rect_.minY, rect_.maxX, rect_.minY + h};
}

UiRect Spinner::downRect() const {
    const float h = rect_.height() * 0.5f;
    return UiRect{rect_.maxX - nubWidthPx, rect_.minY + h, rect_.maxX, rect_.maxY};
}

std::string Spinner::formattedValue() const {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", std::max(0, decimalPlaces), value);
    return std::string{buf};
}

void Spinner::step_by(int dir) {
    const double next = std::clamp(value + static_cast<double>(dir) * step, minValue, maxValue);
    if (next != value) {
        value = next;
        if (onChange) {
            onChange(value);
        }
    }
}

void Spinner::update(float dt) {
    if (heldDir_ == 0) {
        return;
    }
    const float prevHeldTime = heldTimeSec_;
    heldTimeSec_ += dt;
    if (heldTimeSec_ < repeatDelaySec) {
        return;
    }
    // Fire once per repeat interval crossed since the last update() call.
    const float interval = std::max(repeatIntervalSec, 0.001f);
    const float prevSinceStart = std::max(0.0f, prevHeldTime - repeatDelaySec);
    const float nowSinceStart = heldTimeSec_ - repeatDelaySec;
    if (static_cast<int>(nowSinceStart / interval) > static_cast<int>(prevSinceStart / interval)) {
        step_by(heldDir_);
    }
}

void Spinner::draw(UiDrawList& dl) const {
    const UiRect fr = fieldRect();
    const UiRect ur = upRect();
    const UiRect dr = downRect();

    dl.addRoundRectFilled(fr, fieldBg, cornerRadiusPx);
    dl.addRoundRect(fr, fieldBorderColor, cornerRadiusPx, 1.0f);

    dl.addRectFilled(ur, hoveredUp_ ? nubHoverBg : nubBg);
    dl.addRectFilled(dr, hoveredDown_ ? nubHoverBg : nubBg);
    dl.addRect(UiRect{ur.minX, ur.minY, ur.maxX, dr.maxY}, fieldBorderColor, 1.0f);

    if (font_ != nullptr) {
        const std::string text = formattedValue();
        const float ty = fr.minY + (fr.height() - font_->lineHeightPx()) * 0.5f;
        dl.addText(*font_, text, UiVec2{fr.minX + paddingX, ty}, textColor);

        const float chevY = (ur.minY + ur.maxY) * 0.5f - font_->lineHeightPx() * 0.35f;
        const float chevYDown = (dr.minY + dr.maxY) * 0.5f - font_->lineHeightPx() * 0.35f;
        const float cx = ur.minX + (ur.width() - font_->measureText("^")) * 0.5f;
        dl.addText(*font_, "^", UiVec2{cx, chevY}, chevronColor);
        dl.addText(*font_, "v", UiVec2{cx, chevYDown}, chevronColor);
    }

    drawChildren(dl);
}

bool Spinner::onEvent(UiEvent& ev) {
    if (!visible) {
        return false;
    }
    if (ev.type == UiEvent::Type::MouseMove) {
        hoveredUp_ = upRect().contains(ev.mousePx);
        hoveredDown_ = downRect().contains(ev.mousePx);
        return false;
    }
    if (ev.type == UiEvent::Type::MouseDown && ev.button == UiMouseButton::Left) {
        if (upRect().contains(ev.mousePx)) {
            heldDir_ = 1;
            heldTimeSec_ = 0.0f;
            step_by(1);
            ev.handled = true;
            return true;
        }
        if (downRect().contains(ev.mousePx)) {
            heldDir_ = -1;
            heldTimeSec_ = 0.0f;
            step_by(-1);
            ev.handled = true;
            return true;
        }
    }
    if (ev.type == UiEvent::Type::MouseUp) {
        heldDir_ = 0;
        heldTimeSec_ = 0.0f;
    }
    return false;
}

}  // namespace odai::ui
