#include "ui/widgets/toggle.h"

#include "ui/ui_draw_list.h"

namespace odai::ui {

void Toggle::draw(UiDrawList& dl) const {
    const float r = rect_.height() * 0.5f;
    const float thumbR = r - 2.0f;

    const UiColor& tc = checked ? trackOn : trackOff;
    dl.addRoundRectFilled(rect_, tc, r);

    // Thumb slides left (off) → right (on) based on animated tween.
    const float t = thumbTween_.eased();
    const float thumbX = rect_.minX + r + t * (rect_.width() - 2.0f * r);
    dl.addCircleFilled(UiVec2{thumbX, rect_.minY + r}, thumbR, thumbColor);

    drawChildren(dl);
}

bool Toggle::onEvent(UiEvent& ev) {
    if (!visible) {
        return false;
    }
    if (ev.type == UiEvent::Type::MouseDown && ev.button == UiMouseButton::Left
        && rect_.contains(ev.mousePx)) {
        checked = !checked;
        thumbTween_.setTarget(checked ? 1.0f : 0.0f);
        if (onChange) {
            onChange(checked);
        }
        ev.handled = true;
        return true;
    }
    return false;
}

}  // namespace odai::ui
