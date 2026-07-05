#include "ui/widgets/radio_button.h"

#include "ui/ui_draw_list.h"

#include <algorithm>

namespace odai::ui {

void RadioButton::draw(UiDrawList& dl) const {
    const UiVec2 center{(rect_.minX + rect_.maxX) * 0.5f, (rect_.minY + rect_.maxY) * 0.5f};
    const float radius = std::min(rect_.width(), rect_.height()) * 0.5f;
    const UiColor& ring = hovered_ ? ringHoverColor : ringColor;
    dl.addCircle(center, radius, ring, ringThicknessPx);
    if (selected) {
        dl.addCircleFilled(center, radius * 0.5f, dotColor);
    }
    drawChildren(dl);
}

bool RadioButton::onEvent(UiEvent& ev) {
    if (!visible) {
        return false;
    }
    if (ev.type == UiEvent::Type::MouseMove) {
        hovered_ = rect_.contains(ev.mousePx);
        return false;
    }
    if (ev.type == UiEvent::Type::MouseDown && ev.button == UiMouseButton::Left &&
        rect_.contains(ev.mousePx)) {
        if (onSelect) {
            onSelect();
        }
        ev.handled = true;
        return true;
    }
    return false;
}

void ButtonGroup::add(RadioButton* button) {
    const std::size_t index = buttons_.size();
    buttons_.push_back(button);
    button->onSelect = [this, index]() { selectIndex(index); };
}

void ButtonGroup::selectIndex(std::size_t index) {
    if (index >= buttons_.size() || static_cast<int>(index) == selectedIndex_) {
        return;
    }
    for (std::size_t i = 0; i < buttons_.size(); ++i) {
        buttons_[i]->selected = (i == index);
    }
    selectedIndex_ = static_cast<int>(index);
    if (onChange) {
        onChange(selectedIndex_);
    }
}

}  // namespace odai::ui
