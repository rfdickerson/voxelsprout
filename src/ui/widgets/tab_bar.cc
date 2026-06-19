#include "ui/widgets/tab_bar.h"

#include "ui/ui_draw_list.h"

namespace odai::ui {

int TabBar::addTab(std::string label) {
    const int idx = static_cast<int>(tabs_.size());
    tabs_.push_back(std::move(label));
    return idx;
}

UiRect TabBar::tabRect(int index) const {
    if (tabs_.empty()) {
        return {};
    }
    const float tabW = rect_.width() / static_cast<float>(tabs_.size());
    const float x0 = rect_.minX + static_cast<float>(index) * tabW;
    return UiRect{x0, rect_.minY, x0 + tabW, rect_.maxY};
}

void TabBar::draw(UiDrawList& dl) const {
    dl.addRectFilled(rect_, inactiveTabColor);
    dl.addRectFilled(UiRect{rect_.minX, rect_.maxY - 1.0f, rect_.maxX, rect_.maxY}, dividerColor);

    for (int i = 0; i < static_cast<int>(tabs_.size()); ++i) {
        const UiRect tr = tabRect(i);
        const bool active = (i == activeTab);

        if (active) {
            dl.addRectFilled(tr, activeTabColor);
            dl.addRectFilled(
                UiRect{tr.minX, tr.maxY - indicatorThicknessPx, tr.maxX, tr.maxY},
                indicatorColor);
        }

        if (font_ != nullptr && !tabs_[i].empty()) {
            const float tw = font_->measureText(tabs_[i]);
            const float tx = tr.minX + (tr.width() - tw) * 0.5f;
            const float ty = tr.minY + (tr.height() - font_->lineHeightPx()) * 0.5f;
            dl.addText(*font_, tabs_[i], UiVec2{tx, ty}, active ? textActiveColor : textInactiveColor);
        }
    }

    drawChildren(dl);
}

bool TabBar::onEvent(UiEvent& ev) {
    if (!visible) {
        return false;
    }
    if (ev.type == UiEvent::Type::MouseDown && ev.button == UiMouseButton::Left
        && rect_.contains(ev.mousePx)) {
        for (int i = 0; i < static_cast<int>(tabs_.size()); ++i) {
            if (tabRect(i).contains(ev.mousePx)) {
                if (i != activeTab) {
                    activeTab = i;
                    if (onTabChanged) {
                        onTabChanged(i);
                    }
                }
                ev.handled = true;
                return true;
            }
        }
    }
    return false;
}

}  // namespace odai::ui
