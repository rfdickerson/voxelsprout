#include "ui/widgets/context_menu.h"

#include "ui/ui_draw_list.h"

namespace odai::ui {

float ContextMenu::itemY(std::size_t index) const {
    float y = origin_.y;
    for (std::size_t i = 0; i < index && i < items.size(); ++i) {
        y += items[i].separator ? separatorHeightPx : itemHeightPx;
    }
    return y;
}

UiRect ContextMenu::menuRect() const {
    const float bottom = itemY(items.size());
    return UiRect{origin_.x, origin_.y, origin_.x + widthPx, bottom};
}

UiRect ContextMenu::itemRect(std::size_t index) const {
    const float top = itemY(index);
    const float h = (index < items.size() && items[index].separator) ? separatorHeightPx : itemHeightPx;
    return UiRect{origin_.x, top, origin_.x + widthPx, top + h};
}

void ContextMenu::openAt(const UiVec2& screenPx) {
    origin_ = screenPx;
    open_ = true;
    hoveredItem_ = -1;
}

void ContextMenu::close() {
    open_ = false;
    hoveredItem_ = -1;
}

void ContextMenu::draw(UiDrawList& dl) const {
    if (!open_ || items.empty()) {
        drawChildren(dl);
        return;
    }

    const UiRect mr = menuRect();
    dl.addRectFilled(mr, bgColor);
    dl.addRect(mr, borderColor, 1.0f);

    for (std::size_t i = 0; i < items.size(); ++i) {
        const ContextMenuItem& item = items[i];
        const UiRect ir = itemRect(i);
        if (item.separator) {
            const float cy = (ir.minY + ir.maxY) * 0.5f;
            dl.addRectFilled(UiRect{ir.minX + paddingX * 0.5f, cy - 0.5f,
                                     ir.maxX - paddingX * 0.5f, cy + 0.5f},
                              separatorColor);
            continue;
        }
        if (static_cast<int>(i) == hoveredItem_ && item.enabled) {
            dl.addRectFilled(ir, itemHoverColor);
        }
        if (font_ != nullptr) {
            const float ty = ir.minY + (ir.height() - font_->lineHeightPx()) * 0.5f;
            dl.addText(*font_, item.label, UiVec2{ir.minX + paddingX, ty},
                       item.enabled ? textColor : textDisabledColor);
        }
    }

    drawChildren(dl);
}

bool ContextMenu::onEvent(UiEvent& ev) {
    if (!open_) {
        return false;
    }
    if (ev.type == UiEvent::Type::MouseMove) {
        hoveredItem_ = -1;
        for (std::size_t i = 0; i < items.size(); ++i) {
            if (!items[i].separator && itemRect(i).contains(ev.mousePx)) {
                hoveredItem_ = static_cast<int>(i);
                break;
            }
        }
        return false;
    }
    if (ev.type == UiEvent::Type::MouseDown) {
        const UiRect mr = menuRect();
        if (!mr.contains(ev.mousePx)) {
            // Click outside the menu dismisses it without consuming the event, so
            // the click can still reach whatever is underneath.
            close();
            return false;
        }
        for (std::size_t i = 0; i < items.size(); ++i) {
            if (items[i].separator || !items[i].enabled) {
                continue;
            }
            if (itemRect(i).contains(ev.mousePx)) {
                if (items[i].onClick) {
                    items[i].onClick();
                }
                close();
                ev.handled = true;
                return true;
            }
        }
        // Click inside the menu but not on an item (e.g. a separator row):
        // consume it so it doesn't fall through to widgets behind the menu.
        ev.handled = true;
        return true;
    }
    return false;
}

}  // namespace odai::ui
