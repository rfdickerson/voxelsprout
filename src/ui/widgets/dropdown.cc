#include "ui/widgets/dropdown.h"

#include "ui/ui_draw_list.h"

#include <algorithm>

namespace odai::ui {

int Dropdown::visibleCount() const {
    return std::min(static_cast<int>(items.size()), maxVisibleItems);
}

UiRect Dropdown::popupRect() const {
    const float h = static_cast<float>(visibleCount()) * itemHeightPx;
    return UiRect{rect_.minX, rect_.maxY, rect_.maxX, rect_.maxY + h};
}

UiRect Dropdown::itemRect(int index) const {
    const UiRect pr = popupRect();
    return UiRect{pr.minX, pr.minY + static_cast<float>(index) * itemHeightPx,
                  pr.maxX, pr.minY + static_cast<float>(index + 1) * itemHeightPx};
}

void Dropdown::draw(UiDrawList& dl) const {
    // Header.
    const UiColor& hbg = headerHovered_ ? headerHoverBg : headerBg;
    dl.addRoundRectFilled(rect_, hbg, cornerRadiusPx);
    dl.addRoundRect(rect_, headerBorderColor, cornerRadiusPx, 1.0f);

    if (font_ != nullptr) {
        const float ty = rect_.minY + (rect_.height() - font_->lineHeightPx()) * 0.5f;
        const bool valid = selectedIndex >= 0 && selectedIndex < static_cast<int>(items.size());
        if (valid) {
            dl.addText(*font_, items[selectedIndex], UiVec2{rect_.minX + paddingX, ty}, textColor);
        }
        // Chevron indicator — fall back to ASCII ^ / v for broad font compatibility.
        const std::string_view arrow = open_ ? "^" : "v";
        const float aw = font_->measureText(arrow);
        dl.addText(*font_, arrow, UiVec2{rect_.maxX - paddingX - aw, ty}, chevronColor);
    }

    // Floating popup — drawn last so it layers above siblings.
    if (open_ && !items.empty()) {
        const UiRect pr = popupRect();
        dl.addRectFilled(pr, popupBg);
        dl.addRect(pr, popupBorderColor, 1.0f);

        const int n = visibleCount();
        for (int i = 0; i < n; ++i) {
            const UiRect ir = itemRect(i);
            if (i == hoveredItem_) {
                dl.addRectFilled(ir, itemHoverColor);
            }
            if (font_ != nullptr) {
                const float ty = ir.minY + (ir.height() - font_->lineHeightPx()) * 0.5f;
                dl.addText(*font_, items[i], UiVec2{ir.minX + paddingX, ty}, textColor);
            }
        }
    }

    drawChildren(dl);
}

bool Dropdown::onEvent(UiEvent& ev) {
    if (!visible) {
        return false;
    }
    if (ev.type == UiEvent::Type::MouseMove) {
        headerHovered_ = rect_.contains(ev.mousePx);
        hoveredItem_ = -1;
        if (open_) {
            const int n = visibleCount();
            for (int i = 0; i < n; ++i) {
                if (itemRect(i).contains(ev.mousePx)) {
                    hoveredItem_ = i;
                    break;
                }
            }
        }
        return false;
    }
    if (ev.type == UiEvent::Type::MouseDown && ev.button == UiMouseButton::Left) {
        if (open_) {
            // Click on an item → select and close.
            const int n = visibleCount();
            for (int i = 0; i < n; ++i) {
                if (itemRect(i).contains(ev.mousePx)) {
                    selectedIndex = i;
                    open_ = false;
                    hoveredItem_ = -1;
                    if (onSelect) {
                        onSelect(i);
                    }
                    ev.handled = true;
                    return true;
                }
            }
            // Click outside header and popup → close without selecting.
            if (!rect_.contains(ev.mousePx) && !popupRect().contains(ev.mousePx)) {
                open_ = false;
                hoveredItem_ = -1;
                return false;
            }
        }
        // Click on header → toggle popup.
        if (rect_.contains(ev.mousePx)) {
            open_ = !open_;
            hoveredItem_ = -1;
            ev.handled = true;
            return true;
        }
    }
    return false;
}

}  // namespace odai::ui
