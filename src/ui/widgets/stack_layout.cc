#include "ui/widgets/stack_layout.h"

#include "ui/ui_draw_list.h"

namespace odai::ui {

void StackLayout::layoutChildren(bool isHorizontal) const {
    float pen = isHorizontal ? rect_.minX : rect_.minY;
    const float crossMin = isHorizontal ? rect_.minY : rect_.minX;
    const float crossSize = isHorizontal ? rect_.height() : rect_.width();

    bool first = true;
    for (const std::unique_ptr<Widget>& child : children_) {
        if (!child->visible) continue;
        const UiRect& cr = child->rect();
        const float childMain = isHorizontal ? cr.width() : cr.height();
        const float childCross = isHorizontal ? cr.height() : cr.width();

        if (!first) pen += gap;
        first = false;

        float crossOffset = crossMin;
        if (crossAlign == Align::Center) {
            crossOffset = crossMin + (crossSize - childCross) * 0.5f;
        } else if (crossAlign == Align::End) {
            crossOffset = crossMin + crossSize - childCross;
        }

        UiRect newRect;
        if (isHorizontal) {
            newRect = UiRect{pen, crossOffset, pen + childMain, crossOffset + childCross};
        } else {
            newRect = UiRect{crossOffset, pen, crossOffset + childCross, pen + childMain};
        }
        child->repositionAndResize(newRect);
        pen += childMain;
    }
}

void HorizontalStack::draw(UiDrawList& dl) const {
    if (!visible) return;
    layoutChildren(true);
    drawChildren(dl);
}

void VerticalStack::draw(UiDrawList& dl) const {
    if (!visible) return;
    layoutChildren(false);
    drawChildren(dl);
}

}  // namespace odai::ui
