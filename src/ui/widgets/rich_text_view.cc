#include "ui/widgets/rich_text_view.h"

namespace odai::ui {

void RichTextView::draw(UiDrawList& drawList) const {
    if (!cache_.hasFont()) {
        return;
    }
    syncCache();
    cache_.emit(drawList, rect_);
    // Underline the hovered link as an affordance that it is interactive.
    if (hovered_) {
        const float underlineY = hoveredRect_.maxY - 2.0f;
        drawList.pushClip(rect_);
        drawList.addRectFilled(
            UiRect{hoveredRect_.minX, underlineY, hoveredRect_.maxX, underlineY + 1.0f},
            UiColor{1.0f, 0.6f, 0.2f, 0.85f});
        drawList.popClip();
    }
}

bool RichTextView::onEvent(UiEvent& event) {
    if (!visible || event.type != UiEvent::Type::MouseMove) {
        return false;
    }
    syncCache();
    cache_.ensure(rect_);
    hovered_ = false;
    hoveredTooltip_.clear();
    for (const RichTextLink& link : cache_.linksFor(rect_)) {
        if (link.rect.contains(event.mousePx)) {
            hovered_ = true;
            hoveredTooltip_ = link.tooltip;
            hoveredAnchor_ = event.mousePx;
            hoveredRect_ = link.rect;
            break;
        }
    }
    return false;  // Hover is non-consuming; let other widgets see the move too.
}

}  // namespace odai::ui
