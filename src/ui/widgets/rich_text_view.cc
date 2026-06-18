#include "ui/widgets/rich_text_view.h"

namespace odai::ui {

void RichTextView::draw(UiDrawList& drawList) const {
    if (!cache_.hasFont()) {
        return;
    }
    syncCache();
    cache_.emit(drawList, rect_);
    // Hover: re-draw the hovered tooltip runs in a bright highlight color so the
    // text brightens rather than the background.
    if (hovered_) {
        cache_.drawHighlightedTooltip(drawList, rect_, hoveredTooltip_,
                                      UiColor{1.0f, 0.88f, 0.52f, 1.0f});
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
