#pragma once

#include "ui/cached_rich_text.h"
#include "ui/font.h"
#include "ui/rich_text.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <string>

// A multi-line rich-text widget that also tracks hoverable <tip=...> runs. On
// mouse-over of a tooltip-bearing run it records the tooltip text + cursor anchor
// (queried by the app to draw a tooltip overlay) and underlines the hovered run.
// Layout/geometry are cached via CachedRichText.
//
// When content is taller than the widget rect, the view becomes scrollable:
// mouse-wheel scrolls the text, and a thin scrollbar is drawn on the right edge.
namespace odai::ui {

class RichTextView : public Widget {
public:
    RichTextView(const FontSet& fonts, std::string markup) : cache_(fonts, std::move(markup)) {}

    UiColor color{0.85f, 0.90f, 0.93f, 1.0f};
    UiTextAlign align = UiTextAlign::Left;
    bool wrap = true;
    UiVec2 padding{0.0f, 0.0f};

    // Scrollbar appearance.
    float   scrollBarWidthPx = 6.0f;
    UiColor scrollBarTrackColor{0.0f, 0.0f, 0.0f, 0.20f};
    UiColor scrollBarThumbColor{0.70f, 0.58f, 0.30f, 0.55f};
    bool    showScrollBar = true;

    // Current vertical scroll offset in pixels (clamped to [0, maxScroll]).
    float scrollOffsetY = 0.0f;

    void setText(std::string markup) { cache_.setMarkup(std::move(markup)); }

    // Natural content height (padding excluded) — only accurate after first draw.
    [[nodiscard]] float contentHeight() const { return cache_.naturalHeight(); }

    void draw(UiDrawList& drawList) const override;
    bool onEvent(UiEvent& event) override;

    [[nodiscard]] bool hasTooltip() const { return !hoveredTooltip_.empty(); }
    [[nodiscard]] const std::string& tooltipText() const { return hoveredTooltip_; }
    [[nodiscard]] UiVec2 tooltipAnchor() const { return hoveredAnchor_; }

private:
    void syncCache() const {
        cache_.setColor(color);
        cache_.setAlign(align);
        cache_.setWrap(wrap);
        cache_.setPadding(padding);
    }

    float maxScroll() const;

    mutable CachedRichText cache_;

    std::string hoveredTooltip_;
    UiVec2 hoveredAnchor_{};
    UiRect hoveredRect_{};
    bool hovered_ = false;
};

}  // namespace odai::ui
