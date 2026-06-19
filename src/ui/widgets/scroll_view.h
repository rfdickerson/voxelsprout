#pragma once

#include "ui/ui_types.h"
#include "ui/widget.h"

namespace odai::ui {

// Vertical (and optionally horizontal) scroll container.
// Children are arranged in a VerticalStack inside; the view clips to its rect.
// Scroll bar: a thin translucent rect on the right edge when content overflows.
// Virtualization: draw() skips children whose rect doesn't intersect the clip.
class ScrollView : public Widget {
public:
    float scrollOffsetY = 0.0f;     // Current scroll offset in pixels (>= 0).
    float scrollOffsetX = 0.0f;
    float childGap = 4.0f;          // Gap between stacked children.
    float scrollBarWidthPx = 6.0f;
    UiColor scrollBarColor{0.6f, 0.6f, 0.6f, 0.5f};
    UiColor scrollBarBg{0.0f, 0.0f, 0.0f, 0.2f};
    bool showScrollBar = true;
    bool fadeEdges = true;          // Alpha gradient at top/bottom when overflowing.

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent& ev) override;

    // Total content height (updated each draw).
    float contentHeight() const { return m_contentHeight; }

    // Scroll to a pixel offset (clamped).
    void scrollTo(float y);
    void scrollToTop()    { scrollTo(0.0f); }
    void scrollToBottom();

private:
    mutable float m_contentHeight = 0.0f;

    // Scrollbar thumb drag state.
    bool  m_thumbDragging      = false;
    float m_thumbDragStartMouseY  = 0.0f;
    float m_thumbDragStartOffset  = 0.0f;
    mutable bool m_thumbHovered   = false;

    void layoutChildren() const;
    float maxScroll() const;
    [[nodiscard]] UiRect thumbRect() const;
};

}  // namespace odai::ui
