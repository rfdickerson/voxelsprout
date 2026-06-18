#include "ui/widgets/scroll_view.h"

#include "ui/ui_draw_list.h"

#include <algorithm>
#include <cmath>

namespace odai::ui {

void ScrollView::layoutChildren() const {
    float pen = rect_.minY - scrollOffsetY;
    bool first = true;
    for (const std::unique_ptr<Widget>& child : children_) {
        if (!child->visible) continue;
        const float h = child->rect().height();
        if (!first) pen += childGap;
        first = false;
        child->setRect(UiRect{rect_.minX, pen, rect_.maxX - scrollBarWidthPx, pen + h});
        pen += h;
    }
    m_contentHeight = (pen + scrollOffsetY) - rect_.minY;
}

float ScrollView::maxScroll() const {
    return std::max(0.0f, m_contentHeight - rect_.height());
}

void ScrollView::scrollTo(float y) {
    scrollOffsetY = std::clamp(y, 0.0f, std::max(0.0f, m_contentHeight - rect_.height()));
}

void ScrollView::scrollToBottom() {
    scrollTo(m_contentHeight - rect_.height());
}

void ScrollView::draw(UiDrawList& dl) const {
    if (!visible) return;

    layoutChildren();

    dl.pushClip(rect_);

    // Draw only children that intersect the clip rect.
    for (const std::unique_ptr<Widget>& child : children_) {
        if (!child->visible) continue;
        const UiRect childRect = child->rect();
        // Cull children entirely above or below the viewport.
        if (childRect.maxY < rect_.minY || childRect.minY > rect_.maxY) continue;
        const bool fade = child->opacity < 1.0f;
        if (fade) dl.pushOpacity(child->opacity);
        child->draw(dl);
        if (fade) dl.popOpacity();
    }

    dl.popClip();

    // Scroll bar.
    if (showScrollBar && m_contentHeight > rect_.height()) {
        const float trackX = rect_.maxX - scrollBarWidthPx;
        const UiRect track{trackX, rect_.minY, rect_.maxX, rect_.maxY};
        dl.addRectFilled(track, scrollBarBg);

        const float viewFrac = rect_.height() / m_contentHeight;
        const float thumbH = std::max(20.0f, rect_.height() * viewFrac);
        const float scrollFrac = (maxScroll() > 0.0f) ? scrollOffsetY / maxScroll() : 0.0f;
        const float thumbY = rect_.minY + scrollFrac * (rect_.height() - thumbH);
        const UiRect thumb{trackX, thumbY, rect_.maxX, thumbY + thumbH};
        dl.addRectFilled(thumb, scrollBarColor);
    }

    // Fade edges.
    if (fadeEdges && m_contentHeight > rect_.height()) {
        constexpr float kFadeH = 24.0f;
        if (scrollOffsetY > 1.0f) {
            // Top fade.
            for (int i = 0; i < 8; ++i) {
                const float t = static_cast<float>(i) / 8.0f;
                const float y0 = rect_.minY + t * kFadeH;
                const float y1 = rect_.minY + (t + 1.0f / 8.0f) * kFadeH;
                dl.addRectFilled(UiRect{rect_.minX, y0, rect_.maxX, y1},
                                 UiColor{0.0f, 0.0f, 0.0f, 0.35f * (1.0f - t)});
            }
        }
        if (scrollOffsetY < maxScroll() - 1.0f) {
            // Bottom fade.
            for (int i = 0; i < 8; ++i) {
                const float t = static_cast<float>(i) / 8.0f;
                const float y0 = rect_.maxY - kFadeH + t * kFadeH;
                const float y1 = rect_.maxY - kFadeH + (t + 1.0f / 8.0f) * kFadeH;
                dl.addRectFilled(UiRect{rect_.minX, y0, rect_.maxX, y1},
                                 UiColor{0.0f, 0.0f, 0.0f, 0.35f * t});
            }
        }
    }
}

bool ScrollView::onEvent(UiEvent& ev) {
    if (ev.type == UiEvent::Type::Scroll && rect_.contains(ev.mousePx)) {
        // Scroll-wheel: 40px per tick.
        scrollOffsetY = std::clamp(
            scrollOffsetY - ev.scroll * 40.0f,
            0.0f,
            std::max(0.0f, m_contentHeight - rect_.height()));
        ev.handled = true;
        return true;
    }
    return Widget::onEvent(ev);
}

}  // namespace odai::ui
