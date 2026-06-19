#include "ui/widgets/rich_text_view.h"

#include <algorithm>
#include <cmath>
#include <string_view>

namespace odai::ui {

float RichTextView::maxScroll() const {
    const float contentH = cache_.naturalHeight() + padding.y * 2.0f;
    return std::max(0.0f, contentH - rect_.height());
}

void RichTextView::draw(UiDrawList& drawList) const {
    if (!cache_.hasFont()) return;
    syncCache();
    cache_.ensure(rect_);

    const float ms = maxScroll();
    const bool scrollable = ms > 0.5f;

    // Viewport rect — shrink right edge to leave room for scrollbar when needed.
    const float barReserve = (showScrollBar && scrollable) ? scrollBarWidthPx + 2.0f : 0.0f;
    const UiRect viewport{rect_.minX, rect_.minY, rect_.maxX - barReserve, rect_.maxY};

    if (scrollable) {
        cache_.emitScrolled(drawList, viewport, scrollOffsetY);
        if (hovered_) {
            cache_.drawHighlightedTooltipScrolled(drawList, viewport, scrollOffsetY,
                                                  hoveredTooltip_, UiColor{1.0f, 0.88f, 0.52f, 1.0f});
        }
    } else {
        cache_.emit(drawList, viewport);
        if (hovered_) {
            cache_.drawHighlightedTooltip(drawList, viewport, hoveredTooltip_,
                                          UiColor{1.0f, 0.88f, 0.52f, 1.0f});
        }
    }

    // Scrollbar.
    if (showScrollBar && scrollable) {
        const float trackX = rect_.maxX - scrollBarWidthPx;
        const UiRect track{trackX, rect_.minY, rect_.maxX, rect_.maxY};
        drawList.addRoundRectFilled(track, scrollBarTrackColor, scrollBarWidthPx * 0.5f);

        const float contentH = cache_.naturalHeight() + padding.y * 2.0f;
        const float viewFrac = rect_.height() / contentH;
        const float thumbH   = std::max(20.0f, rect_.height() * viewFrac);
        const float scrollFrac = ms > 0.0f ? scrollOffsetY / ms : 0.0f;
        const float thumbY   = rect_.minY + scrollFrac * (rect_.height() - thumbH);
        const UiRect thumb{trackX + 1.0f, thumbY, rect_.maxX - 1.0f, thumbY + thumbH};
        drawList.addRoundRectFilled(thumb, scrollBarThumbColor, scrollBarWidthPx * 0.5f - 1.0f);
    }

    // Fade top/bottom edges when content is clipped.
    if (scrollable) {
        constexpr float kFadeH = 20.0f;
        constexpr int kSteps = 6;
        const float barW = rect_.maxX - rect_.minX;
        if (scrollOffsetY > 0.5f) {
            for (int i = 0; i < kSteps; ++i) {
                const float t  = static_cast<float>(i) / static_cast<float>(kSteps);
                const float y0 = rect_.minY + t * kFadeH;
                const float y1 = rect_.minY + (t + 1.0f / static_cast<float>(kSteps)) * kFadeH;
                drawList.addRectFilled(UiRect{rect_.minX, y0, rect_.minX + barW, y1},
                                       UiColor{0.0f, 0.0f, 0.0f, 0.30f * (1.0f - t)});
            }
        }
        if (scrollOffsetY < ms - 0.5f) {
            for (int i = 0; i < kSteps; ++i) {
                const float t  = static_cast<float>(i) / static_cast<float>(kSteps);
                const float y0 = rect_.maxY - kFadeH + t * kFadeH;
                const float y1 = rect_.maxY - kFadeH + (t + 1.0f / static_cast<float>(kSteps)) * kFadeH;
                drawList.addRectFilled(UiRect{rect_.minX, y0, rect_.minX + barW, y1},
                                       UiColor{0.0f, 0.0f, 0.0f, 0.30f * t});
            }
        }
    }
}

bool RichTextView::onEvent(UiEvent& event) {
    if (!visible) return false;

    if (event.type == UiEvent::Type::Scroll && rect_.contains(event.mousePx)) {
        const float ms = maxScroll();
        if (ms > 0.5f) {
            scrollOffsetY = std::clamp(scrollOffsetY - event.scroll * 40.0f, 0.0f, ms);
            event.handled = true;
            return true;
        }
    }

    // Click on a link: <tip=link:ID>...</tip> — fire the onLinkClick callback.
    if (event.type == UiEvent::Type::MouseDown &&
        event.button == UiMouseButton::Left &&
        rect_.contains(event.mousePx) && onLinkClick) {
        syncCache();
        cache_.ensure(rect_);
        const float barReserve = showScrollBar ? scrollBarWidthPx + 2.0f : 0.0f;
        const UiRect vp{rect_.minX, rect_.minY, rect_.maxX - barReserve, rect_.maxY};
        for (RichTextLink lnk : cache_.linksFor(vp)) {
            lnk.rect.minY -= scrollOffsetY;
            lnk.rect.maxY -= scrollOffsetY;
            static constexpr std::string_view kPrefix = "link:";
            if (lnk.rect.contains(event.mousePx) &&
                lnk.tooltip.size() > kPrefix.size() &&
                std::string_view{lnk.tooltip}.substr(0, kPrefix.size()) == kPrefix) {
                onLinkClick(lnk.tooltip.substr(kPrefix.size()));
                event.handled = true;
                return true;
            }
        }
    }

    if (event.type != UiEvent::Type::MouseMove) return false;

    syncCache();
    cache_.ensure(rect_);
    hovered_ = false;
    hoveredTooltip_.clear();

    // Translate tooltip link rects by the scroll offset before hit-testing.
    const float barReserve = showScrollBar ? scrollBarWidthPx + 2.0f : 0.0f;
    const UiRect viewport{rect_.minX, rect_.minY, rect_.maxX - barReserve, rect_.maxY};

    for (RichTextLink link : cache_.linksFor(viewport)) {
        link.rect.minY -= scrollOffsetY;
        link.rect.maxY -= scrollOffsetY;
        if (link.rect.contains(event.mousePx)) {
            hovered_ = true;
            hoveredTooltip_ = link.tooltip;
            hoveredAnchor_ = event.mousePx;
            hoveredRect_ = link.rect;
            break;
        }
    }
    return false;
}

}  // namespace odai::ui
