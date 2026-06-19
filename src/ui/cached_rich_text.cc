#include "ui/cached_rich_text.h"

#include <utility>

namespace odai::ui {

void CachedRichText::ensure(const UiRect& rect) {
    const float w = rect.width();
    const float h = rect.height();
    if (!dirty_ && builtForW_ == w && builtForH_ == h) {
        return;  // Content and size unchanged — reuse the cache.
    }

    layout_ = RichTextLayout{};
    localLinks_.clear();
    block_ = UiGeometryBlock{};

    if (fonts_.regular != nullptr) {
        const float wrapWidth = wrap_ ? (w - (padding_.x * 2.0f)) : 0.0f;
        layout_ = layoutRichText(markup_, color_, fonts_, wrapWidth, align_);
        localLinks_ = collectRichTextLinks(layout_, fonts_, UiVec2{0.0f, 0.0f});

        // Generate the glyph geometry once, in local coordinates (origin 0,0), by
        // drawing into a scratch list. The scratch's clip rects are discarded on
        // replay, so the framebuffer size here is irrelevant.
        UiDrawList scratch;
        scratch.reset(UiVec2{65535.0f, 65535.0f});
        drawRichText(scratch, layout_, fonts_, UiVec2{0.0f, 0.0f});
        UiDrawData& data = scratch.data();
        block_.vertices = std::move(data.vertices);
        block_.indices = std::move(data.indices);
        block_.commands = std::move(data.commands);
    }

    builtForW_ = w;
    builtForH_ = h;
    dirty_ = false;
    ++rebuildCount_;
}

void CachedRichText::emit(UiDrawList& drawList, const UiRect& rect) {
    if (fonts_.regular == nullptr) {
        return;
    }
    ensure(rect);
    drawList.pushClip(rect);
    drawList.appendCached(block_, contentOrigin(rect));
    drawList.popClip();
}

void CachedRichText::emitScrolled(UiDrawList& drawList, const UiRect& viewportRect,
                                  float scrollOffsetY) {
    if (fonts_.regular == nullptr) return;
    ensure(viewportRect);
    const UiVec2 origin = contentOrigin(viewportRect);
    const UiVec2 translate{origin.x, origin.y - scrollOffsetY};
    // Visible local-Y window: invert the translate to find which local coords map
    // into [viewportRect.minY, viewportRect.maxY] after translation.
    const float yLocalMin = viewportRect.minY - translate.y;
    const float yLocalMax = viewportRect.maxY - translate.y;
    drawList.pushClip(viewportRect);
    drawList.appendCachedClipped(block_, translate, yLocalMin, yLocalMax);
    drawList.popClip();
}

void CachedRichText::drawHighlightedTooltipScrolled(UiDrawList& dl, const UiRect& viewportRect,
                                                     float scrollOffsetY,
                                                     std::string_view tooltip,
                                                     const UiColor& highlightColor) const {
    if (fonts_.regular == nullptr || layout_.lines.empty()) return;
    const UiVec2 origin = contentOrigin(viewportRect);
    const float refAscent = fonts_.regular->ascentPx();
    dl.pushClip(viewportRect);
    for (const RichLine& line : layout_.lines) {
        for (const RichRun& run : line.runs) {
            if (run.tooltip != tooltip || run.text.empty()) continue;
            const Font* runFont = fonts_.select(run.bold, run.italic);
            if (runFont == nullptr) continue;
            const float topY = origin.y - scrollOffsetY + line.y + (refAscent - runFont->ascentPx());
            dl.addText(*runFont, run.text, UiVec2{origin.x + run.x, topY}, highlightColor);
        }
    }
    dl.popClip();
}

void CachedRichText::drawHighlightedTooltip(UiDrawList& dl, const UiRect& rect,
                                             std::string_view tooltip,
                                             const UiColor& highlightColor) const {
    if (fonts_.regular == nullptr || layout_.lines.empty()) {
        return;
    }
    const UiVec2 origin = contentOrigin(rect);
    const float refAscent = fonts_.regular->ascentPx();
    dl.pushClip(rect);
    for (const RichLine& line : layout_.lines) {
        for (const RichRun& run : line.runs) {
            if (run.tooltip != tooltip || run.text.empty()) {
                continue;
            }
            const Font* runFont = fonts_.select(run.bold, run.italic);
            if (runFont == nullptr) {
                continue;
            }
            const float topY = origin.y + line.y + (refAscent - runFont->ascentPx());
            dl.addText(*runFont, run.text, UiVec2{origin.x + run.x, topY}, highlightColor);
        }
    }
    dl.popClip();
}

std::vector<RichTextLink> CachedRichText::linksFor(const UiRect& rect) const {
    const UiVec2 origin = contentOrigin(rect);
    std::vector<RichTextLink> out;
    out.reserve(localLinks_.size());
    for (const RichTextLink& link : localLinks_) {
        out.push_back(RichTextLink{
            UiRect{link.rect.minX + origin.x, link.rect.minY + origin.y,
                   link.rect.maxX + origin.x, link.rect.maxY + origin.y},
            link.tooltip});
    }
    return out;
}

}  // namespace odai::ui
