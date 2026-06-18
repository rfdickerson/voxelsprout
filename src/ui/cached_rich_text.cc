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
