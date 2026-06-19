#pragma once

#include "ui/font.h"
#include "ui/rich_text.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"

#include <cstdint>
#include <string>
#include <vector>

// Caches the layout and generated geometry of a rich-text block so the expensive
// work (parse + per-glyph measure + word-wrap + quad generation) runs only when
// the content or the box SIZE changes. A pure move (same size, new position) reuses
// the cache and only re-translates. Both Label and RichTextView own one of these.
namespace odai::ui {

class CachedRichText {
public:
    CachedRichText() = default;
    CachedRichText(const FontSet& fonts, std::string markup)
        : fonts_(fonts), markup_(std::move(markup)) {}

    // Setters are guarded: they only invalidate the cache when the value actually
    // changes, so a widget can re-sync its public fields every frame for free.
    void setFonts(const FontSet& fonts) { fonts_ = fonts; dirty_ = true; }
    void setMarkup(std::string markup) {
        if (markup != markup_) {
            markup_ = std::move(markup);
            dirty_ = true;
        }
    }
    void setColor(const UiColor& color) {
        if (color.r != color_.r || color.g != color_.g || color.b != color_.b || color.a != color_.a) {
            color_ = color;
            dirty_ = true;
        }
    }
    void setAlign(UiTextAlign align) {
        if (align != align_) {
            align_ = align;
            dirty_ = true;
        }
    }
    void setWrap(bool wrap) {
        if (wrap != wrap_) {
            wrap_ = wrap;
            dirty_ = true;
        }
    }
    void setPadding(const UiVec2& padding) {
        if (padding.x != padding_.x || padding.y != padding_.y) {
            padding_ = padding;
            dirty_ = true;
        }
    }

    [[nodiscard]] const std::string& markup() const { return markup_; }
    [[nodiscard]] bool hasFont() const { return fonts_.regular != nullptr; }

    // Rebuild the cache for `rect` if content or size changed; cheap no-op otherwise.
    void ensure(const UiRect& rect);

    // Emit the cached geometry into the draw list, clipped to `rect`.
    void emit(UiDrawList& drawList, const UiRect& rect);

    [[nodiscard]] const RichTextLayout& layout() const { return layout_; }
    // Tooltip-run rects translated into screen space for `rect`.
    [[nodiscard]] std::vector<RichTextLink> linksFor(const UiRect& rect) const;

    // Like emit(), but shifts the content up by scrollOffsetY so the viewport
    // shows a scrolled window into the full laid-out text.
    void emitScrolled(UiDrawList& drawList, const UiRect& viewportRect, float scrollOffsetY);

    // Natural height of the laid-out content in pixels (valid after ensure()).
    // Does NOT include padding — add padding.y*2 for the full preferred height.
    [[nodiscard]] float naturalHeight() const { return layout_.height; }

    // Re-draw only the runs whose tooltip matches `tooltip`, using `highlightColor`
    // instead of their stored color. Called after emit() to tint hovered link text.
    void drawHighlightedTooltip(UiDrawList& dl, const UiRect& rect,
                                std::string_view tooltip, const UiColor& highlightColor) const;
    void drawHighlightedTooltipScrolled(UiDrawList& dl, const UiRect& viewportRect,
                                        float scrollOffsetY, std::string_view tooltip,
                                        const UiColor& highlightColor) const;

    // Count of (re)builds; used by tests to assert caching behaviour.
    [[nodiscard]] std::uint32_t rebuildCount() const { return rebuildCount_; }

private:
    [[nodiscard]] UiVec2 contentOrigin(const UiRect& rect) const {
        return {rect.minX + padding_.x, rect.minY + padding_.y};
    }

    FontSet fonts_{};
    std::string markup_;
    UiColor color_{0.85f, 0.90f, 0.93f, 1.0f};
    UiTextAlign align_ = UiTextAlign::Left;
    bool wrap_ = true;
    UiVec2 padding_{0.0f, 0.0f};

    // Cache (all in local coords, origin 0,0); rebuilt lazily by ensure().
    RichTextLayout layout_{};
    std::vector<RichTextLink> localLinks_{};
    UiGeometryBlock block_{};
    float builtForW_ = -1.0f;
    float builtForH_ = -1.0f;
    bool dirty_ = true;
    std::uint32_t rebuildCount_ = 0;
};

}  // namespace odai::ui
