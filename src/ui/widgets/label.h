#pragma once

#include "ui/cached_rich_text.h"
#include "ui/font.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <string>

// A text widget. Renders rich-text markup wrapped to the widget's width. Layout
// and geometry are cached and only regenerated when the text/style/size changes.
namespace odai::ui {

class Label : public Widget {
public:
    // Single face: <b>/<i> runs fall back to it.
    Label(const Font* font, std::string markup)
        : cache_(FontSet{font, font, font, font}, std::move(markup)) {}
    // Full family: <b>/<i> markup renders in the real bold/italic faces.
    Label(const FontSet& fonts, std::string markup) : cache_(fonts, std::move(markup)) {}

    UiColor color{0.85f, 0.90f, 0.93f, 1.0f};
    UiTextAlign align = UiTextAlign::Left;
    bool wrap = true;
    UiVec2 padding{0.0f, 0.0f};

    void setText(std::string markup) { cache_.setMarkup(std::move(markup)); }
    [[nodiscard]] const std::string& text() const { return cache_.markup(); }

    void draw(UiDrawList& drawList) const override;

private:
    mutable CachedRichText cache_;
};

}  // namespace odai::ui
