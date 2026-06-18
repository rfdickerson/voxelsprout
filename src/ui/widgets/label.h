#pragma once

#include "ui/font.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <string>

// A text widget. Renders rich-text markup wrapped to the widget's width.
namespace odai::ui {

class Label : public Widget {
public:
    Label(const Font* font, std::string markup) : font_(font), markup_(std::move(markup)) {}

    UiColor color{0.85f, 0.90f, 0.93f, 1.0f};
    UiTextAlign align = UiTextAlign::Left;
    bool wrap = true;
    UiVec2 padding{0.0f, 0.0f};

    void setText(std::string markup) { markup_ = std::move(markup); }
    [[nodiscard]] const std::string& text() const { return markup_; }

    void draw(UiDrawList& drawList) const override;

private:
    const Font* font_ = nullptr;
    std::string markup_;
};

}  // namespace odai::ui
