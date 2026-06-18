#pragma once

#include "ui/font.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"

#include <string>
#include <string_view>
#include <vector>

// Rich-text layout: parses a small markup dialect into styled spans, lays them
// out into wrapped/aligned lines using a Font, and emits glyph quads to a draw
// list. Markup: <b>..</b>, <i>..</i>, <color=#RRGGBB>..</color>, <br>.
// Pure CPU; no Vulkan.
namespace odai::ui {

struct RichSpan {
    std::string text;       // May contain '\n' for explicit line breaks.
    UiColor color{};
    bool bold = false;
    bool italic = false;
};

// Parse markup into contiguous styled spans. Unknown tags are ignored.
std::vector<RichSpan> parseRichText(std::string_view markup, const UiColor& defaultColor);

struct RichRun {
    std::string text;
    UiColor color{};
    bool bold = false;
    bool italic = false;
    float x = 0.0f;  // Pen x within the layout box (alignment already applied).
};

struct RichLine {
    std::vector<RichRun> runs;
    float width = 0.0f;
    float y = 0.0f;  // Top of the line within the layout box.
};

struct RichTextLayout {
    std::vector<RichLine> lines;
    float width = 0.0f;
    float height = 0.0f;
};

// Lay out spans. wrapWidth <= 0 disables wrapping (only explicit '\n' breaks).
RichTextLayout layoutRichText(const std::vector<RichSpan>& spans, const Font& font,
                              float wrapWidth, UiTextAlign align = UiTextAlign::Left);

// Convenience: parse + layout in one call.
RichTextLayout layoutRichText(std::string_view markup, const UiColor& defaultColor, const Font& font,
                              float wrapWidth, UiTextAlign align = UiTextAlign::Left);

// Emit the laid-out glyphs to the draw list with the box origin at posPx.
void drawRichText(UiDrawList& drawList, const RichTextLayout& layout, const Font& font,
                  const UiVec2& posPx);

}  // namespace odai::ui
