#pragma once

#include "ui/font.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"

#include <string>
#include <string_view>
#include <vector>

// Rich-text layout: parses a small markup dialect into styled spans, lays them
// out into wrapped/aligned lines using a Font, and emits glyph quads to a draw
// list.
//
// Angle-bracket markup: <b>..</b>, <i>..</i>, <num>..</num>, <color=#RRGGBB>..</color>,
//   <tip=text>..</tip>, <br>
// Square-bracket inline icons: [icon=food] or [icon=food 48] (optional pixel size override)
//
// Pure CPU; no Vulkan.
namespace odai::ui {

struct RichSpan {
    std::string text;       // May contain '\n' for explicit line breaks.
    UiColor color{};
    bool bold = false;
    bool italic = false;
    bool numeric = false;
    std::string tooltip;    // Non-empty => hoverable; shown on mouse-over.
    std::string iconName;   // Non-empty => this span is an inline icon, text is empty.
    float iconSizePx = 0.0f; // Override icon size (0 = use lineHeight).
};

// Parse markup into contiguous styled spans. Unknown tags are ignored.
// Markup: <b>, <i>, <num>, <color=#RRGGBB>, <tip=text...>, <br>. <num> selects
// FontSet::numeric when supplied. <tip=...> marks a
// hoverable span whose tooltip is the text up to the closing '>'.
std::vector<RichSpan> parseRichText(std::string_view markup, const UiColor& defaultColor);

struct RichRun {
    std::string text;
    UiColor color{};
    bool bold = false;
    bool italic = false;
    bool numeric = false;
    float x = 0.0f;      // Pen x within the layout box (alignment already applied).
    float width = 0.0f;  // Pixel width of this run (for hit-testing).
    std::string tooltip;
    std::string iconName;    // Non-empty => inline icon; text is empty, width = icon size.
    float iconSizePx = 0.0f; // Override icon size (0 = use lineHeight).
};

// A laid-out, screen-space rect of a tooltip-bearing run (for hover hit-testing).
struct RichTextLink {
    UiRect rect{};
    std::string tooltip;
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

// FontSet variants: each run is measured/drawn with the style-matched font, so
// <b>/<i> markup renders in the real bold/italic faces. Line metrics come from
// the regular face; baselines are aligned across mixed-style runs.
RichTextLayout layoutRichText(const std::vector<RichSpan>& spans, const FontSet& fonts,
                              float wrapWidth, UiTextAlign align = UiTextAlign::Left);

RichTextLayout layoutRichText(std::string_view markup, const UiColor& defaultColor, const FontSet& fonts,
                              float wrapWidth, UiTextAlign align = UiTextAlign::Left);

// Collect screen rects of all tooltip-bearing runs, given the layout's box origin.
std::vector<RichTextLink> collectRichTextLinks(const RichTextLayout& layout, const FontSet& fonts,
                                               const UiVec2& posPx);

void drawRichText(UiDrawList& drawList, const RichTextLayout& layout, const FontSet& fonts,
                  const UiVec2& posPx);

}  // namespace odai::ui
