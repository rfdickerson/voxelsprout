#include "ui/rich_text.h"

#include "ui/icon_atlas.h"

#include <cctype>

namespace odai::ui {

namespace {

bool parseHexColor(std::string_view body, UiColor& outColor) {
    if (!body.empty() && body.front() == '#') {
        body.remove_prefix(1);
    }
    if (body.size() != 6) {
        return false;
    }
    std::uint32_t value = 0;
    for (char c : body) {
        std::uint32_t digit = 0;
        if (c >= '0' && c <= '9') {
            digit = static_cast<std::uint32_t>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            digit = static_cast<std::uint32_t>(c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            digit = static_cast<std::uint32_t>(c - 'A' + 10);
        } else {
            return false;
        }
        value = (value << 4) | digit;
    }
    outColor = UiColor::fromRgbHex(value);
    return true;
}

}  // namespace

std::vector<RichSpan> parseRichText(std::string_view markup, const UiColor& defaultColor) {
    std::vector<RichSpan> spans;
    std::string buffer;
    UiColor color = defaultColor;
    std::vector<UiColor> colorStack;
    std::string tooltip;
    std::vector<std::string> tooltipStack;
    int bold = 0;
    int italic = 0;

    // sRGB #6ab0f5 — a readable light-blue used for navigation hyperlinks.
    static const UiColor kLinkBlue = UiColor::fromRgbHex(0x6ab0f5);

    const auto flush = [&]() {
        if (!buffer.empty()) {
            // Auto-apply link blue to <tip=link:…> spans when no explicit <color> is active,
            // so navigation links are visually distinct without requiring markup color tags.
            const bool isLink = tooltip.rfind("link:", 0) == 0;
            const UiColor spanColor = (isLink && colorStack.empty()) ? kLinkBlue : color;
            spans.push_back(RichSpan{buffer, spanColor, bold > 0, italic > 0, tooltip, {}, 0.0f});
            buffer.clear();
        }
    };

    std::size_t i = 0;
    while (i < markup.size()) {
        const char c = markup[i];

        // Square-bracket inline icon: [icon=name]
        if (c == '[') {
            const std::size_t close = markup.find(']', i);
            if (close != std::string_view::npos) {
                const std::string_view tag = markup.substr(i + 1, close - i - 1);
                if (tag.rfind("icon=", 0) == 0) {
                    flush();
                    RichSpan iconSpan;
                    iconSpan.color = color;
                    iconSpan.bold = bold > 0;
                    iconSpan.italic = italic > 0;
                    iconSpan.tooltip = tooltip;
                    // [icon=name] or [icon=name SIZE] — SIZE is an integer pixel override.
                    std::string_view rest = tag.substr(5);
                    const std::size_t sp = rest.find(' ');
                    if (sp != std::string_view::npos) {
                        iconSpan.iconName = std::string(rest.substr(0, sp));
                        float sz = 0.0f;
                        for (char ch : rest.substr(sp + 1)) {
                            if (ch >= '0' && ch <= '9') sz = sz * 10.0f + static_cast<float>(ch - '0');
                        }
                        iconSpan.iconSizePx = sz;
                    } else {
                        iconSpan.iconName = std::string(rest);
                    }
                    spans.push_back(std::move(iconSpan));
                    i = close + 1;
                    continue;
                }
            }
            buffer.push_back(c);
            ++i;
            continue;
        }

        // HTML entity decoding: &amp; → &, &lt; → <, &gt; → >, &quot; → ", &apos; → '
        if (c == '&') {
            const std::size_t semi = markup.find(';', i + 1);
            if (semi != std::string_view::npos && (semi - i) <= 6) {
                const std::string_view entity = markup.substr(i + 1, semi - i - 1);
                char decoded = 0;
                if (entity == "amp")       decoded = '&';
                else if (entity == "lt")   decoded = '<';
                else if (entity == "gt")   decoded = '>';
                else if (entity == "quot") decoded = '"';
                else if (entity == "apos") decoded = '\'';
                if (decoded != 0) {
                    buffer.push_back(decoded);
                    i = semi + 1;
                    continue;
                }
            }
            buffer.push_back(c);
            ++i;
            continue;
        }

        if (c != '<') {
            buffer.push_back(c);
            ++i;
            continue;
        }
        const std::size_t close = markup.find('>', i);
        if (close == std::string_view::npos) {
            buffer.push_back(c);
            ++i;
            continue;
        }
        const std::string_view tag = markup.substr(i + 1, close - i - 1);
        if (tag == "b") {
            flush();
            ++bold;
        } else if (tag == "/b") {
            flush();
            if (bold > 0) {
                --bold;
            }
        } else if (tag == "i") {
            flush();
            ++italic;
        } else if (tag == "/i") {
            flush();
            if (italic > 0) {
                --italic;
            }
        } else if (tag == "br") {
            buffer.push_back('\n');
        } else if (tag.rfind("color=", 0) == 0) {
            UiColor parsed{};
            if (parseHexColor(tag.substr(6), parsed)) {
                flush();
                colorStack.push_back(color);
                color = parsed;
            }
        } else if (tag == "/color") {
            flush();
            if (!colorStack.empty()) {
                color = colorStack.back();
                colorStack.pop_back();
            } else {
                color = defaultColor;
            }
        } else if (tag.rfind("tip=", 0) == 0) {
            flush();
            tooltipStack.push_back(tooltip);
            tooltip = std::string(tag.substr(4));
        } else if (tag == "/tip") {
            flush();
            if (!tooltipStack.empty()) {
                tooltip = tooltipStack.back();
                tooltipStack.pop_back();
            } else {
                tooltip.clear();
            }
        }
        // Unknown tags are dropped.
        i = close + 1;
    }
    flush();
    return spans;
}

RichTextLayout layoutRichText(const std::vector<RichSpan>& spans, const FontSet& fonts, float wrapWidth,
                              UiTextAlign align) {
    RichTextLayout layout;
    const Font* regular = fonts.regular;
    const float lineHeight = (regular != nullptr) ? regular->lineHeightPx() : 0.0f;

    RichLine current;
    float penX = 0.0f;

    const auto finishLine = [&]() {
        current.width = penX;
        layout.lines.push_back(std::move(current));
        current = RichLine{};
        penX = 0.0f;
    };

    const auto appendRun = [&](std::string_view token, const RichSpan& span, float tokenWidth) {
        RichRun run;
        run.text.assign(token.begin(), token.end());
        run.color = span.color;
        run.bold = span.bold;
        run.italic = span.italic;
        run.x = penX;
        run.width = tokenWidth;
        run.tooltip = span.tooltip;
        current.runs.push_back(std::move(run));
        penX += tokenWidth;
    };

    for (const RichSpan& span : spans) {
        // Inline icon span: emit as a single fixed-width run.
        if (!span.iconName.empty()) {
            const float iconSize = (span.iconSizePx > 0.0f) ? span.iconSizePx : lineHeight;
            if (wrapWidth > 0.0f && penX > 0.0f && (penX + iconSize) > wrapWidth) {
                finishLine();
            }
            RichRun run;
            run.iconName = span.iconName;
            run.iconSizePx = iconSize;
            run.color = span.color;
            run.tooltip = span.tooltip;
            run.x = penX;
            run.width = iconSize;
            current.runs.push_back(std::move(run));
            penX += iconSize;
            continue;
        }

        std::size_t i = 0;
        while (i < span.text.size()) {
            const char c = span.text[i];
            if (c == '\n') {
                finishLine();
                ++i;
                continue;
            }
            const bool isSpace = (static_cast<unsigned char>(c) == ' ');
            std::size_t end = i;
            while (end < span.text.size() && span.text[end] != '\n' &&
                   ((static_cast<unsigned char>(span.text[end]) == ' ') == isSpace)) {
                ++end;
            }
            const std::string_view token(span.text.data() + i, end - i);
            const Font* spanFont = fonts.select(span.bold, span.italic);
            const float tokenWidth = (spanFont != nullptr) ? spanFont->measureText(token) : 0.0f;

            if (isSpace) {
                // A space that would overflow the wrap width is swallowed at the
                // line break rather than padding the line past the limit.
                if (wrapWidth > 0.0f && penX > 0.0f && (penX + tokenWidth) > wrapWidth) {
                    i = end;
                    continue;
                }
            } else if (wrapWidth > 0.0f && penX > 0.0f && (penX + tokenWidth) > wrapWidth) {
                finishLine();
            }
            // Skip leading space tokens at the start of a wrapped line.
            if (!(isSpace && penX == 0.0f && !layout.lines.empty())) {
                appendRun(token, span, tokenWidth);
            }
            i = end;
        }
    }
    finishLine();

    // Drop a trailing empty line produced when the text ends exactly on a break.
    if (layout.lines.size() > 1 && layout.lines.back().runs.empty()) {
        layout.lines.pop_back();
    }

    float maxWidth = 0.0f;
    for (const RichLine& line : layout.lines) {
        maxWidth = line.width > maxWidth ? line.width : maxWidth;
    }
    const float alignBox = wrapWidth > 0.0f ? wrapWidth : maxWidth;

    for (std::size_t lineIndex = 0; lineIndex < layout.lines.size(); ++lineIndex) {
        RichLine& line = layout.lines[lineIndex];
        line.y = static_cast<float>(lineIndex) * lineHeight;
        float offset = 0.0f;
        if (align == UiTextAlign::Center) {
            offset = (alignBox - line.width) * 0.5f;
        } else if (align == UiTextAlign::Right) {
            offset = alignBox - line.width;
        }
        if (offset != 0.0f) {
            for (RichRun& run : line.runs) {
                run.x += offset;
            }
        }
    }

    layout.width = maxWidth;
    layout.height = static_cast<float>(layout.lines.size()) * lineHeight;
    return layout;
}

RichTextLayout layoutRichText(std::string_view markup, const UiColor& defaultColor, const FontSet& fonts,
                              float wrapWidth, UiTextAlign align) {
    return layoutRichText(parseRichText(markup, defaultColor), fonts, wrapWidth, align);
}

void drawRichText(UiDrawList& drawList, const RichTextLayout& layout, const FontSet& fonts,
                  const UiVec2& posPx) {
    const Font* regular = fonts.regular;
    const float lineHeight = (regular != nullptr) ? regular->lineHeightPx() : 0.0f;
    const float refAscent = (regular != nullptr) ? regular->ascentPx() : 0.0f;
    for (const RichLine& line : layout.lines) {
        for (const RichRun& run : line.runs) {
            // Inline icon run. Vertically centered on the text line.
            if (!run.iconName.empty()) {
                UiIconEntry entry;
                if (UiIconRegistry::global().resolve(run.iconName, entry)) {
                    const float iconSz = (run.iconSizePx > 0.0f) ? run.iconSizePx : lineHeight;
                    const float x0 = posPx.x + run.x;
                    const float y0 = posPx.y + line.y + (lineHeight - iconSz) * 0.5f;
                    const UiRect dst{x0, y0, x0 + iconSz, y0 + iconSz};
                    drawList.addImage(dst, entry.textureId, run.color, entry.uv);
                }
                continue;
            }

            const Font* runFont = fonts.select(run.bold, run.italic);
            if (runFont == nullptr) {
                continue;
            }
            // Offset the run's top so every run on the line shares one baseline,
            // even if the variant face has a slightly different ascent. addText
            // adds the font's own ascent back, landing all runs on refAscent.
            const float topY = posPx.y + line.y + (refAscent - runFont->ascentPx());
            drawList.addText(*runFont, run.text, UiVec2{posPx.x + run.x, topY}, run.color);
        }
    }
}

std::vector<RichTextLink> collectRichTextLinks(const RichTextLayout& layout, const FontSet& fonts,
                                               const UiVec2& posPx) {
    std::vector<RichTextLink> links;
    const Font* regular = fonts.regular;
    const float lineHeight = (regular != nullptr) ? regular->lineHeightPx() : 0.0f;
    for (const RichLine& line : layout.lines) {
        for (const RichRun& run : line.runs) {
            if (run.tooltip.empty() || run.width <= 0.0f) {
                continue;
            }
            const float x0 = posPx.x + run.x;
            const float y0 = posPx.y + line.y;
            links.push_back(RichTextLink{UiRect{x0, y0, x0 + run.width, y0 + lineHeight}, run.tooltip});
        }
    }
    return links;
}

// Single-font overloads: treat the one face as the whole family (bold/italic
// runs fall back to it). Preserves the original API for callers and tests.
RichTextLayout layoutRichText(const std::vector<RichSpan>& spans, const Font& font, float wrapWidth,
                              UiTextAlign align) {
    return layoutRichText(spans, FontSet{&font, &font, &font, &font}, wrapWidth, align);
}

RichTextLayout layoutRichText(std::string_view markup, const UiColor& defaultColor, const Font& font,
                              float wrapWidth, UiTextAlign align) {
    return layoutRichText(parseRichText(markup, defaultColor), font, wrapWidth, align);
}

void drawRichText(UiDrawList& drawList, const RichTextLayout& layout, const Font& font,
                  const UiVec2& posPx) {
    drawRichText(drawList, layout, FontSet{&font, &font, &font, &font}, posPx);
}

}  // namespace odai::ui
