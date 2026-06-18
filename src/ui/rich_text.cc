#include "ui/rich_text.h"

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
    int bold = 0;
    int italic = 0;

    const auto flush = [&]() {
        if (!buffer.empty()) {
            spans.push_back(RichSpan{buffer, color, bold > 0, italic > 0});
            buffer.clear();
        }
    };

    std::size_t i = 0;
    while (i < markup.size()) {
        const char c = markup[i];
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
        }
        // Unknown tags are dropped.
        i = close + 1;
    }
    flush();
    return spans;
}

RichTextLayout layoutRichText(const std::vector<RichSpan>& spans, const Font& font, float wrapWidth,
                              UiTextAlign align) {
    RichTextLayout layout;
    const float lineHeight = font.lineHeightPx();

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
        current.runs.push_back(std::move(run));
        penX += tokenWidth;
    };

    for (const RichSpan& span : spans) {
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
            const float tokenWidth = font.measureText(token);

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

RichTextLayout layoutRichText(std::string_view markup, const UiColor& defaultColor, const Font& font,
                              float wrapWidth, UiTextAlign align) {
    return layoutRichText(parseRichText(markup, defaultColor), font, wrapWidth, align);
}

void drawRichText(UiDrawList& drawList, const RichTextLayout& layout, const Font& font,
                  const UiVec2& posPx) {
    for (const RichLine& line : layout.lines) {
        for (const RichRun& run : line.runs) {
            drawList.addText(font, run.text, UiVec2{posPx.x + run.x, posPx.y + line.y}, run.color);
        }
    }
}

}  // namespace odai::ui
