#include "ui/widgets/text_box.h"

#include "ui/font.h"
#include "ui/ui_draw_list.h"

#include <cstdint>

namespace odai::ui {

namespace {

void appendUtf8(std::string& s, std::uint32_t cp) {
    if (cp < 0x80u) {
        s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800u) {
        s.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp < 0x10000u) {
        s.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        s.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        s.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        s.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        s.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

// Remove the last UTF-8 codepoint: drop trailing continuation bytes (10xxxxxx),
// then the lead byte.
void popUtf8(std::string& s) {
    while (!s.empty() && (static_cast<unsigned char>(s.back()) & 0xC0u) == 0x80u) {
        s.pop_back();
    }
    if (!s.empty()) {
        s.pop_back();
    }
}

}  // namespace

void TextBox::draw(UiDrawList& dl) const {
    if (!visible) {
        return;
    }
    dl.addRoundRectFilled(rect_, background, cornerRadiusPx);
    dl.addRoundRect(rect_, focused_ ? borderFocusedColor : borderColor, cornerRadiusPx,
                    borderThicknessPx);

    if (font_ != nullptr) {
        const float textX = rect_.minX + padding.x + leftInset;
        const float textY = rect_.minY + (rect_.height() - font_->lineHeightPx()) * 0.5f;
        const UiRect inner{rect_.minX + padding.x * 0.5f, rect_.minY,
                           rect_.maxX - padding.x * 0.5f, rect_.maxY};
        dl.pushClip(inner);
        if (value_.empty() && !focused_) {
            dl.addText(*font_, placeholder_, UiVec2{textX, textY}, placeholderColor);
        } else {
            const float advance = dl.addText(*font_, value_, UiVec2{textX, textY}, textColor);
            if (focused_) {
                const float caretX = textX + advance + 1.0f;
                const float ch = font_->lineHeightPx();
                dl.addRectFilled(UiRect{caretX, textY + ch * 0.12f, caretX + 2.0f, textY + ch * 0.88f},
                                 caretColor);
            }
        }
        dl.popClip();
    }
    drawChildren(dl);
}

bool TextBox::onEvent(UiEvent& e) {
    if (!visible) {
        return false;
    }
    switch (e.type) {
        case UiEvent::Type::MouseDown: {
            const bool inside = rect_.contains(e.mousePx);
            focused_ = inside;
            if (inside) {
                e.handled = true;
                return true;
            }
            return false;
        }
        case UiEvent::Type::Text: {
            if (!focused_) {
                return false;
            }
            const std::uint32_t cp = e.codepoint;
            if (cp == 8u) {
                popUtf8(value_);
            } else if (cp == 13u || cp == 10u) {
                if (onSubmit) {
                    onSubmit();
                }
                e.handled = true;
                return true;
            } else if (cp >= 32u) {
                appendUtf8(value_, cp);
            } else {
                return false;
            }
            if (onChange) {
                onChange(value_);
            }
            e.handled = true;
            return true;
        }
        default:
            return false;
    }
}

}  // namespace odai::ui
