#pragma once

#include "ui/font.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>
#include <string>

// A single-line editable text input. The frame is drawn with vector (SDF)
// rounded-rect primitives, so it stays crisp at any DPI. It shows a placeholder
// when empty, the typed value otherwise, and a caret while focused. Focus is
// gained by clicking inside and lost by clicking elsewhere. Text events carry
// Unicode codepoints; codepoint 8 (backspace) deletes the last character and
// 13/10 (enter) fires onSubmit.
namespace odai::ui {

class TextBox : public Widget {
public:
    explicit TextBox(const Font* font, std::string placeholder = {})
        : font_(font), placeholder_(std::move(placeholder)) {}

    UiColor background{0.10f, 0.16f, 0.22f, 1.0f};
    UiColor borderColor{0.40f, 0.55f, 0.62f, 1.0f};
    UiColor borderFocusedColor{0.91f, 0.66f, 0.30f, 1.0f};
    UiColor textColor{0.90f, 0.94f, 0.96f, 1.0f};
    UiColor placeholderColor{0.55f, 0.62f, 0.66f, 1.0f};
    UiColor caretColor{0.95f, 0.80f, 0.45f, 1.0f};
    float cornerRadiusPx = 8.0f;
    float borderThicknessPx = 1.5f;
    UiVec2 padding{12.0f, 0.0f};  // x: horizontal text inset; y unused (centred).
    float leftInset = 0.0f;       // Extra left space, e.g. to clear a search icon.

    std::function<void(const std::string&)> onChange;
    std::function<void()> onSubmit;

    [[nodiscard]] const std::string& value() const { return value_; }
    void setValue(std::string v) { value_ = std::move(v); }
    [[nodiscard]] bool focused() const { return focused_; }
    void setFocused(bool f) { focused_ = f; }

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent& e) override;

private:
    const Font* font_ = nullptr;
    std::string value_;
    std::string placeholder_;
    bool focused_ = false;
};

}  // namespace odai::ui
