#pragma once

#include "ui/font.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>
#include <string>

// A clickable button: 4 visual states and an onClick callback fired on a
// press-then-release inside the button.
namespace odai::ui {

class Button : public Widget {
public:
    enum class State { Normal, Hover, Pressed, Disabled };

    Button(const Font* font, std::string label, std::function<void()> onClick)
        : font_(font), label_(std::move(label)), onClick_(std::move(onClick)) {}

    UiColor colorNormal{0.12f, 0.18f, 0.24f, 0.90f};
    UiColor colorHover{0.18f, 0.26f, 0.34f, 0.95f};
    UiColor colorPressed{0.10f, 0.14f, 0.18f, 0.95f};
    UiColor colorDisabled{0.10f, 0.12f, 0.14f, 0.60f};
    UiColor borderColor{0.85f, 0.72f, 0.44f, 0.55f};
    UiColor labelColor{0.91f, 0.82f, 0.51f, 1.0f};
    float borderThicknessPx = 1.0f;

    void setEnabled(bool enabled);
    [[nodiscard]] bool enabled() const { return enabled_; }
    [[nodiscard]] State state() const { return state_; }
    void setLabel(std::string label) { label_ = std::move(label); }

    void draw(UiDrawList& drawList) const override;
    bool onEvent(UiEvent& event) override;

private:
    [[nodiscard]] UiColor backgroundForState() const;

    const Font* font_ = nullptr;
    std::string label_;
    std::function<void()> onClick_;
    State state_ = State::Normal;
    bool enabled_ = true;
    bool hovered_ = false;
    bool pressedInside_ = false;
};

}  // namespace odai::ui
