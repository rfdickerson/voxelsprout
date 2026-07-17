#pragma once

#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>
#include <string>

namespace odai::ui {

// Numeric field with up/down nub buttons. Holding a nub repeats the step after
// an initial delay. Call update(dt) each frame to drive the repeat timer.
class Spinner : public Widget {
public:
    explicit Spinner(const Font* font) : font_(font) {}

    double value = 0.0;
    double minValue = 0.0;
    double maxValue = 100.0;
    double step = 1.0;
    int decimalPlaces = 0;

    UiColor fieldBg{0.14f, 0.14f, 0.14f, 1.0f};
    UiColor fieldBorderColor{0.08f, 0.08f, 0.08f, 1.0f};
    UiColor nubBg{0.22f, 0.22f, 0.22f, 1.0f};
    UiColor nubHoverBg{0.30f, 0.30f, 0.30f, 1.0f};
    UiColor textColor{0.90f, 0.90f, 0.90f, 1.0f};
    UiColor chevronColor{0.75f, 0.75f, 0.75f, 1.0f};

    float nubWidthPx = 18.0f;
    float paddingX = 8.0f;
    // 2px matches the shared corner-radius token used by Button/Window/TextBox/
    // Panel/Toast/IconButton/Dropdown/Slider — was 3px, a stray one-off value.
    float cornerRadiusPx = 2.0f;

    // Delay before repeat kicks in, and the interval between repeats, in seconds.
    float repeatDelaySec = 0.4f;
    float repeatIntervalSec = 0.06f;

    std::function<void(double)> onChange;

    void update(float dt);

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent& ev) override;

private:
    const Font* font_ = nullptr;
    int heldDir_ = 0;  // -1, 0, +1
    bool hoveredUp_ = false;
    bool hoveredDown_ = false;
    float heldTimeSec_ = 0.0f;

    void step_by(int dir);
    [[nodiscard]] UiRect fieldRect() const;
    [[nodiscard]] UiRect upRect() const;
    [[nodiscard]] UiRect downRect() const;
    [[nodiscard]] std::string formattedValue() const;
};

}  // namespace odai::ui
