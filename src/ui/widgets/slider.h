#pragma once

#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>

namespace odai::ui {

// Interactive continuous-value control. Rounded track + filled portion + circular
// knob. Value is in [0, 1]. Drag the knob or click anywhere on the track.
class Slider : public Widget {
public:
    float value = 0.5f;  // Current value in [0, 1].

    UiColor trackColor{0.15f, 0.15f, 0.15f, 1.0f};
    UiColor fillColor{0.38f, 0.65f, 0.30f, 1.0f};
    UiColor knobColor{0.88f, 0.88f, 0.88f, 1.0f};
    UiColor knobHoverColor{1.0f, 1.0f, 1.0f, 1.0f};
    UiColor glowColor{0.42f, 0.72f, 0.35f, 0.35f};

    float knobRadiusPx = 8.0f;
    float cornerRadiusPx = 3.0f;

    std::function<void(float)> onChange;

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent& ev) override;

private:
    bool dragging_ = false;
    bool hovered_ = false;

    UiRect trackRect() const;
    float thumbX() const;
};

}  // namespace odai::ui
