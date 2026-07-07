#pragma once

#include "ui/animation.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>

namespace odai::ui {

// Pill-shaped on/off switch with a smoothly animated sliding thumb. The thumb
// animation advances automatically each frame via onTick (UiContext::tick(dt)
// drives this from the root) — no manual per-toggle update call needed.
class Toggle : public Widget {
public:
    bool checked = false;

    UiColor trackOn{0.28f, 0.65f, 0.30f, 1.0f};
    UiColor trackOff{0.22f, 0.22f, 0.22f, 1.0f};
    UiColor thumbColor{0.95f, 0.95f, 0.95f, 1.0f};

    std::function<void(bool)> onChange;

    void onTick(float dt) override {
        thumbTween_.update(dt);
        tickChildren(dt);
    }

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent& ev) override;

private:
    Tween thumbTween_{0.0f, 0.0f, 0.15f, Easing::EaseOut};
};

}  // namespace odai::ui
