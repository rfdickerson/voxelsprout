#pragma once

#include "ui/animation.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <optional>

namespace odai::ui {

// Horizontal fill bar. value is clamped to [0, 1].
//
// Gradient fill: set foregroundEnd.a > 0 to draw the fill as a left→right
// horizontal gradient from `foreground` to `foregroundEnd`. Leave it at default
// {0,0,0,0} for a solid fill using `foreground`.
//
// Animated colors: call foregroundAnim.set(targetColor, durationSec) to smoothly
// transition the fill color, then drive it with update(dt) each frame.
// foregroundEndAnim drives the right-end color of the gradient independently.
// While either tween is in flight, its current() overrides the static field.
class ProgressBar : public Widget {
public:
    float value = 0.0f;           // 0 = empty, 1 = full.
    UiColor foreground{0.3f, 0.7f, 0.3f, 1.0f};
    // Right-end gradient color. When a > 0 the fill is drawn as a horizontal
    // gradient from `foreground` (left) to `foregroundEnd` (right at value=1).
    // At partial fills the right edge is lerped so the full-range spectrum is
    // always visible from 0 to the current fill position.
    UiColor foregroundEnd{0.0f, 0.0f, 0.0f, 0.0f};

    UiColor background{0.1f, 0.1f, 0.1f, 0.8f};
    UiColor borderColor{0.0f, 0.0f, 0.0f, 0.5f};
    float borderThicknessPx = 1.0f;
    // Corner radius in pixels (DPI-scaled by the caller). > 0 draws the track and
    // fill as anti-aliased SDF rounded rects; a large value yields a pill bar.
    float cornerRadiusPx = 0.0f;
    std::optional<UiNineSlice> frame;

    // Animated color tweens. Set a target color with .set(color, durationSec),
    // then call update(dt) each frame. While in flight, current() overrides the
    // corresponding static color field.
    ColorTween foregroundAnim;
    ColorTween foregroundEndAnim;

    // Step all active color tweens forward by dt seconds. Call once per frame.
    void update(float dt);

    void draw(UiDrawList& dl) const override;

private:
    // Returns the effective fill left/right colors, resolving active tweens.
    [[nodiscard]] UiColor effectiveForeground() const;
    [[nodiscard]] UiColor effectiveForegroundEnd() const;
};

}  // namespace odai::ui
