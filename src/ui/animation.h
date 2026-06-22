#pragma once

#include "ui/ui_types.h"

#include <algorithm>

// Lightweight UI animation utilities: easing curves and a normalized [0,1] tween
// that advances toward a target at a fixed rate. Pure CPU, header-only.
namespace odai::ui {

enum class Easing {
    Linear,
    EaseIn,     // Quadratic ease-in (slow start).
    EaseOut,    // Quadratic ease-out (slow stop).
    EaseInOut,  // Smoothstep (slow start and stop).
};

// Map a normalized progress t in [0,1] through an easing curve.
[[nodiscard]] inline float applyEasing(Easing easing, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    switch (easing) {
        case Easing::Linear:
            return t;
        case Easing::EaseIn:
            return t * t;
        case Easing::EaseOut:
            return 1.0f - (1.0f - t) * (1.0f - t);
        case Easing::EaseInOut:
            return t * t * (3.0f - 2.0f * t);  // smoothstep
    }
    return t;
}

// A scalar that eases toward `target` (both in [0,1]) over `durationSec`. Drive it
// each frame with update(dt); read the eased value for an alpha/opacity multiplier.
struct Tween {
    float value = 0.0f;       // Raw linear progress [0,1].
    float target = 0.0f;      // Goal in [0,1].
    float durationSec = 0.15f;
    Easing easing = Easing::EaseOut;

    void setTarget(float t) { target = std::clamp(t, 0.0f, 1.0f); }

    void update(float dt) {
        if (durationSec <= 0.0f) {
            value = target;
            return;
        }
        const float step = dt / durationSec;
        if (value < target) {
            value = std::min(target, value + step);
        } else if (value > target) {
            value = std::max(target, value - step);
        }
    }

    [[nodiscard]] float eased() const { return applyEasing(easing, value); }
    [[nodiscard]] bool idle() const { return value == target; }
};

// Smoothly interpolates a UiColor from one value toward another over durationSec.
// Drive with update(dt); read current() for the interpolated color each frame.
// Calling set() mid-flight captures the current color as the new start, so
// re-targeting during a transition is seamless with no discontinuity.
struct ColorTween {
    UiColor from{};
    UiColor to{};
    Tween   tween;

    // Snap to `color` immediately (no animation).
    void snap(const UiColor& color) {
        from = color;
        to   = color;
        tween.value  = 1.0f;
        tween.target = 1.0f;
    }

    // Begin a smooth transition to `target` starting from the current interpolated
    // color. Re-calling while in flight starts the new transition from wherever the
    // color currently is, avoiding jumps.
    void set(const UiColor& target, float durationSec = 0.25f,
             Easing easing = Easing::EaseInOut) {
        from              = current();
        to                = target;
        tween.value       = 0.0f;
        tween.target      = 1.0f;
        tween.durationSec = durationSec;
        tween.easing      = easing;
    }

    void update(float dt) { tween.update(dt); }
    [[nodiscard]] bool idle() const { return tween.idle(); }

    [[nodiscard]] UiColor current() const {
        const float t = tween.eased();
        return UiColor{
            from.r + (to.r - from.r) * t,
            from.g + (to.g - from.g) * t,
            from.b + (to.b - from.b) * t,
            from.a + (to.a - from.a) * t,
        };
    }

    // Lerp a color between two points given an external t in [0,1]. Does not
    // modify this tween's state; useful for value-driven color ramps (e.g. a
    // progress bar that goes green→red as it fills).
    [[nodiscard]] static UiColor lerp(const UiColor& a, const UiColor& b, float t) {
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        return UiColor{
            a.r + (b.r - a.r) * t,
            a.g + (b.g - a.g) * t,
            a.b + (b.b - a.b) * t,
            a.a + (b.a - a.a) * t,
        };
    }
};

}  // namespace odai::ui
