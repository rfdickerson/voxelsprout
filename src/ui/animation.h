#pragma once

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

}  // namespace odai::ui
