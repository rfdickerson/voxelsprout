#pragma once

#include "ui/ui_types.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

// Lightweight UI animation utilities: easing curves, a normalized [0,1] tween
// that advances toward a target at a fixed rate, from/to value tweens built on
// top of it, and a composable multi-step Sequence. Pure CPU, header-only.
namespace odai::ui {

enum class Easing {
    Linear,
    EaseIn,     // Quadratic ease-in (slow start).
    EaseOut,    // Quadratic ease-out (slow stop).
    EaseInOut,  // Smoothstep (slow start and stop).
    CubicIn,    // Cubic ease-in — sharper slow start than EaseIn.
    CubicOut,   // Cubic ease-out — sharper slow stop than EaseOut.
    BackOut,    // Overshoots past 1.0 then settles back — a soft "pop" landing
                // for pop-in cards/dialogs.
    Spring,     // Critically-damped spring response: snappier attack than
                // EaseOut with no bounce, for a crisp "settle" feel.
};

// Map a normalized progress t in [0,1] through an easing curve. BackOut is the
// one curve that legitimately returns values outside [0,1] mid-flight (the
// overshoot); callers driving a visual property (scale, position) should let
// that through rather than clamping the result.
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
        case Easing::CubicIn:
            return t * t * t;
        case Easing::CubicOut: {
            const float inv = 1.0f - t;
            return 1.0f - inv * inv * inv;
        }
        case Easing::BackOut: {
            // Standard "back" ease-out (Penner-style): overshoots ~10% past 1.0
            // around t=0.7 before settling.
            constexpr float kOvershoot = 1.70158f;
            const float inv = t - 1.0f;
            return 1.0f + inv * inv * ((kOvershoot + 1.0f) * inv + kOvershoot);
        }
        case Easing::Spring: {
            // Critically-damped spring settling to 1.0 (no oscillation, just a
            // snappier attack than a quadratic ease-out).
            constexpr float kOmega = 8.0f;  // higher = faster settle
            return 1.0f - std::exp(-kOmega * t) * (1.0f + kOmega * t);
        }
    }
    return t;
}

// A scalar that eases toward `target` (both in [0,1]) over `durationSec`. Drive it
// each frame with update(dt); read the eased value for an alpha/opacity multiplier.
//
// Retargeting: calling setTarget() mid-flight is always safe when `value` IS the
// displayed quantity (opacity, a thumb's [0,1] slide position, ...) — eased() is a
// monotonic function of value, so reversing direction traces the same curve back
// with no jump. It is NOT safe to build a from/to interpolant on top of a raw
// Tween (`current = from + (to - from) * tween.eased()`) and just swap `to` on
// retarget — the blend fraction no longer matches where the value visually is,
// causing a jump. Use ColorTween/Vec2Tween/RectTween below for that case; they
// recapture `from = current()` on every set() so retargeting is always seamless.
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

// Smoothly interpolates a UiVec2 (e.g. a widget's slide-in offset) from one value
// toward another over durationSec. Same seamless-retarget behavior as ColorTween:
// set() mid-flight recaptures the current position as the new start.
struct Vec2Tween {
    UiVec2 from{};
    UiVec2 to{};
    Tween  tween;

    void snap(const UiVec2& v) {
        from = v;
        to   = v;
        tween.value  = 1.0f;
        tween.target = 1.0f;
    }

    void set(const UiVec2& target, float durationSec = 0.25f,
             Easing easing = Easing::EaseOut) {
        from              = current();
        to                = target;
        tween.value       = 0.0f;
        tween.target      = 1.0f;
        tween.durationSec = durationSec;
        tween.easing      = easing;
    }

    void update(float dt) { tween.update(dt); }
    [[nodiscard]] bool idle() const { return tween.idle(); }

    [[nodiscard]] UiVec2 current() const {
        const float t = tween.eased();
        return UiVec2{from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t};
    }

    [[nodiscard]] static UiVec2 lerp(const UiVec2& a, const UiVec2& b, float t) {
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        return UiVec2{a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t};
    }
};

// Smoothly interpolates a UiRect (position + size together — a window slide-in,
// a card popping to its full size, an accordion opening) from one value toward
// another over durationSec. Same seamless-retarget behavior as ColorTween.
struct RectTween {
    UiRect from{};
    UiRect to{};
    Tween  tween;

    void snap(const UiRect& r) {
        from = r;
        to   = r;
        tween.value  = 1.0f;
        tween.target = 1.0f;
    }

    void set(const UiRect& target, float durationSec = 0.25f,
             Easing easing = Easing::EaseOut) {
        from              = current();
        to                = target;
        tween.value       = 0.0f;
        tween.target      = 1.0f;
        tween.durationSec = durationSec;
        tween.easing      = easing;
    }

    void update(float dt) { tween.update(dt); }
    [[nodiscard]] bool idle() const { return tween.idle(); }

    [[nodiscard]] UiRect current() const {
        const float t = tween.eased();
        return UiRect{
            from.minX + (to.minX - from.minX) * t,
            from.minY + (to.minY - from.minY) * t,
            from.maxX + (to.maxX - from.maxX) * t,
            from.maxY + (to.maxY - from.maxY) * t,
        };
    }

    [[nodiscard]] static UiRect lerp(const UiRect& a, const UiRect& b, float t) {
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        return UiRect{
            a.minX + (b.minX - a.minX) * t,
            a.minY + (b.minY - a.minY) * t,
            a.maxX + (b.maxX - a.maxX) * t,
            a.maxY + (b.maxY - a.maxY) * t,
        };
    }
};

// Composable animation timeline: chain timed steps, each driving an arbitrary
// callback with normalized progress, instead of hand-rolling a bespoke Tween
// field per animated effect. Build with append() (runs after the current end of
// the timeline) and join() (runs alongside the most recently added step), then
// call update(dt) once per frame.
//
//   Sequence seq;
//   seq.append(0.15f, [&](float t){ backdropAlpha = t; });
//   seq.join(0.25f, [&](float t){ dialogScale = applyEasing(Easing::BackOut, t); },
//            [&]{ dialogReady = true; });
//   // each frame:
//   seq.update(dt);
class Sequence {
public:
    using StepFn = std::function<void(float t)>;

    // Add a step starting at the current end of the timeline, running for
    // durationSec and calling fn(t) each frame with t in [0,1]. onComplete (if
    // set) fires exactly once, the frame elapsed time crosses the step's end.
    // durationSec <= 0 fires fn(1.0f) and onComplete once, instantly, when the
    // timeline reaches that point (a zero-length "do this, then continue").
    void append(float durationSec, StepFn fn, std::function<void()> onComplete = {}) {
        addStep(endTime_, durationSec, std::move(fn), std::move(onComplete));
    }

    // Add a step starting at the same time as the most recently added step (runs
    // in parallel with it) instead of after it.
    void join(float durationSec, StepFn fn, std::function<void()> onComplete = {}) {
        const float start = steps_.empty() ? 0.0f : steps_.back().start;
        addStep(start, durationSec, std::move(fn), std::move(onComplete));
    }

    // Insert a silent gap of durationSec before the next append()'d step.
    void delay(float durationSec) { endTime_ += durationSec; }

    // Advance the timeline by dt seconds, driving every step whose window
    // contains the new elapsed time and firing onComplete exactly once per step
    // as elapsed crosses its end.
    void update(float dt) {
        const float prevElapsed = elapsed_;
        elapsed_ += dt;
        for (Step& step : steps_) {
            const float end = step.start + step.duration;
            if (step.duration <= 0.0f) {
                if (prevElapsed < step.start && elapsed_ >= step.start) {
                    if (step.fn) step.fn(1.0f);
                    if (step.onComplete) step.onComplete();
                }
                continue;
            }
            if (elapsed_ <= step.start || prevElapsed >= end) {
                continue;
            }
            const float t = std::clamp((elapsed_ - step.start) / step.duration, 0.0f, 1.0f);
            if (step.fn) step.fn(t);
            if (elapsed_ >= end && prevElapsed < end && step.onComplete) {
                step.onComplete();
            }
        }
    }

    [[nodiscard]] bool done() const { return elapsed_ >= endTime_; }
    [[nodiscard]] float elapsed() const { return elapsed_; }

    // Rewind to the start without touching the authored steps, so the same
    // Sequence can be replayed.
    void reset() { elapsed_ = 0.0f; }

private:
    struct Step {
        float start = 0.0f;
        float duration = 0.0f;
        StepFn fn;
        std::function<void()> onComplete;
    };

    void addStep(float start, float duration, StepFn fn, std::function<void()> onComplete) {
        steps_.push_back(Step{start, duration, std::move(fn), std::move(onComplete)});
        endTime_ = std::max(endTime_, start + duration);
    }

    std::vector<Step> steps_;
    float elapsed_ = 0.0f;
    float endTime_ = 0.0f;
};

}  // namespace odai::ui
