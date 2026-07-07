#include "ui/widgets/progress_bar.h"

#include "ui/ui_draw_list.h"

#include <algorithm>

namespace odai::ui {

void ProgressBar::onTick(float dt) {
    foregroundAnim.update(dt);
    foregroundEndAnim.update(dt);
    tickChildren(dt);
}

UiColor ProgressBar::effectiveForeground() const {
    return foregroundAnim.idle() ? foreground : foregroundAnim.current();
}

UiColor ProgressBar::effectiveForegroundEnd() const {
    return foregroundEndAnim.idle() ? foregroundEnd : foregroundEndAnim.current();
}

void ProgressBar::draw(UiDrawList& dl) const {
    if (!visible) return;

    const float r = cornerRadiusPx;
    if (frame.has_value()) {
        dl.add9Slice(rect_, *frame, UiColor{1.0f, 1.0f, 1.0f, 1.0f});
    } else if (r > 0.0f) {
        dl.addRoundRectFilled(rect_, background, r);
    } else {
        dl.addRectFilled(rect_, background);
        if (borderThicknessPx > 0.0f) {
            dl.addRect(rect_, borderColor, borderThicknessPx);
        }
    }

    const float clamped = std::clamp(value, 0.0f, 1.0f);
    if (clamped > 0.0f) {
        const float fillWidth = rect_.width() * clamped;
        const UiRect fill{rect_.minX, rect_.minY, rect_.minX + fillWidth, rect_.maxY};

        const UiColor fgLeft  = effectiveForeground();
        const UiColor fgEnd   = effectiveForegroundEnd();
        const bool useGradient = fgEnd.a > 0.0f;

        if (r > 0.0f) {
            // Clip to the rounded track so a partial fill's square right edge never
            // pokes past the track's rounded corners.
            dl.pushClip(rect_);
            if (useGradient) {
                // Right-edge color: lerp between fgLeft and fgEnd by fill fraction so
                // the full color ramp sweeps from left to fill position (not full width).
                const UiColor fgRight = ColorTween::lerp(fgLeft, fgEnd, clamped);
                dl.addRoundRectFilledHGradient(fill, fgLeft, fgRight, r);
            } else {
                dl.addRoundRectFilled(fill, fgLeft, r);
            }
            dl.popClip();
        } else {
            if (useGradient) {
                const UiColor fgRight = ColorTween::lerp(fgLeft, fgEnd, clamped);
                dl.addRectFilledHGradient(fill, fgLeft, fgRight);
            } else {
                dl.addRectFilled(fill, fgLeft);
            }
        }
    }

    // Stroke the rounded track on top so the border sits above the fill.
    if (!frame.has_value() && r > 0.0f && borderThicknessPx > 0.0f && borderColor.a > 0.0f) {
        dl.addRoundRect(rect_, borderColor, r, borderThicknessPx);
    }
    drawChildren(dl);
}

}  // namespace odai::ui
