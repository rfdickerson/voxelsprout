#include "ui/widgets/panel.h"

#include <algorithm>

namespace odai::ui {

void Panel::styleOrnate(float s, float alpha) {
    // Parchment/gilt palette mirroring assets/ui/themes/theme_parchment.json.
    bgTop    = UiColor{0.208f, 0.157f, 0.094f, alpha};  // warm lit parchment-brown
    bgBottom = UiColor{0.090f, 0.063f, 0.031f, alpha};  // darker base
    background = *bgBottom;                              // fallback if gradient unused
    borderColor       = UiColor{0.478f, 0.361f, 0.165f, 0.95f};  // panel.border #7A5C2A
    borderThicknessPx = 2.0f * s;
    innerBorderColor   = UiColor{0.784f, 0.588f, 0.227f, 0.85f};  // accent #C8963A
    innerBorderInsetPx = 3.5f * s;
    cornerAccentColor       = UiColor{0.941f, 0.753f, 0.251f, 0.95f};  // gold #F0C040
    cornerAccentPx          = 11.0f * s;
    cornerAccentThicknessPx = 2.0f * s;
    showShadow    = true;
    shadowColor   = UiColor{0.0f, 0.0f, 0.0f, 0.55f};
    shadowBlurPx  = 10.0f * s;
    shadowOffsetY = 5.0f * s;
    cornerRadiusPx = 0.0f;  // ornate frames are angular
}

void Panel::draw(UiDrawList& drawList) const {
    if (showShadow) {
        drawList.addDropShadow(rect_, shadowColor, shadowBlurPx, shadowOffsetX, shadowOffsetY);
    }
    const bool gradient = bgTop.has_value() && bgBottom.has_value();
    if (nineSlice.has_value()) {
        drawList.add9Slice(rect_, *nineSlice, background);
    } else if (gradient) {
        drawList.addRectFilledVGradient(rect_, *bgTop, *bgBottom);
        if (borderThicknessPx > 0.0f && borderColor.a > 0.0f) {
            drawList.addRect(rect_, borderColor, borderThicknessPx);
        }
    } else if (cornerRadiusPx > 0.0f) {
        drawList.addRoundRectFilled(rect_, background, cornerRadiusPx);
        if (borderThicknessPx > 0.0f && borderColor.a > 0.0f) {
            drawList.addRoundRect(rect_, borderColor, cornerRadiusPx, borderThicknessPx);
        }
    } else {
        drawList.addRectFilled(rect_, background);
        if (borderThicknessPx > 0.0f && borderColor.a > 0.0f) {
            drawList.addRect(rect_, borderColor, borderThicknessPx);
        }
    }

    // Inner gilt border line (the engraved second frame).
    if (innerBorderInsetPx > 0.0f && innerBorderColor.a > 0.0f) {
        const float in = innerBorderInsetPx;
        const UiRect inner{rect_.minX + in, rect_.minY + in, rect_.maxX - in, rect_.maxY - in};
        if (inner.valid()) {
            const float t = borderThicknessPx > 0.0f ? std::max(borderThicknessPx - 1.0f, 1.0f) : 1.0f;
            if (!gradient && cornerRadiusPx > 0.0f) {
                drawList.addRoundRect(inner, innerBorderColor,
                                      std::max(cornerRadiusPx - in, 0.0f), t);
            } else {
                drawList.addRect(inner, innerBorderColor, t);
            }
        }
    }

    // Corner accents: short L-shaped gold ticks just inside the frame.
    if (cornerAccentPx > 0.0f && cornerAccentColor.a > 0.0f) {
        const float len = cornerAccentPx;
        const float t   = cornerAccentThicknessPx;
        const float off = (innerBorderInsetPx > 0.0f ? innerBorderInsetPx : 0.0f) + 3.0f;
        const float x0 = rect_.minX + off, y0 = rect_.minY + off;
        const float x1 = rect_.maxX - off, y1 = rect_.maxY - off;
        const UiColor c = cornerAccentColor;
        // Top-left.
        drawList.addRectFilled(UiRect{x0, y0, x0 + len, y0 + t}, c);
        drawList.addRectFilled(UiRect{x0, y0, x0 + t, y0 + len}, c);
        // Top-right.
        drawList.addRectFilled(UiRect{x1 - len, y0, x1, y0 + t}, c);
        drawList.addRectFilled(UiRect{x1 - t, y0, x1, y0 + len}, c);
        // Bottom-left.
        drawList.addRectFilled(UiRect{x0, y1 - t, x0 + len, y1}, c);
        drawList.addRectFilled(UiRect{x0, y1 - len, x0 + t, y1}, c);
        // Bottom-right.
        drawList.addRectFilled(UiRect{x1 - len, y1 - t, x1, y1}, c);
        drawList.addRectFilled(UiRect{x1 - t, y1 - len, x1, y1}, c);
    }

    if (clipContents) {
        drawList.pushClip(rect_);
        drawChildren(drawList);
        drawList.popClip();
    } else {
        drawChildren(drawList);
    }
}

}  // namespace odai::ui
