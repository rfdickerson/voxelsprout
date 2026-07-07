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

void Panel::styleCard(float s, float alpha) {
    // Clean-modern palette mirroring assets/ui/themes/theme_modern.json: a
    // translucent slate card so the colorful map glows through, separated by a
    // hairline edge and a single soft shadow. No gilt frame, inner line, or
    // corner ticks — the rounded fill carries the whole look.
    bgTop.reset();    // no gradient -> draw() takes the rounded-fill path
    bgBottom.reset();
    background        = UiColor{0.071f, 0.086f, 0.110f, alpha};  // slate #12161C
    borderColor       = UiColor{1.0f, 1.0f, 1.0f, 0.10f};        // hairline edge
    borderThicknessPx = 1.0f * s;
    cornerRadiusPx    = 2.0f * s;  // squared-off Victorian look; just softens the edge
    innerBorderColor   = UiColor{};  // clear the engraved inner frame
    innerBorderInsetPx = 0.0f;
    cornerAccentColor       = UiColor{};  // clear the gold corner ticks
    cornerAccentPx          = 0.0f;
    showShadow    = true;
    shadowColor   = UiColor{0.0f, 0.0f, 0.0f, 0.45f};
    shadowBlurPx  = 8.0f * s;   // tighter, less diffuse than before
    shadowOffsetX = 0.0f;
    shadowOffsetY = 3.0f * s;
}

void Panel::styleSoft(float s, UiColor tint) {
    // Soft "neumorphic" card: a light fill that sits on a same-family light
    // background, lifted by a symmetric pair of soft shadows — dark down-right,
    // light up-left — for a gently extruded, low-contrast look. Mirrors the
    // reference palette (fill #E4EBF1, dark shadow #161B1D@23%, light #FAFBFF).
    bgTop.reset();    // solid fill -> draw() takes the rounded-fill path
    bgBottom.reset();
    background        = tint;
    borderColor       = UiColor{};   // no hard border; the shadows define the edge
    borderThicknessPx = 0.0f;
    cornerRadiusPx    = 18.0f * s;   // generously rounded, the soft-UI signature
    innerBorderColor   = UiColor{};
    innerBorderInsetPx = 0.0f;
    cornerAccentColor  = UiColor{};
    cornerAccentPx     = 0.0f;
    showBevel     = false;
    showShadow    = true;
    shadowColor   = UiColor{0.086f, 0.106f, 0.114f, 0.23f};  // #161B1D @ 23%
    shadowBlurPx  = 9.0f * s;
    shadowOffsetX = 6.0f * s;        // cast down-right...
    shadowOffsetY = 6.0f * s;
    liftShadowColor = UiColor{0.980f, 0.984f, 1.0f, 0.9f};   // #FAFBFF highlight up-left
}

void Panel::styleGradientCard(float s, const UiColor& top, const UiColor& bottom,
                              float alpha) {
    // Full-bleed duotone tile: a vertical gradient clipped to rounded corners,
    // no border, one soft shadow for separation. The gradient stops carry the
    // whole look (e.g. sunset orange→pink, ocean blue→teal).
    bgTop    = UiColor{top.r,    top.g,    top.b,    top.a    * alpha};
    bgBottom = UiColor{bottom.r, bottom.g, bottom.b, bottom.a * alpha};
    background = *bgBottom;           // fallback if the gradient path is bypassed
    borderColor       = UiColor{};
    borderThicknessPx = 0.0f;
    cornerRadiusPx    = 14.0f * s;
    innerBorderColor   = UiColor{};
    innerBorderInsetPx = 0.0f;
    cornerAccentColor  = UiColor{};
    cornerAccentPx     = 0.0f;
    showBevel     = false;
    showShadow    = true;
    shadowColor   = UiColor{0.0f, 0.0f, 0.0f, 0.30f};
    shadowBlurPx  = 12.0f * s;
    shadowOffsetX = 0.0f;
    shadowOffsetY = 6.0f * s;
    liftShadowColor = UiColor{};
}

void Panel::styleWin95(float s, bool raised) {
    // Win95 "Redmond" palette: COLOR_BTNFACE silver fill, outer black border,
    // two-tone bevel (white highlight / #808080 shadow). No gradient, no radius.
    bgTop.reset();
    bgBottom.reset();
    background           = UiColor{0.753f, 0.753f, 0.753f, 1.0f};  // #C0C0C0
    borderColor          = UiColor{0.0f,   0.0f,   0.0f,   1.0f};  // #000000
    borderThicknessPx    = 1.0f * s;
    cornerRadiusPx       = 0.0f;
    innerBorderColor     = UiColor{};
    innerBorderInsetPx   = 0.0f;
    cornerAccentColor    = UiColor{};
    cornerAccentPx       = 0.0f;
    showBevel            = true;
    bevelHighlightColor  = UiColor{1.0f,   1.0f,   1.0f,   1.0f};  // #FFFFFF
    bevelShadowColor     = UiColor{0.502f, 0.502f, 0.502f, 1.0f};  // #808080
    bevelThicknessPx     = 2.0f * s;
    bevelInward          = !raised;
    showShadow           = false;
}

void Panel::styleMotif(float s, bool raised) {
    // Motif / CDE palette: blue-gray fill (#AEB2C3), dark outer stroke,
    // two-tone bevel using the characteristic CDE lighter/darker tones.
    bgTop.reset();
    bgBottom.reset();
    background           = UiColor{0.682f, 0.698f, 0.765f, 1.0f};  // #AEB2C3
    borderColor          = UiColor{0.298f, 0.314f, 0.376f, 1.0f};  // #4C5060
    borderThicknessPx    = 1.0f * s;
    cornerRadiusPx       = 0.0f;
    innerBorderColor     = UiColor{};
    innerBorderInsetPx   = 0.0f;
    cornerAccentColor    = UiColor{};
    cornerAccentPx       = 0.0f;
    showBevel            = true;
    bevelHighlightColor  = UiColor{0.871f, 0.878f, 0.914f, 1.0f};  // #DDE0EB
    bevelShadowColor     = UiColor{0.416f, 0.431f, 0.498f, 1.0f};  // #6A6E7F
    bevelThicknessPx     = 2.0f * s;
    bevelInward          = !raised;
    showShadow           = false;
}

void Panel::styleClassicMac(float s) {
    bgTop.reset();
    bgBottom.reset();
    background           = UiColor{1.0f, 1.0f, 1.0f, 1.0f};  // white
    borderColor          = UiColor{0.0f, 0.0f, 0.0f, 1.0f};  // black
    borderThicknessPx    = 1.0f * s;
    cornerRadiusPx       = 0.0f;
    innerBorderColor     = UiColor{};
    innerBorderInsetPx   = 0.0f;
    cornerAccentColor    = UiColor{};
    cornerAccentPx       = 0.0f;
    showBevel            = false;
    bevelInward          = false;
    showShadow           = false;  // callers draw Mac window shadows manually
}

void Panel::onTick(float dt) {
    backgroundAnim.update(dt);
    bgTopAnim.update(dt);
    bgBotAnim.update(dt);
    tickChildren(dt);
}

void Panel::draw(UiDrawList& drawList) const {
    // Neumorphic lift highlight: a light shadow cast opposite the main one, drawn
    // first so the darker main shadow layers over it where they meet.
    if (liftShadowColor.a > 0.0f && shadowBlurPx > 0.0f) {
        drawList.addDropShadow(rect_, liftShadowColor, shadowBlurPx,
                               -shadowOffsetX, -shadowOffsetY);
    }
    if (showShadow) {
        drawList.addDropShadow(rect_, shadowColor, shadowBlurPx, shadowOffsetX, shadowOffsetY);
    }
    // Resolve animated colors: if a tween is in flight use current(), else use the static field.
    const UiColor effectiveBg  = backgroundAnim.idle() ? background   : backgroundAnim.current();
    const UiColor effectiveBgT = bgTopAnim.idle()      ? bgTop.value_or(UiColor{}) : bgTopAnim.current();
    const UiColor effectiveBgB = bgBotAnim.idle()      ? bgBottom.value_or(UiColor{}) : bgBotAnim.current();
    const bool gradient = (bgTop.has_value() && bgBottom.has_value())
                       || (!bgTopAnim.idle() && !bgBotAnim.idle());
    if (nineSlice.has_value()) {
        drawList.add9Slice(rect_, *nineSlice, effectiveBg);
    } else if (gradient && cornerRadiusPx > 0.0f) {
        // Rounded duotone gradient card: the SDF fill clips the gradient to the
        // corners; a matching rounded stroke when a border is configured.
        drawList.addRoundRectFilledVGradient(rect_, effectiveBgT, effectiveBgB, cornerRadiusPx);
        if (borderThicknessPx > 0.0f && borderColor.a > 0.0f) {
            drawList.addRoundRect(rect_, borderColor, cornerRadiusPx, borderThicknessPx);
        }
    } else if (gradient) {
        drawList.addRectFilledVGradient(rect_, effectiveBgT, effectiveBgB);
        if (borderThicknessPx > 0.0f && borderColor.a > 0.0f) {
            drawList.addRect(rect_, borderColor, borderThicknessPx);
        }
    } else if (cornerRadiusPx > 0.0f) {
        drawList.addRoundRectFilled(rect_, effectiveBg, cornerRadiusPx);
        if (borderThicknessPx > 0.0f && borderColor.a > 0.0f) {
            drawList.addRoundRect(rect_, borderColor, cornerRadiusPx, borderThicknessPx);
        }
    } else {
        drawList.addRectFilled(rect_, effectiveBg);
        if (borderThicknessPx > 0.0f && borderColor.a > 0.0f) {
            drawList.addRect(rect_, borderColor, borderThicknessPx);
        }
    }

    if (showBevel) {
        drawList.addBevel(rect_, bevelHighlightColor, bevelShadowColor,
                          cornerRadiusPx, bevelThicknessPx, bevelInward);
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
