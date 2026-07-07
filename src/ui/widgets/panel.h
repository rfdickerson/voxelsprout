#pragma once

#include "ui/animation.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <optional>

// A rectangular container with a solid or 9-slice background and optional border.
namespace odai::ui {

class Panel : public Widget {
public:
    Panel() = default;

    UiColor background{0.05f, 0.10f, 0.14f, 0.86f};
    UiColor borderColor{0.85f, 0.72f, 0.44f, 0.25f};
    float borderThicknessPx = 1.0f;
    // Corner radius in pixels (DPI-scaled by the caller). > 0 draws the solid fill
    // and border as anti-aliased SDF rounded rects. Ignored when nineSlice is set,
    // and ignored for the gradient (ornate) fill, which is always sharp-cornered.
    float cornerRadiusPx = 0.0f;
    std::optional<UiNineSlice> nineSlice;  // If set, used instead of the solid fill.

    // --- Ornate (gilt-frame) styling, all off by default ---------------------
    // When both are set, the panel is filled with a vertical gradient (top->bottom)
    // instead of the solid `background`. Gives panels a parchment sheen.
    std::optional<UiColor> bgTop;
    std::optional<UiColor> bgBottom;
    // A second, inset border line drawn `innerBorderInsetPx` inside the outer edge
    // -> the gilded double-line frame look. Drawn only when inset > 0 and alpha > 0.
    UiColor innerBorderColor{};
    float   innerBorderInsetPx = 0.0f;
    // Short L-shaped accent ticks at the four corners. Drawn only when length > 0.
    UiColor cornerAccentColor{};
    float   cornerAccentPx = 0.0f;
    float   cornerAccentThicknessPx = 2.0f;

    // Configure this panel with the standard parchment/gilt "framed card" look:
    // warm gradient fill, gold outer + inner borders, corner accents, drop shadow.
    // `s` is the DPI scale; pass `alpha` to tune translucency over the map.
    void styleOrnate(float s, float alpha = 0.95f);

    // Configure this panel with the clean-modern "flat card" look: a translucent
    // slate fill, rounded corners, a hairline border, and one soft drop shadow.
    // Deliberately flat — clears the gilt gradient, inner border, and corner
    // accents that styleOrnate sets. `s` is the DPI scale; `alpha` tunes how much
    // of the map shows through.
    void styleCard(float s, float alpha = 0.82f);

    // Configure this panel with the soft "neumorphic" look: a light, near-opaque
    // fill, generously rounded corners, no hard border, and a *pair* of soft
    // shadows — a dark shadow cast down-right plus a light highlight lifted
    // up-left (see liftShadowColor) — so the card reads as gently extruded from
    // the surface. Works only over a light background of the same family. `s` is
    // the DPI scale; pass the panel's fill `tint` to recolor it (defaults to the
    // #E4EBF1 cool-grey from the reference palette).
    void styleSoft(float s, UiColor tint = UiColor{0.894f, 0.922f, 0.945f, 1.0f});

    // Configure this panel as a "duotone gradient card": a vertical color
    // gradient fill (top→bottom) clipped to rounded corners, no border, and one
    // soft drop shadow. Use for full-bleed feature tiles in the modern/travel
    // style. `s` is the DPI scale; `top`/`bottom` are the gradient stops and
    // `alpha` tunes overall translucency.
    void styleGradientCard(float s, const UiColor& top, const UiColor& bottom,
                           float alpha = 1.0f);

    // Configure this panel with the Windows 95 / Redmond look: opaque silver-gray
    // fill, square corners, a thin black outer border, and a two-tone raised bevel
    // (white top-left / #808080 bottom-right). Pass raised=false for a recessed /
    // pressed look (e.g. group-boxes or sunken fields).
    void styleWin95(float s, bool raised = true);

    // Configure this panel with the Motif / CDE look: opaque blue-gray fill,
    // square corners, a dark outer stroke, and a two-tone raised bevel using the
    // characteristic CDE highlight and shadow tones.
    void styleMotif(float s, bool raised = true);

    // Configure this panel with the Mac System 6/7 "Platinum" look: opaque white
    // fill, 1px black border, square corners, no bevel. Drop shadows on windows
    // are drawn manually as offset solid-black rects in the calling code.
    void styleClassicMac(float s);

    // --- Bevel (raised / recessed 3-D edge) -----------------------------------
    // When showBevel is true, a two-tone border is drawn over the fill using
    // addBevel(). highlightColor lights the top edge; shadowColor darkens the
    // bottom edge. Both follow the panel's cornerRadiusPx. Set bevelInward=true
    // for a pressed/recessed look. Coexists with the uniform borderColor stroke.
    bool    showBevel             = false;
    UiColor bevelHighlightColor   {1.0f, 1.0f, 1.0f, 0.28f};
    UiColor bevelShadowColor      {0.0f, 0.0f, 0.0f, 0.45f};
    float   bevelThicknessPx      = 2.0f;
    bool    bevelInward           = false;

    bool    showShadow    = false;
    UiColor shadowColor   {0.0f, 0.0f, 0.0f, 0.55f};
    float   shadowBlurPx  = 8.0f;
    float   shadowOffsetX = 0.0f;
    float   shadowOffsetY = 4.0f;

    // Optional second "lift" shadow for the soft/neumorphic look: a light
    // highlight cast in the direction *opposite* the main shadow (the main
    // offsets are negated), sharing shadowBlurPx. Drawn under the main shadow
    // only when its alpha > 0. Off by default; set by styleSoft.
    // NOTE: UiColor{} default-constructs to OPAQUE WHITE, not transparent, so
    // the alpha must be zeroed explicitly here or every panel (styleCard,
    // styleOrnate, plain Panel) draws an unwanted white lift-shadow underneath
    // its real shadow, since none of them ever touch this field.
    UiColor liftShadowColor{1.0f, 1.0f, 1.0f, 0.0f};

    // When true, child drawing is clipped to rect_ — useful for animated panels
    // whose height changes each frame (accordion, slide-in drawers, etc.).
    bool clipContents = false;

    // --- Animated background --------------------------------------------------
    // Call backgroundAnim.set(targetColor, durationSec) to smoothly cross-fade the
    // panel's background to a new color. While the tween is in flight, the draw
    // method uses backgroundAnim.current() instead of `background`. Advanced
    // automatically each frame via onTick (UiContext::tick(dt) drives this from
    // the root) — no manual per-panel update call needed.
    // bgTopAnim / bgBotAnim animate the ornate gradient stops independently.
    ColorTween backgroundAnim;
    ColorTween bgTopAnim;
    ColorTween bgBotAnim;

    void onTick(float dt) override;
    void draw(UiDrawList& drawList) const override;
};

}  // namespace odai::ui
