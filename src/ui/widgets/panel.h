#pragma once

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

    bool    showShadow    = false;
    UiColor shadowColor   {0.0f, 0.0f, 0.0f, 0.55f};
    float   shadowBlurPx  = 8.0f;
    float   shadowOffsetX = 0.0f;
    float   shadowOffsetY = 4.0f;

    // When true, child drawing is clipped to rect_ — useful for animated panels
    // whose height changes each frame (accordion, slide-in drawers, etc.).
    bool clipContents = false;

    void draw(UiDrawList& drawList) const override;
};

}  // namespace odai::ui
