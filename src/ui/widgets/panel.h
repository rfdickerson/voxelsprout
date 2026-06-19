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
    // and border as anti-aliased SDF rounded rects. Ignored when nineSlice is set.
    float cornerRadiusPx = 0.0f;
    std::optional<UiNineSlice> nineSlice;  // If set, used instead of the solid fill.

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
