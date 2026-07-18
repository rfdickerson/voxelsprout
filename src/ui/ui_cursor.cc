#include "ui/ui_cursor.h"

#include "ui/ui_draw_list.h"
#include "ui/vector/vector_path.h"
#include "ui/vector/vector_tessellator.h"

namespace odai::ui {

void drawCursor(UiDrawList& dl, const UiVec2& posPx, float scale) {
    // Classic arrow-pointer silhouette, tip at the local origin (the hotspot),
    // authored in local units and scaled/translated into place below. Units
    // roughly match pixels at scale=1.0 (~19px tall, ~11px wide).
    const float u = scale;
    const auto x = [&](float lx) { return posPx.x + lx * u; };
    const auto y = [&](float ly) { return posPx.y + ly * u; };

    VectorPath arrow;
    arrow.setTessellationTolerancePx(0.3f);
    arrow.moveTo(x(0.0f), y(0.0f));    // tip (hotspot)
    arrow.lineTo(x(0.0f), y(14.0f));   // down the left edge
    arrow.lineTo(x(3.5f), y(11.0f));   // inward notch
    arrow.lineTo(x(6.0f), y(17.0f));   // tail point
    arrow.lineTo(x(8.0f), y(16.0f));
    arrow.lineTo(x(5.0f), y(10.0f));
    arrow.lineTo(x(11.0f), y(10.0f));  // right shoulder
    arrow.close();

    dl.addPathFilled(arrow, UiColor{0.05f, 0.05f, 0.05f, 1.0f});

    StrokeOptions outline;
    outline.widthPx = 1.25f * scale;
    outline.join = LineJoin::Round;
    outline.cap = LineCap::Round;
    dl.addPathStroked(arrow, UiColor{1.0f, 1.0f, 1.0f, 1.0f}, outline);
}

}  // namespace odai::ui
