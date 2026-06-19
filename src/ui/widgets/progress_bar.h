#pragma once

#include "ui/ui_types.h"
#include "ui/widget.h"

#include <optional>

namespace odai::ui {

// Horizontal fill bar. value is clamped to [0, 1].
class ProgressBar : public Widget {
public:
    float value = 0.0f;           // 0 = empty, 1 = full.
    UiColor foreground{0.3f, 0.7f, 0.3f, 1.0f};
    UiColor background{0.1f, 0.1f, 0.1f, 0.8f};
    UiColor borderColor{0.0f, 0.0f, 0.0f, 0.5f};
    float borderThicknessPx = 1.0f;
    // Corner radius in pixels (DPI-scaled by the caller). > 0 draws the track and
    // fill as anti-aliased SDF rounded rects; a large value yields a pill bar.
    float cornerRadiusPx = 0.0f;
    std::optional<UiNineSlice> frame;

    void draw(UiDrawList& dl) const override;
};

}  // namespace odai::ui
