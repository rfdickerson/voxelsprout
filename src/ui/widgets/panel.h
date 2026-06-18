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
    std::optional<UiNineSlice> nineSlice;  // If set, used instead of the solid fill.

    void draw(UiDrawList& drawList) const override;
};

}  // namespace odai::ui
