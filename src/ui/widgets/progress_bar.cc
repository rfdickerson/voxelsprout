#include "ui/widgets/progress_bar.h"

#include "ui/ui_draw_list.h"

#include <algorithm>

namespace odai::ui {

void ProgressBar::draw(UiDrawList& dl) const {
    if (!visible) return;

    if (frame.has_value()) {
        dl.add9Slice(rect_, *frame, UiColor{1.0f, 1.0f, 1.0f, 1.0f});
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
        dl.addRectFilled(fill, foreground);
    }
    drawChildren(dl);
}

}  // namespace odai::ui
