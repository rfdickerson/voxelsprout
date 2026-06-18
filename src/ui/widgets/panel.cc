#include "ui/widgets/panel.h"

namespace odai::ui {

void Panel::draw(UiDrawList& drawList) const {
    if (nineSlice.has_value()) {
        drawList.add9Slice(rect_, *nineSlice, background);
    } else {
        drawList.addRectFilled(rect_, background);
        if (borderThicknessPx > 0.0f && borderColor.a > 0.0f) {
            drawList.addRect(rect_, borderColor, borderThicknessPx);
        }
    }
    drawChildren(drawList);
}

}  // namespace odai::ui
