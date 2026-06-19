#include "ui/widgets/panel.h"

namespace odai::ui {

void Panel::draw(UiDrawList& drawList) const {
    if (showShadow) {
        drawList.addDropShadow(rect_, shadowColor, shadowBlurPx, shadowOffsetX, shadowOffsetY);
    }
    if (nineSlice.has_value()) {
        drawList.add9Slice(rect_, *nineSlice, background);
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
    drawChildren(drawList);
}

}  // namespace odai::ui
