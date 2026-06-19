#include "ui/widgets/stat_badge.h"

#include "ui/icon_atlas.h"
#include "ui/ui_draw_list.h"

namespace odai::ui {

void StatBadgeRow::draw(UiDrawList& dl) const {
    if (backgroundColor.a > 0.0f) {
        dl.addRectFilled(rect_, backgroundColor);
    }

    float x = rect_.minX;
    const float iconY = rect_.minY + (rect_.height() - iconSizePx) * 0.5f;
    const float textY = font_ != nullptr
                        ? rect_.minY + (rect_.height() - font_->lineHeightPx()) * 0.5f
                        : rect_.minY;

    for (const Stat& stat : stats) {
        // Icon.
        UiIconEntry icon;
        if (!stat.iconName.empty() && UiIconRegistry::global().resolve(stat.iconName, icon)) {
            dl.addImage(UiRect{x, iconY, x + iconSizePx, iconY + iconSizePx},
                        icon.textureId, UiColor{1, 1, 1, 1}, icon.uv);
            x += iconSizePx + gapPx;
        }

        // Value text.
        if (font_ != nullptr && !stat.value.empty()) {
            dl.addText(*font_, stat.value, UiVec2{x, textY}, stat.valueColor);
            x += font_->measureText(stat.value) + statGapPx;
        } else {
            x += statGapPx;
        }
    }

    drawChildren(dl);
}

}  // namespace odai::ui
