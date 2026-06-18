#include "ui/widgets/label.h"

#include "ui/rich_text.h"

namespace odai::ui {

void Label::draw(UiDrawList& drawList) const {
    if (font_ == nullptr) {
        return;
    }
    const float wrapWidth = wrap ? (rect_.width() - (padding.x * 2.0f)) : 0.0f;
    const RichTextLayout layout = layoutRichText(markup_, color, *font_, wrapWidth, align);
    drawList.pushClip(rect_);
    drawRichText(drawList, layout, *font_, UiVec2{rect_.minX + padding.x, rect_.minY + padding.y});
    drawList.popClip();
}

}  // namespace odai::ui
