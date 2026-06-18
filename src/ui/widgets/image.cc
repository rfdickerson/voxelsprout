#include "ui/widgets/image.h"

#include "ui/ui_draw_list.h"

namespace odai::ui {

Image::Image(UiTextureId id) : textureId(id) {}

void Image::draw(UiDrawList& dl) const {
    if (!visible || textureId == kUiNoTexture) return;
    dl.addImage(rect_, textureId, tint, uvRect);
    drawChildren(dl);
}

}  // namespace odai::ui
