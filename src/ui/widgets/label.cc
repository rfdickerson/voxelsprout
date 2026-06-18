#include "ui/widgets/label.h"

namespace odai::ui {

void Label::draw(UiDrawList& drawList) const {
    if (!cache_.hasFont()) {
        return;
    }
    // Sync the public style fields into the cache (guarded setters are no-ops when
    // unchanged), then emit the cached geometry.
    cache_.setColor(color);
    cache_.setAlign(align);
    cache_.setWrap(wrap);
    cache_.setPadding(padding);
    cache_.emit(drawList, rect_);
}

}  // namespace odai::ui
