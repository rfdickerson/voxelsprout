#include "ui/widgets/modal.h"

#include "ui/ui_draw_list.h"

namespace odai::ui {

Modal::Modal(const Font* font, std::string title, std::function<void()> onClose)
    : onClose_(std::move(onClose)) {
    visible = false;
    auto window = std::make_unique<Window>(font, std::move(title), [this]() { close(); });
    dialog_ = static_cast<Window*>(addChild(std::move(window)));
}

void Modal::layout() {
    const UiVec2 center{(rect_.minX + rect_.maxX) * 0.5f, (rect_.minY + rect_.maxY) * 0.5f};
    const UiRect dialogRect{center.x - dialogSizePx.x * 0.5f, center.y - dialogSizePx.y * 0.5f,
                            center.x + dialogSizePx.x * 0.5f, center.y + dialogSizePx.y * 0.5f};
    dialog_->setRect(dialogRect);
}

void Modal::open() {
    visible = true;
    layout();
}

void Modal::close() {
    visible = false;
    if (onClose_) {
        onClose_();
    }
}

void Modal::draw(UiDrawList& dl) const {
    dl.addRectFilled(rect_, backdropColor);
    drawChildren(dl);
}

bool Modal::onEvent(UiEvent& ev) {
    dispatchToChildren(ev);
    if (blocking) {
        ev.handled = true;
        return true;
    }
    return ev.handled;
}

}  // namespace odai::ui
