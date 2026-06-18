#include "ui/widgets/window.h"

#include "ui/font.h"
#include "ui/ui_draw_list.h"

namespace odai::ui {

Window::Window(const Font* font, std::string title, std::function<void()> onClose)
    : font_(font), title_(std::move(title)), onClose_(std::move(onClose)) {}

UiRect Window::frameRect() const {
    return UiRect{rect_.minX + margin, rect_.minY + margin, rect_.maxX - margin, rect_.maxY - margin};
}

UiRect Window::titleBarRect() const {
    const UiRect f = frameRect();
    return UiRect{f.minX, f.minY, f.maxX, f.minY + titleBarH};
}

UiRect Window::closeBtnRect() const {
    const UiRect f = frameRect();
    const float inset = titleBarH * 0.14f;
    const float sz = titleBarH - inset * 2.0f;
    return UiRect{f.maxX - inset - sz, f.minY + inset, f.maxX - inset, f.minY + inset + sz};
}

UiRect Window::contentRect() const {
    const UiRect f = frameRect();
    return UiRect{f.minX + padding.x, f.minY + titleBarH + padding.y, f.maxX - padding.x, f.maxY - padding.y};
}

void Window::draw(UiDrawList& dl) const {
    const UiRect f = frameRect();
    const float edge = titleBarH * 0.14f;  // Keep the toolbar/separator inside the frame border.

    // Frame: 9-slice if provided, else a solid fill + 1px border.
    if (frame.has_value()) {
        dl.add9Slice(f, *frame, frameTint);
    } else {
        dl.addRectFilled(f, bodyColor);
        dl.addRect(f, borderColor, 1.0f);
    }

    // Toolbar strip + separator line beneath it.
    const UiRect toolbar{f.minX + edge, f.minY + edge, f.maxX - edge, f.minY + titleBarH};
    dl.addRectFilled(toolbar, titleBarColor);
    const float sepY = f.minY + titleBarH;
    dl.addRectFilled(UiRect{f.minX + edge, sepY, f.maxX - edge, sepY + 1.0f}, borderColor);

    if (font_ != nullptr && !title_.empty()) {
        const float textY = f.minY + (titleBarH - font_->lineHeightPx()) * 0.5f;
        dl.addText(*font_, title_, UiVec2{toolbar.minX + edge + 4.0f, textY}, titleColor);
    }

    if (showCloseButton) {
        const UiRect cb = closeBtnRect();
        if (closeHovered_) {
            dl.addRectFilled(cb, UiColor{0.85f, 0.30f, 0.24f, 0.35f});
        }
        if (font_ != nullptr) {
            const float tw = font_->measureText("x");
            const float tx = cb.minX + (cb.width() - tw) * 0.5f;
            const float ty = cb.minY + (cb.height() - font_->lineHeightPx()) * 0.5f;
            dl.addText(*font_, "x", UiVec2{tx, ty}, closeHovered_ ? closeHoverColor : closeColor);
        }
    }

    dl.pushClip(contentRect());
    drawChildren(dl);
    dl.popClip();
}

bool Window::onEvent(UiEvent& e) {
    if (!visible) {
        return false;
    }

    if (e.type == UiEvent::Type::MouseMove) {
        closeHovered_ = showCloseButton && closeBtnRect().contains(e.mousePx);
    }

    if (e.type == UiEvent::Type::MouseDown && e.button == UiMouseButton::Left) {
        if (showCloseButton && closeBtnRect().contains(e.mousePx)) {
            if (onClose_) {
                onClose_();
            }
            e.handled = true;
            return true;
        }
        if (draggable && titleBarRect().contains(e.mousePx)) {
            dragging_ = true;
            dragOffset_ = {e.mousePx.x - rect_.minX, e.mousePx.y - rect_.minY};
            e.handled = true;
            return true;
        }
    }

    if (e.type == UiEvent::Type::MouseMove && dragging_) {
        const float nx = e.mousePx.x - dragOffset_.x;
        const float ny = e.mousePx.y - dragOffset_.y;
        // Move the whole subtree so the toolbar, frame, and content stay together.
        translate(nx - rect_.minX, ny - rect_.minY);
        return true;
    }

    if (e.type == UiEvent::Type::MouseUp) {
        dragging_ = false;
    }

    if (!contentRect().contains(e.mousePx) && e.type != UiEvent::Type::MouseMove) {
        return false;
    }
    return dispatchToChildren(e);
}

}  // namespace odai::ui
