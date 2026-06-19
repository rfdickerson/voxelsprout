#include "ui/widgets/window.h"

#include "ui/font.h"
#include "ui/ui_draw_list.h"

#include <algorithm>

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
    // Always anchored in the title bar — square, vertically centered, right-aligned.
    // Uses a fixed fraction of titleBarH so it scales with DPI.
    const float sz = titleBarH * 0.55f;
    const float cy = f.minY + titleBarH * 0.5f;
    const float cx = f.maxX - titleBarH * 0.5f;
    return UiRect{cx - sz * 0.5f, cy - sz * 0.5f, cx + sz * 0.5f, cy + sz * 0.5f};
}

UiRect Window::contentRect() const {
    const UiRect f = frameRect();
    return UiRect{f.minX + padding.x, f.minY + titleBarH + padding.y, f.maxX - padding.x, f.maxY - padding.y};
}

void Window::draw(UiDrawList& dl) const {
    const UiRect f = frameRect();

    if (showShadow) {
        dl.addDropShadow(f, shadowColor, shadowBlurPx, shadowOffsetX, shadowOffsetY);
    }

    // Frame: 9-slice if provided, else rounded solid fill + border.
    if (frame.has_value()) {
        dl.add9Slice(f, *frame, frameTint);
    } else {
        const float r = cornerRadiusPx;
        dl.addRoundRectFilled(f, bodyColor, r);
        dl.addRoundRect(f, borderColor, r, 1.5f);
    }

    // Title bar strip + separator.  The strip is inset slightly from the window
    // sides so its straight edges stay within the rounded corners of the body.
    const float edge = std::max(cornerRadiusPx, 1.0f);
    const UiRect toolbar{f.minX + edge, f.minY + edge, f.maxX - edge, f.minY + titleBarH};
    if (!frame.has_value()) {
        // Cast a soft shadow from the titlebar ledge down into the content area.
        // Clip to the body below so the Gaussian blur doesn't bleed upward.
        const UiRect bodyBelow{f.minX, f.minY + titleBarH, f.maxX, f.maxY};
        dl.pushClip(bodyBelow);
        dl.addDropShadow(toolbar, UiColor{0.0f, 0.0f, 0.0f, 0.50f}, 6.0f, 0.0f, 3.0f);
        dl.popClip();

        // Base fill.
        dl.addRectFilled(toolbar, titleBarColor);

        // Bevel: 1.5 px highlight along top edge — simulates top-lit raised ledge.
        dl.addRectFilled(
            UiRect{toolbar.minX, toolbar.minY, toolbar.maxX, toolbar.minY + 1.5f},
            UiColor{1.0f, 1.0f, 1.0f, 0.18f});
        // Bevel: 1 px darkening along bottom edge — where the ledge recedes.
        dl.addRectFilled(
            UiRect{toolbar.minX, toolbar.maxY - 1.5f, toolbar.maxX, toolbar.maxY},
            UiColor{0.0f, 0.0f, 0.0f, 0.35f});
    }
    const float sepY = f.minY + titleBarH;
    dl.addRectFilled(UiRect{f.minX + edge, sepY, f.maxX - edge, sepY + 1.5f}, borderColor);

    // Title text.
    if (font_ != nullptr && !title_.empty()) {
        const Font& tf = (titleFont != nullptr) ? *titleFont : *font_;
        const float topInset  = frame.has_value() ? frame->borderTopPx  : 0.0f;
        const float leftInset = frame.has_value() ? frame->borderLeftPx : 0.0f;
        const float textY = f.minY + topInset + (titleBarH - topInset - tf.lineHeightPx()) * 0.5f;
        // Reserve space for close button so title doesn't underlap it.
        const float closeReserve = showCloseButton ? titleBarH : 0.0f;
        if (frame.has_value()) {
            dl.addText(tf, title_, UiVec2{toolbar.minX + edge + 4.0f + leftInset, textY}, titleColor);
        } else {
            const float usableW = toolbar.width() - closeReserve;
            const float tw = tf.measureText(title_);
            const float centerX = toolbar.minX + (usableW - tw) * 0.5f;
            dl.addText(tf, title_, UiVec2{centerX, textY}, titleColor);
        }
    }

    // Close button: beveled rounded square with a "×" glyph.
    // The bevel is drawn as lighter top/left edges and darker bottom/right edges —
    // classic raised-button look using only solid filled rects (no textures).
    if (showCloseButton && font_ != nullptr) {
        const UiRect cb = closeBtnRect();
        const float r = cb.width() * 0.20f;
        const float bev = 1.5f;  // bevel edge width in pixels

        // Drop shadow cast behind the button — gives it physical elevation above the titlebar.
        dl.addDropShadow(cb, UiColor{0.0f, 0.0f, 0.0f, closeHovered_ ? 0.35f : 0.55f},
                         2.5f, 0.5f, 1.5f);

        // Base fill: neutral gray, darkened on hover to signal press-readiness.
        const UiColor baseFill = closeHovered_
            ? UiColor{0.48f, 0.14f, 0.10f, 0.98f}
            : UiColor{0.30f, 0.32f, 0.36f, 0.92f};
        dl.addRoundRectFilled(cb, baseFill, r);

        // Outer dark border (recessed feel).
        dl.addRoundRect(cb, UiColor{0.0f, 0.0f, 0.0f, 0.60f}, r, 1.0f);

        // Bevel highlight: top and left inner edges — lighter, simulates top-light.
        const UiColor hlColor{1.0f, 1.0f, 1.0f, closeHovered_ ? 0.10f : 0.30f};
        dl.addRectFilled(UiRect{cb.minX + bev, cb.minY + bev, cb.maxX - bev, cb.minY + bev + bev}, hlColor);
        dl.addRectFilled(UiRect{cb.minX + bev, cb.minY + bev, cb.minX + bev + bev, cb.maxY - bev}, hlColor);

        // Bevel shadow: bottom and right inner edges — darker, simulates depth.
        const UiColor shColor{0.0f, 0.0f, 0.0f, closeHovered_ ? 0.15f : 0.50f};
        dl.addRectFilled(UiRect{cb.minX + bev, cb.maxY - bev - bev, cb.maxX - bev, cb.maxY - bev}, shColor);
        dl.addRectFilled(UiRect{cb.maxX - bev - bev, cb.minY + bev, cb.maxX - bev, cb.maxY - bev}, shColor);

        // "×" glyph centered in the button.
        const char kTimes[] = "\xc3\x97";
        const UiColor glyphColor = closeHovered_ ? closeHoverColor : closeColor;
        const float tw = font_->measureText(kTimes);
        const float tx = cb.minX + (cb.width()  - tw)               * 0.5f;
        const float ty = cb.minY + (cb.height() - font_->lineHeightPx()) * 0.5f;
        dl.addText(*font_, kTimes, UiVec2{tx, ty}, glyphColor);
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
