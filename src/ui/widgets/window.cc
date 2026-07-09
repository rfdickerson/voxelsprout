#include "ui/widgets/window.h"

#include "ui/font.h"
#include "ui/ui_draw_list.h"

#include <cctype>

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

    // Frame: 9-slice if provided, else rounded solid fill + layered border.
    if (frame.has_value()) {
        dl.add9Slice(f, *frame, frameTint);
    } else {
        const float r = cornerRadiusPx;
        dl.addRoundRectFilled(f, bodyColor, r);
        // Top-lit bevel just inside the silhouette: a soft highlight along the top
        // edge and a recessed shadow along the bottom give the frame physical depth
        // before the defining stroke is laid over it.
        dl.addBevel(f, frameBevelHighlightColor, frameBevelShadowColor, r, frameBevelThicknessPx);
        // Crisp defining edge — the window's accent color reads as a thin gilt rule.
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
            toolbarBevelHighlightColor);
        // Bevel: 1 px darkening along bottom edge — where the ledge recedes.
        dl.addRectFilled(
            UiRect{toolbar.minX, toolbar.maxY - 1.5f, toolbar.maxX, toolbar.maxY},
            toolbarBevelShadowColor);
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
        const Font& sc = (titleSmallCapsFont != nullptr) ? *titleSmallCapsFont : tf;
        const auto isLower = [](unsigned char c) { return c >= 'a' && c <= 'z'; };
        const auto measureSmallCaps = [&]() {
            float width = 0.0f;
            for (unsigned char c : title_) {
                const char uppercase = static_cast<char>(std::toupper(c));
                width += isLower(c) ? sc.measureText(std::string_view(&uppercase, 1))
                                    : tf.measureText(std::string_view(&uppercase, 1));
            }
            return width;
        };
        const auto drawSmallCaps = [&](float x) {
            for (unsigned char c : title_) {
                const char uppercase = static_cast<char>(std::toupper(c));
                const Font& glyphFont = isLower(c) ? sc : tf;
                const float glyphY = textY + tf.ascentPx() - glyphFont.ascentPx();
                x += dl.addText(glyphFont, std::string_view(&uppercase, 1), UiVec2{x, glyphY}, titleColor);
            }
        };
        if (frame.has_value()) {
            drawSmallCaps(toolbar.minX + edge + 4.0f + leftInset);
        } else {
            const float usableW = toolbar.width() - closeReserve;
            const float tw = measureSmallCaps();
            const float centerX = toolbar.minX + (usableW - tw) * 0.5f;
            drawSmallCaps(centerX);
        }
    }

    // Close button: beveled rounded square with a "×" glyph.
    // Resting: raised key (top-lit highlight, bottom shadow). Hover: warm red,
    // bevel flips inward so the button visibly depresses.
    if (showCloseButton && font_ != nullptr) {
        const UiRect cb = closeBtnRect();
        const float r = cb.width() * 0.08f;

        // Base fill.
        const UiColor baseFill = closeHovered_
            ? UiColor{0.55f, 0.16f, 0.12f, 1.0f}
            : UiColor{0.27f, 0.30f, 0.35f, 1.0f};
        dl.addRoundRectFilled(cb, baseFill, r);

        // Outer silhouette drawn first so the bevel highlight isn't occluded by it.
        dl.addRoundRect(cb, UiColor{0.0f, 0.0f, 0.0f, 0.55f}, r, 1.0f);

        // Bevel drawn on an inset rect so it sits cleanly inside the silhouette
        // and the two don't compete on the same pixels.
        constexpr float kInset = 1.0f;
        constexpr float kBev   = 1.5f;
        const UiRect inner{cb.minX + kInset, cb.minY + kInset,
                           cb.maxX - kInset, cb.maxY - kInset};
        const float ri = std::max(0.0f, r - kInset);
        const UiColor hlColor{1.0f, 1.0f, 1.0f, 0.50f};
        const UiColor shColor{0.0f, 0.0f, 0.0f, 0.65f};
        dl.addBevel(inner, hlColor, shColor, ri, kBev, /*inward=*/closeHovered_);

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

void Window::bringToFront() {
    // Shared across every Window instance: siblings compare zOrder directly
    // (see Widget::drawChildren / dispatchToChildren), so a single ever-
    // increasing counter is enough to place whichever window last claimed it
    // above all others, regardless of the order windows were originally added
    // to their parent.
    static int s_topZOrder = 0;
    zOrder = ++s_topZOrder;
}

bool Window::onEvent(UiEvent& e) {
    if (!visible) {
        return false;
    }

    if (e.type == UiEvent::Type::MouseMove) {
        closeHovered_ = showCloseButton && closeBtnRect().contains(e.mousePx);
    }

    // Clicking anywhere inside the window's silhouette — title bar, close
    // button, or content — brings it to the front of its sibling windows,
    // matching desktop window-manager behavior. Checked before the specific
    // close/drag handling below so it applies regardless of which of those
    // paths (if any) ends up consuming the event.
    if (e.type == UiEvent::Type::MouseDown && e.button == UiMouseButton::Left &&
        frameRect().contains(e.mousePx)) {
        bringToFront();
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
