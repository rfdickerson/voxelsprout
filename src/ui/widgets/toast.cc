#include "ui/widgets/toast.h"

#include "ui/icon_atlas.h"
#include "ui/ui_draw_list.h"

#include <algorithm>

namespace odai::ui {

void ToastManager::push(std::string iconName, std::string message) {
    ToastEntry entry;
    entry.iconName = std::move(iconName);
    entry.message = std::move(message);
    // Lifetime = display + fade-out tail so the last 0.3 s is always the fade.
    entry.lifetimeRemaining = displaySeconds + 0.3f;
    entry.fadeTween = Tween{0.0f, 1.0f, 0.3f, Easing::EaseOut};
    toasts_.push_back(std::move(entry));
}

void ToastManager::onTick(float dt) {
    for (ToastEntry& t : toasts_) {
        t.fadeTween.update(dt);
        t.lifetimeRemaining -= dt;
        // Start fading out 0.3 s before expiry.
        if (t.lifetimeRemaining <= 0.3f) {
            t.fadeTween.setTarget(0.0f);
        }
    }
    toasts_.erase(
        std::remove_if(toasts_.begin(), toasts_.end(),
                       [](const ToastEntry& e) { return e.lifetimeRemaining <= 0.0f; }),
        toasts_.end());
    tickChildren(dt);
}

void ToastManager::draw(UiDrawList& dl) const {
    // Toasts stack downward from the top-right of rect_.
    const float anchorX = rect_.maxX - toastWidthPx;
    float y = rect_.minY;
    for (const ToastEntry& t : toasts_) {
        const UiRect tr{anchorX, y, anchorX + toastWidthPx, y + toastHeightPx};
        dl.pushOpacity(t.fadeTween.eased());
        dl.addRoundRectFilled(tr, background, cornerRadiusPx);
        dl.addRoundRect(tr, borderColor, cornerRadiusPx, 1.0f);

        if (font_ != nullptr) {
            const float ty = tr.minY + (toastHeightPx - font_->lineHeightPx()) * 0.5f;
            float tx = tr.minX + paddingX;

            UiIconEntry icon;
            if (!t.iconName.empty() && UiIconRegistry::global().resolve(t.iconName, icon)) {
                const float isz = font_->lineHeightPx();
                dl.addImage(UiRect{tx, ty, tx + isz, ty + isz}, icon.textureId,
                            UiColor{1, 1, 1, 1}, icon.uv);
                tx += isz + 6.0f;
            }

            dl.addText(*font_, t.message, UiVec2{tx, ty}, textColor);
        }

        dl.popOpacity();
        y += toastHeightPx + toastGapPx;
    }

    drawChildren(dl);
}

}  // namespace odai::ui
