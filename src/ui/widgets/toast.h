#pragma once

#include "ui/animation.h"
#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <string>
#include <vector>

namespace odai::ui {

struct ToastEntry {
    std::string iconName;
    std::string message;
    float lifetimeRemaining = 3.3f;  // Includes fade-out tail (displaySeconds + 0.3f).
    Tween fadeTween{0.0f, 1.0f, 0.3f, Easing::EaseOut};
};

// Stack of self-dismissing notification panels. Anchor this widget in the top-
// right of the root so toasts stack downward. Fade/expiry is advanced
// automatically each frame via onTick (UiContext::tick(dt) drives this from the
// root); call push() to enqueue a new notification.
class ToastManager : public Widget {
public:
    explicit ToastManager(const Font* font) : font_(font) {}

    float displaySeconds = 3.0f;  // How long each toast is fully visible.
    float toastWidthPx = 260.0f;
    float toastHeightPx = 44.0f;
    float toastGapPx = 6.0f;
    float cornerRadiusPx = 2.0f;
    float paddingX = 10.0f;

    UiColor background{0.12f, 0.12f, 0.12f, 0.92f};
    UiColor borderColor{0.28f, 0.28f, 0.28f, 1.0f};
    UiColor textColor{0.92f, 0.92f, 0.92f, 1.0f};

    void push(std::string iconName, std::string message);
    void onTick(float dt) override;

    int activeCount() const { return static_cast<int>(toasts_.size()); }

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent&) override { return false; }

private:
    const Font* font_ = nullptr;
    std::vector<ToastEntry> toasts_;
};

}  // namespace odai::ui
