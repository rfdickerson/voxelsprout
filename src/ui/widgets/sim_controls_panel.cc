#include "ui/widgets/sim_controls_panel.h"

#include "ui/ui_draw_list.h"
#include "ui/widgets/button.h"
#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"

#include <memory>
#include <string>
#include <utility>

namespace odai::ui {

void SimControlsPanel::setState(const UiRect& rect, float s, const State& state) {
    children_.clear();
    speed_ = state.speed;
    setRect(rect);

    auto bg = std::make_unique<Panel>();
    bg->setRect(rect);
    bg->styleCard(s);
    bg_ = static_cast<Panel*>(addChild(std::move(bg)));

    const Font* font = fonts_.bold ? fonts_.bold : fonts_.regular;
    const float pad  = 8.0f * s;
    const float btnH = rect.height() - 2.0f * pad;
    const float btnW = btnH;  // square speed chips

    float x = rect.minX + pad;

    // Date / turn label on the left
    if (fonts_.regular && !state.dateLabel.empty()) {
        auto lbl = std::make_unique<Label>(fonts_, state.dateLabel);
        lbl->color = UiColor{0.80f, 0.76f, 0.60f, 1.0f};
        const float lw = fonts_.regular->measureText(state.dateLabel) + 16.0f * s;
        lbl->setRect(UiRect::fromXYWH(x, rect.minY + pad, lw, btnH));
        dateLabel_ = static_cast<Label*>(addChild(std::move(lbl)));
        x += lw + pad;
    }

    // Pause / Normal / Fast / Ultrafast speed chips
    if (state.showSpeedButtons) {
        struct SpeedChip { SimSpeed sp; const char* label; };
        const SpeedChip chips[4] = {
            {SimSpeed::Paused,    "\xE2\x80\x96"},  // ❚❚ pause
            {SimSpeed::Normal,    "\xE2\x96\xBA"},  // ▶
            {SimSpeed::Fast,      "\xE2\x96\xBA\xE2\x96\xBA"},  // ▶▶
            {SimSpeed::Ultrafast, "\xE2\x96\xBA\xE2\x96\xBA\xE2\x96\xBA"},  // ▶▶▶
        };

        auto makeChip = [&](SimSpeed sp, const char* lbl, Button*& out) {
            auto btn = std::make_unique<Button>(font, lbl, [this, sp, cb = state.onSpeedChange]() {
                speed_ = sp;
                setSpeed(sp);
                if (cb) cb(sp);
            });
            btn->setRect(UiRect::fromXYWH(x, rect.minY + pad, btnW, btnH));
            btn->cornerRadiusPx    = 3.0f * s;
            btn->borderThicknessPx = 1.0f * s;
            const bool active = (sp == speed_);
            btn->colorNormal  = active ? UiColor{0.22f, 0.38f, 0.58f, 1.0f}
                                       : UiColor{0.12f, 0.14f, 0.18f, 0.85f};
            btn->colorHover   = UiColor{0.28f, 0.42f, 0.62f, 1.0f};
            btn->colorPressed = UiColor{0.16f, 0.28f, 0.46f, 1.0f};
            btn->borderColor  = active ? UiColor{0.45f, 0.65f, 0.90f, 0.80f}
                                       : UiColor{0.30f, 0.34f, 0.44f, 0.55f};
            out = static_cast<Button*>(addChild(std::move(btn)));
            x += btnW + 4.0f * s;
        };

        makeChip(chips[0].sp, chips[0].label, pauseBtn_);
        makeChip(chips[1].sp, chips[1].label, normalBtn_);
        makeChip(chips[2].sp, chips[2].label, fastBtn_);
        makeChip(chips[3].sp, chips[3].label, ultrafastBtn_);
    }

    // End Turn button on the right
    if (state.showEndTurn) {
        const std::string etLabel = state.endTurnLabel.empty() ? "End Turn" : state.endTurnLabel;
        const float etW = font ? font->measureText(etLabel) + 32.0f * s : 110.0f * s;
        const float etX = rect.maxX - pad - etW;

        auto btn = std::make_unique<Button>(font, etLabel, state.onEndTurn ? state.onEndTurn : [] {});
        btn->setRect(UiRect::fromXYWH(etX, rect.minY + pad, etW, btnH));
        btn->cornerRadiusPx    = 3.0f * s;
        btn->borderThicknessPx = 1.0f * s;
        btn->colorNormal       = UiColor{0.14f, 0.24f, 0.12f, 0.95f};
        btn->colorHover        = UiColor{0.22f, 0.38f, 0.18f, 1.00f};
        btn->colorPressed      = UiColor{0.10f, 0.18f, 0.09f, 1.00f};
        btn->borderColor       = UiColor{0.40f, 0.78f, 0.36f, 0.80f};
        btn->setEnabled(state.endTurnEnabled);
        endTurnBtn_ = static_cast<Button*>(addChild(std::move(btn)));
    }
}

void SimControlsPanel::setSpeed(SimSpeed speed) {
    speed_ = speed;
    auto update = [&](Button* btn, SimSpeed sp) {
        if (!btn) return;
        const bool active = (sp == speed_);
        btn->colorNormal = active ? UiColor{0.22f, 0.38f, 0.58f, 1.0f}
                                  : UiColor{0.12f, 0.14f, 0.18f, 0.85f};
        btn->borderColor = active ? UiColor{0.45f, 0.65f, 0.90f, 0.80f}
                                  : UiColor{0.30f, 0.34f, 0.44f, 0.55f};
    };
    update(pauseBtn_,    SimSpeed::Paused);
    update(normalBtn_,   SimSpeed::Normal);
    update(fastBtn_,     SimSpeed::Fast);
    update(ultrafastBtn_, SimSpeed::Ultrafast);
}

}  // namespace odai::ui
