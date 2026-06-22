#include "ui/widgets/event_tracker_panel.h"

#include "ui/icon_atlas.h"
#include "ui/ui_draw_list.h"
#include "ui/widgets/image.h"
#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"

#include <algorithm>
#include <string>
#include <utility>

namespace odai::ui {

void EventTrackerPanel::setEntries(const UiRect& rect, float s, std::string_view heading,
                                   const std::vector<Entry>& entries) {
    children_.clear();
    setRect(rect);

    const float pad = 14.0f * s;

    auto bg = std::make_unique<Panel>();
    bg->setRect(rect);
    bg->styleCard(s);
    bg->showBevel           = true;
    bg->bevelHighlightColor = UiColor{1.0f, 1.0f, 1.0f, 0.18f};
    bg->bevelShadowColor    = UiColor{0.0f, 0.0f, 0.0f, 0.40f};
    bg->bevelThicknessPx    = 1.5f;
    bg_ = static_cast<Panel*>(addChild(std::move(bg)));

    const float x0  = rect.minX + pad;
    const float wIn = rect.width() - 2.0f * pad;

    const Font* bf  = fonts_.bold ? fonts_.bold : fonts_.regular;
    const Font* rf  = fonts_.regular;
    const float boldH = bf ? bf->lineHeightPx() : 22.0f * s;
    const float regH  = rf ? rf->lineHeightPx() : 20.0f * s;

    float y = rect.minY + pad;

    auto head = std::make_unique<Label>(fonts_, std::string("<b>") + std::string(heading) + "</b>");
    head->color = UiColor{0.941f, 0.753f, 0.251f, 1.0f};
    head->setRect(UiRect::fromXYWH(x0, y, wIn, boldH));
    addChild(std::move(head));
    y += boldH + 6.0f * s;

    auto sep = std::make_unique<Panel>();
    sep->setRect(UiRect::fromXYWH(x0, y, wIn, 1.5f * s));
    sep->background = UiColor{0.784f, 0.588f, 0.227f, 0.55f};
    sep->borderThicknessPx = 0.0f;
    addChild(std::move(sep));
    y += 1.5f * s + 10.0f * s;

    const float discR    = 19.0f * s;
    const float rowGap   = 12.0f * s;
    const float textGap  = 2.0f * s;
    const float captionH = std::max(regH * 0.8f, 12.0f * s);
    const float descBudget = regH * 2.0f;

    for (const Entry& e : entries) {
        const float blockH = captionH + textGap + boldH + textGap + descBudget;
        const float rowH   = std::max(blockH, discR * 2.0f);
        const float rowCy  = y + rowH * 0.5f;

        const float discCx = x0 + discR;
        auto disc = std::make_unique<Panel>();
        disc->setRect(UiRect{discCx - discR, rowCy - discR, discCx + discR, rowCy + discR});
        disc->background        = UiColor{0.118f, 0.086f, 0.039f, 0.95f};
        disc->borderColor       = UiColor{0.784f, 0.588f, 0.227f, 0.85f};
        disc->borderThicknessPx = 1.5f * s;
        disc->cornerRadiusPx    = discR;
        addChild(std::move(disc));

        UiIconEntry icon{};
        if (!e.iconName.empty() && UiIconRegistry::global().resolve(e.iconName, icon)) {
            const float iconR = discR * 0.72f;
            auto img = std::make_unique<Image>(icon.textureId);
            img->uvRect = icon.uv;
            img->setRect(UiRect{discCx - iconR, rowCy - iconR, discCx + iconR, rowCy + iconR});
            addChild(std::move(img));
        }

        const float tx = discCx + discR + 12.0f * s;
        const float tw = rect.maxX - pad - tx;
        float ty = rowCy - blockH * 0.5f;

        auto sub = std::make_unique<Label>(rf, e.subtitle);
        sub->color = UiColor{0.627f, 0.502f, 0.376f, 1.0f};
        sub->setRect(UiRect::fromXYWH(tx, ty, tw, captionH));
        addChild(std::move(sub));
        ty += captionH + textGap;

        auto title = std::make_unique<Label>(fonts_, std::string("<b>") + e.title + "</b>");
        title->color = UiColor{0.910f, 0.851f, 0.690f, 1.0f};
        title->setRect(UiRect::fromXYWH(tx, ty, tw, boldH));
        addChild(std::move(title));
        ty += boldH + textGap;

        auto desc = std::make_unique<Label>(rf, e.description);
        desc->color = UiColor{0.741f, 0.682f, 0.557f, 1.0f};
        desc->setRect(UiRect::fromXYWH(tx, ty, tw, descBudget));
        addChild(std::move(desc));

        y += rowH + rowGap;
    }
}

}  // namespace odai::ui
