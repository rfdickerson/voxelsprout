#include "ui/widgets/command_view_panel.h"

#include "ui/icon_atlas.h"
#include "ui/ui_draw_list.h"
#include "ui/widgets/icon_button.h"
#include "ui/widgets/image.h"
#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

namespace odai::ui {
namespace {

constexpr UiColor kGold{0.941f, 0.753f, 0.251f, 1.0f};
constexpr UiColor kText{0.910f, 0.851f, 0.690f, 1.0f};
constexpr UiColor kDim{0.627f, 0.502f, 0.376f, 1.0f};
constexpr UiColor kDesc{0.741f, 0.682f, 0.557f, 1.0f};
constexpr UiColor kCardFill{0.071f, 0.051f, 0.024f, 0.55f};
constexpr UiColor kCardBorder{0.478f, 0.361f, 0.165f, 0.55f};

}  // namespace

void CommandViewPanel::setState(const UiRect& rect, float s, const State& state) {
    children_.clear();
    setRect(rect);

    const float pad = 14.0f * s;
    const Font* bf  = fonts_.bold ? fonts_.bold : fonts_.regular;
    const Font* rf  = fonts_.regular;
    const float boldH = bf ? bf->lineHeightPx() : 22.0f * s;
    const float regH  = rf ? rf->lineHeightPx() : 20.0f * s;
    const float capH  = std::max(regH * 0.8f, 12.0f * s);

    // Ornate background frame.
    auto bg = std::make_unique<Panel>();
    bg->setRect(rect);
    bg->styleOrnate(s);
    bg_ = static_cast<Panel*>(addChild(std::move(bg)));

    const float x0  = rect.minX + pad;
    const float wIn = rect.width() - 2.0f * pad;

    // --- small helpers -------------------------------------------------------
    auto addLabel = [&](const std::string& markup, const UiColor& col, const FontSet& fs,
                        UiTextAlign align, const UiRect& r) {
        auto l = std::make_unique<Label>(fs, markup);
        l->color = col;
        l->align = align;
        l->setRect(r);
        addChild(std::move(l));
    };
    auto addCaption = [&](const std::string& text, float& y) {
        addLabel("<b>" + text + "</b>", kDim, fonts_, UiTextAlign::Left,
                 UiRect::fromXYWH(x0, y, wIn, capH));
        y += capH + 4.0f * s;
    };
    auto addCardBg = [&](float top, float height) {
        auto card = std::make_unique<Panel>();
        card->setRect(UiRect::fromXYWH(x0, top, wIn, height));
        card->background        = kCardFill;
        card->borderColor       = kCardBorder;
        card->borderThicknessPx = 1.0f * s;
        card->cornerRadiusPx    = 4.0f * s;
        addChild(std::move(card));
    };
    auto addMedallion = [&](const std::string& iconName, const UiRect& r) {
        auto disc = std::make_unique<Panel>();
        disc->setRect(r);
        disc->background        = UiColor{0.118f, 0.086f, 0.039f, 0.95f};
        disc->borderColor       = UiColor{0.784f, 0.588f, 0.227f, 0.85f};
        disc->borderThicknessPx = 1.5f * s;
        disc->cornerRadiusPx    = std::min(r.width(), r.height()) * 0.5f;
        addChild(std::move(disc));
        UiIconEntry icon{};
        if (!iconName.empty() && UiIconRegistry::global().resolve(iconName, icon)) {
            const float inset = std::min(r.width(), r.height()) * 0.16f;
            auto img = std::make_unique<Image>(icon.textureId);
            img->uvRect = icon.uv;
            img->setRect(UiRect{r.minX + inset, r.minY + inset, r.maxX - inset, r.maxY - inset});
            addChild(std::move(img));
        }
    };
    // A 3-up stat grid: each cell shows a dim caption over a bold value.
    auto addStatGrid = [&](const std::vector<Stat>& stats, float& y) {
        if (stats.empty()) return;
        const int n = static_cast<int>(stats.size());
        const float gap   = 6.0f * s;
        const float cellW = (wIn - gap * (n - 1)) / static_cast<float>(n);
        const float cellH = capH + 3.0f * s + boldH + 10.0f * s;
        for (int i = 0; i < n; ++i) {
            const float cx = x0 + static_cast<float>(i) * (cellW + gap);
            auto cell = std::make_unique<Panel>();
            cell->setRect(UiRect::fromXYWH(cx, y, cellW, cellH));
            cell->background        = UiColor{0.094f, 0.067f, 0.031f, 0.70f};
            cell->borderColor       = UiColor{0.478f, 0.361f, 0.165f, 0.50f};
            cell->borderThicknessPx = 1.0f * s;
            cell->cornerRadiusPx    = 3.0f * s;
            addChild(std::move(cell));
            addLabel(stats[i].label, kDim, fonts_, UiTextAlign::Center,
                     UiRect::fromXYWH(cx, y + 5.0f * s, cellW, capH));
            addLabel("<b>" + stats[i].value + "</b>", kText, fonts_, UiTextAlign::Center,
                     UiRect::fromXYWH(cx, y + 5.0f * s + capH + 3.0f * s, cellW, boldH));
        }
        y += cellH + 10.0f * s;
    };

    float y = rect.minY + pad;

    // Heading + underline.
    addLabel("<b>COMMAND VIEW</b>", kGold, fonts_, UiTextAlign::Left,
             UiRect::fromXYWH(x0, y, wIn, boldH));
    y += boldH + 6.0f * s;
    {
        auto sep = std::make_unique<Panel>();
        sep->setRect(UiRect::fromXYWH(x0, y, wIn, 1.5f * s));
        sep->background = UiColor{0.784f, 0.588f, 0.227f, 0.55f};
        sep->borderThicknessPx = 0.0f;
        addChild(std::move(sep));
        y += 1.5f * s + 10.0f * s;
    }

    // Nothing selected -> hint and stop.
    if (!state.hasHex && !state.hasUnit) {
        addLabel(state.emptyHint.empty() ? "<i>Select a unit or hex.</i>" : state.emptyHint,
                 kDim, fonts_, UiTextAlign::Left, UiRect::fromXYWH(x0, y, wIn, regH * 3.0f));
        return;
    }

    // --- Selected hex card ---------------------------------------------------
    if (state.hasHex) {
        addCaption("SELECTED HEX", y);
        const float medR    = 16.0f * s;
        const float cardH   = std::max(medR * 2.0f, boldH + capH + 8.0f * s) + 12.0f * s;
        const float cardTop = y;
        addCardBg(cardTop, cardH);
        const float cpad = 10.0f * s;
        const float medCx = x0 + cpad + medR;
        const float medCy = cardTop + cpad + medR;
        addMedallion(state.hex.iconName,
                     UiRect{medCx - medR, medCy - medR, medCx + medR, medCy + medR});
        const float tx = medCx + medR + 10.0f * s;
        addLabel("<b>" + state.hex.name + "</b>", kText, fonts_, UiTextAlign::Left,
                 UiRect::fromXYWH(tx, cardTop + cpad, rect.maxX - pad - tx, boldH));
        // Yields row beneath the name.
        float yx = tx;
        const float yieldY = cardTop + cpad + boldH + 4.0f * s;
        const float chipIcon = capH;
        for (const Yield& yld : state.hex.yields) {
            UiIconEntry icon{};
            if (!yld.iconName.empty() && UiIconRegistry::global().resolve(yld.iconName, icon)) {
                auto img = std::make_unique<Image>(icon.textureId);
                img->uvRect = icon.uv;
                img->setRect(UiRect::fromXYWH(yx, yieldY, chipIcon, chipIcon));
                addChild(std::move(img));
                yx += chipIcon + 3.0f * s;
            }
            const float vw = rf ? rf->measureText(yld.value) + 6.0f * s : 24.0f * s;
            addLabel(yld.value, kText, fonts_, UiTextAlign::Left,
                     UiRect::fromXYWH(yx, yieldY, vw, capH));
            yx += vw + 8.0f * s;
        }
        y = cardTop + cardH + 10.0f * s;
    }

    // --- Recommended action card --------------------------------------------
    if (!state.action.title.empty()) {
        addCaption("RECOMMENDED ACTION", y);
        const float descH = regH * 2.0f;
        const float btn   = 30.0f * s;
        const bool hasBtns = !state.action.actions.empty();
        const float cardH = 10.0f * s + boldH + 4.0f * s + descH
                            + (hasBtns ? 8.0f * s + btn : 0.0f) + 10.0f * s;
        const float cardTop = y;
        addCardBg(cardTop, cardH);
        const float cpad = 10.0f * s;
        float cy = cardTop + cpad;
        addLabel("<b>" + state.action.title + "</b>", kGold, fonts_, UiTextAlign::Left,
                 UiRect::fromXYWH(x0 + cpad, cy, wIn - 2.0f * cpad, boldH));
        cy += boldH + 4.0f * s;
        addLabel(state.action.description, kDesc, fonts_, UiTextAlign::Left,
                 UiRect::fromXYWH(x0 + cpad, cy, wIn - 2.0f * cpad, descH));
        cy += descH + 8.0f * s;
        float bx = x0 + cpad;
        for (const Action& a : state.action.actions) {
            UiIconEntry icon{};
            const bool resolved = !a.iconName.empty()
                                  && UiIconRegistry::global().resolve(a.iconName, icon);
            auto ib = std::make_unique<IconButton>(a.onClick ? a.onClick : [] {});
            if (resolved) { ib->textureId = icon.textureId; ib->uvRect = icon.uv; }
            ib->cornerRadiusPx    = 4.0f * s;
            ib->borderThicknessPx = 1.5f * s;
            ib->iconPaddingPx     = 4.0f * s;
            ib->glowSizePx        = 10.0f * s;
            ib->setRect(UiRect::fromXYWH(bx, cy, btn, btn));
            addChild(std::move(ib));
            bx += btn + 6.0f * s;
        }
        y = cardTop + cardH + 10.0f * s;
    }

    // --- Unit card -----------------------------------------------------------
    if (state.hasUnit) {
        const float portR   = 22.0f * s;
        const float headTop = y;
        addMedallion(state.unit.portraitName,
                     UiRect{x0, headTop, x0 + portR * 2.0f, headTop + portR * 2.0f});
        const float tx = x0 + portR * 2.0f + 12.0f * s;
        addLabel("<b>" + state.unit.name + "</b>", kText, fonts_, UiTextAlign::Left,
                 UiRect::fromXYWH(tx, headTop + 2.0f * s, rect.maxX - pad - tx, boldH));
        addLabel(state.unit.klass, kDim, fonts_, UiTextAlign::Left,
                 UiRect::fromXYWH(tx, headTop + 2.0f * s + boldH + 2.0f * s,
                                  rect.maxX - pad - tx, capH));
        y = headTop + portR * 2.0f + 10.0f * s;

        addStatGrid(state.unit.primaryStats, y);
        addStatGrid(state.unit.combatStats, y);

        if (!state.unit.abilities.empty()) {
            addLabel(state.unit.abilities, kDesc, fonts_, UiTextAlign::Left,
                     UiRect::fromXYWH(x0, y, wIn, regH * 3.0f));
            y += regH * 3.0f + 8.0f * s;
        }
        if (!state.unit.settlementPreview.empty()) {
            addCaption("SETTLEMENT PREVIEW", y);
            addLabel(state.unit.settlementPreview, kDesc, fonts_, UiTextAlign::Left,
                     UiRect::fromXYWH(x0, y, wIn, regH * 5.0f));
            y += regH * 5.0f;
        }
    }
}

}  // namespace odai::ui
