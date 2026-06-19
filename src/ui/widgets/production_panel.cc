#include "ui/widgets/production_panel.h"

#include "ui/icon_atlas.h"
#include "ui/ui_draw_list.h"
#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"
#include "ui/widgets/scroll_view.h"

#include <string>
#include <utility>

namespace odai::ui {
namespace {

// A thin section-divider bar (e.g. "Districts & Buildings", "Units").
class SectionHeader : public Widget {
public:
    SectionHeader(const Font* font, std::string label, float s)
        : font_(font), label_(std::move(label)), s_(s) {}

    void draw(UiDrawList& dl) const override {
        dl.addRectFilled(rect_, UiColor{0.08f, 0.11f, 0.18f, 1.0f});
        // Top hairline separator
        dl.addRect(UiRect{rect_.minX, rect_.minY, rect_.maxX, rect_.minY + 1.0f},
                   UiColor{0.75f, 0.60f, 0.28f, 0.40f}, 1.0f);
        if (font_) {
            const float gy = rect_.minY + (rect_.height() - font_->lineHeightPx()) * 0.5f;
            dl.addText(*font_, label_, UiVec2{rect_.minX + 10.0f * s_, gy},
                       UiColor{0.80f, 0.70f, 0.45f, 1.0f});
        }
    }

private:
    const Font* font_ = nullptr;
    std::string label_;
    float s_ = 1.0f;
};

// A single production-queue row. Self-drawing, self-hit-testing. The parent
// ScrollView reassigns rects as it scrolls, so all layout is rect-relative.
class ProductionRow : public Widget {
public:
    ProductionRow(const FontSet& fonts, float scale) : fonts_(fonts), s_(scale) {}

    std::string id;
    std::string name;
    std::string info;  // "60 shields · 18 turns"
    UiTextureId iconTex = kUiNoTexture;
    UiRect iconUv{0.0f, 0.0f, 1.0f, 1.0f};
    bool selected = false;
    std::function<void()> onSelect;
    std::function<void()> onOpenPedia;

    void draw(UiDrawList& dl) const override {
        const float pad = 8.0f * s_;
        const float radius = 4.0f * s_;

        // Row background
        if (selected) {
            dl.addRoundRectFilled(rect_, UiColor{0.20f, 0.16f, 0.06f, 0.95f}, radius);
            dl.addRoundRect(rect_, UiColor{0.90f, 0.74f, 0.30f, 0.88f}, radius, 1.5f * s_);
        } else if (hovered_) {
            dl.addRoundRectFilled(rect_, UiColor{0.13f, 0.17f, 0.22f, 0.90f}, radius);
        }

        // Icon — fixed 44px, vertically centered
        const float iconSide = 44.0f * s_;
        const float iconY = rect_.minY + (rect_.height() - iconSide) * 0.5f;
        const UiRect iconRect = UiRect::fromXYWH(rect_.minX + pad, iconY, iconSide, iconSide);
        if (iconTex != kUiNoTexture) {
            dl.addImage(iconRect, iconTex, UiColor{1.0f, 1.0f, 1.0f, 1.0f}, iconUv);
        } else {
            dl.addRoundRectFilled(iconRect, UiColor{0.20f, 0.22f, 0.26f, 1.0f}, 3.0f * s_);
        }

        // CivPedia info button (compute before text so we know the right boundary)
        const UiRect pr = pediaRect();

        // Text region: from after icon to just before the info button
        const float textX = iconRect.maxX + 10.0f * s_;
        const float textMaxX = pr.minX - 8.0f * s_;

        dl.pushClip(UiRect{textX, rect_.minY, textMaxX, rect_.maxY});

        // Stack name + info vertically and centre the block within the row so
        // they never overlap regardless of font size.
        const Font* nameFont = fonts_.bold ? fonts_.bold : fonts_.regular;
        const float nameH = nameFont       ? nameFont->lineHeightPx()       : 0.0f;
        const float infoH = fonts_.regular ? fonts_.regular->lineHeightPx() : 0.0f;
        const float gap   = 3.0f * s_;
        const float blockH = nameH + (infoH > 0.0f ? gap + infoH : 0.0f);
        const float blockY = rect_.minY + (rect_.height() - blockH) * 0.5f;

        if (nameFont) {
            dl.addText(*nameFont, name, UiVec2{textX, blockY},
                       UiColor{0.94f, 0.88f, 0.72f, 1.0f});
        }
        if (fonts_.regular) {
            dl.addText(*fonts_.regular, info, UiVec2{textX, blockY + nameH + gap},
                       UiColor{0.60f, 0.68f, 0.78f, 1.0f});
        }

        dl.popClip();

        // Info button — circle with "i" glyph
        const UiColor fill = pediaHovered_ ? UiColor{0.30f, 0.46f, 0.68f, 1.0f}
                                           : UiColor{0.16f, 0.24f, 0.36f, 0.92f};
        dl.addRoundRectFilled(pr, fill, pr.width() * 0.5f);
        dl.addRoundRect(pr, UiColor{0.50f, 0.66f, 0.88f, 0.72f}, pr.width() * 0.5f, 1.2f * s_);
        if (fonts_.regular) {
            const float gw = fonts_.regular->measureText("i");
            const float gh = fonts_.regular->lineHeightPx();
            dl.addText(*fonts_.regular, "i",
                       UiVec2{pr.minX + (pr.width() - gw) * 0.5f,
                               pr.minY + (pr.height() - gh) * 0.5f},
                       UiColor{0.85f, 0.92f, 1.0f, 1.0f});
        }
    }

    bool onEvent(UiEvent& ev) override {
        switch (ev.type) {
            case UiEvent::Type::MouseMove:
                hovered_ = rect_.contains(ev.mousePx);
                pediaHovered_ = pediaRect().contains(ev.mousePx);
                return false;
            case UiEvent::Type::MouseDown:
                if (ev.button == UiMouseButton::Left && rect_.contains(ev.mousePx)) {
                    pressedPedia_ = pediaRect().contains(ev.mousePx);
                    pressedInside_ = true;
                    ev.handled = true;
                    return true;
                }
                return false;
            case UiEvent::Type::MouseUp: {
                if (ev.button != UiMouseButton::Left || !pressedInside_) return false;
                const bool inPedia = pediaRect().contains(ev.mousePx);
                const bool wasPedia = pressedPedia_;
                pressedInside_ = false;
                pressedPedia_ = false;
                if (inPedia && wasPedia) {
                    if (onOpenPedia) onOpenPedia();
                    ev.handled = true;
                    return true;
                }
                if (rect_.contains(ev.mousePx)) {
                    if (onSelect) onSelect();
                    ev.handled = true;
                    return true;
                }
                return false;
            }
            default:
                return false;
        }
    }

private:
    // Info button: 24px circle, 14px from the right edge, vertically centered.
    UiRect pediaRect() const {
        const float side   = 24.0f * s_;
        const float margin = 14.0f * s_;
        const float cy     = (rect_.minY + rect_.maxY) * 0.5f;
        const float x1     = rect_.maxX - margin - side;
        return UiRect{x1, cy - side * 0.5f, x1 + side, cy + side * 0.5f};
    }

    FontSet fonts_;
    float s_ = 1.0f;
    bool hovered_       = false;
    bool pediaHovered_  = false;
    bool pressedInside_ = false;
    bool pressedPedia_  = false;
};

// City info header: city name, yield badges, and governor. Drawn directly on
// top of the panel background — no separate background fill to preserve the
// panel's rounded corners.
class CityHeader : public Widget {
public:
    CityHeader(const FontSet& fonts, float s, const ProductionPanel::CityInfo& city)
        : fonts_(fonts), s_(s), name_(city.name),
          govName_(city.governorName.empty() ? "Unassigned" : city.governorName) {

        const char* kNames[6] = {"food","production","gold","science","faith","culture"};
        const int   kVals[6]  = {city.food, city.production, city.gold,
                                  city.science, city.faith, city.culture};
        for (int i = 0; i < 6; ++i) {
            Yield y{};
            y.label = std::to_string(kVals[i]);
            UiIconEntry entry{};
            if (UiIconRegistry::global().resolve(kNames[i], entry)) {
                y.tex = entry.textureId;
                y.uv  = entry.uv;
            }
            yields_.push_back(y);
        }
    }

    void draw(UiDrawList& dl) const override {
        const Font* bf  = fonts_.bold    ? fonts_.bold    : fonts_.regular;
        const Font* rf  = fonts_.regular;
        const Font* itf = fonts_.italic  ? fonts_.italic  : rf;

        float y = rect_.minY + 8.0f * s_;

        // City name — bold gold, centered
        if (bf && !name_.empty()) {
            const float nameW = bf->measureText(name_);
            dl.addText(*bf, name_,
                       UiVec2{rect_.minX + (rect_.width() - nameW) * 0.5f, y},
                       UiColor{1.0f, 0.88f, 0.48f, 1.0f});
            y += bf->lineHeightPx() + 7.0f * s_;
        }

        // Yield badges: [icon][value] × 6, centered as a group
        if (!yields_.empty()) {
            const float iconSide = 16.0f * s_;
            const float iconGap  = 2.0f * s_;
            const float itemGap  = 9.0f * s_;

            // Measure total width for centering
            float totalW = 0.0f;
            for (int i = 0; i < 6; ++i) {
                totalW += iconSide + iconGap + (rf ? rf->measureText(yields_[i].label) : 18.0f);
                if (i < 5) totalW += itemGap;
            }
            float x = rect_.minX + (rect_.width() - totalW) * 0.5f;

            const float rowH    = iconSide;
            const float textOffY = y + (rowH - (rf ? rf->lineHeightPx() : 0.0f)) * 0.5f;

            for (int i = 0; i < 6; ++i) {
                const Yield& yld = yields_[i];
                const UiRect ir{x, y, x + iconSide, y + iconSide};
                if (yld.tex != kUiNoTexture) {
                    dl.addImage(ir, yld.tex, UiColor{1.0f, 1.0f, 1.0f, 1.0f}, yld.uv);
                } else {
                    dl.addRoundRectFilled(ir, UiColor{0.35f, 0.38f, 0.42f, 0.6f}, 2.0f * s_);
                }
                x += iconSide + iconGap;
                if (rf) {
                    dl.addText(*rf, yld.label, UiVec2{x, textOffY},
                               UiColor{0.92f, 0.88f, 0.72f, 1.0f});
                    x += rf->measureText(yld.label) + itemGap;
                }
            }
            y += rowH + 7.0f * s_;
        }

        // Governor row — italic, muted blue-grey, centered
        if (itf) {
            const std::string govStr = "Governor:  " + govName_;
            const float gw = itf->measureText(govStr);
            dl.addText(*itf, govStr,
                       UiVec2{rect_.minX + (rect_.width() - gw) * 0.5f, y},
                       UiColor{0.60f, 0.68f, 0.80f, 0.88f});
        }
    }

private:
    struct Yield {
        std::string label;
        UiTextureId tex = kUiNoTexture;
        UiRect uv{0.0f, 0.0f, 1.0f, 1.0f};
    };

    FontSet fonts_;
    float s_;
    std::string name_;
    std::string govName_;
    std::vector<Yield> yields_;
};

}  // namespace

void ProductionPanel::setItems(const UiRect& rect, float s, const std::string& title,
                               const std::vector<Row>& rows,
                               const CityInfo city) {
    children_.clear();
    rows_.clear();
    title_ = title;
    setRect(rect);

    const float pad    = 10.0f * s;
    const float cityH  = 88.0f * s;   // name + yields + governor
    const float titleH = 30.0f * s;   // "Choose Production" label

    // Outer background panel
    auto bg = std::make_unique<Panel>();
    bg->setRect(rect);
    bg->background        = UiColor{0.07f, 0.09f, 0.13f, 0.97f};
    bg->borderColor       = UiColor{0.75f, 0.62f, 0.34f, 0.55f};
    bg->borderThicknessPx = 1.5f * s;
    bg->cornerRadiusPx    = 4.0f * s;
    bg->showShadow        = true;
    bg_ = static_cast<Panel*>(addChild(std::move(bg)));

    // City info header (name, yields, governor)
    auto cityHdr = std::make_unique<CityHeader>(fonts_, s, city);
    cityHdr->setRect(UiRect::fromXYWH(rect.minX, rect.minY, rect.width(), cityH));
    addChild(std::move(cityHdr));

    // Gold separator below city header
    auto sep1 = std::make_unique<Panel>();
    sep1->setRect(UiRect::fromXYWH(rect.minX + pad, rect.minY + cityH,
                                    rect.width() - 2.0f * pad, 1.5f * s));
    sep1->background        = UiColor{0.78f, 0.62f, 0.30f, 0.38f};
    sep1->borderThicknessPx = 0.0f;
    sep1->cornerRadiusPx    = 0.0f;
    addChild(std::move(sep1));

    // "Choose Production" title label — centred, two-line when a build is active
    const float titleY = rect.minY + cityH + 1.5f * s;
    auto tl = std::make_unique<Label>(
        fonts_, "<b><color=#ceb96a>" + title + "</color></b>");
    tl->align = UiTextAlign::Center;
    tl->setRect(UiRect::fromXYWH(rect.minX + pad, titleY + 4.0f * s,
                                  rect.width() - 2.0f * pad, titleH));
    titleLabel_ = static_cast<Label*>(addChild(std::move(tl)));

    // Gold separator line below title
    auto sep2 = std::make_unique<Panel>();
    sep2->setRect(UiRect::fromXYWH(rect.minX + pad, titleY + titleH + 2.0f * s,
                                    rect.width() - 2.0f * pad, 1.5f * s));
    sep2->background        = UiColor{0.78f, 0.62f, 0.30f, 0.40f};
    sep2->borderThicknessPx = 0.0f;
    sep2->cornerRadiusPx    = 0.0f;
    addChild(std::move(sep2));

    // Scrollable list
    const float listTop = titleY + titleH + 10.0f * s;
    auto sv = std::make_unique<ScrollView>();
    sv->setRect(UiRect::fromXYWH(rect.minX + 4.0f * s, listTop,
                                  rect.width() - 8.0f * s,
                                  rect.maxY - listTop - pad));
    sv->childGap       = 4.0f * s;
    sv->scrollBarColor = UiColor{0.72f, 0.58f, 0.30f, 0.55f};
    sv->scrollBarBg    = UiColor{0.0f, 0.0f, 0.0f, 0.20f};

    // Row height: tall enough to fit name + info stacked, with equal top/bottom
    // padding. Using font metrics means the row scales correctly with any DPI.
    const Font* nameFontRef = fonts_.bold ? fonts_.bold : fonts_.regular;
    const float nameLineH   = nameFontRef       ? nameFontRef->lineHeightPx()       : 28.0f * s;
    const float infoLineH   = fonts_.regular    ? fonts_.regular->lineHeightPx()    : 22.0f * s;
    const float rowPad      = 18.0f * s;  // total vertical whitespace (top + bottom)
    const float rowH        = std::max(nameLineH + infoLineH + 3.0f * s + rowPad,
                                       52.0f * s);  // never shorter than the icon
    const float sectionH = std::max(26.0f * s, infoLineH + 10.0f * s);
    const float rowW     = sv->rect().width();
    std::string currentSection;

    for (const Row& r : rows) {
        // Insert a section header when the section label changes
        if (r.section != currentSection && !r.section.empty()) {
            currentSection = r.section;
            const Font* sf = fonts_.bold ? fonts_.bold : fonts_.regular;
            auto hdr = std::make_unique<SectionHeader>(sf, r.section, s);
            hdr->setRect(UiRect::fromXYWH(0.0f, 0.0f, rowW, sectionH));
            sv->addChild(std::move(hdr));
        }

        auto row = std::make_unique<ProductionRow>(fonts_, s);
        row->setRect(UiRect::fromXYWH(0.0f, 0.0f, rowW, rowH));
        row->id       = r.id;
        row->name     = r.name;
        row->info     = std::to_string(r.productionCost) + " shields  \xC2\xB7  " +
                        std::to_string(r.turns) + (r.turns == 1 ? " turn" : " turns");
        row->selected = r.selected;
        row->onSelect     = r.onSelect;
        row->onOpenPedia  = r.onOpenPedia;

        UiIconEntry entry{};
        if (UiIconRegistry::global().resolve(r.iconName, entry)) {
            row->iconTex = entry.textureId;
            row->iconUv  = entry.uv;
        }
        rows_.push_back(row.get());
        sv->addChild(std::move(row));
    }
    list_ = static_cast<ScrollView*>(addChild(std::move(sv)));
}

void ProductionPanel::setSelected(const std::string& id) {
    std::string currentName;
    for (Widget* w : rows_) {
        auto* row = static_cast<ProductionRow*>(w);
        row->selected = (row->id == id);
        if (row->selected) currentName = row->name;
    }
    if (titleLabel_) {
        std::string markup = "<b><color=#ceb96a>" + title_ + "</color></b>";
        if (!currentName.empty()) {
            markup += "\n<color=#7a909e><i>Building: " + currentName + "</i></color>";
        }
        titleLabel_->setText(markup);
    }
}

}  // namespace odai::ui
