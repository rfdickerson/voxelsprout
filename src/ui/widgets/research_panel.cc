#include "ui/widgets/research_panel.h"

#include "ui/ui_draw_list.h"
#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"
#include "ui/widgets/progress_bar.h"
#include "ui/widgets/rich_text_view.h"
#include "ui/widgets/scroll_view.h"

#include <string>
#include <utility>

namespace odai::ui {
namespace {

using ItemState = ResearchPanel::ItemState;

// A thin category-divider bar ("Ancient Era", "Technology Tier 2", etc.).
class SectionHeader : public Widget {
public:
    SectionHeader(const Font* font, std::string label, float s)
        : font_(font), label_(std::move(label)), s_(s) {}

    void draw(UiDrawList& dl) const override {
        dl.addRectFilled(rect_, UiColor{0.08f, 0.11f, 0.18f, 1.0f});
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

// One selectable research item row. Self-drawing and self-hit-testing.
class ResearchRow : public Widget {
public:
    ResearchRow(const FontSet& fonts, float scale) : fonts_(fonts), s_(scale) {}

    std::string id;
    std::string name;
    std::string info;
    ItemState   baseState    = ItemState::Available;
    bool        showSelected = false;
    std::function<void()> onSelect;
    std::function<void()> onOpenDetail;

    [[nodiscard]] bool selectable() const { return baseState == ItemState::Available; }
    [[nodiscard]] ItemState displayState() const {
        if (selectable() && showSelected) return ItemState::Selected;
        return baseState;
    }

    void draw(UiDrawList& dl) const override {
        const float pad    = 10.0f * s_;
        const float radius = 4.0f * s_;
        const ItemState st = displayState();

        if (st == ItemState::Selected) {
            dl.addRoundRectFilled(rect_, UiColor{0.20f, 0.16f, 0.06f, 0.95f}, radius);
            dl.addRoundRect(rect_, UiColor{0.90f, 0.74f, 0.30f, 0.88f}, radius, 1.5f * s_);
        } else if (st == ItemState::Completed) {
            dl.addRoundRectFilled(rect_, UiColor{0.08f, 0.16f, 0.10f, 0.70f}, radius);
        } else if (hovered_ && selectable()) {
            dl.addRoundRectFilled(rect_, UiColor{0.13f, 0.17f, 0.22f, 0.90f}, radius);
        } else if (st == ItemState::Locked) {
            dl.addRoundRectFilled(rect_, UiColor{0.09f, 0.10f, 0.13f, 0.55f}, radius);
        }

        // Status pip on the left.
        const float pipR = 5.0f * s_;
        const float pipCx = rect_.minX + pad + pipR;
        const float pipCy = (rect_.minY + rect_.maxY) * 0.5f;
        UiColor pip{0.35f, 0.62f, 0.85f, 1.0f};
        if (st == ItemState::Completed) pip = UiColor{0.40f, 0.74f, 0.42f, 1.0f};
        else if (st == ItemState::Locked) pip = UiColor{0.42f, 0.44f, 0.50f, 1.0f};
        dl.addRoundRectFilled(UiRect{pipCx - pipR, pipCy - pipR, pipCx + pipR, pipCy + pipR},
                              pip, pipR);

        const UiRect pr = detailRect();

        const float textX    = pipCx + pipR + 10.0f * s_;
        const float textMaxX = pr.minX - 8.0f * s_;
        dl.pushClip(UiRect{textX, rect_.minY, textMaxX, rect_.maxY});

        const Font* nameFont = fonts_.bold ? fonts_.bold : fonts_.regular;
        const float nameH = nameFont       ? nameFont->lineHeightPx()       : 0.0f;
        const float infoH = fonts_.regular ? fonts_.regular->lineHeightPx() : 0.0f;
        const float gap   = 3.0f * s_;
        const float blockH = nameH + (infoH > 0.0f ? gap + infoH : 0.0f);
        const float blockY = rect_.minY + (rect_.height() - blockH) * 0.5f;

        UiColor nameCol{0.92f, 0.88f, 0.74f, 1.0f};
        UiColor infoCol{0.60f, 0.68f, 0.78f, 1.0f};
        if (st == ItemState::Completed) {
            nameCol = UiColor{0.62f, 0.80f, 0.62f, 1.0f};
            infoCol = UiColor{0.46f, 0.64f, 0.48f, 1.0f};
        } else if (st == ItemState::Locked) {
            nameCol = UiColor{0.56f, 0.59f, 0.66f, 1.0f};
            infoCol = UiColor{0.80f, 0.66f, 0.34f, 0.92f};
        }

        if (nameFont) dl.addText(*nameFont, name, UiVec2{textX, blockY}, nameCol);
        if (fonts_.regular)
            dl.addText(*fonts_.regular, info, UiVec2{textX, blockY + nameH + gap}, infoCol);

        dl.popClip();

        // Detail (i) button — circle with "i" glyph.
        const UiColor fill = detailHovered_ ? UiColor{0.30f, 0.46f, 0.68f, 1.0f}
                                            : UiColor{0.16f, 0.24f, 0.36f, 0.92f};
        dl.addRoundRectFilled(pr, fill, pr.width() * 0.5f);
        dl.addRoundRect(pr, UiColor{0.50f, 0.66f, 0.88f, 0.72f}, pr.width() * 0.5f, 1.2f * s_);
        if (fonts_.regular) {
            const float gw = fonts_.regular->measureText("i");
            const float gh = fonts_.regular->lineHeightPx();
            dl.addText(*fonts_.regular, "i",
                       UiVec2{pr.minX + (pr.width() - gw) * 0.5f, pr.minY + (pr.height() - gh) * 0.5f},
                       UiColor{0.85f, 0.92f, 1.0f, 1.0f});
        }
    }

    bool onEvent(UiEvent& ev) override {
        switch (ev.type) {
            case UiEvent::Type::MouseMove:
                hovered_       = rect_.contains(ev.mousePx);
                detailHovered_ = detailRect().contains(ev.mousePx);
                return false;
            case UiEvent::Type::MouseDown:
                if (ev.button == UiMouseButton::Left && rect_.contains(ev.mousePx)) {
                    pressedDetail_ = detailRect().contains(ev.mousePx);
                    pressedInside_ = true;
                    ev.handled = true;
                    return true;
                }
                return false;
            case UiEvent::Type::MouseUp: {
                if (ev.button != UiMouseButton::Left || !pressedInside_) return false;
                const bool inDetail  = detailRect().contains(ev.mousePx);
                const bool wasDetail = pressedDetail_;
                pressedInside_ = false;
                pressedDetail_ = false;
                if (inDetail && wasDetail) {
                    if (onOpenDetail) onOpenDetail();
                    ev.handled = true;
                    return true;
                }
                if (rect_.contains(ev.mousePx)) {
                    if (selectable() && onSelect) onSelect();
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
    UiRect detailRect() const {
        const float side   = 24.0f * s_;
        const float margin = 14.0f * s_;
        const float cy     = (rect_.minY + rect_.maxY) * 0.5f;
        const float x1     = rect_.maxX - margin - side;
        return UiRect{x1, cy - side * 0.5f, x1 + side, cy + side * 0.5f};
    }

    FontSet fonts_;
    float s_ = 1.0f;
    bool hovered_       = false;
    bool detailHovered_ = false;
    bool pressedInside_ = false;
    bool pressedDetail_ = false;
};

}  // namespace

void ResearchPanel::applyHeader(const ResearchProgress& progress) {
    if (currentLabel_) {
        const std::string title = progress.title.empty()
            ? std::string("<b><color=#ceb96a>Choose an item to research</color></b>")
            : "<color=#9fa8a8>Researching</color>  <b><color=#ceb96a>" + progress.title + "</color></b>";
        currentLabel_->setText(title);
    }
    if (progress_) progress_->value = progress.fraction;
    if (statusLabel_) statusLabel_->setText("<color=#7a909e>" + progress.status + "</color>");
    if (descView_) {
        descView_->setText(progress.description);
        descView_->scrollOffsetY = 0.0f;
    }
}

void ResearchPanel::applyProgress(const std::string& selectedId, const ResearchProgress& progress) {
    for (Widget* w : rows_) {
        auto* row = static_cast<ResearchRow*>(w);
        row->showSelected = (row->id == selectedId);
    }
    applyHeader(progress);
}

void ResearchPanel::setItems(const UiRect& rect, float s,
                             const std::vector<Row>& rows, const ResearchProgress& progress) {
    children_.clear();
    rows_.clear();
    setRect(rect);

    const float pad = 12.0f * s;

    auto bg = std::make_unique<Panel>();
    bg->setRect(rect);
    bg->background        = UiColor{0.07f, 0.09f, 0.13f, 0.92f};
    bg->borderColor       = UiColor{0.75f, 0.62f, 0.34f, 0.45f};
    bg->borderThicknessPx = 1.0f * s;
    bg->cornerRadiusPx    = 2.0f * s;
    bg_ = static_cast<Panel*>(addChild(std::move(bg)));

    const float x0   = rect.minX + pad;
    const float wIn  = rect.width() - 2.0f * pad;
    const Font* bf   = fonts_.bold ? fonts_.bold : fonts_.regular;
    const float boldH = bf             ? bf->lineHeightPx()             : 22.0f * s;
    const float regH  = fonts_.regular ? fonts_.regular->lineHeightPx() : 20.0f * s;

    float y = rect.minY + pad;

    auto cur = std::make_unique<Label>(fonts_, "");
    cur->setRect(UiRect::fromXYWH(x0, y, wIn, boldH));
    currentLabel_ = static_cast<Label*>(addChild(std::move(cur)));
    y += boldH + 6.0f * s;

    auto pb = std::make_unique<ProgressBar>();
    pb->setRect(UiRect::fromXYWH(x0, y, wIn, 16.0f * s));
    pb->foreground      = UiColor{0.33f, 0.66f, 0.86f, 1.0f};
    pb->background      = UiColor{0.05f, 0.07f, 0.10f, 0.95f};
    pb->borderColor     = UiColor{0.50f, 0.66f, 0.88f, 0.45f};
    pb->cornerRadiusPx  = 2.0f * s;
    progress_ = static_cast<ProgressBar*>(addChild(std::move(pb)));
    y += 16.0f * s + 4.0f * s;

    auto st = std::make_unique<Label>(fonts_, "");
    st->setRect(UiRect::fromXYWH(x0, y, wIn, regH));
    statusLabel_ = static_cast<Label*>(addChild(std::move(st)));
    y += regH + 8.0f * s;

    auto sep1 = std::make_unique<Panel>();
    sep1->setRect(UiRect::fromXYWH(x0, y, wIn, 1.5f * s));
    sep1->background = UiColor{0.78f, 0.62f, 0.30f, 0.38f};
    sep1->borderThicknessPx = 0.0f;
    sep1->cornerRadiusPx = 0.0f;
    addChild(std::move(sep1));
    y += 1.5f * s + 8.0f * s;

    const float descH = std::max(regH * 5.0f, 96.0f * s);
    auto desc = std::make_unique<RichTextView>(fonts_, "");
    desc->padding = {4.0f * s, 2.0f * s};
    desc->scrollBarThumbColor = UiColor{0.70f, 0.56f, 0.28f, 0.6f};
    desc->scrollBarTrackColor = UiColor{0.0f, 0.0f, 0.0f, 0.15f};
    desc->setRect(UiRect::fromXYWH(x0, y, wIn, descH));
    descView_ = static_cast<RichTextView*>(addChild(std::move(desc)));
    y += descH + 8.0f * s;

    auto sep2 = std::make_unique<Panel>();
    sep2->setRect(UiRect::fromXYWH(x0, y, wIn, 1.5f * s));
    sep2->background = UiColor{0.78f, 0.62f, 0.30f, 0.40f};
    sep2->borderThicknessPx = 0.0f;
    sep2->cornerRadiusPx = 0.0f;
    addChild(std::move(sep2));
    y += 1.5f * s + 8.0f * s;

    auto sv = std::make_unique<ScrollView>();
    sv->setRect(UiRect::fromXYWH(rect.minX + 4.0f * s, y,
                                 rect.width() - 8.0f * s, rect.maxY - y - pad));
    sv->childGap       = 4.0f * s;
    sv->scrollBarColor = UiColor{0.72f, 0.58f, 0.30f, 0.55f};
    sv->scrollBarBg    = UiColor{0.0f, 0.0f, 0.0f, 0.20f};

    const float nameLineH = bf             ? bf->lineHeightPx()             : 26.0f * s;
    const float rowH      = std::max(nameLineH + regH + 3.0f * s + 16.0f * s, 48.0f * s);
    const float sectionH  = std::max(26.0f * s, regH + 10.0f * s);
    const float rowW      = sv->rect().width();
    std::string currentSection;

    for (const Row& r : rows) {
        if (r.section != currentSection && !r.section.empty()) {
            currentSection = r.section;
            const Font* sf = fonts_.bold ? fonts_.bold : fonts_.regular;
            auto hdr = std::make_unique<SectionHeader>(sf, r.section, s);
            hdr->setRect(UiRect::fromXYWH(0.0f, 0.0f, rowW, sectionH));
            sv->addChild(std::move(hdr));
        }

        auto row = std::make_unique<ResearchRow>(fonts_, s);
        row->setRect(UiRect::fromXYWH(0.0f, 0.0f, rowW, rowH));
        row->id           = r.id;
        row->name         = r.name;
        row->info         = r.info;
        row->baseState    = (r.state == ItemState::Selected) ? ItemState::Available : r.state;
        row->showSelected = (r.state == ItemState::Selected);
        row->onSelect     = r.onSelect;
        row->onOpenDetail = r.onOpenDetail;
        rows_.push_back(row.get());
        sv->addChild(std::move(row));
    }
    list_ = static_cast<ScrollView*>(addChild(std::move(sv)));

    applyHeader(progress);
}

}  // namespace odai::ui
