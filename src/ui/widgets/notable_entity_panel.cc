#include "ui/widgets/notable_entity_panel.h"

#include "ui/icon_atlas.h"
#include "ui/ui_draw_list.h"
#include "ui/widgets/button.h"
#include "ui/widgets/image.h"
#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"
#include "ui/widgets/rich_text_view.h"
#include "ui/widgets/scroll_view.h"

#include <algorithm>
#include <string>
#include <utility>

namespace odai::ui {
namespace {

// One selectable entity in the left rail.
class EntityRow : public Widget {
public:
    EntityRow(const FontSet& fonts, float scale) : fonts_(fonts), s_(scale) {}

    std::string id;
    std::string name;
    std::string subtitle;
    std::string portraitName;
    bool pending = false;
    bool showSelected = false;
    std::function<void()> onSelect;

    void draw(UiDrawList& dl) const override {
        const float pad    = 8.0f * s_;
        const float radius = 5.0f * s_;

        if (showSelected) {
            dl.addRoundRectFilled(rect_, UiColor{0.20f, 0.16f, 0.06f, 0.95f}, radius);
            dl.addRoundRect(rect_, UiColor{0.90f, 0.74f, 0.30f, 0.88f}, radius, 1.5f * s_);
        } else if (hovered_) {
            dl.addRoundRectFilled(rect_, UiColor{0.15f, 0.13f, 0.09f, 0.85f}, radius);
        } else {
            dl.addRoundRectFilled(rect_, UiColor{0.09f, 0.08f, 0.06f, 0.55f}, radius);
        }

        // Portrait thumbnail on the left.
        const float thumb = rect_.height() - pad * 2.0f;
        const UiRect pr{rect_.minX + pad, rect_.minY + pad,
                        rect_.minX + pad + thumb, rect_.minY + pad + thumb};
        UiIconEntry icon;
        if (!portraitName.empty() && UiIconRegistry::global().resolve(portraitName, icon)) {
            dl.addImage(pr, icon.textureId, UiColor{1, 1, 1, 1}, icon.uv);
        } else {
            dl.addRoundRectFilled(pr, UiColor{0.24f, 0.20f, 0.16f, 1.0f}, 3.0f * s_);
        }
        dl.addRoundRect(pr, UiColor{0.70f, 0.56f, 0.28f, 0.55f}, 3.0f * s_, 1.0f * s_);

        const float textX    = pr.maxX + 8.0f * s_;
        const float dotW     = pending ? 20.0f * s_ : 0.0f;
        const float textMaxX = rect_.maxX - pad - dotW;
        const Font* nameFont = fonts_.bold ? fonts_.bold : fonts_.regular;
        const float nameH = nameFont       ? nameFont->lineHeightPx()       : 0.0f;
        const float infoH = fonts_.regular ? fonts_.regular->lineHeightPx() : 0.0f;
        const float gap   = 2.0f * s_;
        const float blockH = nameH + (infoH > 0.0f ? gap + infoH : 0.0f);
        const float blockY = rect_.minY + (rect_.height() - blockH) * 0.5f;

        dl.pushClip(UiRect{textX, rect_.minY, textMaxX, rect_.maxY});
        if (nameFont)
            dl.addText(*nameFont, name, UiVec2{textX, blockY}, UiColor{0.92f, 0.88f, 0.74f, 1.0f});
        if (fonts_.regular)
            dl.addText(*fonts_.regular, subtitle, UiVec2{textX, blockY + nameH + gap},
                       UiColor{0.62f, 0.58f, 0.46f, 1.0f});
        dl.popClip();

        // "Pending" amber dot on the right.
        if (pending) {
            const float r  = 6.0f * s_;
            const float cx = rect_.maxX - pad - r;
            const float cy = (rect_.minY + rect_.maxY) * 0.5f;
            dl.addRoundRectFilled(UiRect{cx - r, cy - r, cx + r, cy + r},
                                  UiColor{0.92f, 0.74f, 0.26f, 1.0f}, r);
        }
    }

    bool onEvent(UiEvent& ev) override {
        switch (ev.type) {
            case UiEvent::Type::MouseMove:
                hovered_ = rect_.contains(ev.mousePx);
                return false;
            case UiEvent::Type::MouseDown:
                if (ev.button == UiMouseButton::Left && rect_.contains(ev.mousePx)) {
                    pressedInside_ = true;
                    ev.handled = true;
                    return true;
                }
                return false;
            case UiEvent::Type::MouseUp: {
                if (ev.button != UiMouseButton::Left || !pressedInside_) return false;
                pressedInside_ = false;
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
    FontSet fonts_;
    float s_ = 1.0f;
    bool hovered_ = false;
    bool pressedInside_ = false;
};

}  // namespace

void NotableEntityPanel::applyHeaderAndBody(const Detail& detail) {
    if (portrait_ != nullptr) {
        UiIconEntry icon;
        if (!detail.portraitName.empty() &&
            UiIconRegistry::global().resolve(detail.portraitName, icon)) {
            portrait_->textureId = icon.textureId;
            portrait_->uvRect    = icon.uv;
            portrait_->tint      = UiColor{1, 1, 1, 1};
        } else {
            portrait_->textureId = kUiNoTexture;
            portrait_->uvRect    = UiRect{0.0f, 0.0f, 1.0f, 1.0f};
            portrait_->tint      = UiColor{0.24f, 0.20f, 0.16f, 1.0f};
        }
    }
    if (nameLabel_ != nullptr)
        nameLabel_->setText("<b><color=#e6d6a4>" + detail.name + "</color></b>");
    if (titleLabel_ != nullptr)
        titleLabel_->setText("<color=#a89878>" + detail.title + "</color>");
    if (bodyView_ != nullptr) {
        bodyView_->setText(detail.body);
        bodyView_->scrollOffsetY = 0.0f;
    }
    action_ = detail.onAction;
    if (actionBtn_ != nullptr) {
        actionBtn_->visible = detail.showAction;
        actionBtn_->setEnabled(detail.actionEnabled);
        actionBtn_->setLabel(detail.actionLabel.empty() ? std::string("Use ability")
                                                        : detail.actionLabel);
    }
}

void NotableEntityPanel::applyDetail(const std::string& selectedId, const Detail& detail) {
    for (Widget* w : rows_) {
        auto* row = static_cast<EntityRow*>(w);
        row->showSelected = (row->id == selectedId);
    }
    applyHeaderAndBody(detail);
}

void NotableEntityPanel::setEntries(const UiRect& rect, float s,
                                    const std::vector<Entry>& entries, const Detail& detail) {
    children_.clear();
    rows_.clear();
    scale_ = s;
    setRect(rect);

    const float pad = 12.0f * s;

    auto bg = std::make_unique<Panel>();
    bg->setRect(rect);
    bg->background        = UiColor{0.07f, 0.06f, 0.04f, 0.92f};
    bg->borderColor       = UiColor{0.75f, 0.62f, 0.34f, 0.45f};
    bg->borderThicknessPx = 1.0f * s;
    bg->cornerRadiusPx    = 2.0f * s;
    bg_ = static_cast<Panel*>(addChild(std::move(bg)));

    const Font* bf    = fonts_.bold ? fonts_.bold : fonts_.regular;
    const float boldH = bf             ? bf->lineHeightPx()             : 22.0f * s;
    const float regH  = fonts_.regular ? fonts_.regular->lineHeightPx() : 20.0f * s;

    // Left rail
    const float railW = 200.0f * s;
    auto sv = std::make_unique<ScrollView>();
    sv->setRect(UiRect::fromXYWH(rect.minX + pad, rect.minY + pad, railW,
                                 rect.height() - 2.0f * pad));
    sv->childGap       = 6.0f * s;
    sv->scrollBarColor = UiColor{0.72f, 0.58f, 0.30f, 0.55f};
    sv->scrollBarBg    = UiColor{0.0f, 0.0f, 0.0f, 0.20f};

    const float rowH = 56.0f * s;
    const float rowW = sv->rect().width();
    for (const Entry& e : entries) {
        auto row = std::make_unique<EntityRow>(fonts_, s);
        row->setRect(UiRect::fromXYWH(0.0f, 0.0f, rowW, rowH));
        row->id           = e.id;
        row->name         = e.name;
        row->subtitle     = e.subtitle;
        row->portraitName = e.portraitName;
        row->pending      = e.pending;
        row->showSelected = (e.id == detail.id);
        row->onSelect     = e.onSelect;
        rows_.push_back(row.get());
        sv->addChild(std::move(row));
    }
    rail_ = static_cast<ScrollView*>(addChild(std::move(sv)));

    // Right detail pane
    const float detailX = rect.minX + pad + railW + pad;
    const float portSz  = 96.0f * s;
    float y = rect.minY + pad;

    auto portrait = std::make_unique<Image>();
    portrait->setRect(UiRect::fromXYWH(detailX, y, portSz, portSz));
    portrait->fitMode = Image::FitMode::Stretch;
    portrait_ = static_cast<Image*>(addChild(std::move(portrait)));

    const float headTextX = detailX + portSz + 12.0f * s;
    const float headTextW = rect.maxX - pad - headTextX;
    auto nameL = std::make_unique<Label>(fonts_, "");
    nameL->setRect(UiRect::fromXYWH(headTextX, y + 10.0f * s, headTextW, boldH));
    nameLabel_ = static_cast<Label*>(addChild(std::move(nameL)));

    auto titleL = std::make_unique<Label>(fonts_, "");
    titleL->setRect(UiRect::fromXYWH(headTextX, y + 10.0f * s + boldH + 2.0f * s, headTextW, regH));
    titleLabel_ = static_cast<Label*>(addChild(std::move(titleL)));

    y += portSz + 10.0f * s;

    const float detailW = rect.maxX - pad - detailX;
    auto sep = std::make_unique<Panel>();
    sep->setRect(UiRect::fromXYWH(detailX, y, detailW, 1.5f * s));
    sep->background = UiColor{0.78f, 0.62f, 0.30f, 0.40f};
    sep->borderThicknessPx = 0.0f;
    sep->cornerRadiusPx = 0.0f;
    addChild(std::move(sep));
    y += 1.5f * s + 8.0f * s;

    const float btnH = 34.0f * s;
    const float btnBand = btnH + 10.0f * s;
    const float bodyBottom = rect.maxY - pad - btnBand;

    auto body = std::make_unique<RichTextView>(fonts_, "");
    body->padding = {4.0f * s, 2.0f * s};
    body->scrollBarThumbColor = UiColor{0.70f, 0.56f, 0.28f, 0.6f};
    body->scrollBarTrackColor = UiColor{0.0f, 0.0f, 0.0f, 0.15f};
    body->setRect(UiRect::fromXYWH(detailX, y, detailW, std::max(0.0f, bodyBottom - y)));
    bodyView_ = static_cast<RichTextView*>(addChild(std::move(body)));

    const Font* btnFont = fonts_.bold ? fonts_.bold : fonts_.regular;
    auto btn = std::make_unique<Button>(btnFont, "Use ability", [this]() {
        if (action_) action_();
    });
    btn->setRect(UiRect::fromXYWH(detailX, rect.maxY - pad - btnH, detailW, btnH));
    btn->cornerRadiusPx = 2.0f * s;
    btn->borderThicknessPx = 1.0f * s;
    actionBtn_ = static_cast<Button*>(addChild(std::move(btn)));

    applyHeaderAndBody(detail);
}

}  // namespace odai::ui
