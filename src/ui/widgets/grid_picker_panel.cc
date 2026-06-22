#include "ui/widgets/grid_picker_panel.h"

#include "ui/icon_atlas.h"
#include "ui/ui_draw_list.h"
#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"
#include "ui/widgets/scroll_view.h"

#include <memory>
#include <string>
#include <utility>

namespace odai::ui {
namespace {

// One cell in the grid: icon + label beneath it.
class GridCell : public Widget {
public:
    GridCell(const Font* font, float scale) : font_(font), s_(scale) {}

    std::string id;
    std::string label;
    UiTextureId iconTex = kUiNoTexture;
    UiRect iconUv{0.0f, 0.0f, 1.0f, 1.0f};
    bool enabled  = true;
    bool selected = false;
    std::function<void()> onClick;

    void draw(UiDrawList& dl) const override {
        const float radius = 4.0f * s_;
        const float alpha  = enabled ? 1.0f : 0.40f;

        if (selected) {
            dl.addRoundRectFilled(rect_, UiColor{0.20f, 0.16f, 0.06f, 0.95f * alpha}, radius);
            dl.addRoundRect(rect_, UiColor{0.90f, 0.74f, 0.30f, 0.88f * alpha}, radius, 1.5f * s_);
        } else if (hovered_ && enabled) {
            dl.addRoundRectFilled(rect_, UiColor{0.16f, 0.19f, 0.24f, 0.90f}, radius);
        } else {
            dl.addRoundRectFilled(rect_, UiColor{0.09f, 0.10f, 0.13f, 0.60f}, radius);
        }

        const float pad = 6.0f * s_;
        const float labelH = font_ ? font_->lineHeightPx() : 14.0f * s_;
        const float iconSize = rect_.height() - 2.0f * pad - labelH - 2.0f * s_;
        const float iconX    = rect_.minX + (rect_.width() - iconSize) * 0.5f;
        const float iconY    = rect_.minY + pad;

        if (iconTex != kUiNoTexture) {
            dl.addImage(UiRect::fromXYWH(iconX, iconY, iconSize, iconSize),
                        iconTex, UiColor{1, 1, 1, alpha}, iconUv);
        } else {
            dl.addRoundRectFilled(UiRect::fromXYWH(iconX, iconY, iconSize, iconSize),
                                  UiColor{0.26f, 0.28f, 0.34f, 0.8f * alpha}, 3.0f * s_);
        }

        if (font_ && !label.empty()) {
            const float lw  = font_->measureText(label);
            const float lx  = rect_.minX + (rect_.width() - lw) * 0.5f;
            const float ly  = rect_.minY + pad + iconSize + 2.0f * s_;
            dl.addText(*font_, label, UiVec2{lx, ly},
                       UiColor{0.82f, 0.78f, 0.64f, alpha});
        }
    }

    bool onEvent(UiEvent& ev) override {
        switch (ev.type) {
            case UiEvent::Type::MouseMove:
                hovered_ = rect_.contains(ev.mousePx);
                return false;
            case UiEvent::Type::MouseDown:
                if (enabled && ev.button == UiMouseButton::Left && rect_.contains(ev.mousePx)) {
                    pressedInside_ = true;
                    ev.handled = true;
                    return true;
                }
                return false;
            case UiEvent::Type::MouseUp:
                if (ev.button != UiMouseButton::Left || !pressedInside_) return false;
                pressedInside_ = false;
                if (enabled && rect_.contains(ev.mousePx) && onClick) {
                    onClick();
                    ev.handled = true;
                    return true;
                }
                return false;
            default:
                return false;
        }
    }

private:
    const Font* font_ = nullptr;
    float s_ = 1.0f;
    bool hovered_ = false;
    bool pressedInside_ = false;
};

// Category section header row.
class CategoryHeader : public Widget {
public:
    CategoryHeader(const Font* font, std::string label, float s)
        : font_(font), label_(std::move(label)), s_(s) {}

    void draw(UiDrawList& dl) const override {
        dl.addRectFilled(rect_, UiColor{0.08f, 0.10f, 0.15f, 1.0f});
        dl.addRect(UiRect{rect_.minX, rect_.minY, rect_.maxX, rect_.minY + 1.0f},
                   UiColor{0.60f, 0.50f, 0.24f, 0.40f}, 1.0f);
        if (font_) {
            const float gy = rect_.minY + (rect_.height() - font_->lineHeightPx()) * 0.5f;
            dl.addText(*font_, label_, UiVec2{rect_.minX + 10.0f * s_, gy},
                       UiColor{0.75f, 0.65f, 0.40f, 1.0f});
        }
    }

private:
    const Font* font_ = nullptr;
    std::string label_;
    float s_ = 1.0f;
};

// A horizontal row of N cells inside the ScrollView.
class GridRow : public Widget {
public:
    void draw(UiDrawList&) const override { /* children draw themselves */ }
};

}  // namespace

void GridPickerPanel::setItems(const UiRect& rect, float s, int columns,
                               const std::vector<GridCategory>& categories) {
    children_.clear();
    cells_.clear();
    setRect(rect);
    if (columns < 1) columns = 1;

    auto bg = std::make_unique<Panel>();
    bg->setRect(rect);
    bg->styleCard(s);
    bg_ = static_cast<Panel*>(addChild(std::move(bg)));

    const float pad     = 8.0f * s;
    const Font* font    = fonts_.regular;
    const Font* hdrFont = fonts_.bold ? fonts_.bold : fonts_.regular;
    const float regH    = font ? font->lineHeightPx() : 14.0f * s;
    const float hdrH    = std::max(22.0f * s, regH + 8.0f * s);
    const float gap     = 4.0f * s;
    const float cellW   = (rect.width() - 2.0f * pad - gap * (columns - 1)) / columns;
    const float cellH   = cellW + regH + 4.0f * s;  // square icon + label row

    auto sv = std::make_unique<ScrollView>();
    sv->setRect(UiRect::fromXYWH(rect.minX + pad, rect.minY + pad,
                                 rect.width() - 2.0f * pad,
                                 rect.height() - 2.0f * pad));
    sv->childGap       = 0.0f;
    sv->scrollBarColor = UiColor{0.60f, 0.48f, 0.26f, 0.55f};
    sv->scrollBarBg    = UiColor{0.0f, 0.0f, 0.0f, 0.20f};
    scroll_ = static_cast<ScrollView*>(addChild(std::move(sv)));

    const float rowW = scroll_->rect().width();

    for (const GridCategory& cat : categories) {
        if (!cat.label.empty()) {
            auto hdr = std::make_unique<CategoryHeader>(hdrFont, cat.label, s);
            hdr->setRect(UiRect::fromXYWH(0.0f, 0.0f, rowW, hdrH));
            scroll_->addChild(std::move(hdr));
        }

        // Lay items into rows of `columns` cells each.
        for (std::size_t i = 0; i < cat.items.size();) {
            auto row = std::make_unique<GridRow>();
            row->setRect(UiRect::fromXYWH(0.0f, 0.0f, rowW, cellH));

            for (int col = 0; col < columns && i < cat.items.size(); ++col, ++i) {
                const GridItem& item = cat.items[i];
                const float cx = static_cast<float>(col) * (cellW + gap);

                auto cell = std::make_unique<GridCell>(font, s);
                cell->setRect(UiRect::fromXYWH(cx, 0.0f, cellW, cellH));
                cell->id       = item.id;
                cell->label    = item.label;
                cell->enabled  = item.enabled;
                cell->selected = item.selected;
                cell->onClick  = item.onClick;

                UiIconEntry icon{};
                if (UiIconRegistry::global().resolve(item.iconName, icon)) {
                    cell->iconTex = icon.textureId;
                    cell->iconUv  = icon.uv;
                }
                cells_.push_back(cell.get());
                row->addChild(std::move(cell));
            }
            scroll_->addChild(std::move(row));
        }
    }
}

void GridPickerPanel::setSelected(const std::string& id) {
    for (Widget* w : cells_) {
        auto* cell = static_cast<GridCell*>(w);
        cell->selected = (cell->id == id);
    }
}

}  // namespace odai::ui
