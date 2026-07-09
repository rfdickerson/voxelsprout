#include "ui/widgets/toolbar.h"

#include "ui/font.h"
#include "ui/icon_atlas.h"
#include "ui/ui_draw_list.h"
#include "ui/vector/vector_icon_registry.h"

namespace odai::ui {

namespace {

UiColor scaleRgb(const UiColor& c, float f) {
    return UiColor{c.r * f, c.g * f, c.b * f, c.a};
}

UiColor mixWhite(const UiColor& c, float t) {
    return UiColor{c.r + (1.0f - c.r) * t, c.g + (1.0f - c.g) * t, c.b + (1.0f - c.b) * t, c.a};
}

const char* iconName(Toolbar::IconKind kind) {
    switch (kind) {
        case Toolbar::IconKind::Coin:
            return "gold";
        case Toolbar::IconKind::Science:
            return "science";
        case Toolbar::IconKind::Culture:
            return "culture";
        case Toolbar::IconKind::Faith:
            return "faith";
        case Toolbar::IconKind::Food:
            return "food";
        case Toolbar::IconKind::Production:
            return "production";
        case Toolbar::IconKind::Dot:
            return "";
    }
    return "";
}

}  // namespace

void Toolbar::drawIcon(UiDrawList& dl, IconKind kind, const UiColor& color, const UiRect& box) const {
    const char* name = iconName(kind);
    if (name[0] != '\0') {
        // Vector icons scale crisp at any DPI; check registry first.
        if (VectorIconRegistry::global().resolve(name) != nullptr) {
            dl.addVectorIcon(name, box);
            return;
        }
        // Fall back to PNG atlas.
        UiIconEntry entry;
        if (UiIconRegistry::global().resolve(name, entry)) {
            dl.addImage(box, entry.textureId, UiColor{1.0f, 1.0f, 1.0f, color.a}, entry.uv);
            return;
        }
    }

    const UiVec2 c{(box.minX + box.maxX) * 0.5f, (box.minY + box.maxY) * 0.5f};
    const float r = box.width() * 0.5f;
    switch (kind) {
        case IconKind::Coin:
            dl.addCircleFilled(c, r, color);
            dl.addCircle(c, r * 0.62f, scaleRgb(color, 0.55f), r * 0.16f);
            break;
        case IconKind::Science:  // disc with a bright bubble highlight
            dl.addCircleFilled(c, r, color);
            dl.addCircleFilled(UiVec2{c.x - r * 0.28f, c.y - r * 0.28f}, r * 0.30f, mixWhite(color, 0.7f));
            break;
        case IconKind::Faith:  // disc with a soft inner ring
            dl.addCircleFilled(c, r, color);
            dl.addCircle(c, r * 0.55f, mixWhite(color, 0.6f), r * 0.14f);
            break;
        case IconKind::Food:  // disc with a small leaf stem on top
            dl.addCircleFilled(c, r, color);
            dl.addRoundRectFilled(UiRect{c.x - r * 0.12f, box.minY - r * 0.22f, c.x + r * 0.12f, c.y - r * 0.3f},
                                  scaleRgb(color, 0.7f), r * 0.12f);
            break;
        case IconKind::Culture:  // rounded "tablet"
            dl.addRoundRectFilled(box, color, r * 0.45f);
            dl.addRoundRect(box, scaleRgb(color, 0.55f), r * 0.45f, r * 0.16f);
            break;
        case IconKind::Production:  // rounded square (cog-ish block)
            dl.addRoundRectFilled(box, color, r * 0.30f);
            dl.addCircleFilled(c, r * 0.30f, scaleRgb(color, 0.5f));
            break;
        case IconKind::Dot:
            dl.addCircleFilled(c, r * 0.7f, color);
            break;
    }
}

bool Toolbar::onEvent(UiEvent& event) {
    if (event.type == UiEvent::Type::MouseMove) {
        hoveredItem_ = -1;
        if (font_ != nullptr && rect_.contains(event.mousePx)) {
            const float iconSz = rect_.height() * iconScale;
            float x = rect_.minX + paddingXPx;
            for (int i = 0; i < static_cast<int>(items_.size()); ++i) {
                const float textW = font_->measureText(items_[i].value);
                const float badgeW = iconSz + iconGapPx + textW;
                const UiRect badgeRect{x, rect_.minY, x + badgeW, rect_.maxY};
                if (!items_[i].tooltip.empty() && badgeRect.contains(event.mousePx)) {
                    hoveredItem_ = i;
                    hoveredAnchor_ = {event.mousePx.x, rect_.maxY};
                    break;
                }
                x += badgeW + itemGapPx;
            }
        }
        for (const auto& child : children_) {
            if (child->visible) child->onEvent(event);
        }
        return false;
    }
    return Widget::onEvent(event);
}

void Toolbar::draw(UiDrawList& dl) const {
    if (!visible) {
        return;
    }

    // Background band.
    if (cornerRadiusPx > 0.0f) {
        dl.addRoundRectFilled(rect_, background, cornerRadiusPx);
    } else {
        dl.addRectFilled(rect_, background);
    }
    // Bottom accent rule (skipped for a rounded floating bar).
    if (accentThicknessPx > 0.0f && accentLine.a > 0.0f && cornerRadiusPx <= 0.0f) {
        dl.addRectFilled(UiRect{rect_.minX, rect_.maxY - accentThicknessPx, rect_.maxX, rect_.maxY},
                         accentLine);
    }

    if (font_ != nullptr) {
        const float h = rect_.height();
        const float iconSz = h * iconScale;
        const float iconTop = rect_.minY + (h - iconSz) * 0.5f;
        const float textY = rect_.minY + (h - font_->lineHeightPx()) * 0.5f;
        float x = rect_.minX + paddingXPx;

        dl.pushClip(rect_);
        for (const Item& item : items_) {
            const float textW = font_->measureText(item.value);
            const float badgeW = iconSz + iconGapPx + textW;

            if (showItemChips) {
                const UiRect chipRect{x - chipPadXPx, rect_.minY + chipInsetYPx,
                                      x + badgeW + chipPadXPx, rect_.maxY - chipInsetYPx};
                dl.addRoundRectFilled(chipRect, chipBackground, chipRadiusPx);
                if (chipBorder.a > 0.0f) {
                    dl.addRoundRect(chipRect, chipBorder, chipRadiusPx, 1.0f);
                }
                // Thin bottom trim tinted to this item's icon color — the one
                // touch of per-stat identity on an otherwise neutral chip.
                const UiColor trim{item.iconColor.r, item.iconColor.g, item.iconColor.b, chipTrimAlpha};
                const float trimInset = chipRadiusPx * 0.6f;
                dl.addRectFilled(UiRect{chipRect.minX + trimInset, chipRect.maxY - chipTrimThicknessPx,
                                        chipRect.maxX - trimInset, chipRect.maxY},
                                 trim);
            }

            const UiRect iconBox{x, iconTop, x + iconSz, iconTop + iconSz};
            drawIcon(dl, item.icon, item.iconColor, iconBox);
            x += iconSz + iconGapPx;
            const float advance = dl.addText(*font_, item.value, UiVec2{x, textY}, item.valueColor);
            x += advance + itemGapPx;
        }
        dl.popClip();
    }

    drawChildren(dl);
}

}  // namespace odai::ui
