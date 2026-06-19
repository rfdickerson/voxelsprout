#include "ui/widgets/toolbar.h"

#include "ui/font.h"
#include "ui/icon_atlas.h"
#include "ui/ui_draw_list.h"

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
    UiIconEntry entry;
    const char* name = iconName(kind);
    if (name[0] != '\0' && UiIconRegistry::global().resolve(name, entry)) {
        dl.addImage(box, entry.textureId, UiColor{1.0f, 1.0f, 1.0f, color.a}, entry.uv);
        return;
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
