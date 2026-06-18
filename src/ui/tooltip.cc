#include "ui/tooltip.h"

#include "ui/ui_draw_list.h"
#include "ui/rich_text.h"

#include <algorithm>
#include <cmath>

namespace odai::ui {

void UiTooltipManager::requestTooltip(const UiRect& anchor, std::string markup,
                                       float delaySeconds) {
    if (m_hoveredThisFrame) return;  // First request wins per frame.
    m_hoveredThisFrame = true;
    if (m_primary.markup != markup) {
        m_primary.markup = std::move(markup);
        m_primary.anchor = anchor;
        m_primary.delaySeconds = delaySeconds;
        m_primary.elapsed = 0.0f;
        m_primary.showing = false;
        m_primary.fade = Tween{};
        m_primary.layoutDirty = true;
    } else {
        m_primary.anchor = anchor;
    }
}

void UiTooltipManager::clearTooltip() {
    m_hoveredThisFrame = false;
}

void UiTooltipManager::update(float dt) {
    if (!m_hoveredThisFrame) {
        // Start fade out.
        m_primary.fade.setTarget(0.0f);
        m_primary.fade.update(dt);
        if (m_primary.fade.value <= 0.0f && !m_primary.showing) {
            m_primary.markup.clear();
        }
        if (m_primary.fade.value <= 0.0f) {
            m_primary.showing = false;
        }
    } else if (!m_primary.showing) {
        m_primary.elapsed += dt;
        if (m_primary.elapsed >= m_primary.delaySeconds) {
            m_primary.showing = true;
            m_primary.fade.setTarget(1.0f);
        }
    } else {
        m_primary.fade.setTarget(1.0f);
        m_primary.fade.update(dt);
    }

    if (m_hoveredThisFrame && m_primary.showing) {
        m_primary.fade.update(dt);
    }

    // Reset hover flag; will be set again next frame by requestTooltip().
    m_hoveredThisFrame = false;
}

bool UiTooltipManager::isVisible() const {
    return m_primary.showing && m_primary.fade.value > 0.0f;
}

UiRect UiTooltipManager::positionTooltip(const UiRect& contentRect,
                                          const UiRect& anchor,
                                          const UiVec2& viewport) const {
    const float w = contentRect.width();
    const float h = contentRect.height();

    // Prefer placing below-right of the anchor.
    float x = anchor.maxX + 4.0f;
    float y = anchor.minY;

    // Flip left if overflowing right edge.
    if (x + w > viewport.x - 8.0f) {
        x = anchor.minX - w - 4.0f;
    }
    // Flip above if overflowing bottom.
    if (y + h > viewport.y - 8.0f) {
        y = anchor.maxY - h;
    }
    // Clamp to viewport.
    x = std::clamp(x, 8.0f, viewport.x - w - 8.0f);
    y = std::clamp(y, 8.0f, viewport.y - h - 8.0f);

    return UiRect{x, y, x + w, y + h};
}

void UiTooltipManager::draw(UiDrawList& dl, const UiVec2& viewport) const {
    if (!isVisible() || m_primary.markup.empty()) return;
    if (!m_fonts.regular) return;

    const float alpha = m_primary.fade.eased();

    // Lay out the tooltip text.
    if (m_primary.layoutDirty) {
        m_primary.layout = layoutRichText(m_primary.markup, m_textColor, m_fonts,
                                          m_maxWidth - 2.0f * kPadding,
                                          UiTextAlign::Left);
        m_primary.layoutDirty = false;
    }

    const float textW = m_primary.layout.width;
    const float textH = m_primary.layout.height;
    const float boxW = textW + 2.0f * kPadding;
    const float boxH = textH + 2.0f * kPadding;

    const UiRect contentRect{0, 0, boxW, boxH};
    const UiRect pos = positionTooltip(contentRect, m_primary.anchor, viewport);

    dl.pushOpacity(alpha);

    // Background.
    if (m_frame.has_value()) {
        dl.add9Slice(pos, m_frame.value());
    } else {
        UiColor bg = m_background;
        bg.a *= alpha;
        dl.addRectFilled(pos, bg);
    }

    // Text.
    const UiVec2 textOrigin{pos.minX + kPadding, pos.minY + kPadding};
    drawRichText(dl, m_primary.layout, m_fonts, textOrigin);

    dl.popOpacity();
}

}  // namespace odai::ui
