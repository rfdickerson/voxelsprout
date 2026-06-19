#include "ui/widgets/donut_chart.h"

#include "ui/ui_draw_list.h"

#include <algorithm>
#include <cmath>

namespace odai::ui {

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float k2Pi = kPi * 2.0f;
constexpr float kHalfPi = kPi * 0.5f;
}  // namespace

UiVec2 DonutChart::center() const {
    return UiVec2{(rect_.minX + rect_.maxX) * 0.5f, (rect_.minY + rect_.maxY) * 0.5f};
}

float DonutChart::outerRadiusPx() const {
    return std::min(rect_.width(), rect_.height()) * 0.5f;
}

float DonutChart::innerRadiusPx() const {
    return outerRadiusPx() * innerRadiusFraction;
}

void DonutChart::draw(UiDrawList& dl) const {
    if (segments.empty()) {
        return;
    }
    const UiVec2 c = center();
    const float outerR = outerRadiusPx();
    const float innerR = innerRadiusPx();

    float angleStart = -kHalfPi;  // 12 o'clock.
    for (int i = 0; i < static_cast<int>(segments.size()); ++i) {
        const DonutSegment& seg = segments[i];
        if (seg.fraction <= 0.0f) {
            continue;
        }
        const float sweep = seg.fraction * k2Pi;
        const float angleEnd = angleStart + sweep;
        dl.addSectorFilled(c, innerR, outerR, angleStart, angleEnd, seg.color, 32);
        if (i == hoveredSeg_) {
            dl.addSectorFilled(c, innerR, outerR, angleStart, angleEnd, hoverHighlightColor, 32);
        }
        angleStart = angleEnd;
    }

    // Center label for donut style.
    if (font_ != nullptr && !centerLabel.empty() && innerRadiusFraction > 0.0f) {
        const float tw = font_->measureText(centerLabel);
        const float tx = c.x - tw * 0.5f;
        const float ty = c.y - font_->lineHeightPx() * 0.5f;
        dl.addText(*font_, centerLabel, UiVec2{tx, ty}, centerLabelColor);
    }

    drawChildren(dl);
}

bool DonutChart::onEvent(UiEvent& ev) {
    if (!visible || ev.type != UiEvent::Type::MouseMove) {
        return false;
    }
    hoveredSeg_ = -1;
    const UiVec2 c = center();
    const float dx = ev.mousePx.x - c.x;
    const float dy = ev.mousePx.y - c.y;
    const float dist = std::sqrt(dx * dx + dy * dy);
    const float outerR = outerRadiusPx();
    const float innerR = innerRadiusPx();
    if (dist < innerR || dist > outerR) {
        return false;
    }
    // Map mouse angle to a segment (start at 12 o'clock, clockwise).
    float angle = std::atan2(dy, dx) + kHalfPi;
    if (angle < 0.0f) {
        angle += k2Pi;
    }
    float cursor = 0.0f;
    for (int i = 0; i < static_cast<int>(segments.size()); ++i) {
        cursor += segments[i].fraction * k2Pi;
        if (angle <= cursor + 1e-4f) {
            hoveredSeg_ = i;
            break;
        }
    }
    return false;
}

}  // namespace odai::ui
