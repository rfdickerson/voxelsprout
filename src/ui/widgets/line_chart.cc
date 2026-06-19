#include "ui/widgets/line_chart.h"

#include "ui/ui_draw_list.h"

#include <algorithm>
#include <cmath>

namespace odai::ui {

UiRect LineChart::chartArea() const {
    return UiRect{rect_.minX + paddingPx, rect_.minY + paddingPx,
                  rect_.maxX - paddingPx, rect_.maxY - paddingPx};
}

float LineChart::valueToY(float v, float minV, float maxV, const UiRect& area) const {
    if (maxV <= minV) {
        return area.maxY;
    }
    const float t = (v - minV) / (maxV - minV);
    return area.maxY - t * area.height();
}

float LineChart::indexToX(int i, int count, const UiRect& area) const {
    if (count <= 1) {
        return area.minX + area.width() * 0.5f;
    }
    return area.minX + (static_cast<float>(i) / static_cast<float>(count - 1)) * area.width();
}

void LineChart::draw(UiDrawList& dl) const {
    dl.addRoundRectFilled(rect_, backgroundColor, 4.0f);
    dl.addRoundRect(rect_, borderColor, 4.0f, 1.0f);

    const UiRect area = chartArea();
    if (!area.valid()) {
        return;
    }

    // Compute auto range.
    float minV = minValue;
    float maxV = maxValue;
    if (autoRange) {
        bool first = true;
        for (const ChartSeries& s : series) {
            for (float val : s.values) {
                if (first) { minV = maxV = val; first = false; }
                else { minV = std::min(minV, val); maxV = std::max(maxV, val); }
            }
        }
        if (!std::isfinite(minV) || !std::isfinite(maxV) || minV == maxV) {
            minV -= 1.0f; maxV += 1.0f;
        }
    }

    dl.pushClip(area);

    // Horizontal grid lines.
    if (showGrid && gridLineCount > 0) {
        for (int g = 0; g <= gridLineCount; ++g) {
            const float t = static_cast<float>(g) / static_cast<float>(gridLineCount);
            const float y = area.minY + t * area.height();
            dl.addRectFilled(UiRect{area.minX, y - 0.5f, area.maxX, y + 0.5f}, gridColor);
        }
    }

    const float half = lineThicknessPx * 0.5f;
    for (const ChartSeries& s : series) {
        const int n = static_cast<int>(s.values.size());
        if (n < 2) {
            continue;
        }
        // Segments: connect consecutive points with a bounding rect. Works well
        // for the gentle slopes typical of turn-by-turn charts; dots cover gaps.
        for (int i = 0; i < n - 1; ++i) {
            const float x0 = indexToX(i, n, area);
            const float y0 = valueToY(s.values[i], minV, maxV, area);
            const float x1 = indexToX(i + 1, n, area);
            const float y1 = valueToY(s.values[i + 1], minV, maxV, area);
            dl.addRectFilled(
                UiRect{std::min(x0, x1), std::min(y0, y1) - half,
                       std::max(x0, x1) + 1.0f, std::max(y0, y1) + half},
                s.color);
        }
        // Dots at each sample.
        if (showDots) {
            for (int i = 0; i < n; ++i) {
                const float x = indexToX(i, n, area);
                const float y = valueToY(s.values[i], minV, maxV, area);
                dl.addCircleFilled(UiVec2{x, y}, dotRadiusPx, s.color);
            }
        }
    }

    dl.popClip();
    drawChildren(dl);
}

}  // namespace odai::ui
