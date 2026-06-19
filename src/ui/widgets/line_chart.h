#pragma once

#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <string>
#include <vector>

namespace odai::ui {

struct ChartSeries {
    std::vector<float> values;
    UiColor color{0.50f, 0.80f, 0.50f, 1.0f};
    std::string label;
};

// Multi-series polyline trend chart. Values are normalized to [minValue, maxValue]
// and drawn as connected data points within the widget rect.
class LineChart : public Widget {
public:
    explicit LineChart(const Font* font = nullptr) : font_(font) {}

    std::vector<ChartSeries> series;

    float minValue = 0.0f;
    float maxValue = 100.0f;
    bool autoRange = true;  // Derive min/max from data each frame.
    bool showGrid = true;
    bool showDots = true;
    int gridLineCount = 4;
    float lineThicknessPx = 2.0f;
    float dotRadiusPx = 3.0f;

    UiColor gridColor{0.25f, 0.25f, 0.25f, 0.55f};
    UiColor backgroundColor{0.10f, 0.10f, 0.10f, 0.85f};
    UiColor borderColor{0.20f, 0.20f, 0.20f, 1.0f};

    float paddingPx = 6.0f;

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent&) override { return false; }

private:
    const Font* font_ = nullptr;

    UiRect chartArea() const;
    float valueToY(float v, float minV, float maxV, const UiRect& area) const;
    float indexToX(int i, int count, const UiRect& area) const;
};

}  // namespace odai::ui
