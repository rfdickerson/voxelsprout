#pragma once

#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <string>
#include <vector>

namespace odai::ui {

struct DonutSegment {
    float fraction = 0.0f;  // Fraction of the whole in [0, 1]. Segments should sum to ~1.
    UiColor color;
    std::string label;
};

// Pie or donut chart rendered as filled arc sectors. Arcs are triangulated on
// CPU using addSectorFilled; no new GPU primitives are required.
// innerRadiusFraction = 0 → pie, 0.5 → donut (hole half the outer radius).
class DonutChart : public Widget {
public:
    explicit DonutChart(const Font* font = nullptr) : font_(font) {}

    std::vector<DonutSegment> segments;
    float innerRadiusFraction = 0.5f;
    std::string centerLabel;
    UiColor centerLabelColor{0.90f, 0.90f, 0.90f, 1.0f};
    UiColor hoverHighlightColor{1.0f, 1.0f, 1.0f, 0.14f};

    int hoveredSegment() const { return hoveredSeg_; }

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent& ev) override;

private:
    const Font* font_ = nullptr;
    mutable int hoveredSeg_ = -1;

    UiVec2 center() const;
    float outerRadiusPx() const;
    float innerRadiusPx() const;
};

}  // namespace odai::ui
