#pragma once

#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <string>
#include <vector>

namespace odai::ui {

struct Stat {
    std::string iconName;    // Resolved from UiIconRegistry::global().
    std::string value;       // Displayed text (e.g. "12", "+3/turn").
    UiColor valueColor{0.92f, 0.92f, 0.92f, 1.0f};
};

// Horizontal row of (icon, value) pairs. Reusable for unit stat bars, city
// yield strips, etc. Icons are resolved from the global UiIconRegistry.
class StatBadgeRow : public Widget {
public:
    explicit StatBadgeRow(const Font* font) : font_(font) {}

    std::vector<Stat> stats;
    float iconSizePx = 20.0f;
    float gapPx = 4.0f;     // Icon-to-label spacing.
    float statGapPx = 12.0f; // Spacing between individual stats.

    UiColor backgroundColor{0.0f, 0.0f, 0.0f, 0.0f};  // Transparent by default.

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent&) override { return false; }

private:
    const Font* font_ = nullptr;
};

}  // namespace odai::ui
