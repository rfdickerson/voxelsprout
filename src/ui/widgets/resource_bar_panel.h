#pragma once

#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>
#include <string>
#include <vector>

namespace odai::ui {

class Panel;
class Label;

// Top-of-screen HUD strip displaying N named resource slots. Each slot shows an
// icon, a formatted value label, and an optional tooltip + click callback.
//
// Suitable for any game that shows resources in a persistent HUD: 4X yields
// (food/gold/science…), city-builder finances (money/population/happiness),
// colony sims (food/materials/medicine), RTS resources (wood/stone/gold), etc.
//
// Pure UI (Vulkan-free, no game types). The app provides fully-formatted value
// strings; the panel handles layout and rendering only.
class ResourceBarPanel : public Widget {
public:
    struct ResourceEntry {
        std::string iconName;  // UiIconRegistry name.
        std::string value;     // Pre-formatted, e.g. "3,241" or "+47/turn".
        std::string tooltip;   // Shown on hover (via UiTooltipManager if set).
        UiColor     color     = UiColor{0.92f, 0.88f, 0.72f, 1.0f};
        std::function<void()> onClick;  // Optional: open detail panel.
    };

    explicit ResourceBarPanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the bar into `rect` at DPI scale `s`.
    void setResources(const UiRect& rect, float s,
                      const std::vector<ResourceEntry>& entries);

    // Expose the background Panel for per-game styling overrides.
    Panel* bgPanel() const { return bg_; }

private:
    FontSet fonts_;
    Panel* bg_ = nullptr;
};

}  // namespace odai::ui
