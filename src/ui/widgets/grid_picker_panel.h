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
class ScrollView;

// Scrollable N-column icon grid for selecting buildable / placeable items,
// grouped into named categories with dividers.
//
// The SimCity zone-picker toolbar, the Age of Empires build menu, the RimWorld
// architect tab, the Civ district/improvement chooser — all share this pattern.
//
// Each item shows an icon (from the UiIconRegistry), a short label beneath it,
// and a tooltip on hover. Disabled items are shown greyed-out. The active item
// gets a selection highlight.
//
// Pure UI (Vulkan-free, no game types).
class GridPickerPanel : public Widget {
public:
    struct GridItem {
        std::string id;
        std::string iconName;   // UiIconRegistry name.
        std::string label;      // Short name shown beneath the icon.
        std::string tooltip;    // Shown on hover.
        bool enabled  = true;
        bool selected = false;
        std::function<void()> onClick;
    };

    struct GridCategory {
        std::string            label;  // Section header ("Buildings", "Zones", "Infrastructure").
        std::vector<GridItem>  items;
    };

    explicit GridPickerPanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the grid into `rect` at DPI scale `s` with `columns` cells per row.
    void setItems(const UiRect& rect, float s, int columns,
                  const std::vector<GridCategory>& categories);

    // Update the selection highlight without rebuilding.
    void setSelected(const std::string& id);

private:
    FontSet fonts_;
    Panel* bg_ = nullptr;
    ScrollView* scroll_ = nullptr;
    std::vector<Widget*> cells_;  // Raw pointers into the ScrollView for re-highlight.
};

}  // namespace odai::ui
