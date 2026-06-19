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

// Docked city production menu: a scrollable list of buildable units and
// buildings. Each row shows an icon, the item name, its production cost in
// shields and the turns-to-build, and a CivPedia link. Clicking a row selects
// it for production; clicking the link opens the CivPedia window.
//
// Composite widget: owns a background Panel, a title Label, and a ScrollView of
// self-laying-out rows. Vulkan-free; participates in the normal widget tree.
class ProductionPanel : public Widget {
public:
    struct Row {
        std::string id;
        std::string name;
        std::string iconName;          // Resolved from the global UiIconRegistry.
        int productionCost = 0;        // Shields required.
        int turns = 0;                 // Pre-computed by the caller.
        bool selected = false;         // Currently the city's active build.
        std::string section;           // Section header label (e.g. "Units"). Header inserted on change.
        std::function<void()> onSelect;     // Row clicked -> set production.
        std::function<void()> onOpenPedia;  // Link clicked -> open CivPedia.
    };

    struct CityInfo {
        std::string name;          // e.g. "Rome"
        int food = 0;
        int production = 0;
        int gold = 0;
        int science = 0;
        int faith = 0;
        int culture = 0;
        std::string governorName;  // "" = unassigned
    };

    explicit ProductionPanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the panel into `rect` at DPI scale `s` with the given rows and city info.
    void setItems(const UiRect& rect, float s, const std::string& title,
                  const std::vector<Row>& rows,
                  const CityInfo& city = {});

    // Highlight the row with this id as the city's active build and update the
    // header to name it. No-op if no row matches.
    void setSelected(const std::string& id);

private:
    FontSet fonts_;
    std::string title_;
    Panel* bg_ = nullptr;
    Label* titleLabel_ = nullptr;
    ScrollView* list_ = nullptr;
    std::vector<Widget*> rows_;  // Owned by the ScrollView; raw pointers for selection.
};

}  // namespace odai::ui
