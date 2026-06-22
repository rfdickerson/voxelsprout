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

// Generic build / production queue panel: a scrollable list of items that can
// be queued for construction, training, or research in the selected location.
// Each row shows an icon, item name, cost, time estimate, and an optional detail
// link. A location header at the top summarises name, user-supplied resource
// stats (as icon + value pairs), and an optional manager/governor field.
//
// Originally ProductionPanel. Suitable for: 4X city queues (buildings, units),
// city-builder construction menus, RTS production buildings, etc.
//
// Composite widget: owns a background Panel, a title Label, and a ScrollView of
// self-laying-out rows. Vulkan-free; participates in the normal widget tree.
class BuildQueuePanel : public Widget {
public:
    // One row in the scrollable list.
    struct Row {
        std::string id;
        std::string name;
        std::string iconName;          // Resolved from the global UiIconRegistry.
        int cost = 0;                  // Generic build cost (shields, money, materials…).
        int turns = 0;                 // Pre-computed time estimate by the caller.
        bool selected = false;         // Currently the active build.
        std::string section;           // Section header label (e.g. "Units"). Header inserted on change.
        std::function<void()> onSelect;      // Row clicked → set as active build.
        std::function<void()> onOpenDetail;  // Link clicked → open detail / pedia.
    };

    // One icon+value stat shown in the location header row.
    struct LocationStat {
        std::string iconName;  // Resolved from UiIconRegistry.
        std::string value;     // Pre-formatted string, e.g. "47" or "+3".
    };

    // Summary info shown above the queue list.
    struct LocationInfo {
        std::string name;                    // e.g. "Rome", "District 4", "Barracks"
        std::vector<LocationStat> stats;     // Resource icons + values (any game-defined set).
        std::string managerName;             // "" = unassigned (governor, commander, etc.)
    };

    explicit BuildQueuePanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the panel into `rect` at DPI scale `s` with the given rows and location info.
    void setItems(const UiRect& rect, float s, const std::string& title,
                  const std::vector<Row>& rows,
                  const LocationInfo& location);

    // Highlight the row with this id as the active build and update the header.
    // No-op if no row matches.
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
