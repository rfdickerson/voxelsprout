#pragma once

#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <string>
#include <vector>

namespace odai::ui {

class Panel;
class Label;

// Generic event / status tracker panel: a left-rail list of notable events,
// active effects, or world-state items. Each entry shows a circular medallion
// icon, a bold title, a dim subtitle, and a wrapped description.
//
// Originally WorldTrackerPanel. Suitable for: 4X world tracker (research,
// wonders, great people), city-builder alert log, colony-sim event feed, etc.
//
// Pure UI (Vulkan-free, no game types): the app maps its game state into the
// Entry structs below. Icon names are resolved against the global UiIconRegistry.
class EventTrackerPanel : public Widget {
public:
    struct Entry {
        std::string iconName;     // UiIconRegistry name; "" → blank medallion.
        std::string title;        // Bold primary line.
        std::string subtitle;     // Small dim caps line (category / type label).
        std::string description;  // Wrapped detail line.
    };

    explicit EventTrackerPanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the whole panel. `heading` is drawn at the top; entries follow.
    void setEntries(const UiRect& rect, float s, std::string_view heading,
                    const std::vector<Entry>& entries);

    // Expose the background Panel for per-game styling overrides.
    Panel* bgPanel() const { return bg_; }

private:
    FontSet fonts_;
    Panel* bg_ = nullptr;
};

}  // namespace odai::ui
