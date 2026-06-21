#pragma once

#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <string>
#include <vector>

namespace odai::ui {

class Panel;
class Label;

// Left-rail status list in the style of a 4X "world tracker": an ornate framed
// panel with a heading and a vertical stack of entry rows. Each row is a circular
// medallion icon plus a bold title, a small dim subtitle, and a wrapped
// description line.
//
// Pure UI (Vulkan-free, no game types): the app maps its game state into the
// Entry structs below, exactly as it does for ProductionPanel / TechTreePanel.
// Icon names are resolved against the global UiIconRegistry at build time; an
// unknown name simply draws the medallion disc with no glyph.
class WorldTrackerPanel : public Widget {
public:
    struct Entry {
        std::string iconName;     // UiIconRegistry name; "" -> blank medallion.
        std::string title;        // Bold primary line (e.g. "Oral Tradition").
        std::string subtitle;     // Small dim caps line (e.g. "RESEARCH").
        std::string description;  // Wrapped detail line.
    };

    explicit WorldTrackerPanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the whole panel. Heading is drawn at the top; entries follow.
    void setEntries(const UiRect& rect, float s, std::string_view heading,
                    const std::vector<Entry>& entries);

private:
    FontSet fonts_;
    Panel* bg_ = nullptr;
};

}  // namespace odai::ui
