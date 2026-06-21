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

// Right-rail "command view" in the style of a 4X selection inspector: an ornate
// framed panel containing stacked cards describing the current selection — the
// hex under inspection, a recommended action with action-icon buttons, and the
// selected unit (portrait, primary stats, combat stats, abilities, settlement
// preview). Cards are shown only when the corresponding data is present.
//
// Pure UI (Vulkan-free, no game types): the app maps its selection state into the
// State struct below. Icon/portrait names resolve against the global
// UiIconRegistry at build time; unknown names draw an empty chip.
class CommandViewPanel : public Widget {
public:
    struct Yield {
        std::string iconName;  // UiIconRegistry name (food/production/gold/...).
        std::string value;     // "+7", "1", etc.
    };
    struct HexInfo {
        std::string name;             // "Forest".
        std::string iconName;         // Optional terrain icon medallion.
        std::vector<Yield> yields;    // Per-yield icon + value chips.
    };
    struct Action {
        std::string iconName;            // Action icon for the chip.
        std::function<void()> onClick;   // Optional; chip is inert when empty.
    };
    struct ActionInfo {
        std::string title;             // "Found City".
        std::string description;       // Wrapped recommendation text.
        std::vector<Action> actions;   // Row of action-icon buttons.
    };
    struct Stat {
        std::string label;  // "HP", "MOV", "STR", ...
        std::string value;  // "100/100", "2/2", "1", ...
    };
    struct UnitInfo {
        std::string portraitName;        // unit_icons / civ_leaders name.
        std::string name;                // "Clan Settler".
        std::string klass;               // "Human Scout".
        std::vector<Stat> primaryStats;  // HP / MOV / ORDERS (rendered as a grid).
        std::vector<Stat> combatStats;   // STR / DEF / COSTS.
        std::string abilities;           // Rich-text block.
        std::string settlementPreview;   // Rich-text block.
    };
    struct State {
        bool hasHex = false;
        bool hasUnit = false;
        HexInfo hex;
        ActionInfo action;
        UnitInfo unit;
        std::string emptyHint;  // Shown when nothing is selected.
    };

    explicit CommandViewPanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the whole panel from the current selection state.
    void setState(const UiRect& rect, float s, const State& state);

private:
    FontSet fonts_;
    Panel* bg_ = nullptr;
};

}  // namespace odai::ui
