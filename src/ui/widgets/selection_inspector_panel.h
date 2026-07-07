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

// Generic selection inspector panel: stacked cards describing the current
// selection — the tile/hex under inspection, a recommended action with icon
// buttons, and the selected entity (portrait, primary stats, combat stats,
// abilities, placement preview). Cards appear only when the corresponding data
// is present.
//
// Originally CommandViewPanel. Suitable for: 4X unit/hex inspector, RTS
// selection panel, city-builder building inspector, colony-sim entity detail, etc.
//
// Pure UI (Vulkan-free, no game types): the app maps its selection state into the
// State struct below. Icon/portrait names resolve against the global
// UiIconRegistry.
class SelectionInspectorPanel : public Widget {
public:
    struct ResourceValue {
        std::string iconName;  // UiIconRegistry name.
        std::string value;     // "+7", "1", etc.
    };
    struct TileInfo {
        std::string name;             // "Forest", "Plains", "District 4", etc.
        std::string iconName;         // Optional terrain/tile icon medallion.
        UiTextureId previewTexture = kUiNoTexture;  // Optional square artwork.
        std::vector<ResourceValue> yields;           // Per-resource icon + value chips.
    };
    struct Action {
        std::string iconName;            // Action icon for the chip.
        std::function<void()> onClick;   // Optional; chip is inert when empty.
    };
    struct ActionInfo {
        std::string title;             // E.g. "Found City", "Build Farm".
        std::string description;       // Wrapped recommendation text.
        std::vector<Action> actions;   // Row of action-icon buttons.
    };
    struct Stat {
        std::string label;  // "HP", "MOV", "STR", "Power", etc.
        std::string value;  // "100/100", "2/2", "1", etc.
    };
    struct EntityInfo {
        std::string portraitName;        // Icon-registry name for portrait/icon.
        std::string name;                // Unit, building, or entity name.
        std::string klass;               // Type/class label.
        std::vector<Stat> primaryStats;  // Rendered as a stat grid.
        std::vector<Stat> secondaryStats; // Second stat grid (e.g. combat stats).
        std::string abilities;           // Rich-text block.
        std::string placementPreview;    // Rich-text block (e.g. "Found City" preview).
    };
    struct State {
        std::string title;    // Panel heading. Defaults to "Inspector" when empty.
        bool hasTile   = false;
        bool hasEntity = false;
        TileInfo   tile;
        ActionInfo action;
        EntityInfo entity;
        std::string emptyHint;  // Shown when nothing is selected.
        // Caption overrides (shown above each card section).
        std::string tileCaption    = "Selected Tile";
        std::string actionCaption  = "Recommended Action";
        std::string entityCaption  = "";  // "" → not shown
    };

    explicit SelectionInspectorPanel(const FontSet& fonts) : fonts_(fonts) {}

    // Background card frame styling. setState() rebuilds bg_ from scratch on
    // every call (it's the only way to refresh this panel), so per-game style
    // overrides must live here as persistent members rather than being poked
    // onto bgPanel() after construction — they'd be wiped on the next refresh.
    // Defaults match the panel's original hardcoded look.
    UiColor borderColor{1.0f, 1.0f, 1.0f, 0.10f};
    float   borderThicknessPx = 1.0f;
    UiColor bevelHighlightColor{1.0f, 1.0f, 1.0f, 0.18f};
    UiColor bevelShadowColor{0.0f, 0.0f, 0.0f, 0.40f};
    float   bevelThicknessPx = 1.5f;
    bool    bevelInward = false;  // true = sunken "well" look instead of raised.

    // (Re)build the whole panel from the current selection state.
    void setState(const UiRect& rect, float s, const State& state);

    // Expose the background Panel for per-game styling overrides (mirrors
    // ResearchPanel::bgPanel()). Safe to read after setState(); anything
    // touching bevel/border colors should go through the persistent fields
    // above instead, since this pointer is replaced on every setState() call.
    [[nodiscard]] Panel* bgPanel() const { return bg_; }

private:
    FontSet fonts_;
    Panel* bg_ = nullptr;
};

}  // namespace odai::ui
