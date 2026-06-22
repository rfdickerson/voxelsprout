#pragma once

#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>
#include <string>
#include <vector>

namespace odai::ui {

class Button;
class Label;
class Panel;
class RichTextView;
class ScrollView;

// Generic faction / affiliation panel: shows available groups, ideologies, or
// belief systems the player can join. A left rail lists all options with an
// availability status pip; the right pane shows the selected faction's flavor
// text, bonus summary, and an optional Join / Adopt / Switch button.
//
// Originally ReligionPanel. Suitable for: 4X religion, ideology, or alliance
// systems; grand-strategy diplomacy blocs; colony-sim faction allegiances, etc.
//
// Pure UI: the app maps its faction catalog + current allegiance into
// Entry/Detail structs.
class FactionPanel : public Widget {
public:
    enum class Status {
        Current,    // The player's current allegiance.
        Available,  // Can be joined now (prerequisites satisfied).
        Locked,     // Prerequisites not yet met.
        NotYet,     // Requires tech / progress not yet reached.
    };

    // One faction in the left rail.
    struct Entry {
        std::string id;
        std::string name;
        std::string subtitle;     // E.g. "+2 science, +1 happiness" or "Reform from Judaism"
        Status status = Status::NotYet;
        std::function<void()> onSelect;
    };

    // Right-hand detail pane for the selected faction.
    struct Detail {
        std::string id;            // Drives rail highlight.
        std::string name;
        std::string bonusSummary;  // One-line effect summary.
        std::string body;          // Rich-text flavor + description.
        bool showJoin = false;     // Show the join/adopt button at all.
        bool joinEnabled = false;
        std::string joinLabel;     // E.g. "Adopt this Faith" / "Join the Alliance".
        std::function<void()> onJoin;
    };

    explicit FactionPanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the whole panel from the current catalog + allegiance state.
    void setEntries(const UiRect& rect, float s,
                    const std::vector<Entry>& entries, const Detail& detail);

    // Swap in a new detail pane without rebuilding the rail (safe from onSelect).
    void applyDetail(const std::string& selectedId, const Detail& detail);

private:
    void applyDetailContent(const Detail& detail);

    FontSet fonts_;
    float scale_ = 1.0f;
    Panel*        bg_         = nullptr;
    Label*        nameLabel_  = nullptr;
    Label*        bonusLabel_ = nullptr;
    RichTextView* bodyView_   = nullptr;
    Button*       joinBtn_    = nullptr;
    ScrollView*   rail_       = nullptr;
    std::vector<Widget*> rows_;
    std::function<void()> joinAction_;
};

}  // namespace odai::ui
