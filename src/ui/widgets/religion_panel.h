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

// The Religion window: shows available and adopted faiths, their historical
// bonuses/penalties, and lineage (which religions a player can reform to).
// The left rail lists all religions grouped by availability; the right pane
// shows the selected faith's flavor text, bonuses, and an Adopt button.
//
// Pure UI: the app maps ReligionDef catalog + empire stateReligion into
// Entry/Detail structs, exactly as it does for GreatPeoplePanel.
class ReligionPanel : public Widget {
public:
    enum class Status {
        Current,     // this is the empire's current state religion
        Available,   // can be adopted now (has tech, parent satisfied)
        Locked,      // requires a different parent religion
        NotYet,      // required tech not yet researched
    };

    // One faith in the left rail.
    struct Entry {
        std::string id;
        std::string name;
        std::string subtitle;     // e.g. "+2 science, +1 happiness" or "Reform from Judaism"
        Status status = Status::NotYet;
        std::function<void()> onSelect;
    };

    // Right-hand detail pane for the selected faith.
    struct Detail {
        std::string id;           // drives rail highlight
        std::string name;
        std::string bonusSummary; // one-line yield/happiness summary
        std::string body;         // rich-text flavor + historical description
        bool showAdopt = false;   // show the Adopt button at all
        bool adoptEnabled = false;
        std::string adoptLabel;   // "Adopt this Faith" / "Reform to Christianity"
        std::function<void()> onAdopt;
    };

    explicit ReligionPanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the whole panel from the current catalog + empire state.
    void setEntries(const UiRect& rect, float s,
                    const std::vector<Entry>& entries, const Detail& detail);

    // Swap in a new detail pane without rebuilding the rail (safe from onSelect).
    void applyDetail(const std::string& selectedId, const Detail& detail);

private:
    void applyDetailContent(const Detail& detail);

    FontSet fonts_;
    float scale_ = 1.0f;
    Panel*        bg_        = nullptr;
    Label*        nameLabel_ = nullptr;
    Label*        bonusLabel_ = nullptr;
    RichTextView* bodyView_  = nullptr;
    Button*       adoptBtn_  = nullptr;
    ScrollView*   rail_      = nullptr;
    std::vector<Widget*> rows_;
    std::function<void()> adoptAction_;
};

}  // namespace odai::ui
