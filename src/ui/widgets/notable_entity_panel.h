#pragma once

#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>
#include <string>
#include <vector>

namespace odai::ui {

class Button;
class Image;
class Label;
class Panel;
class RichTextView;
class ScrollView;

// Generic notable-entity panel: a roster of special characters, heroes, or
// figures with portraits, names, titles, and an optional action button (settle,
// assign, deploy, recruit, etc.). A left rail shows the portrait list; the right
// pane shows the selected entity's full detail.
//
// Originally GreatPeoplePanel. Suitable for: 4X great people, grand-strategy
// generals/admirals, colony-sim colonist specialists, RPG party members, etc.
//
// Pure UI (Vulkan-free, no game types): the app maps its entity roster into the
// Entry/Detail structs below.
class NotableEntityPanel : public Widget {
public:
    // One entity in the left rail.
    struct Entry {
        std::string id;
        std::string name;
        std::string subtitle;            // E.g. "Great Scientist" or role/class
        std::string portraitName;        // Icon-registry name; resolved at build time.
        bool pending = false;            // Unassigned / awaiting placement → amber dot.
        std::function<void()> onSelect;  // Rail entry clicked.
    };

    // The right-hand detail for the currently selected entity.
    struct Detail {
        std::string id;             // Selected entity id (drives rail highlight).
        std::string name;
        std::string title;          // Class, role, or epithet.
        std::string portraitName;
        std::string body;           // Rich-text flavor + stats + status.
        bool showAction = false;    // Show the action button.
        bool actionEnabled = true;  // False greys it out.
        std::string actionLabel;    // E.g. "Settle in City" / "Assign to base" / "Deploy".
        std::function<void()> onAction;
    };

    explicit NotableEntityPanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the whole panel. Safe to call from anywhere EXCEPT inside a rail
    // entry's own onSelect callback. For in-callback selection changes use applyDetail().
    void setEntries(const UiRect& rect, float s,
                    const std::vector<Entry>& entries, const Detail& detail);

    // Re-highlight the selected rail entry and swap the detail pane in place,
    // WITHOUT rebuilding the child tree. Safe to call from a rail entry's onSelect.
    void applyDetail(const std::string& selectedId, const Detail& detail);

private:
    void applyHeaderAndBody(const Detail& detail);

    FontSet fonts_;
    float scale_ = 1.0f;
    Panel*        bg_         = nullptr;
    Image*        portrait_   = nullptr;
    Label*        nameLabel_  = nullptr;
    Label*        titleLabel_ = nullptr;
    RichTextView* bodyView_   = nullptr;
    Button*       actionBtn_  = nullptr;
    ScrollView*   rail_       = nullptr;
    std::vector<Widget*> rows_;
    std::function<void()> action_;
};

}  // namespace odai::ui
