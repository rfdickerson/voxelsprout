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
class Image;
class RichTextView;
class ScrollView;

// The Council of Houses window content: a left rail of advisor portraits and a
// right pane showing the selected advisor's portrait, name/title, and current
// advice as rich text. Clicking a rail entry selects that advisor.
//
// Pure UI (Vulkan-free, no game types): the app maps its advisor catalog +
// evaluated advice into the Advisor/Detail structs below, exactly as it does for
// the tech tree (TechTreePanel) and production (ProductionPanel) panels.
class AdvisorsPanel : public Widget {
public:
    // One advisor in the left rail.
    struct Advisor {
        std::string id;
        std::string name;
        std::string title;
        std::string portraitName;       // icon-registry name; resolved at build time
        int  adviceCount = 0;            // badge count
        bool hasUrgent = false;          // tints the badge red
        std::function<void()> onSelect;  // rail entry clicked
    };

    // The right-hand detail for the currently selected advisor.
    struct Detail {
        std::string id;                  // selected advisor id (drives rail highlight)
        std::string name;
        std::string title;
        std::string portraitName;
        std::string body;                // rich-text: greeting + advice
    };

    explicit AdvisorsPanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the whole panel. Safe to call from anywhere EXCEPT inside a rail
    // entry's own onSelect callback (it rebuilds the child tree). For in-callback
    // selection changes use applyDetail().
    void setAdvisors(const UiRect& rect, float s,
                     const std::vector<Advisor>& advisors, const Detail& detail);

    // Re-highlight the selected rail entry and swap the detail pane in place,
    // WITHOUT rebuilding the child tree. Safe to call from a rail entry's onSelect.
    void applyDetail(const std::string& selectedId, const Detail& detail);

private:
    void applyHeaderAndBody(const Detail& detail);

    FontSet fonts_;
    Panel*        bg_         = nullptr;
    Image*        portrait_   = nullptr;
    Label*        nameLabel_  = nullptr;
    Label*        titleLabel_ = nullptr;
    RichTextView* bodyView_   = nullptr;
    ScrollView*   rail_       = nullptr;
    std::vector<Widget*> rows_;  // owned by the rail; raw pointers for re-highlight
};

}  // namespace odai::ui
