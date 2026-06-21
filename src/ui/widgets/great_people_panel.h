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

// The Great People window content: a left rail of portraits (the empire's settled
// figures and any still awaiting a home) and a right pane showing the selected
// figure's portrait, name, class/epithet, flavor + bonus as rich text, and -- when
// the figure is born but not yet placed -- a button to settle them into a city.
//
// Pure UI (Vulkan-free, no game types): the app maps its great-people catalog +
// per-empire roster into the Entry/Detail structs below, exactly as it does for the
// advisors (AdvisorsPanel) and tech tree (TechTreePanel) panels.
class GreatPeoplePanel : public Widget {
public:
    // One figure in the left rail.
    struct Entry {
        std::string id;
        std::string name;
        std::string subtitle;            // e.g. "Great Scientist - Balmora" or "Great Scientist - unborn"
        std::string portraitName;        // icon-registry name; resolved at build time
        bool pending = false;            // born but not yet settled -> amber "new" dot
        std::function<void()> onSelect;  // rail entry clicked
    };

    // The right-hand detail for the currently selected figure.
    struct Detail {
        std::string id;                  // selected figure id (drives rail highlight)
        std::string name;
        std::string title;               // class or epithet
        std::string portraitName;
        std::string body;                // rich-text: flavor + bonus + status
        bool showIntegrate = false;      // figure is pending -> offer the settle button
        bool integrateEnabled = true;    // false greys it out (e.g. no eligible city)
        std::string integrateLabel;      // e.g. "Settle in Balmora"
        std::function<void()> onIntegrate;
    };

    explicit GreatPeoplePanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the whole panel. Safe to call from anywhere EXCEPT inside a rail
    // entry's own onSelect callback (it rebuilds the child tree). For in-callback
    // selection changes use applyDetail().
    void setEntries(const UiRect& rect, float s,
                    const std::vector<Entry>& entries, const Detail& detail);

    // Re-highlight the selected rail entry and swap the detail pane in place,
    // WITHOUT rebuilding the child tree. Safe to call from a rail entry's onSelect.
    void applyDetail(const std::string& selectedId, const Detail& detail);

private:
    void applyHeaderAndBody(const Detail& detail);

    FontSet fonts_;
    float scale_ = 1.0f;
    Panel*        bg_           = nullptr;
    Image*        portrait_     = nullptr;
    Label*        nameLabel_    = nullptr;
    Label*        titleLabel_   = nullptr;
    RichTextView* bodyView_     = nullptr;
    Button*       integrateBtn_ = nullptr;
    ScrollView*   rail_         = nullptr;
    std::vector<Widget*> rows_;  // owned by the rail; raw pointers for re-highlight
    // The selected figure's settle action, swapped per selection so the one
    // persistent button can serve every figure without rebuilding the tree.
    std::function<void()> integrateAction_;
};

}  // namespace odai::ui
