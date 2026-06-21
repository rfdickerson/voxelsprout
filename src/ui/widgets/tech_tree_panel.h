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
class ProgressBar;
class RichTextView;

// Tech-tree research window content: a header showing the technology currently
// being researched (name + progress bar + turns remaining), a description of that
// technology, and a scrollable list of selectable techs grouped by era. Clicking
// an available tech sets it as the next research target; clicking the (i) link
// opens its CivPedia article. Locked and already-researched techs are shown but
// not selectable.
//
// Pure UI (Vulkan-free, no game types): the app maps its research state into the
// Row/Research structs below, exactly as it does for ProductionPanel.
class TechTreePanel : public Widget {
public:
    // Per-tech availability. Drives row color, selectability, and the status pip.
    enum class TechState { Available, Selected, Researched, Locked };

    struct Row {
        std::string id;
        std::string name;
        std::string info;          // "34 science  ·  3 turns" / "Requires Writing" / "Researched"
        TechState   state = TechState::Available;
        std::string section;       // Era header label; a divider is inserted when it changes.
        std::function<void()> onSelect;      // Available row clicked -> set as research target.
        std::function<void()> onOpenPedia;   // (i) clicked -> open CivPedia.
    };

    // The current-research header: progress bar + status line + the blurb shown
    // for the selected/next technology.
    struct Research {
        std::string title;         // "" -> "Choose a technology to research".
        float       fraction = 0.0f;  // 0..1 progress toward `title`.
        std::string status;        // "120 / 250 science  ·  9 turns left".
        std::string description;   // Rich-text blurb for the selected tech.
    };

    explicit TechTreePanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the whole panel. Safe to call from anywhere EXCEPT inside a row's
    // own onSelect callback (it rebuilds the child tree). For in-callback updates
    // use applyResearch().
    void setItems(const UiRect& rect, float s,
                  const std::vector<Row>& rows, const Research& research);

    // Update the header (progress / status / description) and re-highlight the
    // selected row in place, WITHOUT rebuilding the child tree. This is what a row
    // calls when clicked, so the widget handling the click is never destroyed.
    void applyResearch(const std::string& selectedId, const Research& research);

private:
    void applyHeader(const Research& research);

    FontSet fonts_;
    Panel*        bg_          = nullptr;
    Label*        currentLabel_ = nullptr;
    ProgressBar*  progress_    = nullptr;
    Label*        statusLabel_ = nullptr;
    RichTextView* descView_    = nullptr;
    ScrollView*   list_        = nullptr;
    std::vector<Widget*> rows_;  // Owned by the ScrollView; raw pointers for re-highlight.
};

}  // namespace odai::ui
