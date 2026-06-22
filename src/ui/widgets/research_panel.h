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

// Generic research / technology panel: a header showing the current research
// target with a progress bar, a rich-text description pane, and a scrollable
// list of selectable items grouped by category. Available items can be clicked
// to set them as the research target; locked and completed items are shown but
// not selectable.
//
// Originally TechTreePanel. Suitable for: 4X tech trees, city-builder upgrade
// menus, colony-sim technology paths, etc.
//
// Pure UI (Vulkan-free, no game types): the app maps its research state into the
// Row/ResearchProgress structs below.
class ResearchPanel : public Widget {
public:
    // Per-item availability. Drives row color, selectability, and the status pip.
    enum class ItemState { Available, Selected, Completed, Locked };

    struct Row {
        std::string id;
        std::string name;
        std::string info;          // E.g. "34 science  ·  3 turns" / "Requires Writing"
        ItemState   state = ItemState::Available;
        std::string section;       // Category header; a divider is inserted when it changes.
        std::function<void()> onSelect;      // Available row clicked → set as research target.
        std::function<void()> onOpenDetail;  // Detail link clicked → open description/pedia.
    };

    // Current research header: progress bar + status line + description blurb.
    struct ResearchProgress {
        std::string title;         // "" → show "Choose an item to research".
        float       fraction = 0.0f;  // 0..1 progress toward `title`.
        std::string status;        // E.g. "120 / 250 science  ·  9 turns left".
        std::string description;   // Rich-text blurb for the selected/next item.
    };

    explicit ResearchPanel(const FontSet& fonts) : fonts_(fonts) {}

    // (Re)build the whole panel. Safe to call from anywhere EXCEPT inside a row's
    // own onSelect callback (it rebuilds the child tree). For in-callback updates
    // use applyProgress().
    void setItems(const UiRect& rect, float s,
                  const std::vector<Row>& rows, const ResearchProgress& progress);

    // Update the header (progress / status / description) and re-highlight the
    // selected row in place, WITHOUT rebuilding the child tree. Safe to call from
    // a row's onSelect callback.
    void applyProgress(const std::string& selectedId, const ResearchProgress& progress);

private:
    void applyHeader(const ResearchProgress& progress);

    FontSet fonts_;
    Panel*        bg_           = nullptr;
    Label*        currentLabel_ = nullptr;
    ProgressBar*  progress_     = nullptr;
    Label*        statusLabel_  = nullptr;
    RichTextView* descView_     = nullptr;
    ScrollView*   list_         = nullptr;
    std::vector<Widget*> rows_;  // Owned by the ScrollView; raw pointers for re-highlight.
};

}  // namespace odai::ui
