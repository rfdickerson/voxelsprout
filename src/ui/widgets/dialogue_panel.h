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

// A Dragon-Age-style conversation box: speaker name + rich-text line on top, a
// stack of choice buttons below. Pure UI — the app drives it from a
// dialogue::DialogueRuntime by mapping currentNode()/availableChoices() into
// Line/ChoiceEntry each time the conversation advances, the same way
// FactionPanel is driven from a game's own faction catalog via Entry/Detail.
class DialoguePanel : public Widget {
public:
    struct Line {
        std::string speaker;  // "" = narrator/no name shown
        std::string text;     // rich-text markup
    };

    struct ChoiceEntry {
        std::string text;
        std::function<void()> onChoose;
    };

    explicit DialoguePanel(const FontSet& fonts) : fonts_(fonts) {}

    // Rebuilds the whole panel: the current line plus one button per choice.
    // Pass an empty choices list for a narrator-only line with no response
    // yet (mid-autoNext beat, or the conversation's final line).
    void setContent(const UiRect& rect, float s, const Line& line,
                     const std::vector<ChoiceEntry>& choices);

private:
    FontSet fonts_;
    Panel* bg_ = nullptr;
    Label* speakerLabel_ = nullptr;
    RichTextView* bodyView_ = nullptr;
    std::vector<Widget*> choiceButtons_;
};

}  // namespace odai::ui
