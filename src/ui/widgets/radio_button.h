#pragma once

#include "ui/ui_types.h"
#include "ui/widget.h"

#include <cstddef>
#include <functional>
#include <vector>

namespace odai::ui {

// Single circular radio option. Mutually-exclusive selection across a set of
// RadioButtons is coordinated by ButtonGroup, not by RadioButton itself.
class RadioButton : public Widget {
public:
    bool selected = false;

    UiColor ringColor{0.55f, 0.55f, 0.55f, 1.0f};
    UiColor ringHoverColor{0.75f, 0.75f, 0.75f, 1.0f};
    UiColor dotColor{0.42f, 0.72f, 0.38f, 1.0f};
    float ringThicknessPx = 1.5f;

    // Fired when the user clicks this radio button (not when selection changes
    // due to a sibling being clicked). ButtonGroup listens here.
    std::function<void()> onSelect;

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent& ev) override;

private:
    bool hovered_ = false;
};

// Coordinates mutually-exclusive selection across a set of RadioButtons that
// are already children of some parent widget. Does not own or position the
// buttons — add() just wires each one's onSelect to clear its siblings.
class ButtonGroup {
public:
    // Register a radio button at the given index. Index order determines what
    // onChange reports; it does not need to match child draw order.
    void add(RadioButton* button);

    // Select by index, clearing all siblings and firing onChange. No-op if the
    // index is already selected or out of range.
    void selectIndex(std::size_t index);

    [[nodiscard]] int selectedIndex() const { return selectedIndex_; }

    std::function<void(int)> onChange;

private:
    std::vector<RadioButton*> buttons_;
    int selectedIndex_ = -1;
};

}  // namespace odai::ui
