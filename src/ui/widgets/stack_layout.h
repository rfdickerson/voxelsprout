#pragma once

#include "ui/ui_types.h"
#include "ui/widget.h"

namespace odai::ui {

// Shared base for horizontal and vertical stacks. Child rects are computed each
// frame in draw() so they respond automatically to parent resizes.
class StackLayout : public Widget {
public:
    enum class Align { Start, Center, End };

    float gap = 6.0f;        // Pixels between children.
    Align crossAlign = Align::Start;  // Cross-axis alignment.

protected:
    // Called by subclasses to arrange children along the primary axis.
    // isHorizontal=true → arrange left-to-right; false → top-to-bottom.
    void layoutChildren(bool isHorizontal) const;
};

// Horizontal left-to-right stack.
class HorizontalStack : public StackLayout {
public:
    void draw(UiDrawList& dl) const override;
};

// Vertical top-to-bottom stack.
class VerticalStack : public StackLayout {
public:
    void draw(UiDrawList& dl) const override;
};

}  // namespace odai::ui
