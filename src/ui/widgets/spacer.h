#pragma once

#include "ui/widget.h"

namespace odai::ui {

// Invisible widget used to add fixed gaps in stack layouts.
// Set rect width/height to the desired gap size.
class Spacer : public Widget {
public:
    void draw(UiDrawList&) const override {}
};

}  // namespace odai::ui
