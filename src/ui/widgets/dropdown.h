#pragma once

#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>
#include <string>
#include <vector>

namespace odai::ui {

// Rounded-rect header + floating list popup for single-item selection.
// The popup is drawn directly (not as a child) to avoid parent clip rect
// truncation — place the Dropdown last in its draw tree for correct layering.
class Dropdown : public Widget {
public:
    explicit Dropdown(const Font* font) : font_(font) { zOrder = 1; }

    std::vector<std::string> items;
    int selectedIndex = 0;
    int maxVisibleItems = 6;

    UiColor headerBg{0.18f, 0.18f, 0.18f, 1.0f};
    UiColor headerBorderColor{0.08f, 0.08f, 0.08f, 1.0f};
    UiColor headerHoverBg{0.25f, 0.25f, 0.25f, 1.0f};
    UiColor popupBg{0.14f, 0.14f, 0.14f, 0.97f};
    UiColor popupBorderColor{0.08f, 0.08f, 0.08f, 1.0f};
    UiColor itemHoverColor{0.28f, 0.28f, 0.28f, 1.0f};
    UiColor textColor{0.90f, 0.90f, 0.90f, 1.0f};
    UiColor chevronColor{0.55f, 0.55f, 0.55f, 1.0f};

    float itemHeightPx = 24.0f;
    float paddingX = 8.0f;
    float cornerRadiusPx = 3.0f;

    std::function<void(int)> onSelect;

    bool isOpen() const { return open_; }
    void close() { open_ = false; hoveredItem_ = -1; }

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent& ev) override;

private:
    const Font* font_ = nullptr;
    bool open_ = false;
    bool headerHovered_ = false;
    int hoveredItem_ = -1;

    UiRect popupRect() const;
    UiRect itemRect(int index) const;
    int visibleCount() const;
};

}  // namespace odai::ui
