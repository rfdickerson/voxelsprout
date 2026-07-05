#pragma once

#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace odai::ui {

struct ContextMenuItem {
    std::string label;
    std::function<void()> onClick;
    bool separator = false;  // When true, label/onClick are ignored: draws a divider.
    bool enabled = true;
};

// Right-click popup menu. Closed by default; call openAt() (typically from a
// MouseDown-right handler elsewhere in the tree) to show it at a screen point.
// Like Dropdown, place this last in its parent's draw order so it layers above
// siblings, and give it a high zOrder so it receives input first.
class ContextMenu : public Widget {
public:
    explicit ContextMenu(const Font* font) : font_(font) { zOrder = 100; }

    std::vector<ContextMenuItem> items;

    UiColor bgColor{0.14f, 0.14f, 0.14f, 0.98f};
    UiColor borderColor{0.08f, 0.08f, 0.08f, 1.0f};
    UiColor itemHoverColor{0.28f, 0.28f, 0.28f, 1.0f};
    UiColor textColor{0.90f, 0.90f, 0.90f, 1.0f};
    UiColor textDisabledColor{0.45f, 0.45f, 0.45f, 1.0f};
    UiColor separatorColor{0.30f, 0.30f, 0.30f, 1.0f};

    float itemHeightPx = 24.0f;
    float separatorHeightPx = 7.0f;
    float paddingX = 10.0f;
    float widthPx = 160.0f;

    void openAt(const UiVec2& screenPx);
    void close();
    [[nodiscard]] bool isOpen() const { return open_; }

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent& ev) override;

private:
    const Font* font_ = nullptr;
    bool open_ = false;
    UiVec2 origin_{};
    int hoveredItem_ = -1;

    [[nodiscard]] UiRect menuRect() const;
    [[nodiscard]] UiRect itemRect(std::size_t index) const;
    [[nodiscard]] float itemY(std::size_t index) const;
};

}  // namespace odai::ui
