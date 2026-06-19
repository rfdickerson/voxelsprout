#pragma once

#include "ui/font.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <functional>
#include <string>
#include <vector>

namespace odai::ui {

// Horizontal tab row. addTab() returns a tab index; the caller uses the index
// in onTabChanged to show/hide its own content panels. Does not manage content.
class TabBar : public Widget {
public:
    explicit TabBar(const Font* font) : font_(font) {}

    // Add a tab and return its index.
    int addTab(std::string label);

    int activeTab = 0;

    UiColor activeTabColor{0.22f, 0.22f, 0.22f, 1.0f};
    UiColor inactiveTabColor{0.12f, 0.12f, 0.12f, 1.0f};
    UiColor indicatorColor{0.42f, 0.72f, 0.38f, 1.0f};
    UiColor textActiveColor{0.95f, 0.95f, 0.95f, 1.0f};
    UiColor textInactiveColor{0.50f, 0.50f, 0.50f, 1.0f};
    UiColor dividerColor{0.08f, 0.08f, 0.08f, 1.0f};

    float indicatorThicknessPx = 2.0f;

    std::function<void(int)> onTabChanged;

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent& ev) override;

private:
    const Font* font_ = nullptr;
    std::vector<std::string> tabs_;

    UiRect tabRect(int index) const;
};

}  // namespace odai::ui
