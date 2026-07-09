#pragma once

#include "ui/font.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

#include <string>
#include <vector>

// A horizontal status/resource bar in the style of a 4X game's top strip: a
// background band followed by a row of "badges", each an icon plus a value
// (e.g. a gold coin and "+39.3"). Resource icons resolve through the global
// icon registry when available, with DPI-aware SDF vector primitives as a
// fallback. Items are laid out left to right; any child widgets (e.g. a
// right-aligned turn label) draw on top afterwards.
namespace odai::ui {

class Toolbar : public Widget {
public:
    enum class IconKind { Coin, Science, Culture, Faith, Food, Production, Dot };

    struct Item {
        IconKind icon = IconKind::Dot;
        UiColor iconColor{1.0f, 1.0f, 1.0f, 1.0f};
        std::string value;
        UiColor valueColor{0.92f, 0.95f, 0.97f, 1.0f};
        std::string tooltip;  // Rich-text markup shown on hover; empty = no tooltip.
    };

    explicit Toolbar(const Font* font) : font_(font) {}

    UiColor background{0.04f, 0.07f, 0.10f, 0.94f};
    UiColor accentLine{0.82f, 0.66f, 0.34f, 0.55f};  // thin rule along the bottom edge
    float accentThicknessPx = 2.0f;
    float cornerRadiusPx = 0.0f;  // > 0 rounds the band (e.g. a floating pill bar)
    float paddingXPx = 16.0f;     // inset of the first/last item from the band edge
    float itemGapPx = 26.0f;      // space between adjacent badges
    float iconGapPx = 8.0f;       // space between an icon and its value text
    float iconScale = 0.52f;      // icon diameter as a fraction of the band height

    // "Chip" backing: a rounded pill grouping each icon+value pair, in the style
    // of a modern 4X resource bar (e.g. Civ 6) instead of the bare icon/text
    // floating directly on the band. Kept neutral/dark so it doesn't compete
    // with the icon's own color; a thin bottom trim tinted to the item's
    // iconColor still gives each stat its own identity.
    bool showItemChips = true;
    UiColor chipBackground{0.0f, 0.0f, 0.0f, 0.24f};
    UiColor chipBorder{1.0f, 1.0f, 1.0f, 0.08f};
    float chipRadiusPx = 9.0f;
    float chipInsetYPx = 5.0f;   // vertical inset from the band's top/bottom edges
    float chipPadXPx = 9.0f;     // horizontal padding beyond the icon/text bounds
    float chipTrimThicknessPx = 2.0f;
    float chipTrimAlpha = 0.75f;  // alpha of the per-item colored bottom trim

    std::size_t addItem(IconKind icon, const UiColor& iconColor, std::string value,
                        const UiColor& valueColor) {
        items_.push_back(Item{icon, iconColor, std::move(value), valueColor});
        return items_.size() - 1;
    }
    void setValue(std::size_t index, std::string value) {
        if (index < items_.size()) {
            items_[index].value = std::move(value);
        }
    }
    void setTooltip(std::size_t index, std::string tooltip) {
        if (index < items_.size()) {
            items_[index].tooltip = std::move(tooltip);
        }
    }
    [[nodiscard]] std::size_t itemCount() const { return items_.size(); }

    [[nodiscard]] bool hasHoveredTooltip() const {
        return hoveredItem_ >= 0 && !items_[static_cast<std::size_t>(hoveredItem_)].tooltip.empty();
    }
    [[nodiscard]] const std::string& hoveredTooltipText() const {
        return items_[static_cast<std::size_t>(hoveredItem_)].tooltip;
    }
    [[nodiscard]] UiVec2 hoveredTooltipAnchor() const { return hoveredAnchor_; }

    void draw(UiDrawList& dl) const override;
    bool onEvent(UiEvent& event) override;

private:
    void drawIcon(UiDrawList& dl, IconKind kind, const UiColor& color, const UiRect& box) const;

    const Font* font_ = nullptr;
    std::vector<Item> items_;
    mutable int hoveredItem_ = -1;
    mutable UiVec2 hoveredAnchor_{};
};

}  // namespace odai::ui
