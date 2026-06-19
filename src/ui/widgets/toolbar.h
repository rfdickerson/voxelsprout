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
    [[nodiscard]] std::size_t itemCount() const { return items_.size(); }

    void draw(UiDrawList& dl) const override;

private:
    void drawIcon(UiDrawList& dl, IconKind kind, const UiColor& color, const UiRect& box) const;

    const Font* font_ = nullptr;
    std::vector<Item> items_;
};

}  // namespace odai::ui
