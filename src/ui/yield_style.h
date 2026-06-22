#pragma once

#include "ui/ui_types.h"

#include <string>

// Single source of truth for how each 4X yield is presented: its icon, its
// accent color, and how a value is formatted. The top yield bar, the selection
// inspector, city views, and rich-text markup all route through here so the
// game speaks one consistent visual language for food/production/gold/etc.
// Pure CPU, no Vulkan — lives in the headless UI library.
namespace odai::ui {

// The core 4X yields. Order is fixed; kYieldStyles is indexed by this enum.
enum class Yield { Food, Production, Gold, Science, Culture, Faith };

struct YieldStyle {
    const char* iconName;  // registered atlas key; also valid in [icon=...] markup
    UiColor     color;     // fixed accent color shown wherever this yield appears
};

// Fixed presentation per yield. Colors authored as sRGB hex (WYSIWYG):
// food green, production amber, gold yellow, science blue, culture purple,
// faith warm grey-white. Icon names match the keys registered from the yield
// atlas (see Toolbar::iconName and app.cc icon registration).
inline const YieldStyle& yieldStyle(Yield y) {
    static const YieldStyle kStyles[] = {
        {"food",       UiColor::fromRgbHex(0x6FCF7B)},  // Food
        {"production", UiColor::fromRgbHex(0xEFB24A)},  // Production
        {"gold",       UiColor::fromRgbHex(0xF2D24A)},  // Gold
        {"science",    UiColor::fromRgbHex(0x5AA6EF)},  // Science
        {"culture",    UiColor::fromRgbHex(0xB58BD6)},  // Culture
        {"faith",      UiColor::fromRgbHex(0xD8DCE2)},  // Faith
    };
    return kStyles[static_cast<int>(y)];
}

inline const char* yieldIconName(Yield y) { return yieldStyle(y).iconName; }
inline UiColor      yieldColor(Yield y) { return yieldStyle(y).color; }

// Format a yield value for display. With `signedDelta` (the convention for
// per-turn rates) a leading '+' is shown for positive values; negatives always
// carry their '-'. Zero renders as an em dash so empty rows read cleanly.
inline std::string yieldText(int value, bool signedDelta = true) {
    if (value == 0) {
        return "\xE2\x80\x94";  // U+2014 em dash
    }
    if (signedDelta && value > 0) {
        return "+" + std::to_string(value);
    }
    return std::to_string(value);
}

}  // namespace odai::ui
