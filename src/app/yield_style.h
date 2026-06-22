#pragma once

// Moved from src/ui/yield_style.h. This file belongs in the app layer
// because Civ-specific yield vocabulary (Food/Production/Gold/Science/Culture/
// Faith) is not a concern of the generic UI library. See src/ui/resource_style.h
// for the generic registry that app code should populate at startup.

#include "ui/resource_style.h"
#include "ui/ui_types.h"

#include <string>

namespace odai::ui {

// The core 4X yields. Order is fixed; yieldStyle() is indexed by this enum.
enum class Yield { Food, Production, Gold, Science, Culture, Faith };

struct YieldStyle {
    const char* iconName;  // registered atlas key; also valid in [icon=...] markup
    UiColor     color;     // fixed accent color shown wherever this yield appears
};

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

inline std::string yieldText(int value, bool signedDelta = true) {
    return resourceText(value, signedDelta);
}

// Register all 4X yield styles with the generic resource registry.
// Call once at startup, after icon atlases are loaded.
inline void registerYieldStyles() {
    for (int i = 0; i <= static_cast<int>(Yield::Faith); ++i) {
        const Yield y = static_cast<Yield>(i);
        registerResourceStyle(yieldStyle(y).iconName,
                              ResourceStyle{yieldStyle(y).iconName, yieldStyle(y).color});
    }
}

}  // namespace odai::ui
