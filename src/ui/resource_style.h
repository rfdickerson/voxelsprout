#pragma once

#include "ui/ui_types.h"

#include <string_view>
#include <unordered_map>
#include <string>

// Generic resource-presentation registry: maps a string key to an icon name and
// accent color, so any game can define its own resource vocabulary (food/gold/
// science for 4X; money/power/water for city builders; wood/stone/ore for RTS).
//
// The registry is global and singleton; call registerResourceStyle() once during
// startup (after icon atlases are loaded) for each resource your game uses.
// Panels and draw helpers call resourceStyle() to look up the accent color and
// icon for a given key.
//
// Pure CPU, no Vulkan — lives in the headless UI library.
namespace odai::ui {

struct ResourceStyle {
    const char* iconName = "";   // Registered atlas key; also valid in [icon=...] markup.
    UiColor     color    = {};   // Accent color shown wherever this resource appears.
};

// Register a resource key and its visual style.
// Overwrites any previous entry for the same key.
void registerResourceStyle(std::string_view key, const ResourceStyle& style);

// Look up a resource key. Returns nullptr if the key has not been registered.
const ResourceStyle* resourceStyle(std::string_view key);

// Format a resource value for display.
//   signedDelta = true  (per-turn rate):  "+7", "-3", or "—" for zero
//   signedDelta = false (absolute amount): "1,200", "0"
std::string resourceText(int value, bool signedDelta = true);
std::string resourceTextFloat(float value, bool signedDelta = true, int decimals = 1);

}  // namespace odai::ui
