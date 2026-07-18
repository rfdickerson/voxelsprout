#pragma once

#include "ui/ui_types.h"

// Custom-rendered mouse cursor: a small procedural arrow glyph drawn via
// UiDrawList, replacing the OS cursor (which callers hide separately with
// GLFW_CURSOR_HIDDEN). Vulkan-free, like the rest of src/ui/.
namespace odai::ui {

class UiDrawList;

// Draws a static arrow-pointer cursor with its hotspot (tip) at posPx, sized
// by `scale` (pass the caller's DPI/UI scale so the cursor stays a consistent
// physical size). Call this LAST each frame, after all other UI geometry, so
// the cursor draws on top of every widget/tooltip it overlaps.
void drawCursor(UiDrawList& dl, const UiVec2& posPx, float scale = 1.0f);

}  // namespace odai::ui
