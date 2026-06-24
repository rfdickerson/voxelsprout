#pragma once

#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"

#include <filesystem>

// Parses an SVG (via nanosvg) and tessellates its shapes into a UiGeometryBlock
// of SolidColor triangles with AA fringes, ready to cache or replay through the
// UI draw list. Vulkan-free, so it runs both at runtime and in the bundler.
//
// Scope (v1): solid fills + strokes with nonzero/even-odd fill rules and
// join/cap styling. Gradients are approximated by their average stop color.
// nanosvg bakes viewBox/transforms, so we only apply a single uniform,
// aspect-preserving scale into a `targetSizePx` square box (origin top-left).
namespace odai::ui {

struct SvgTessellateOptions {
    float targetSizePx = 32.0f;  // The icon is fit into a targetSizePx square.
    float dpiScale = 1.0f;       // Scales the AA fringe (not the geometry).
    float tolerancePx = 0.25f;   // Flattening tolerance (device px).
    UiColor tintOverride{};      // If useTint, every shape uses this color.
    bool useTint = false;
};

// Parse `path` and fill `outBlock`. Returns false if the file can't be parsed.
bool tessellateSvgFile(const std::filesystem::path& path, const SvgTessellateOptions& opts,
                       UiGeometryBlock& outBlock);

// Parse from an in-memory SVG string (used by tests). `svgText` is copied
// internally because nanosvg mutates its input buffer.
bool tessellateSvgString(const char* svgText, const SvgTessellateOptions& opts,
                         UiGeometryBlock& outBlock);

}  // namespace odai::ui
