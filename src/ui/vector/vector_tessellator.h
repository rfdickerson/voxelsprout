#pragma once

#include "ui/vector/vector_mesh_sink.h"
#include "ui/vector/vector_path.h"

#include <cstdint>

// Turns a flattened VectorPath into anti-aliased triangle geometry. Fills are
// triangulated (convex fan fast path, otherwise ear-clipping with hole bridging)
// honoring nonzero/even-odd fill rules; strokes are expanded with joins and caps.
// Both wrap the result in a 1px feathered fringe (extra vertices with alpha=0) so
// the existing straight-alpha SolidColor pipeline anti-aliases them for free.
//
// All work is CPU + Vulkan-free, so the same code runs in the offline bundler.
namespace odai::ui {

struct TessOptions {
    float aaFringePx = 1.25f;  // Feather width in device px; 0 disables AA.
    float dpiScale = 1.0f;     // Multiplies the fringe so it stays ~constant on screen.
    FillRule fillRule = FillRule::NonZero;
};

struct StrokeOptions {
    float widthPx = 1.0f;
    LineJoin join = LineJoin::Miter;
    LineCap cap = LineCap::Butt;
    float miterLimit = 4.0f;
    float aaFringePx = 1.25f;
    float dpiScale = 1.0f;
};

// Triangulate every subpath of `path` (treated as closed) and emit solid + AA
// fringe geometry in `rgba8` (packed ABGR8) into `sink`.
void tessellateFill(const VectorPath& path, std::uint32_t rgba8, const TessOptions& opts,
                    IMeshSink& sink);

// Expand the polylines of `path` into a stroked outline with joins/caps + AA.
void tessellateStroke(const VectorPath& path, std::uint32_t rgba8, const StrokeOptions& opts,
                      IMeshSink& sink);

}  // namespace odai::ui
