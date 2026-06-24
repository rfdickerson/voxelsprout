#pragma once

#include "ui/ui_types.h"

#include <cstddef>
#include <vector>

// Vulkan-free vector path builder. Collects move/line/curve/arc commands and
// flattens beziers and arcs into polylines at a caller-controlled tolerance. The
// resulting subpaths feed the tessellator (vector_tessellator.h), which turns
// them into triangle meshes the UI draw list can stream to the GPU.
//
// Coordinate space matches the rest of the UI: pixels, origin top-left, +Y down.
// Angles are in radians, 0 = +X, increasing clockwise (because +Y is down).
namespace odai::ui {

enum class FillRule : std::uint8_t {
    NonZero = 0,
    EvenOdd = 1,
};

enum class LineJoin : std::uint8_t {
    Miter = 0,
    Round = 1,
    Bevel = 2,
};

enum class LineCap : std::uint8_t {
    Butt = 0,
    Round = 1,
    Square = 2,
};

// One flattened contour: a polyline of points plus whether it forms a closed
// loop. Fills always treat subpaths as closed; strokes honor `closed`.
struct VectorSubPath {
    std::vector<UiVec2> points;
    bool closed = false;
};

class VectorPath {
public:
    // Tolerance is the maximum allowed deviation (in pixels) between a flattened
    // segment and the true curve/arc. Smaller = smoother + more triangles. For
    // crisp output at high DPI, pass baseTolerance / dpiScale.
    void setTessellationTolerancePx(float tolerancePx);
    [[nodiscard]] float tessellationTolerancePx() const { return m_tolerancePx; }

    // --- Path construction (chainable) ---
    VectorPath& moveTo(float x, float y);
    VectorPath& lineTo(float x, float y);
    VectorPath& cubicBezierTo(float c1x, float c1y, float c2x, float c2y, float x, float y);
    VectorPath& quadBezierTo(float cx, float cy, float x, float y);
    // Arc tangent to the lines (current->p1) and (p1->p2), of the given radius,
    // joined to the current point by a straight segment (SVG/HTML5 arcTo).
    VectorPath& arcTo(float x1, float y1, float x2, float y2, float radius);
    // Arc centered at (cx,cy) sweeping from a0 to a1. ccw selects the short/long
    // direction; emitted as a polyline starting a new subpath if none is open.
    VectorPath& arc(float cx, float cy, float radius, float a0, float a1, bool ccw = false);
    VectorPath& ellipse(float cx, float cy, float rx, float ry);
    VectorPath& circle(float cx, float cy, float radius);
    VectorPath& rect(float x, float y, float w, float h);
    VectorPath& roundedRect(float x, float y, float w, float h, float radius);
    VectorPath& close();

    void clear();
    [[nodiscard]] bool empty() const { return m_subPaths.empty(); }
    [[nodiscard]] const std::vector<VectorSubPath>& subPaths() const { return m_subPaths; }

private:
    VectorSubPath& currentSubPath();
    [[nodiscard]] UiVec2 currentPoint() const;
    void appendPoint(float x, float y);
    // Recursive adaptive subdivision; appends points (excluding p0, including p3).
    void flattenCubic(float x0, float y0, float x1, float y1, float x2, float y2,
                      float x3, float y3, int depth);

    std::vector<VectorSubPath> m_subPaths;
    float m_tolerancePx = 0.25f;
    bool m_hasCurrent = false;
    UiVec2 m_current{};
};

}  // namespace odai::ui
