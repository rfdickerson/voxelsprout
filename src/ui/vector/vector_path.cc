#include "ui/vector/vector_path.h"

#include <algorithm>
#include <cmath>

namespace odai::ui {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr int kMaxCubicDepth = 16;

float distToSegmentSq(float px, float py, float ax, float ay, float bx, float by) {
    const float dx = bx - ax;
    const float dy = by - ay;
    const float lenSq = dx * dx + dy * dy;
    if (lenSq <= 1e-12f) {
        const float ex = px - ax;
        const float ey = py - ay;
        return ex * ex + ey * ey;
    }
    float t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
    t = std::clamp(t, 0.0f, 1.0f);
    const float ex = px - (ax + t * dx);
    const float ey = py - (ay + t * dy);
    return ex * ex + ey * ey;
}

}  // namespace

void VectorPath::setTessellationTolerancePx(float tolerancePx) {
    m_tolerancePx = std::max(tolerancePx, 0.01f);
}

VectorSubPath& VectorPath::currentSubPath() {
    if (m_subPaths.empty()) {
        m_subPaths.emplace_back();
    }
    return m_subPaths.back();
}

UiVec2 VectorPath::currentPoint() const {
    return m_hasCurrent ? m_current : UiVec2{0.0f, 0.0f};
}

void VectorPath::appendPoint(float x, float y) {
    VectorSubPath& sp = currentSubPath();
    // Skip points coincident with the previous one — they create degenerate
    // triangles and confuse stroke normals.
    if (!sp.points.empty()) {
        const UiVec2& last = sp.points.back();
        const float dx = x - last.x;
        const float dy = y - last.y;
        if (dx * dx + dy * dy < 1e-8f) {
            return;
        }
    }
    sp.points.push_back(UiVec2{x, y});
    m_current = UiVec2{x, y};
    m_hasCurrent = true;
}

VectorPath& VectorPath::moveTo(float x, float y) {
    // Start a fresh subpath. If the current one is empty, reuse it.
    if (m_subPaths.empty() || !m_subPaths.back().points.empty()) {
        m_subPaths.emplace_back();
    }
    m_subPaths.back().points.push_back(UiVec2{x, y});
    m_current = UiVec2{x, y};
    m_hasCurrent = true;
    return *this;
}

VectorPath& VectorPath::lineTo(float x, float y) {
    if (!m_hasCurrent) {
        return moveTo(x, y);
    }
    appendPoint(x, y);
    return *this;
}

void VectorPath::flattenCubic(float x0, float y0, float x1, float y1, float x2, float y2,
                              float x3, float y3, int depth) {
    // Distance of the control points from the chord; if both are within
    // tolerance the curve is flat enough to approximate with a line.
    const float d1 = distToSegmentSq(x1, y1, x0, y0, x3, y3);
    const float d2 = distToSegmentSq(x2, y2, x0, y0, x3, y3);
    const float tolSq = m_tolerancePx * m_tolerancePx;
    if (depth >= kMaxCubicDepth || (d1 <= tolSq && d2 <= tolSq)) {
        appendPoint(x3, y3);
        return;
    }
    // de Casteljau subdivision at t = 0.5.
    const float x01 = (x0 + x1) * 0.5f, y01 = (y0 + y1) * 0.5f;
    const float x12 = (x1 + x2) * 0.5f, y12 = (y1 + y2) * 0.5f;
    const float x23 = (x2 + x3) * 0.5f, y23 = (y2 + y3) * 0.5f;
    const float x012 = (x01 + x12) * 0.5f, y012 = (y01 + y12) * 0.5f;
    const float x123 = (x12 + x23) * 0.5f, y123 = (y12 + y23) * 0.5f;
    const float xm = (x012 + x123) * 0.5f, ym = (y012 + y123) * 0.5f;
    flattenCubic(x0, y0, x01, y01, x012, y012, xm, ym, depth + 1);
    flattenCubic(xm, ym, x123, y123, x23, y23, x3, y3, depth + 1);
}

VectorPath& VectorPath::cubicBezierTo(float c1x, float c1y, float c2x, float c2y,
                                      float x, float y) {
    const UiVec2 p0 = currentPoint();
    flattenCubic(p0.x, p0.y, c1x, c1y, c2x, c2y, x, y, 0);
    return *this;
}

VectorPath& VectorPath::quadBezierTo(float cx, float cy, float x, float y) {
    // Elevate the quadratic to an equivalent cubic and reuse the flattener.
    const UiVec2 p0 = currentPoint();
    const float c1x = p0.x + (2.0f / 3.0f) * (cx - p0.x);
    const float c1y = p0.y + (2.0f / 3.0f) * (cy - p0.y);
    const float c2x = x + (2.0f / 3.0f) * (cx - x);
    const float c2y = y + (2.0f / 3.0f) * (cy - y);
    return cubicBezierTo(c1x, c1y, c2x, c2y, x, y);
}

VectorPath& VectorPath::arc(float cx, float cy, float radius, float a0, float a1, bool ccw) {
    if (radius <= 0.0f) {
        return *this;
    }
    // Normalize the sweep direction so we always step from a0 toward a1.
    float sweep = a1 - a0;
    if (ccw) {
        // Counter-clockwise: ensure sweep is negative.
        while (sweep > 0.0f) sweep -= 2.0f * kPi;
        if (sweep < -2.0f * kPi) sweep = -2.0f * kPi;
    } else {
        while (sweep < 0.0f) sweep += 2.0f * kPi;
        if (sweep > 2.0f * kPi) sweep = 2.0f * kPi;
    }

    // Segment count from the chord-height (sagitta) tolerance: the max angular
    // step whose chord deviates from the arc by <= tolerance.
    float maxStep = 2.0f * std::acos(std::max(0.0f, 1.0f - m_tolerancePx / radius));
    if (!(maxStep > 1e-4f)) {
        maxStep = kPi * 0.5f;
    }
    const int nSeg = std::max(1, static_cast<int>(std::ceil(std::fabs(sweep) / maxStep)));
    const float step = sweep / static_cast<float>(nSeg);

    for (int i = 0; i <= nSeg; ++i) {
        const float a = a0 + step * static_cast<float>(i);
        const float x = cx + radius * std::cos(a);
        const float y = cy + radius * std::sin(a);
        if (i == 0 && !m_hasCurrent) {
            moveTo(x, y);
        } else if (i == 0) {
            lineTo(x, y);
        } else {
            appendPoint(x, y);
        }
    }
    return *this;
}

VectorPath& VectorPath::arcTo(float x1, float y1, float x2, float y2, float radius) {
    const UiVec2 p0 = currentPoint();
    if (!m_hasCurrent) {
        return moveTo(x1, y1);
    }
    // Vectors from the corner point p1 toward p0 and p2.
    float d0x = p0.x - x1, d0y = p0.y - y1;
    float d2x = x2 - x1, d2y = y2 - y1;
    const float len0 = std::sqrt(d0x * d0x + d0y * d0y);
    const float len2 = std::sqrt(d2x * d2x + d2y * d2y);
    if (len0 < 1e-6f || len2 < 1e-6f || radius <= 0.0f) {
        return lineTo(x1, y1);
    }
    d0x /= len0; d0y /= len0;
    d2x /= len2; d2y /= len2;
    // Half-angle between the two edges.
    const float cosA = std::clamp(d0x * d2x + d0y * d2y, -1.0f, 1.0f);
    const float angle = std::acos(cosA);
    if (angle < 1e-4f || angle > kPi - 1e-4f) {
        return lineTo(x1, y1);  // Colinear: nothing to round.
    }
    const float tanHalf = std::tan(angle * 0.5f);
    const float dist = radius / tanHalf;  // Distance from p1 to each tangent point.
    // Tangent points on each edge.
    const float t0x = x1 + d0x * dist, t0y = y1 + d0y * dist;
    const float t2x = x1 + d2x * dist, t2y = y1 + d2y * dist;
    // Arc center: along the bisector, at radius / sin(half-angle) from p1.
    float bx = d0x + d2x, by = d0y + d2y;
    const float blen = std::sqrt(bx * bx + by * by);
    bx /= blen; by /= blen;
    const float centerDist = radius / std::sin(angle * 0.5f);
    const float ccx = x1 + bx * centerDist;
    const float ccy = y1 + by * centerDist;
    const float startAngle = std::atan2(t0y - ccy, t0x - ccx);
    const float endAngle = std::atan2(t2y - ccy, t2x - ccx);
    // Straight segment to the first tangent point, then the arc to the second.
    lineTo(t0x, t0y);
    // Cross product picks the sweep direction (ccw if positive in +Y-down space).
    const float cross = d0x * d2y - d0y * d2x;
    return arc(ccx, ccy, radius, startAngle, endAngle, cross > 0.0f);
}

VectorPath& VectorPath::ellipse(float cx, float cy, float rx, float ry) {
    if (rx <= 0.0f || ry <= 0.0f) {
        return *this;
    }
    // Approximate via a unit-circle arc scaled — but our arc() assumes a single
    // radius, so flatten directly using the larger radius for the step count.
    const float rMax = std::max(rx, ry);
    float maxStep = 2.0f * std::acos(std::max(0.0f, 1.0f - m_tolerancePx / rMax));
    if (!(maxStep > 1e-4f)) {
        maxStep = kPi * 0.5f;
    }
    const int nSeg = std::max(4, static_cast<int>(std::ceil(2.0f * kPi / maxStep)));
    const float step = 2.0f * kPi / static_cast<float>(nSeg);
    moveTo(cx + rx, cy);
    for (int i = 1; i <= nSeg; ++i) {
        const float a = step * static_cast<float>(i);
        appendPoint(cx + rx * std::cos(a), cy + ry * std::sin(a));
    }
    return close();
}

VectorPath& VectorPath::circle(float cx, float cy, float radius) {
    return ellipse(cx, cy, radius, radius);
}

VectorPath& VectorPath::rect(float x, float y, float w, float h) {
    if (w <= 0.0f || h <= 0.0f) {
        return *this;
    }
    moveTo(x, y);
    lineTo(x + w, y);
    lineTo(x + w, y + h);
    lineTo(x, y + h);
    return close();
}

VectorPath& VectorPath::roundedRect(float x, float y, float w, float h, float radius) {
    if (w <= 0.0f || h <= 0.0f) {
        return *this;
    }
    const float r = std::min(radius, std::min(w, h) * 0.5f);
    if (r <= 0.0f) {
        return rect(x, y, w, h);
    }
    const float x0 = x, y0 = y, x1 = x + w, y1 = y + h;
    // Corner arcs in +Y-down space: angles increase clockwise. Start at the top
    // edge just right of the top-left corner and walk clockwise.
    moveTo(x0 + r, y0);
    lineTo(x1 - r, y0);
    arc(x1 - r, y0 + r, r, -kPi * 0.5f, 0.0f);        // top-right
    lineTo(x1, y1 - r);
    arc(x1 - r, y1 - r, r, 0.0f, kPi * 0.5f);          // bottom-right
    lineTo(x0 + r, y1);
    arc(x0 + r, y1 - r, r, kPi * 0.5f, kPi);           // bottom-left
    lineTo(x0, y0 + r);
    arc(x0 + r, y0 + r, r, kPi, kPi * 1.5f);           // top-left
    return close();
}

VectorPath& VectorPath::close() {
    if (!m_subPaths.empty()) {
        m_subPaths.back().closed = true;
    }
    return *this;
}

void VectorPath::clear() {
    m_subPaths.clear();
    m_hasCurrent = false;
    m_current = UiVec2{};
}

}  // namespace odai::ui
