#include "ui/vector/vector_tessellator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace odai::ui {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Strip the alpha byte of a packed ABGR8 color to zero (transparent fringe edge).
std::uint32_t withZeroAlpha(std::uint32_t rgba8) {
    return rgba8 & 0x00FFFFFFu;
}

float signedArea(const std::vector<UiVec2>& p) {
    float a = 0.0f;
    for (std::size_t i = 0, n = p.size(); i < n; ++i) {
        const UiVec2& u = p[i];
        const UiVec2& v = p[(i + 1) % n];
        a += u.x * v.y - v.x * u.y;
    }
    return a * 0.5f;
}

bool pointInPolygon(const std::vector<UiVec2>& poly, const UiVec2& pt) {
    bool inside = false;
    const std::size_t n = poly.size();
    for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
        const bool straddles = (poly[i].y > pt.y) != (poly[j].y > pt.y);
        if (straddles) {
            const float xCross = (poly[j].x - poly[i].x) * (pt.y - poly[i].y) /
                                     (poly[j].y - poly[i].y) +
                                 poly[i].x;
            if (pt.x < xCross) {
                inside = !inside;
            }
        }
    }
    return inside;
}

UiVec2 normalize(float dx, float dy) {
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-8f) {
        return UiVec2{0.0f, 0.0f};
    }
    return UiVec2{dx / len, dy / len};
}

// --- Polygon triangulation (ear clipping with holes) --------------------------
//
// Index-based port of the mapbox/earcut algorithm: a linked-list ear clipper with
// hole bridging plus the local-intersection-cure and split fallbacks that make it
// robust on thin annuli and other awkward icon geometry. Nodes are referenced by
// integer index (not pointer), so the backing vector may grow during splits
// without invalidating references. Triangles are streamed to an IMeshSink.
class EarClipper {
public:
    void triangulate(const std::vector<std::vector<UiVec2>>& rings, std::uint32_t rgba8,
                     IMeshSink& sink) {
        m_sink = &sink;
        m_rgba = rgba8;
        m_nodes.clear();
        if (rings.empty() || rings[0].size() < 3) {
            return;
        }
        int outerNode = linkedList(rings[0], true);
        if (outerNode < 0 || m_nodes[outerNode].next == m_nodes[outerNode].prev) {
            return;
        }
        if (rings.size() > 1) {
            outerNode = eliminateHoles(rings, outerNode);
        }
        earcutLinked(outerNode, 0);
    }

private:
    struct Node {
        UiVec2 p{};
        std::uint32_t outIndex = 0;  // Vertex index in the sink.
        int prev = -1;
        int next = -1;
        bool steiner = false;
    };

    Node& nd(int i) { return m_nodes[static_cast<std::size_t>(i)]; }

    int createNode(const UiVec2& p) {
        Node n;
        n.p = p;
        n.outIndex = m_sink->pushVertex(p.x, p.y, m_rgba);
        m_nodes.push_back(n);
        return static_cast<int>(m_nodes.size()) - 1;
    }

    int insertNode(const UiVec2& p, int last) {
        const int cur = createNode(p);
        if (last < 0) {
            nd(cur).prev = cur;
            nd(cur).next = cur;
        } else {
            const int nx = nd(last).next;
            nd(cur).next = nx;
            nd(cur).prev = last;
            nd(nx).prev = cur;
            nd(last).next = cur;
        }
        return cur;
    }

    void removeNode(int i) {
        nd(nd(i).next).prev = nd(i).prev;
        nd(nd(i).prev).next = nd(i).next;
    }

    bool equals(int a, int b) { return nd(a).p.x == nd(b).p.x && nd(a).p.y == nd(b).p.y; }

    // mapbox sign convention for triangle orientation.
    double area(int p, int q, int r) {
        return (static_cast<double>(nd(q).p.y) - nd(p).p.y) *
                   (static_cast<double>(nd(r).p.x) - nd(q).p.x) -
               (static_cast<double>(nd(q).p.x) - nd(p).p.x) *
                   (static_cast<double>(nd(r).p.y) - nd(q).p.y);
    }

    static int sign(double v) { return (v > 0.0) - (v < 0.0); }

    static bool pointInTri(double ax, double ay, double bx, double by, double cx, double cy,
                           double px, double py) {
        return (cx - px) * (ay - py) - (ax - px) * (cy - py) >= 0 &&
               (ax - px) * (by - py) - (bx - px) * (ay - py) >= 0 &&
               (bx - px) * (cy - py) - (cx - px) * (by - py) >= 0;
    }

    void emitTri(int a, int b, int c) {
        m_sink->pushTriangle(nd(a).outIndex, nd(b).outIndex, nd(c).outIndex);
    }

    // Build a circular doubly linked list from a ring; orient per `clockwise`.
    int linkedList(const std::vector<UiVec2>& ring, bool clockwise) {
        const int len = static_cast<int>(ring.size());
        double sum = 0.0;
        for (int i = 0, j = len - 1; i < len; j = i++) {
            sum += (static_cast<double>(ring[j].x) - ring[i].x) *
                   (static_cast<double>(ring[i].y) + ring[j].y);
        }
        int last = -1;
        if (clockwise == (sum > 0.0)) {
            for (int i = 0; i < len; ++i) last = insertNode(ring[i], last);
        } else {
            for (int i = len - 1; i >= 0; --i) last = insertNode(ring[i], last);
        }
        if (last >= 0 && equals(last, nd(last).next)) {
            const int nx = nd(last).next;
            removeNode(last);
            last = nx;
        }
        return last;
    }

    // Remove collinear or duplicate points. Returns a valid node in the loop.
    int filterPoints(int start, int end) {
        if (start < 0) {
            return start;
        }
        if (end < 0) {
            end = start;
        }
        int p = start;
        bool again;
        do {
            again = false;
            if (!nd(p).steiner && (equals(p, nd(p).next) || area(nd(p).prev, p, nd(p).next) == 0.0)) {
                removeNode(p);
                p = end = nd(p).prev;
                if (p == nd(p).next) {
                    break;
                }
                again = true;
            } else {
                p = nd(p).next;
            }
        } while (again || p != end);
        return end;
    }

    bool isEar(int ear) {
        const int a = nd(ear).prev;
        const int b = ear;
        const int c = nd(ear).next;
        if (area(a, b, c) >= 0.0) {
            return false;  // Reflex or collinear corner — not an ear.
        }
        const double ax = nd(a).p.x, ay = nd(a).p.y;
        const double bx = nd(b).p.x, by = nd(b).p.y;
        const double cx = nd(c).p.x, cy = nd(c).p.y;
        int p = nd(c).next;
        while (p != a) {
            if (pointInTri(ax, ay, bx, by, cx, cy, nd(p).p.x, nd(p).p.y) &&
                area(nd(p).prev, p, nd(p).next) >= 0.0) {
                return false;
            }
            p = nd(p).next;
        }
        return true;
    }

    void earcutLinked(int ear, int pass) {
        if (ear < 0) {
            return;
        }
        int stop = ear;
        while (nd(ear).prev != nd(ear).next) {
            const int prev = nd(ear).prev;
            const int next = nd(ear).next;
            if (isEar(ear)) {
                emitTri(prev, ear, next);
                removeNode(ear);
                ear = nd(next).next;
                stop = nd(next).next;
                continue;
            }
            ear = next;
            if (ear == stop) {
                // No ear found on a full pass — escalate to a fallback strategy.
                if (pass == 0) {
                    earcutLinked(filterPoints(ear, -1), 1);
                } else if (pass == 1) {
                    ear = cureLocalIntersections(filterPoints(ear, -1));
                    earcutLinked(ear, 2);
                } else if (pass == 2) {
                    splitEarcut(ear);
                }
                break;
            }
        }
    }

    bool onSegment(int p, int q, int r) {
        return nd(q).p.x <= std::max(nd(p).p.x, nd(r).p.x) &&
               nd(q).p.x >= std::min(nd(p).p.x, nd(r).p.x) &&
               nd(q).p.y <= std::max(nd(p).p.y, nd(r).p.y) &&
               nd(q).p.y >= std::min(nd(p).p.y, nd(r).p.y);
    }

    bool intersects(int p1, int q1, int p2, int q2) {
        const int o1 = sign(area(p1, q1, p2));
        const int o2 = sign(area(p1, q1, q2));
        const int o3 = sign(area(p2, q2, p1));
        const int o4 = sign(area(p2, q2, q1));
        if (o1 != o2 && o3 != o4) {
            return true;
        }
        if (o1 == 0 && onSegment(p1, p2, q1)) return true;
        if (o2 == 0 && onSegment(p1, q2, q1)) return true;
        if (o3 == 0 && onSegment(p2, p1, q2)) return true;
        if (o4 == 0 && onSegment(p2, q1, q2)) return true;
        return false;
    }

    bool intersectsPolygon(int a, int b) {
        int p = a;
        do {
            const int pn = nd(p).next;
            if (p != a && pn != a && p != b && pn != b && intersects(p, pn, a, b)) {
                return true;
            }
            p = nd(p).next;
        } while (p != a);
        return false;
    }

    bool locallyInside(int a, int b) {
        return area(nd(a).prev, a, nd(a).next) < 0.0
                   ? area(a, b, nd(a).next) >= 0.0 && area(a, nd(a).prev, b) >= 0.0
                   : area(a, b, nd(a).prev) < 0.0 || area(a, nd(a).next, b) < 0.0;
    }

    bool middleInside(int a, int b) {
        int p = a;
        bool inside = false;
        const double px = (static_cast<double>(nd(a).p.x) + nd(b).p.x) * 0.5;
        const double py = (static_cast<double>(nd(a).p.y) + nd(b).p.y) * 0.5;
        do {
            const int pn = nd(p).next;
            if (((nd(p).p.y > py) != (nd(pn).p.y > py)) && nd(pn).p.y != nd(p).p.y &&
                (px < (static_cast<double>(nd(pn).p.x) - nd(p).p.x) * (py - nd(p).p.y) /
                              (static_cast<double>(nd(pn).p.y) - nd(p).p.y) +
                          nd(p).p.x)) {
                inside = !inside;
            }
            p = nd(p).next;
        } while (p != a);
        return inside;
    }

    bool isValidDiagonal(int a, int b) {
        return nd(a).next != b && nd(a).prev != b && !intersectsPolygon(a, b) &&
               locallyInside(a, b) && locallyInside(b, a) && middleInside(a, b);
    }

    // Splice a diagonal a-b, returning the new node b2 of the second loop.
    int splitPolygon(int a, int b) {
        const int an = nd(a).next;
        const int bp = nd(b).prev;
        const int a2 = createNode(nd(a).p);  // createNode may reallocate; indices stay valid.
        const int b2 = createNode(nd(b).p);
        nd(a).next = b;
        nd(b).prev = a;
        nd(a2).next = an;
        nd(an).prev = a2;
        nd(b2).next = a2;
        nd(a2).prev = b2;
        nd(bp).next = b2;
        nd(b2).prev = bp;
        return b2;
    }

    int cureLocalIntersections(int start) {
        int p = start;
        do {
            const int a = nd(p).prev;
            const int b = nd(nd(p).next).next;
            if (!equals(a, b) && intersects(a, p, nd(p).next, b) && locallyInside(a, b) &&
                locallyInside(b, a)) {
                emitTri(a, p, b);
                removeNode(p);
                removeNode(nd(p).next);
                p = start = b;
            }
            p = nd(p).next;
        } while (p != start);
        return filterPoints(p, -1);
    }

    void splitEarcut(int start) {
        int a = start;
        do {
            int b = nd(nd(a).next).next;
            while (b != nd(a).prev) {
                if (nd(a).outIndex != nd(b).outIndex && isValidDiagonal(a, b)) {
                    int c = splitPolygon(a, b);
                    a = filterPoints(a, nd(a).next);
                    c = filterPoints(c, nd(c).next);
                    earcutLinked(a, 0);
                    earcutLinked(c, 0);
                    return;
                }
                b = nd(b).next;
            }
            a = nd(a).next;
        } while (a != start);
    }

    int getLeftmost(int start) {
        int p = start;
        int leftmost = start;
        do {
            if (nd(p).p.x < nd(leftmost).p.x ||
                (nd(p).p.x == nd(leftmost).p.x && nd(p).p.y < nd(leftmost).p.y)) {
                leftmost = p;
            }
            p = nd(p).next;
        } while (p != start);
        return leftmost;
    }

    int findHoleBridge(int hole, int outerNode) {
        int p = outerNode;
        const double hx = nd(hole).p.x;
        const double hy = nd(hole).p.y;
        double qx = -std::numeric_limits<double>::infinity();
        int m = -1;
        do {
            const int pn = nd(p).next;
            if (hy <= nd(p).p.y && hy >= nd(pn).p.y && nd(pn).p.y != nd(p).p.y) {
                const double x = nd(p).p.x + (hy - nd(p).p.y) *
                                                 (static_cast<double>(nd(pn).p.x) - nd(p).p.x) /
                                                 (static_cast<double>(nd(pn).p.y) - nd(p).p.y);
                if (x <= hx && x > qx) {
                    qx = x;
                    m = nd(p).p.x < nd(pn).p.x ? p : pn;
                    if (x == hx) {
                        return m;
                    }
                }
            }
            p = nd(p).next;
        } while (p != outerNode);

        if (m < 0) {
            return -1;
        }
        // Find a reflex vertex inside the (hole, bridge) triangle, closest by angle,
        // to use as the visible bridge endpoint.
        const int stop = m;
        const double mx = nd(m).p.x;
        const double my = nd(m).p.y;
        double tanMin = std::numeric_limits<double>::infinity();
        p = m;
        do {
            if (hx >= nd(p).p.x && nd(p).p.x >= mx && hx != nd(p).p.x &&
                pointInTri(hy < my ? hx : qx, hy, mx, my, hy < my ? qx : hx, hy, nd(p).p.x,
                           nd(p).p.y)) {
                const double tanCur = std::fabs(hy - nd(p).p.y) / (hx - nd(p).p.x);
                if (locallyInside(p, hole) &&
                    (tanCur < tanMin || (tanCur == tanMin && nd(p).p.x > nd(m).p.x))) {
                    m = p;
                    tanMin = tanCur;
                }
            }
            p = nd(p).next;
        } while (p != stop);
        return m;
    }

    void eliminateHole(int hole, int outerNode) {
        const int bridge = findHoleBridge(hole, outerNode);
        if (bridge < 0) {
            return;
        }
        const int bridgeReverse = splitPolygon(bridge, hole);
        filterPoints(bridge, nd(bridge).next);
        filterPoints(bridgeReverse, nd(bridgeReverse).next);
    }

    int eliminateHoles(const std::vector<std::vector<UiVec2>>& rings, int outerNode) {
        std::vector<int> queue;
        for (std::size_t r = 1; r < rings.size(); ++r) {
            if (rings[r].size() < 3) {
                continue;
            }
            int list = linkedList(rings[r], false);
            if (list >= 0) {
                if (list == nd(list).next) {
                    nd(list).steiner = true;
                }
                queue.push_back(getLeftmost(list));
            }
        }
        std::sort(queue.begin(), queue.end(),
                  [this](int a, int b) { return nd(a).p.x < nd(b).p.x; });
        for (int q : queue) {
            eliminateHole(q, outerNode);
        }
        return outerNode;
    }

    std::vector<Node> m_nodes;
    IMeshSink* m_sink = nullptr;
    std::uint32_t m_rgba = 0;
};

// Emit a fringe quad along edge a->b: the inner edge sits on the path with full
// color, the outer edge is pushed `fringe` px along the outward normal with
// alpha 0. Triangle winding doesn't matter (no culling in the UI pipeline).
void emitFringeEdge(const UiVec2& a, const UiVec2& b, const UiVec2& outwardN, float fringe,
                    std::uint32_t rgbaFull, IMeshSink& sink) {
    const std::uint32_t rgbaZero = withZeroAlpha(rgbaFull);
    const std::uint32_t ia = sink.pushVertex(a.x, a.y, rgbaFull);
    const std::uint32_t ib = sink.pushVertex(b.x, b.y, rgbaFull);
    const std::uint32_t oa =
        sink.pushVertex(a.x + outwardN.x * fringe, a.y + outwardN.y * fringe, rgbaZero);
    const std::uint32_t ob =
        sink.pushVertex(b.x + outwardN.x * fringe, b.y + outwardN.y * fringe, rgbaZero);
    sink.pushTriangle(ia, ib, ob);
    sink.pushTriangle(ia, ob, oa);
}

// Outward (away-from-fill) normal of edge a->b. After orienting solid rings to
// area>0 and holes to area<0, fill is consistently on the left, so the
// right-hand normal (dy,-dx) points away from fill for every ring.
void emitRingFringe(const std::vector<UiVec2>& ring, float fringe, std::uint32_t rgbaFull,
                    IMeshSink& sink) {
    const std::size_t n = ring.size();
    for (std::size_t i = 0; i < n; ++i) {
        const UiVec2& a = ring[i];
        const UiVec2& b = ring[(i + 1) % n];
        const UiVec2 outward = normalize(b.y - a.y, -(b.x - a.x));
        if (outward.x == 0.0f && outward.y == 0.0f) {
            continue;
        }
        emitFringeEdge(a, b, outward, fringe, rgbaFull, sink);
    }
}

}  // namespace

void tessellateFill(const VectorPath& path, std::uint32_t rgba8, const TessOptions& opts,
                    IMeshSink& sink) {
    // Gather closed contours with >= 3 points.
    std::vector<std::vector<UiVec2>> contours;
    for (const VectorSubPath& sp : path.subPaths()) {
        if (sp.points.size() >= 3) {
            contours.push_back(sp.points);
        }
    }
    if (contours.empty()) {
        return;
    }

    const std::size_t n = contours.size();
    std::vector<int> depth(n, 0);
    std::vector<float> areas(n);
    for (std::size_t i = 0; i < n; ++i) {
        areas[i] = signedArea(contours[i]);
    }
    // Containment depth: how many other contours contain contour i's first vertex.
    for (std::size_t i = 0; i < n; ++i) {
        const UiVec2 sample = contours[i][0];
        for (std::size_t j = 0; j < n; ++j) {
            if (j != i && pointInPolygon(contours[j], sample)) {
                ++depth[i];
            }
        }
    }

    // Classify each contour as a filled-region boundary (solid) or a hole.
    std::vector<bool> solid(n, false);
    for (std::size_t i = 0; i < n; ++i) {
        if (opts.fillRule == FillRule::EvenOdd) {
            solid[i] = (depth[i] % 2) == 0;
        } else {
            // Nonzero: winding number at a sample point of contour i = sum of the
            // orientation signs of all contours containing that point (incl. i).
            const UiVec2 sample = contours[i][0];
            int winding = 0;
            for (std::size_t j = 0; j < n; ++j) {
                const bool contains = (j == i) || pointInPolygon(contours[j], sample);
                if (contains) {
                    winding += (areas[j] >= 0.0f) ? 1 : -1;
                }
            }
            solid[i] = winding != 0;
        }
    }

    const float fringe = opts.aaFringePx * std::max(opts.dpiScale, 0.01f);

    // Triangulate each solid contour together with the holes nested directly
    // inside it (one depth deeper and not solid).
    for (std::size_t i = 0; i < n; ++i) {
        if (!solid[i]) {
            continue;
        }
        std::vector<std::vector<UiVec2>> rings;
        rings.push_back(contours[i]);  // outer (re-oriented inside the clipper)
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i || solid[j] || depth[j] != depth[i] + 1) {
                continue;
            }
            if (pointInPolygon(contours[i], contours[j][0])) {
                rings.push_back(contours[j]);
            }
        }
        EarClipper clipper;
        clipper.triangulate(rings, rgba8, sink);

        if (fringe > 0.0f) {
            // Fringe each boundary ring after orienting it the same way the
            // clipper does (solid outer area>0, holes area<0) so the outward
            // normal points away from fill.
            std::vector<UiVec2> outer = rings[0];
            if (signedArea(outer) < 0.0f) std::reverse(outer.begin(), outer.end());
            emitRingFringe(outer, fringe, rgba8, sink);
            for (std::size_t k = 1; k < rings.size(); ++k) {
                std::vector<UiVec2> hole = rings[k];
                if (signedArea(hole) > 0.0f) std::reverse(hole.begin(), hole.end());
                emitRingFringe(hole, fringe, rgba8, sink);
            }
        }
    }
}

void tessellateStroke(const VectorPath& path, std::uint32_t rgba8, const StrokeOptions& opts,
                      IMeshSink& sink) {
    const float halfW = std::max(opts.widthPx * 0.5f, 0.0f);
    if (halfW <= 0.0f) {
        return;
    }
    const float fringe = std::max(opts.aaFringePx * std::max(opts.dpiScale, 0.01f), 0.0f);
    const std::uint32_t rgbaZero = withZeroAlpha(rgba8);

    for (const VectorSubPath& sp : path.subPaths()) {
        // Build the working point list; drop the duplicated closing point.
        std::vector<UiVec2> pts = sp.points;
        if (pts.size() < 2) {
            continue;
        }
        const bool closed = sp.closed;
        if (closed && pts.size() > 1) {
            const UiVec2& f = pts.front();
            const UiVec2& l = pts.back();
            if (std::fabs(f.x - l.x) < 1e-5f && std::fabs(f.y - l.y) < 1e-5f) {
                pts.pop_back();
            }
        }
        const int count = static_cast<int>(pts.size());
        if (count < 2) {
            continue;
        }

        // Per-segment unit direction and left normal (-dy, dx).
        const int segCount = closed ? count : count - 1;
        std::vector<UiVec2> segDir(segCount);
        std::vector<UiVec2> segNrm(segCount);
        for (int s = 0; s < segCount; ++s) {
            const UiVec2& a = pts[s];
            const UiVec2& b = pts[(s + 1) % count];
            const UiVec2 d = normalize(b.x - a.x, b.y - a.y);
            segDir[s] = d;
            segNrm[s] = UiVec2{-d.y, d.x};  // left normal
        }

        // Emit a quad (corners in order) as two triangles, all the same color.
        auto quad = [&](const UiVec2& a0, const UiVec2& a1, const UiVec2& a2, const UiVec2& a3,
                        std::uint32_t c) {
            const std::uint32_t i0 = sink.pushVertex(a0.x, a0.y, c);
            const std::uint32_t i1 = sink.pushVertex(a1.x, a1.y, c);
            const std::uint32_t i2 = sink.pushVertex(a2.x, a2.y, c);
            const std::uint32_t i3 = sink.pushVertex(a3.x, a3.y, c);
            sink.pushTriangle(i0, i1, i2);
            sink.pushTriangle(i0, i2, i3);
        };
        // Fringe quad: inner edge (in0,in1) full alpha, outer edge (out0,out1) alpha 0.
        auto fringeQuad = [&](const UiVec2& in0, const UiVec2& in1, const UiVec2& out1,
                              const UiVec2& out0) {
            const std::uint32_t i0 = sink.pushVertex(in0.x, in0.y, rgba8);
            const std::uint32_t i1 = sink.pushVertex(in1.x, in1.y, rgba8);
            const std::uint32_t i2 = sink.pushVertex(out1.x, out1.y, rgbaZero);
            const std::uint32_t i3 = sink.pushVertex(out0.x, out0.y, rgbaZero);
            sink.pushTriangle(i0, i1, i2);
            sink.pushTriangle(i0, i2, i3);
        };

        // Emit each segment as a core quad plus an AA fringe on both sides.
        for (int s = 0; s < segCount; ++s) {
            const UiVec2& a = pts[s];
            const UiVec2& b = pts[(s + 1) % count];
            const UiVec2 nrm = segNrm[s];
            const UiVec2 left0{a.x + nrm.x * halfW, a.y + nrm.y * halfW};
            const UiVec2 left1{b.x + nrm.x * halfW, b.y + nrm.y * halfW};
            const UiVec2 right0{a.x - nrm.x * halfW, a.y - nrm.y * halfW};
            const UiVec2 right1{b.x - nrm.x * halfW, b.y - nrm.y * halfW};
            // Core (full alpha).
            quad(left0, left1, right1, right0, rgba8);
            if (fringe > 0.0f) {
                const UiVec2 lf0{left0.x + nrm.x * fringe, left0.y + nrm.y * fringe};
                const UiVec2 lf1{left1.x + nrm.x * fringe, left1.y + nrm.y * fringe};
                const UiVec2 rf0{right0.x - nrm.x * fringe, right0.y - nrm.y * fringe};
                const UiVec2 rf1{right1.x - nrm.x * fringe, right1.y - nrm.y * fringe};
                fringeQuad(left0, left1, lf1, lf0);   // left side fade-out
                fringeQuad(right0, right1, rf1, rf0);  // right side fade-out
            }
        }

        // Joins: fill the wedge at each interior vertex with a fan (round) or a
        // single triangle (bevel/miter-fallback). Miter extends the offset lines.
        const int jointStart = closed ? 0 : 1;
        const int jointEnd = closed ? count : count - 1;
        for (int v = jointStart; v < jointEnd; ++v) {
            const int sPrev = (v - 1 + segCount) % segCount;
            const int sCur = v % segCount;
            const UiVec2& nPrev = segNrm[sPrev];
            const UiVec2& nCur = segNrm[sCur];
            const UiVec2& p = pts[v];
            // Turn direction: cross of prev dir and cur dir.
            const float turn = segDir[sPrev].x * segDir[sCur].y - segDir[sPrev].y * segDir[sCur].x;
            // Outer side is opposite the turn. Use both offset points on the outer
            // side and fill the gap.
            const float side = (turn < 0.0f) ? 1.0f : -1.0f;  // +left or -left
            const UiVec2 o0{p.x + nPrev.x * halfW * side, p.y + nPrev.y * halfW * side};
            const UiVec2 o1{p.x + nCur.x * halfW * side, p.y + nCur.y * halfW * side};

            if (opts.join == LineJoin::Round) {
                float a0 = std::atan2(o0.y - p.y, o0.x - p.x);
                float a1 = std::atan2(o1.y - p.y, o1.x - p.x);
                // Sweep the short way around.
                float diff = a1 - a0;
                while (diff > kPi) diff -= 2.0f * kPi;
                while (diff < -kPi) diff += 2.0f * kPi;
                const int steps = std::max(1, static_cast<int>(std::ceil(std::fabs(diff) / 0.4f)));
                UiVec2 prevPt = o0;
                const std::uint32_t ic = sink.pushVertex(p.x, p.y, rgba8);
                for (int k = 1; k <= steps; ++k) {
                    const float a = a0 + diff * (static_cast<float>(k) / static_cast<float>(steps));
                    const UiVec2 cur{p.x + std::cos(a) * halfW, p.y + std::sin(a) * halfW};
                    const std::uint32_t ip = sink.pushVertex(prevPt.x, prevPt.y, rgba8);
                    const std::uint32_t iq = sink.pushVertex(cur.x, cur.y, rgba8);
                    sink.pushTriangle(ic, ip, iq);
                    prevPt = cur;
                }
            } else if (opts.join == LineJoin::Miter) {
                // Miter point = intersection of the two outer offset lines.
                const UiVec2 mid = normalize(nPrev.x * side + nCur.x * side,
                                             nPrev.y * side + nCur.y * side);
                const float denom = mid.x * (nPrev.x * side) + mid.y * (nPrev.y * side);
                if (denom > 1e-4f) {
                    const float miterLen = halfW / denom;
                    if (miterLen <= opts.miterLimit * halfW) {
                        const UiVec2 mpt{p.x + mid.x * miterLen, p.y + mid.y * miterLen};
                        const std::uint32_t ic = sink.pushVertex(p.x, p.y, rgba8);
                        const std::uint32_t i0 = sink.pushVertex(o0.x, o0.y, rgba8);
                        const std::uint32_t im = sink.pushVertex(mpt.x, mpt.y, rgba8);
                        const std::uint32_t i1 = sink.pushVertex(o1.x, o1.y, rgba8);
                        sink.pushTriangle(ic, i0, im);
                        sink.pushTriangle(ic, im, i1);
                        continue;
                    }
                }
                // Fall through to bevel if the miter is too long / degenerate.
                const std::uint32_t ic = sink.pushVertex(p.x, p.y, rgba8);
                const std::uint32_t i0 = sink.pushVertex(o0.x, o0.y, rgba8);
                const std::uint32_t i1 = sink.pushVertex(o1.x, o1.y, rgba8);
                sink.pushTriangle(ic, i0, i1);
            } else {  // Bevel
                const std::uint32_t ic = sink.pushVertex(p.x, p.y, rgba8);
                const std::uint32_t i0 = sink.pushVertex(o0.x, o0.y, rgba8);
                const std::uint32_t i1 = sink.pushVertex(o1.x, o1.y, rgba8);
                sink.pushTriangle(ic, i0, i1);
            }
        }

        // Caps for open paths.
        if (!closed) {
            auto emitCap = [&](const UiVec2& end, const UiVec2& dir, const UiVec2& nrm) {
                if (opts.cap == LineCap::Butt) {
                    return;
                }
                const UiVec2 l{end.x + nrm.x * halfW, end.y + nrm.y * halfW};
                const UiVec2 r{end.x - nrm.x * halfW, end.y - nrm.y * halfW};
                if (opts.cap == LineCap::Square) {
                    const UiVec2 le{l.x + dir.x * halfW, l.y + dir.y * halfW};
                    const UiVec2 re{r.x + dir.x * halfW, r.y + dir.y * halfW};
                    quad(l, le, re, r, rgba8);
                } else {  // Round
                    const float a0 = std::atan2(l.y - end.y, l.x - end.x);
                    const int steps = std::max(2, static_cast<int>(std::ceil(kPi / 0.4f)));
                    const std::uint32_t ic = sink.pushVertex(end.x, end.y, rgba8);
                    UiVec2 prevPt = l;
                    for (int k = 1; k <= steps; ++k) {
                        // Sweep pi around toward r, going through the cap direction.
                        const float a = a0 + kPi * (static_cast<float>(k) / static_cast<float>(steps)) *
                                                 ((dir.x * (-nrm.y) + dir.y * nrm.x) >= 0 ? 1.0f : -1.0f);
                        const UiVec2 cur{end.x + std::cos(a) * halfW, end.y + std::sin(a) * halfW};
                        const std::uint32_t ip = sink.pushVertex(prevPt.x, prevPt.y, rgba8);
                        const std::uint32_t iq = sink.pushVertex(cur.x, cur.y, rgba8);
                        sink.pushTriangle(ic, ip, iq);
                        prevPt = cur;
                    }
                }
            };
            // Start cap points backward along the first segment.
            emitCap(pts.front(), UiVec2{-segDir.front().x, -segDir.front().y}, segNrm.front());
            // End cap points forward along the last segment.
            emitCap(pts.back(), segDir.back(), segNrm.back());
        }
    }
}

}  // namespace odai::ui
