#include "procgen/csg.h"

#include <cmath>
#include <memory>
#include <utility>

namespace odai::procgen {

namespace {

// Buildings are authored in local space (coords roughly 0..3), so an absolute
// epsilon is safe. Kept loose enough that float drift from transforms still
// classifies shared axis-aligned faces as coplanar.
constexpr float kPlaneEpsilon = 1e-4f;
constexpr float kAxisSnapEpsilon = 1e-5f;
constexpr float kMinFragmentArea = 1e-8f;

float snapAxisComponent(float v) {
    if (std::fabs(v) < kAxisSnapEpsilon) {
        return 0.0f;
    }
    if (std::fabs(v - 1.0f) < kAxisSnapEpsilon) {
        return 1.0f;
    }
    if (std::fabs(v + 1.0f) < kAxisSnapEpsilon) {
        return -1.0f;
    }
    return v;
}

// Snap near-axis normals to exact axis vectors so stacked axis-aligned boxes
// classify each other's faces as exactly coplanar instead of producing shards.
Vector3 snapAxis(const Vector3& n) {
    Vector3 snapped{snapAxisComponent(n.x), snapAxisComponent(n.y), snapAxisComponent(n.z)};
    const float len = odai::math::length(snapped);
    if (len <= 0.0f) {
        return n;
    }
    return snapped / len;
}

float polygonArea(const std::vector<Vector3>& vertices) {
    Vector3 sum{};
    for (std::size_t i = 1; i + 1 < vertices.size(); ++i) {
        sum += odai::math::cross(vertices[i] - vertices[0], vertices[i + 1] - vertices[0]);
    }
    return 0.5f * odai::math::length(sum);
}

}  // namespace

Plane Plane::fromVertices(const std::vector<Vector3>& vertices) {
    Vector3 normal{};
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const Vector3& current = vertices[i];
        const Vector3& next = vertices[(i + 1) % vertices.size()];
        normal.x += (current.y - next.y) * (current.z + next.z);
        normal.y += (current.z - next.z) * (current.x + next.x);
        normal.z += (current.x - next.x) * (current.y + next.y);
    }
    Plane plane;
    plane.normal = snapAxis(odai::math::normalize(normal));
    // Averaging w over all vertices resists per-vertex float noise.
    float w = 0.0f;
    for (const Vector3& v : vertices) {
        w += odai::math::dot(plane.normal, v);
    }
    plane.w = vertices.empty() ? 0.0f : w / static_cast<float>(vertices.size());
    return plane;
}

void Plane::flip() {
    normal = -normal;
    w = -w;
}

void Polygon::flip() {
    std::vector<Vector3> reversed(vertices.rbegin(), vertices.rend());
    vertices = std::move(reversed);
    plane.flip();
}

namespace {

enum PolygonClass : int {
    kCoplanar = 0,
    kFront = 1,
    kBack = 2,
    kSpanning = 3,
};

void splitPolygon(const Plane& splitter, const Polygon& polygon,
                  std::vector<Polygon>& coplanarFront, std::vector<Polygon>& coplanarBack,
                  std::vector<Polygon>& front, std::vector<Polygon>& back) {
    int polygonType = 0;
    std::vector<int> types;
    types.reserve(polygon.vertices.size());
    for (const Vector3& v : polygon.vertices) {
        const float t = odai::math::dot(splitter.normal, v) - splitter.w;
        const int type = (t < -kPlaneEpsilon) ? kBack : ((t > kPlaneEpsilon) ? kFront : kCoplanar);
        polygonType |= type;
        types.push_back(type);
    }

    switch (polygonType) {
        case kCoplanar:
            if (odai::math::dot(splitter.normal, polygon.plane.normal) > 0.0f) {
                coplanarFront.push_back(polygon);
            } else {
                coplanarBack.push_back(polygon);
            }
            break;
        case kFront:
            front.push_back(polygon);
            break;
        case kBack:
            back.push_back(polygon);
            break;
        case kSpanning: {
            std::vector<Vector3> frontVerts;
            std::vector<Vector3> backVerts;
            const std::size_t count = polygon.vertices.size();
            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t j = (i + 1) % count;
                const int ti = types[i];
                const int tj = types[j];
                const Vector3& vi = polygon.vertices[i];
                const Vector3& vj = polygon.vertices[j];
                if (ti != kBack) {
                    frontVerts.push_back(vi);
                }
                if (ti != kFront) {
                    backVerts.push_back(vi);
                }
                if ((ti | tj) == kSpanning) {
                    const Vector3 edge = vj - vi;
                    const float denom = odai::math::dot(splitter.normal, edge);
                    const float t = (splitter.w - odai::math::dot(splitter.normal, vi)) / denom;
                    const Vector3 v = vi + edge * t;
                    frontVerts.push_back(v);
                    backVerts.push_back(v);
                }
            }
            // Drop epsilon slivers so shards from near-coplanar splits never
            // reach emission.
            if (frontVerts.size() >= 3 && polygonArea(frontVerts) > kMinFragmentArea) {
                Polygon p;
                p.vertices = std::move(frontVerts);
                p.plane = polygon.plane;
                p.color = polygon.color;
                front.push_back(std::move(p));
            }
            if (backVerts.size() >= 3 && polygonArea(backVerts) > kMinFragmentArea) {
                Polygon p;
                p.vertices = std::move(backVerts);
                p.plane = polygon.plane;
                p.color = polygon.color;
                back.push_back(std::move(p));
            }
            break;
        }
        default:
            break;
    }
}

struct BspNode {
    bool hasPlane = false;
    Plane plane;
    std::unique_ptr<BspNode> front;
    std::unique_ptr<BspNode> back;
    std::vector<Polygon> polygons;

    void invert() {
        for (Polygon& p : polygons) {
            p.flip();
        }
        if (hasPlane) {
            plane.flip();
        }
        if (front) {
            front->invert();
        }
        if (back) {
            back->invert();
        }
        std::swap(front, back);
    }

    std::vector<Polygon> clipPolygons(const std::vector<Polygon>& input) const {
        if (!hasPlane) {
            return input;
        }
        std::vector<Polygon> frontPolys;
        std::vector<Polygon> backPolys;
        for (const Polygon& p : input) {
            // Coplanar polygons route with the front/back sets.
            splitPolygon(plane, p, frontPolys, backPolys, frontPolys, backPolys);
        }
        std::vector<Polygon> result;
        if (front) {
            result = front->clipPolygons(frontPolys);
        } else {
            result = std::move(frontPolys);
        }
        if (back) {
            std::vector<Polygon> clippedBack = back->clipPolygons(backPolys);
            result.insert(result.end(), std::make_move_iterator(clippedBack.begin()),
                          std::make_move_iterator(clippedBack.end()));
        }
        // No back child: back polygons are inside the solid and are discarded.
        return result;
    }

    void clipTo(const BspNode& bsp) {
        polygons = bsp.clipPolygons(polygons);
        if (front) {
            front->clipTo(bsp);
        }
        if (back) {
            back->clipTo(bsp);
        }
    }

    std::vector<Polygon> allPolygons() const {
        std::vector<Polygon> result = polygons;
        if (front) {
            std::vector<Polygon> fp = front->allPolygons();
            result.insert(result.end(), std::make_move_iterator(fp.begin()),
                          std::make_move_iterator(fp.end()));
        }
        if (back) {
            std::vector<Polygon> bp = back->allPolygons();
            result.insert(result.end(), std::make_move_iterator(bp.begin()),
                          std::make_move_iterator(bp.end()));
        }
        return result;
    }

    void build(std::vector<Polygon> input) {
        if (input.empty()) {
            return;
        }
        if (!hasPlane) {
            plane = input[0].plane;
            hasPlane = true;
        }
        std::vector<Polygon> frontPolys;
        std::vector<Polygon> backPolys;
        for (Polygon& p : input) {
            splitPolygon(plane, p, polygons, polygons, frontPolys, backPolys);
        }
        if (!frontPolys.empty()) {
            if (!front) {
                front = std::make_unique<BspNode>();
            }
            front->build(std::move(frontPolys));
        }
        if (!backPolys.empty()) {
            if (!back) {
                back = std::make_unique<BspNode>();
            }
            back->build(std::move(backPolys));
        }
    }
};

}  // namespace

CsgMesh csgUnion(const CsgMesh& a, const CsgMesh& b) {
    BspNode nodeA;
    BspNode nodeB;
    nodeA.build(a.polygons);
    nodeB.build(b.polygons);
    nodeA.clipTo(nodeB);
    nodeB.clipTo(nodeA);
    nodeB.invert();
    nodeB.clipTo(nodeA);
    nodeB.invert();
    nodeA.build(nodeB.allPolygons());
    return CsgMesh{nodeA.allPolygons()};
}

CsgMesh csgSubtract(const CsgMesh& a, const CsgMesh& b) {
    BspNode nodeA;
    BspNode nodeB;
    nodeA.build(a.polygons);
    nodeB.build(b.polygons);
    nodeA.invert();
    nodeA.clipTo(nodeB);
    nodeB.clipTo(nodeA);
    nodeB.invert();
    nodeB.clipTo(nodeA);
    nodeB.invert();
    nodeA.build(nodeB.allPolygons());
    nodeA.invert();
    return CsgMesh{nodeA.allPolygons()};
}

CsgMesh csgIntersect(const CsgMesh& a, const CsgMesh& b) {
    BspNode nodeA;
    BspNode nodeB;
    nodeA.build(a.polygons);
    nodeB.build(b.polygons);
    nodeA.invert();
    nodeB.clipTo(nodeA);
    nodeB.invert();
    nodeA.clipTo(nodeB);
    nodeB.clipTo(nodeA);
    nodeA.build(nodeB.allPolygons());
    nodeA.invert();
    return CsgMesh{nodeA.allPolygons()};
}

void merge(CsgMesh& dst, const CsgMesh& src) {
    dst.polygons.insert(dst.polygons.end(), src.polygons.begin(), src.polygons.end());
}

void translate(CsgMesh& mesh, const Vector3& offset) {
    for (Polygon& p : mesh.polygons) {
        for (Vector3& v : p.vertices) {
            v += offset;
        }
        p.plane.w += odai::math::dot(p.plane.normal, offset);
    }
}

void scaleMesh(CsgMesh& mesh, const Vector3& factors) {
    for (Polygon& p : mesh.polygons) {
        for (Vector3& v : p.vertices) {
            v.x *= factors.x;
            v.y *= factors.y;
            v.z *= factors.z;
        }
        // Negative factors would flip winding; generators only use positive
        // scales, so recomputing the plane from vertices is sufficient.
        p.plane = Plane::fromVertices(p.vertices);
    }
}

void rotateY(CsgMesh& mesh, float radiansAngle) {
    const float c = std::cos(radiansAngle);
    const float s = std::sin(radiansAngle);
    for (Polygon& p : mesh.polygons) {
        for (Vector3& v : p.vertices) {
            const float x = v.x * c + v.z * s;
            const float z = -v.x * s + v.z * c;
            v.x = x;
            v.z = z;
        }
        p.plane = Plane::fromVertices(p.vertices);
    }
}

void paint(CsgMesh& mesh, const Color3& color) {
    for (Polygon& p : mesh.polygons) {
        p.color = color;
    }
}

}  // namespace odai::procgen
