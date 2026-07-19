#pragma once

#include <cstdint>
#include <vector>

#include "math/math.h"

// Pure-CPU constructive solid geometry on low-poly polygon meshes.
// Meshes are closed polygon soups (watertight solid boundaries) of convex CCW
// polygons carrying a flat per-face color. Booleans use the csg.js BSP
// algorithm; inputs are expected to be tiny (tens of polygons) and built in
// local space near the origin so a fixed absolute plane epsilon is meaningful.
namespace odai::procgen {

using odai::math::Vector3;

struct Color3 {
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
};

inline Color3 fromRgbHex(std::uint32_t hex) {
    return Color3{
        static_cast<float>((hex >> 16) & 0xffu) / 255.0f,
        static_cast<float>((hex >> 8) & 0xffu) / 255.0f,
        static_cast<float>(hex & 0xffu) / 255.0f,
    };
}

inline Color3 mix(const Color3& a, const Color3& b, float t) {
    return Color3{
        a.r + (b.r - a.r) * t,
        a.g + (b.g - a.g) * t,
        a.b + (b.b - a.b) * t,
    };
}

struct Plane {
    Vector3 normal;   // unit length
    float w = 0.0f;   // dot(normal, p) == w for points on the plane

    // Newell-style normal over the full vertex loop; more stable than a
    // three-point cross product for quads perturbed by transforms.
    static Plane fromVertices(const std::vector<Vector3>& vertices);
    void flip();
};

// Convex planar polygon; vertices wound CCW seen from the outward normal side.
struct Polygon {
    std::vector<Vector3> vertices;
    Plane plane;
    Color3 color;

    void flip();
};

struct CsgMesh {
    std::vector<Polygon> polygons;
};

CsgMesh csgUnion(const CsgMesh& a, const CsgMesh& b);
CsgMesh csgSubtract(const CsgMesh& a, const CsgMesh& b);
CsgMesh csgIntersect(const CsgMesh& a, const CsgMesh& b);

// Cheap union: concatenates boundaries without removing the overlap. Correct
// for opaque flat-shaded rendering whenever the seam between the two solids is
// enclosed (small attached details); avoids BSP face splitting entirely.
void merge(CsgMesh& dst, const CsgMesh& src);

// Rigid/affine helpers. Planes are recomputed from the transformed vertices.
void translate(CsgMesh& mesh, const Vector3& offset);
void scaleMesh(CsgMesh& mesh, const Vector3& factors);  // about the origin
void rotateY(CsgMesh& mesh, float radiansAngle);        // about the origin
void paint(CsgMesh& mesh, const Color3& color);

}  // namespace odai::procgen
