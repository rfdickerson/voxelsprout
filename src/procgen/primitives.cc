#include "procgen/primitives.h"

#include <cmath>
#include <utility>

namespace odai::procgen {

namespace {

Polygon makePolygon(std::vector<Vector3> vertices, const Color3& color) {
    Polygon p;
    p.plane = Plane::fromVertices(vertices);
    p.vertices = std::move(vertices);
    p.color = color;
    return p;
}

}  // namespace

CsgMesh makeBox(const Vector3& minCorner, const Vector3& maxCorner, const Color3& color) {
    const float x0 = minCorner.x, y0 = minCorner.y, z0 = minCorner.z;
    const float x1 = maxCorner.x, y1 = maxCorner.y, z1 = maxCorner.z;
    CsgMesh mesh;
    mesh.polygons.reserve(6);
    // Bottom (-Y)
    mesh.polygons.push_back(makePolygon({{x0, y0, z0}, {x1, y0, z0}, {x1, y0, z1}, {x0, y0, z1}}, color));
    // Top (+Y)
    mesh.polygons.push_back(makePolygon({{x0, y1, z0}, {x0, y1, z1}, {x1, y1, z1}, {x1, y1, z0}}, color));
    // Front (-Z)
    mesh.polygons.push_back(makePolygon({{x0, y0, z0}, {x0, y1, z0}, {x1, y1, z0}, {x1, y0, z0}}, color));
    // Back (+Z)
    mesh.polygons.push_back(makePolygon({{x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1}}, color));
    // Left (-X)
    mesh.polygons.push_back(makePolygon({{x0, y0, z0}, {x0, y0, z1}, {x0, y1, z1}, {x0, y1, z0}}, color));
    // Right (+X)
    mesh.polygons.push_back(makePolygon({{x1, y0, z0}, {x1, y1, z0}, {x1, y1, z1}, {x1, y0, z1}}, color));
    return mesh;
}

CsgMesh makeGablePrism(float minX, float minZ, float maxX, float maxZ,
                       float y0, float eaveY, float ridgeY, const Color3& color) {
    const float midZ = 0.5f * (minZ + maxZ);
    CsgMesh mesh;
    mesh.polygons.reserve(7);
    // Bottom (-Y)
    mesh.polygons.push_back(makePolygon(
        {{minX, y0, minZ}, {maxX, y0, minZ}, {maxX, y0, maxZ}, {minX, y0, maxZ}}, color));
    // Walls (-Z / +Z)
    mesh.polygons.push_back(makePolygon(
        {{minX, y0, minZ}, {minX, eaveY, minZ}, {maxX, eaveY, minZ}, {maxX, y0, minZ}}, color));
    mesh.polygons.push_back(makePolygon(
        {{minX, y0, maxZ}, {maxX, y0, maxZ}, {maxX, eaveY, maxZ}, {minX, eaveY, maxZ}}, color));
    // Roof planes meeting at the X-aligned ridge
    mesh.polygons.push_back(makePolygon(
        {{minX, eaveY, minZ}, {minX, ridgeY, midZ}, {maxX, ridgeY, midZ}, {maxX, eaveY, minZ}}, color));
    mesh.polygons.push_back(makePolygon(
        {{minX, eaveY, maxZ}, {maxX, eaveY, maxZ}, {maxX, ridgeY, midZ}, {minX, ridgeY, midZ}}, color));
    // Gable end pentagons (-X / +X)
    mesh.polygons.push_back(makePolygon(
        {{minX, y0, maxZ}, {minX, eaveY, maxZ}, {minX, ridgeY, midZ}, {minX, eaveY, minZ}, {minX, y0, minZ}},
        color));
    mesh.polygons.push_back(makePolygon(
        {{maxX, y0, minZ}, {maxX, eaveY, minZ}, {maxX, ridgeY, midZ}, {maxX, eaveY, maxZ}, {maxX, y0, maxZ}},
        color));
    return mesh;
}

CsgMesh makeCylinder(const Vector3& baseCenter, float radius, float height,
                     int segments, const Color3& color) {
    if (segments < 3) {
        segments = 3;
    }
    const float topY = baseCenter.y + height;
    std::vector<Vector3> bottomRing;
    std::vector<Vector3> topRing;
    bottomRing.reserve(static_cast<std::size_t>(segments));
    topRing.reserve(static_cast<std::size_t>(segments));
    for (int i = 0; i < segments; ++i) {
        const float angle = 2.0f * odai::math::kPi * static_cast<float>(i) / static_cast<float>(segments);
        // (cos, -sin) so increasing i winds CCW seen from +Y.
        const float x = baseCenter.x + radius * std::cos(angle);
        const float z = baseCenter.z - radius * std::sin(angle);
        bottomRing.push_back({x, baseCenter.y, z});
        topRing.push_back({x, topY, z});
    }

    CsgMesh mesh;
    mesh.polygons.reserve(static_cast<std::size_t>(segments) + 2);
    mesh.polygons.push_back(makePolygon(topRing, color));
    std::vector<Vector3> bottomReversed(bottomRing.rbegin(), bottomRing.rend());
    mesh.polygons.push_back(makePolygon(std::move(bottomReversed), color));
    for (int i = 0; i < segments; ++i) {
        const int j = (i + 1) % segments;
        mesh.polygons.push_back(makePolygon(
            {bottomRing[static_cast<std::size_t>(i)], bottomRing[static_cast<std::size_t>(j)],
             topRing[static_cast<std::size_t>(j)], topRing[static_cast<std::size_t>(i)]},
            color));
    }
    return mesh;
}

CsgMesh makeConvexPrism(const std::vector<std::array<float, 2>>& footprintCcw,
                        float y0, float y1, const Color3& color, float taperTopScale) {
    std::vector<std::array<float, 2>> footprint = footprintCcw;
    // Normalize winding so the top face comes out +Y regardless of the
    // caller's convention.
    float signedArea = 0.0f;
    for (std::size_t i = 0; i < footprint.size(); ++i) {
        const auto& a = footprint[i];
        const auto& b = footprint[(i + 1) % footprint.size()];
        signedArea += a[0] * b[1] - b[0] * a[1];
    }
    if (signedArea > 0.0f) {
        std::vector<std::array<float, 2>> reversed(footprint.rbegin(), footprint.rend());
        footprint = std::move(reversed);
    }

    float centroidX = 0.0f;
    float centroidZ = 0.0f;
    for (const auto& p : footprint) {
        centroidX += p[0];
        centroidZ += p[1];
    }
    const float invCount = 1.0f / static_cast<float>(footprint.size());
    centroidX *= invCount;
    centroidZ *= invCount;

    std::vector<Vector3> bottomRing;
    std::vector<Vector3> topRing;
    bottomRing.reserve(footprint.size());
    topRing.reserve(footprint.size());
    for (const auto& p : footprint) {
        bottomRing.push_back({p[0], y0, p[1]});
        topRing.push_back({centroidX + (p[0] - centroidX) * taperTopScale, y1,
                           centroidZ + (p[1] - centroidZ) * taperTopScale});
    }

    CsgMesh mesh;
    mesh.polygons.reserve(footprint.size() + 2);
    mesh.polygons.push_back(makePolygon(topRing, color));
    std::vector<Vector3> bottomReversed(bottomRing.rbegin(), bottomRing.rend());
    mesh.polygons.push_back(makePolygon(std::move(bottomReversed), color));
    for (std::size_t i = 0; i < footprint.size(); ++i) {
        const std::size_t j = (i + 1) % footprint.size();
        mesh.polygons.push_back(makePolygon(
            {bottomRing[i], bottomRing[j], topRing[j], topRing[i]}, color));
    }
    return mesh;
}

}  // namespace odai::procgen
