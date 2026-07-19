#include "procgen/mesh_emit.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace odai::procgen {

namespace {

void expandBounds(Vector3& lo, Vector3& hi, const Vector3& p) {
    lo.x = std::min(lo.x, p.x);
    lo.y = std::min(lo.y, p.y);
    lo.z = std::min(lo.z, p.z);
    hi.x = std::max(hi.x, p.x);
    hi.y = std::max(hi.y, p.y);
    hi.z = std::max(hi.z, p.z);
}

#ifndef NDEBUG
// Fan triangulation is only valid for convex polygons; BSP clipping of convex
// input preserves convexity, so a violation here means a generator bug.
bool isConvex(const Polygon& polygon) {
    const std::size_t n = polygon.vertices.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Vector3& a = polygon.vertices[i];
        const Vector3& b = polygon.vertices[(i + 1) % n];
        const Vector3& c = polygon.vertices[(i + 2) % n];
        const Vector3 turn = odai::math::cross(b - a, c - b);
        if (odai::math::dot(turn, polygon.plane.normal) < -1e-6f) {
            return false;
        }
    }
    return true;
}
#endif

}  // namespace

TriMesh triangulate(const CsgMesh& mesh) {
    TriMesh out;
    constexpr float kInf = std::numeric_limits<float>::max();
    out.boundsMin = {kInf, kInf, kInf};
    out.boundsMax = {-kInf, -kInf, -kInf};

    for (const Polygon& polygon : mesh.polygons) {
        if (polygon.vertices.size() < 3) {
            continue;
        }
        assert(isConvex(polygon));
        const std::uint32_t base = static_cast<std::uint32_t>(out.vertices.size());
        for (const Vector3& v : polygon.vertices) {
            odai::importer::ImportedScenePackedVertex vertex;
            vertex.position[0] = v.x;
            vertex.position[1] = v.y;
            vertex.position[2] = v.z;
            vertex.normal[0] = polygon.plane.normal.x;
            vertex.normal[1] = polygon.plane.normal.y;
            vertex.normal[2] = polygon.plane.normal.z;
            vertex.color[0] = polygon.color.r;
            vertex.color[1] = polygon.color.g;
            vertex.color[2] = polygon.color.b;
            out.vertices.push_back(vertex);
            expandBounds(out.boundsMin, out.boundsMax, v);
        }
        for (std::size_t i = 1; i + 1 < polygon.vertices.size(); ++i) {
            out.indices.push_back(base);
            out.indices.push_back(base + static_cast<std::uint32_t>(i));
            out.indices.push_back(base + static_cast<std::uint32_t>(i) + 1u);
        }
    }
    if (out.vertices.empty()) {
        out.boundsMin = {};
        out.boundsMax = {};
    }
    return out;
}

void appendTriMeshRotated(const TriMesh& mesh, const Vector3& offset, int quarterTurns,
                          const Vector3& pivot, const Color3& colorMul,
                          odai::importer::ImportedScene& scene) {
    quarterTurns &= 3;
    if (quarterTurns == 0) {
        appendTriMesh(mesh, offset, colorMul, scene);
        return;
    }
    if (mesh.vertices.empty()) {
        return;
    }
    // cos/sin of quarterTurns * 90 deg, exact.
    static constexpr float kCos[4] = {1.0f, 0.0f, -1.0f, 0.0f};
    static constexpr float kSin[4] = {0.0f, 1.0f, 0.0f, -1.0f};
    const float c = kCos[quarterTurns];
    const float s = kSin[quarterTurns];

    const std::uint32_t base = static_cast<std::uint32_t>(scene.packedVertices.size());
    scene.packedVertices.reserve(scene.packedVertices.size() + mesh.vertices.size());
    for (const odai::importer::ImportedScenePackedVertex& src : mesh.vertices) {
        odai::importer::ImportedScenePackedVertex v = src;
        const float lx = src.position[0] - pivot.x;
        const float lz = src.position[2] - pivot.z;
        v.position[0] = pivot.x + lx * c + lz * s + offset.x;
        v.position[1] = src.position[1] + offset.y;
        v.position[2] = pivot.z - lx * s + lz * c + offset.z;
        const float nx = src.normal[0], nz = src.normal[2];
        v.normal[0] = nx * c + nz * s;
        v.normal[2] = -nx * s + nz * c;
        v.color[0] *= colorMul.r;
        v.color[1] *= colorMul.g;
        v.color[2] *= colorMul.b;
        scene.packedVertices.push_back(v);
        for (int axis = 0; axis < 3; ++axis) {
            scene.boundsMin[axis] = std::min(scene.boundsMin[axis], v.position[axis]);
            scene.boundsMax[axis] = std::max(scene.boundsMax[axis], v.position[axis]);
        }
    }
    scene.packedIndices.reserve(scene.packedIndices.size() + mesh.indices.size());
    for (std::uint32_t index : mesh.indices) {
        scene.packedIndices.push_back(base + index);
    }
}

void appendTriMesh(const TriMesh& mesh, const Vector3& offset, const Color3& colorMul,
                   odai::importer::ImportedScene& scene) {
    if (mesh.vertices.empty()) {
        return;
    }
    const std::uint32_t base = static_cast<std::uint32_t>(scene.packedVertices.size());
    scene.packedVertices.reserve(scene.packedVertices.size() + mesh.vertices.size());
    for (const odai::importer::ImportedScenePackedVertex& src : mesh.vertices) {
        odai::importer::ImportedScenePackedVertex v = src;
        v.position[0] += offset.x;
        v.position[1] += offset.y;
        v.position[2] += offset.z;
        v.color[0] *= colorMul.r;
        v.color[1] *= colorMul.g;
        v.color[2] *= colorMul.b;
        scene.packedVertices.push_back(v);
    }
    scene.packedIndices.reserve(scene.packedIndices.size() + mesh.indices.size());
    for (std::uint32_t index : mesh.indices) {
        scene.packedIndices.push_back(base + index);
    }
    const float lo[3] = {mesh.boundsMin.x + offset.x, mesh.boundsMin.y + offset.y,
                         mesh.boundsMin.z + offset.z};
    const float hi[3] = {mesh.boundsMax.x + offset.x, mesh.boundsMax.y + offset.y,
                         mesh.boundsMax.z + offset.z};
    for (int axis = 0; axis < 3; ++axis) {
        scene.boundsMin[axis] = std::min(scene.boundsMin[axis], lo[axis]);
        scene.boundsMax[axis] = std::max(scene.boundsMax[axis], hi[axis]);
    }
}

}  // namespace odai::procgen
