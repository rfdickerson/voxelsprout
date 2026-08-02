#pragma once

#include <algorithm>
#include <cmath>

#include "math/math.h"

// Math Geometry subsystem
// Responsible for: float-space bounding volumes, rays, and the intersection
// tests that were previously written inline at each call site.
// Should NOT do: acceleration structures, scene traversal, or anything that
// needs to know what a chunk or a draw call is.
//
// core/grid3.h's CellAabb is the integer-cell counterpart of Aabb3f and stays
// separate: it is min-inclusive/max-exclusive over cells, not a float box.
namespace odai::math {

// Field-per-axis rather than two Vector3s, matching the layout of the two
// struct definitions this replaces (app/app.cc and voxelcraft_player.h).
struct Aabb3f {
    float minX = 0.0f;
    float maxX = 0.0f;
    float minY = 0.0f;
    float maxY = 0.0f;
    float minZ = 0.0f;
    float maxZ = 0.0f;

    constexpr bool operator==(const Aabb3f&) const = default;
};

inline constexpr Aabb3f makeAabbFromCenterExtents(const Vector3& center, const Vector3& halfExtents) {
    return Aabb3f{
        center.x - halfExtents.x, center.x + halfExtents.x,
        center.y - halfExtents.y, center.y + halfExtents.y,
        center.z - halfExtents.z, center.z + halfExtents.z
    };
}

inline constexpr Aabb3f makeAabbFromCorners(const Vector3& a, const Vector3& b) {
    return Aabb3f{
        a.x < b.x ? a.x : b.x, a.x < b.x ? b.x : a.x,
        a.y < b.y ? a.y : b.y, a.y < b.y ? b.y : a.y,
        a.z < b.z ? a.z : b.z, a.z < b.z ? b.z : a.z
    };
}

inline constexpr Vector3 aabbCenter(const Aabb3f& box) {
    return Vector3{
        (box.minX + box.maxX) * 0.5f,
        (box.minY + box.maxY) * 0.5f,
        (box.minZ + box.maxZ) * 0.5f
    };
}

inline constexpr Vector3 aabbExtents(const Aabb3f& box) {
    return Vector3{box.maxX - box.minX, box.maxY - box.minY, box.maxZ - box.minZ};
}

inline constexpr void expandAabb(Aabb3f& box, const Vector3& point) {
    if (point.x < box.minX) box.minX = point.x;
    if (point.x > box.maxX) box.maxX = point.x;
    if (point.y < box.minY) box.minY = point.y;
    if (point.y > box.maxY) box.maxY = point.y;
    if (point.z < box.minZ) box.minZ = point.z;
    if (point.z > box.maxZ) box.maxZ = point.z;
}

inline constexpr bool aabbContainsPoint(const Aabb3f& box, const Vector3& point) {
    return point.x >= box.minX && point.x <= box.maxX &&
           point.y >= box.minY && point.y <= box.maxY &&
           point.z >= box.minZ && point.z <= box.maxZ;
}

// Overlap test with an optional shrink applied to rhs on every axis. Both
// collision call sites pass their kCollisionEpsilon so that boxes merely
// touching do not count as colliding; epsilon 0 gives the plain test.
inline constexpr bool aabbOverlaps(const Aabb3f& lhs, const Aabb3f& rhs, float epsilon = 0.0f) {
    return
        lhs.maxX > (rhs.minX + epsilon) &&
        lhs.minX < (rhs.maxX - epsilon) &&
        lhs.maxY > (rhs.minY + epsilon) &&
        lhs.minY < (rhs.maxY - epsilon) &&
        lhs.maxZ > (rhs.minZ + epsilon) &&
        lhs.minZ < (rhs.maxZ - epsilon);
}

struct Ray {
    Vector3 origin{};
    Vector3 direction{};  // caller-normalized; the helpers do not renormalize
};

struct RayTriangleHit {
    bool hit = false;
    float distance = 0.0f;
    float u = 0.0f;  // barycentric weight of v1
    float v = 0.0f;  // barycentric weight of v2
};

// Moller-Trumbore, double-sided. The arithmetic is transcribed verbatim from
// App::raycastImportedSceneFromCamera so results do not shift; do not reorder
// the cross/dot chain when tidying.
inline RayTriangleHit intersectRayTriangle(
    const Ray& ray,
    const Vector3& v0,
    const Vector3& v1,
    const Vector3& v2,
    float epsilon = 1e-4f
) {
    RayTriangleHit result{};
    const Vector3 edge1 = v1 - v0;
    const Vector3 edge2 = v2 - v0;
    const Vector3 pvec = cross(ray.direction, edge2);
    const float det = dot(edge1, pvec);
    if (std::fabs(det) <= epsilon) {
        return result;
    }

    const float invDet = 1.0f / det;
    const Vector3 tvec = ray.origin - v0;
    const float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return result;
    }
    const Vector3 qvec = cross(tvec, edge1);
    const float v = dot(ray.direction, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) {
        return result;
    }
    const float distance = dot(edge2, qvec) * invDet;
    if (distance <= epsilon) {
        return result;
    }

    result.hit = true;
    result.distance = distance;
    result.u = u;
    result.v = v;
    return result;
}

// Slab test. outDistance receives the entry distance, or tMin when the origin
// is already inside the box.
inline bool intersectRayAabb(
    const Ray& ray,
    const Aabb3f& box,
    float tMin,
    float tMax,
    float& outDistance
) {
    // Not named near/far: those collide with the legacy near/far macros
    // Windows SDK headers (windows.h) still define, breaking the MSVC build.
    float nearDistance = tMin;
    float farDistance = tMax;

    const float origin[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
    const float direction[3] = {ray.direction.x, ray.direction.y, ray.direction.z};
    const float boxMin[3] = {box.minX, box.minY, box.minZ};
    const float boxMax[3] = {box.maxX, box.maxY, box.maxZ};

    for (int axis = 0; axis < 3; ++axis) {
        if (std::fabs(direction[axis]) < 1e-8f) {
            // Parallel to this slab: miss unless the origin is already between
            // the planes.
            if (origin[axis] < boxMin[axis] || origin[axis] > boxMax[axis]) {
                return false;
            }
            continue;
        }
        const float invDirection = 1.0f / direction[axis];
        float t0 = (boxMin[axis] - origin[axis]) * invDirection;
        float t1 = (boxMax[axis] - origin[axis]) * invDirection;
        if (t0 > t1) {
            std::swap(t0, t1);
        }
        nearDistance = std::max(nearDistance, t0);
        farDistance = std::min(farDistance, t1);
        if (nearDistance > farDistance) {
            return false;
        }
    }

    outDistance = nearDistance;
    return true;
}

}  // namespace odai::math
