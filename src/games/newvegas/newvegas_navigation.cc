#include "games/newvegas/newvegas_navigation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace odai::games::newvegas {

namespace {

using odai::math::Vector3;

float distanceSquaredXZ(const Vector3& a, const Vector3& b) {
    const float dx = a.x - b.x;
    const float dz = a.z - b.z;
    return (dx * dx) + (dz * dz);
}

Vector3 closestPointOnSegmentXZ(
    const Vector3& point, const Vector3& a, const Vector3& b) {
    const float dx = b.x - a.x;
    const float dz = b.z - a.z;
    const float lengthSquared = (dx * dx) + (dz * dz);
    const float t = lengthSquared > 1e-8f
        ? std::clamp(((point.x - a.x) * dx + (point.z - a.z) * dz) / lengthSquared,
                     0.0f, 1.0f)
        : 0.0f;
    return Vector3{
        a.x + (dx * t),
        a.y + ((b.y - a.y) * t),
        a.z + (dz * t)};
}

// Closest point in the horizontal projection, with Y reconstructed from the
// same barycentric/edge weights. Navmesh triangles describe a height field in
// the small, so this is the point a walking actor wants; a full 3D closest-point
// query can slide sideways toward a steep face solely to reduce vertical error.
Vector3 closestPointOnTriangleXZ(
    const Vector3& point, const Vector3& a, const Vector3& b, const Vector3& c) {
    const float v0x = b.x - a.x;
    const float v0z = b.z - a.z;
    const float v1x = c.x - a.x;
    const float v1z = c.z - a.z;
    const float v2x = point.x - a.x;
    const float v2z = point.z - a.z;
    const float d00 = (v0x * v0x) + (v0z * v0z);
    const float d01 = (v0x * v1x) + (v0z * v1z);
    const float d11 = (v1x * v1x) + (v1z * v1z);
    const float d20 = (v2x * v0x) + (v2z * v0z);
    const float d21 = (v2x * v1x) + (v2z * v1z);
    const float denominator = (d00 * d11) - (d01 * d01);
    if (std::abs(denominator) > 1e-8f) {
        const float v = ((d11 * d20) - (d01 * d21)) / denominator;
        const float w = ((d00 * d21) - (d01 * d20)) / denominator;
        const float u = 1.0f - v - w;
        if (u >= 0.0f && v >= 0.0f && w >= 0.0f) {
            return Vector3{
                (a.x * u) + (b.x * v) + (c.x * w),
                (a.y * u) + (b.y * v) + (c.y * w),
                (a.z * u) + (b.z * v) + (c.z * w)};
        }
    }

    Vector3 closest = closestPointOnSegmentXZ(point, a, b);
    float best = distanceSquaredXZ(point, closest);
    for (const Vector3 candidate : {
             closestPointOnSegmentXZ(point, b, c),
             closestPointOnSegmentXZ(point, c, a)}) {
        const float distance = distanceSquaredXZ(point, candidate);
        if (distance < best) {
            best = distance;
            closest = candidate;
        }
    }
    return closest;
}

template <typename MeshType, typename TriangleType>
Vector3 triangleCentroid(const MeshType& mesh, const TriangleType& triangle) {
    const Vector3& a = mesh.vertices[triangle.vertex[0]];
    const Vector3& b = mesh.vertices[triangle.vertex[1]];
    const Vector3& c = mesh.vertices[triangle.vertex[2]];
    return Vector3{
        (a.x + b.x + c.x) / 3.0f,
        (a.y + b.y + c.y) / 3.0f,
        (a.z + b.z + c.z) / 3.0f};
}

}  // namespace

void ActorNavigationWorld::addCell(
    const importer::CellCoord& cell,
    const std::vector<importer::fnv::FalloutNavMeshRecord>& records) {
    std::vector<Mesh> meshes;
    meshes.reserve(records.size());
    for (const importer::fnv::FalloutNavMeshRecord& record : records) {
        const std::size_t vertexCount = record.vertices.size() / 3u;
        if (vertexCount == 0u || record.triangles.empty()) {
            continue;
        }
        Mesh mesh;
        mesh.formId = record.formId;
        mesh.vertices.reserve(vertexCount);
        for (std::size_t i = 0; i < vertexCount; ++i) {
            const float* source = &record.vertices[i * 3u];
            // Bethesda xyz is Z-up; engine xyz is Y-up with the horizontal Y
            // axis negated, matching CellSceneBuilder's world conversion.
            mesh.vertices.push_back(Vector3{source[0], source[2], -source[1]});
        }
        mesh.triangles.reserve(record.triangles.size());
        for (const importer::fnv::FalloutNavMeshTriangle& source : record.triangles) {
            if (source.vertex[0] >= vertexCount || source.vertex[1] >= vertexCount ||
                source.vertex[2] >= vertexCount) {
                continue;
            }
            Triangle triangle;
            for (int corner = 0; corner < 3; ++corner) {
                triangle.vertex[corner] = source.vertex[corner];
                triangle.neighbour[corner] = source.neighbour[corner];
            }
            mesh.triangles.push_back(triangle);
        }
        if (!mesh.triangles.empty()) {
            meshes.push_back(std::move(mesh));
        }
    }
    if (meshes.empty()) {
        m_cells.erase(cell);
    } else {
        m_cells[cell] = std::move(meshes);
    }
}

void ActorNavigationWorld::removeCell(const importer::CellCoord& cell) {
    m_cells.erase(cell);
}

void ActorNavigationWorld::clear() {
    m_cells.clear();
}

bool ActorNavigationWorld::findNearest(
    const Vector3& point,
    float maxHorizontalDistance,
    float maxVerticalDistance,
    Location& outLocation) const {
    const float maxHorizontalSquared = maxHorizontalDistance * maxHorizontalDistance;
    float bestScore = std::numeric_limits<float>::infinity();
    bool found = false;
    for (const auto& [cell, meshes] : m_cells) {
        (void)cell;
        for (const Mesh& mesh : meshes) {
            for (std::size_t index = 0; index < mesh.triangles.size(); ++index) {
                const Triangle& triangle = mesh.triangles[index];
                const Vector3 candidate = closestPointOnTriangleXZ(
                    point,
                    mesh.vertices[triangle.vertex[0]],
                    mesh.vertices[triangle.vertex[1]],
                    mesh.vertices[triangle.vertex[2]]);
                const float horizontalSquared = distanceSquaredXZ(point, candidate);
                const float vertical = std::abs(candidate.y - point.y);
                if (horizontalSquared > maxHorizontalSquared || vertical > maxVerticalDistance) {
                    continue;
                }
                // Prefer the actor's authored storey, then the closest path in
                // plan view. This keeps battlement guards on battlements while
                // still moving a rock-top guard sideways onto the street.
                const float score = horizontalSquared + (vertical * vertical * 4.0f);
                if (score < bestScore) {
                    bestScore = score;
                    outLocation = Location{&mesh, index, candidate, score};
                    found = true;
                }
            }
        }
    }
    return found;
}

bool ActorNavigationWorld::projectPoint(
    float worldX,
    float worldY,
    float worldZ,
    float maxHorizontalDistance,
    float maxVerticalDistance,
    Vector3& outPoint) const {
    Location location;
    if (!findNearest(
            Vector3{worldX, worldY, worldZ},
            maxHorizontalDistance, maxVerticalDistance, location)) {
        return false;
    }
    outPoint = location.point;
    return true;
}

bool ActorNavigationWorld::buildWanderPath(
    const Vector3& start,
    const Vector3& origin,
    float radius,
    std::uint32_t randomValue,
    std::vector<Vector3>& outWaypoints) const {
    outWaypoints.clear();
    Location startLocation;
    if (!findNearest(start, 180.0f, 250.0f, startLocation)) {
        return false;
    }

    const Mesh& mesh = *startLocation.mesh;
    const std::size_t count = mesh.triangles.size();
    std::vector<int> parent(count, -2);
    std::queue<std::size_t> frontier;
    parent[startLocation.triangle] = -1;
    frontier.push(startLocation.triangle);
    std::vector<std::size_t> candidates;
    const float radiusSquared = radius * radius;
    constexpr float kMinimumTripSquared = 180.0f * 180.0f;

    while (!frontier.empty()) {
        const std::size_t current = frontier.front();
        frontier.pop();
        const Vector3 centroid = triangleCentroid(mesh, mesh.triangles[current]);
        if (distanceSquaredXZ(centroid, origin) <= radiusSquared &&
            distanceSquaredXZ(centroid, startLocation.point) >= kMinimumTripSquared) {
            candidates.push_back(current);
        }
        for (const std::uint16_t neighbour : mesh.triangles[current].neighbour) {
            if (neighbour == importer::fnv::kNavMeshNoNeighbour || neighbour >= count ||
                parent[neighbour] != -2) {
                continue;
            }
            parent[neighbour] = static_cast<int>(current);
            frontier.push(neighbour);
        }
    }
    if (candidates.empty()) {
        return false;
    }

    const std::size_t target = candidates[randomValue % candidates.size()];
    std::vector<std::size_t> triangles;
    for (int current = static_cast<int>(target); current >= 0; current = parent[current]) {
        triangles.push_back(static_cast<std::size_t>(current));
    }
    std::reverse(triangles.begin(), triangles.end());

    // Route through each shared edge midpoint. A triangle centroid connected
    // directly to another triangle's centroid is also valid, but explicit edge
    // crossings keep every finite-precision segment visibly centred on paths.
    for (std::size_t step = 1; step < triangles.size(); ++step) {
        const Triangle& before = mesh.triangles[triangles[step - 1u]];
        const Triangle& after = mesh.triangles[triangles[step]];
        std::uint16_t shared[2] = {};
        int sharedCount = 0;
        for (const std::uint16_t a : before.vertex) {
            for (const std::uint16_t b : after.vertex) {
                if (a == b && sharedCount < 2) {
                    shared[sharedCount++] = a;
                }
            }
        }
        if (sharedCount == 2) {
            const Vector3& a = mesh.vertices[shared[0]];
            const Vector3& b = mesh.vertices[shared[1]];
            outWaypoints.push_back(Vector3{
                (a.x + b.x) * 0.5f,
                (a.y + b.y) * 0.5f,
                (a.z + b.z) * 0.5f});
        }
    }
    outWaypoints.push_back(triangleCentroid(mesh, mesh.triangles[target]));
    return !outWaypoints.empty();
}

std::size_t ActorNavigationWorld::meshCount() const {
    std::size_t count = 0u;
    for (const auto& [cell, meshes] : m_cells) {
        (void)cell;
        count += meshes.size();
    }
    return count;
}

std::size_t ActorNavigationWorld::triangleCount() const {
    std::size_t count = 0u;
    for (const auto& [cell, meshes] : m_cells) {
        (void)cell;
        for (const Mesh& mesh : meshes) {
            count += mesh.triangles.size();
        }
    }
    return count;
}

}  // namespace odai::games::newvegas
