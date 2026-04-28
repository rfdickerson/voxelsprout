#include "world/navmesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>

namespace odai::world {

namespace {

constexpr float kGeometryEpsilon = 1e-4f;
constexpr std::uint32_t kGpuSceneComponentFlagAlphaTest = 1u << 2;

struct QueryAabb {
    float minX = 0.0f;
    float maxX = 0.0f;
    float minY = 0.0f;
    float maxY = 0.0f;
    float minZ = 0.0f;
    float maxZ = 0.0f;
};

struct EdgeEndpoint {
    std::int64_t x = 0;
    std::int64_t z = 0;

    friend bool operator==(const EdgeEndpoint& lhs, const EdgeEndpoint& rhs) {
        return lhs.x == rhs.x && lhs.z == rhs.z;
    }

    friend bool operator<(const EdgeEndpoint& lhs, const EdgeEndpoint& rhs) {
        if (lhs.x != rhs.x) {
            return lhs.x < rhs.x;
        }
        return lhs.z < rhs.z;
    }
};

struct EdgeKey {
    EdgeEndpoint a{};
    EdgeEndpoint b{};

    friend bool operator==(const EdgeKey& lhs, const EdgeKey& rhs) {
        return lhs.a == rhs.a && lhs.b == rhs.b;
    }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& key) const {
        const std::uint64_t ax = static_cast<std::uint64_t>(key.a.x);
        const std::uint64_t az = static_cast<std::uint64_t>(key.a.z);
        const std::uint64_t bx = static_cast<std::uint64_t>(key.b.x);
        const std::uint64_t bz = static_cast<std::uint64_t>(key.b.z);
        std::uint64_t hash = 1469598103934665603ull;
        const auto mix = [&hash](std::uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ull;
        };
        mix(ax);
        mix(az);
        mix(bx);
        mix(bz);
        return static_cast<std::size_t>(hash);
    }
};

struct EdgeRef {
    std::uint32_t triangleIndex = 0;
    std::uint8_t edgeIndex = 0;
};

[[nodiscard]] odai::math::Vector3 transformPoint(const std::array<float, 16>& transform, const float point[3]) {
    return {
        (transform[0] * point[0]) + (transform[1] * point[1]) + (transform[2] * point[2]) + transform[3],
        (transform[4] * point[0]) + (transform[5] * point[1]) + (transform[6] * point[2]) + transform[7],
        (transform[8] * point[0]) + (transform[9] * point[1]) + (transform[10] * point[2]) + transform[11]
    };
}

[[nodiscard]] float minWalkableNormalY(float maxSlopeDegrees) {
    return std::cos(odai::math::radians(std::clamp(maxSlopeDegrees, 0.0f, 89.0f)));
}

[[nodiscard]] QueryAabb makeBounds(
    const odai::math::Vector3& p0,
    const odai::math::Vector3& p1,
    const odai::math::Vector3& p2
) {
    QueryAabb bounds{};
    bounds.minX = std::min({p0.x, p1.x, p2.x});
    bounds.maxX = std::max({p0.x, p1.x, p2.x});
    bounds.minY = std::min({p0.y, p1.y, p2.y});
    bounds.maxY = std::max({p0.y, p1.y, p2.y});
    bounds.minZ = std::min({p0.z, p1.z, p2.z});
    bounds.maxZ = std::max({p0.z, p1.z, p2.z});
    return bounds;
}

[[nodiscard]] bool aabbOverlaps(const QueryAabb& lhs, const QueryAabb& rhs) {
    return
        lhs.maxX >= rhs.minX &&
        lhs.minX <= rhs.maxX &&
        lhs.maxY >= rhs.minY &&
        lhs.minY <= rhs.maxY &&
        lhs.maxZ >= rhs.minZ &&
        lhs.minZ <= rhs.maxZ;
}

[[nodiscard]] bool pointInsideTriangleXz(
    const Navmesh::Triangle& triangle,
    float x,
    float z,
    float& outU,
    float& outV
) {
    const float v0x = triangle.p1.x - triangle.p0.x;
    const float v0z = triangle.p1.z - triangle.p0.z;
    const float v1x = triangle.p2.x - triangle.p0.x;
    const float v1z = triangle.p2.z - triangle.p0.z;
    const float v2x = x - triangle.p0.x;
    const float v2z = z - triangle.p0.z;

    const float denom = (v0x * v1z) - (v1x * v0z);
    if (std::fabs(denom) <= kGeometryEpsilon) {
        return false;
    }

    const float invDenom = 1.0f / denom;
    outU = ((v2x * v1z) - (v1x * v2z)) * invDenom;
    outV = ((v0x * v2z) - (v2x * v0z)) * invDenom;
    return outU >= -0.001f && outV >= -0.001f && (outU + outV) <= 1.001f;
}

[[nodiscard]] bool heightAtXz(const Navmesh::Triangle& triangle, float x, float z, float& outY) {
    float u = 0.0f;
    float v = 0.0f;
    if (!pointInsideTriangleXz(triangle, x, z, u, v)) {
        return false;
    }
    outY = triangle.p0.y + ((triangle.p1.y - triangle.p0.y) * u) + ((triangle.p2.y - triangle.p0.y) * v);
    return true;
}

[[nodiscard]] odai::math::Vector3 closestPointOnSegmentXz(
    const odai::math::Vector3& point,
    const odai::math::Vector3& a,
    const odai::math::Vector3& b
) {
    const float abX = b.x - a.x;
    const float abZ = b.z - a.z;
    const float lengthSq = (abX * abX) + (abZ * abZ);
    if (lengthSq <= kGeometryEpsilon) {
        return a;
    }

    const float t = std::clamp(((point.x - a.x) * abX + (point.z - a.z) * abZ) / lengthSq, 0.0f, 1.0f);
    return {
        a.x + (abX * t),
        a.y + ((b.y - a.y) * t),
        a.z + (abZ * t)
    };
}

[[nodiscard]] odai::math::Vector3 closestPointOnTriangleXz(
    const Navmesh::Triangle& triangle,
    const odai::math::Vector3& point
) {
    float y = 0.0f;
    if (heightAtXz(triangle, point.x, point.z, y)) {
        return {point.x, y, point.z};
    }

    const std::array<odai::math::Vector3, 3> candidates{
        closestPointOnSegmentXz(point, triangle.p0, triangle.p1),
        closestPointOnSegmentXz(point, triangle.p1, triangle.p2),
        closestPointOnSegmentXz(point, triangle.p2, triangle.p0)
    };

    odai::math::Vector3 best = candidates[0];
    float bestDistanceSq = std::numeric_limits<float>::max();
    for (const odai::math::Vector3& candidate : candidates) {
        const float dx = candidate.x - point.x;
        const float dz = candidate.z - point.z;
        const float distanceSq = (dx * dx) + (dz * dz);
        if (distanceSq < bestDistanceSq) {
            best = candidate;
            bestDistanceSq = distanceSq;
        }
    }
    return best;
}

[[nodiscard]] float distanceSquaredXz(const odai::math::Vector3& lhs, const odai::math::Vector3& rhs) {
    const float dx = lhs.x - rhs.x;
    const float dz = lhs.z - rhs.z;
    return (dx * dx) + (dz * dz);
}

[[nodiscard]] Navmesh::Triangle makeTriangle(const NavmeshBuildTriangle& source) {
    Navmesh::Triangle triangle{};
    triangle.p0 = source.p0;
    triangle.p1 = source.p1;
    triangle.p2 = source.p2;
    triangle.area = source.area;
    triangle.center = (triangle.p0 + triangle.p1 + triangle.p2) / 3.0f;

    odai::math::Vector3 normal = odai::math::cross(triangle.p1 - triangle.p0, triangle.p2 - triangle.p0);
    const float normalLength = odai::math::length(normal);
    if (normalLength > kGeometryEpsilon) {
        normal /= normalLength;
    } else {
        normal = {0.0f, 1.0f, 0.0f};
    }
    if (normal.y < 0.0f) {
        normal *= -1.0f;
    }
    triangle.normal = normal;
    return triangle;
}

[[nodiscard]] EdgeEndpoint edgeEndpoint(const odai::math::Vector3& point, float epsilon) {
    const float invEpsilon = 1.0f / std::max(epsilon, 1e-3f);
    return {
        static_cast<std::int64_t>(std::llround(point.x * invEpsilon)),
        static_cast<std::int64_t>(std::llround(point.z * invEpsilon))
    };
}

[[nodiscard]] EdgeKey edgeKey(
    const odai::math::Vector3& a,
    const odai::math::Vector3& b,
    float epsilon
) {
    EdgeEndpoint ea = edgeEndpoint(a, epsilon);
    EdgeEndpoint eb = edgeEndpoint(b, epsilon);
    if (eb < ea) {
        std::swap(ea, eb);
    }
    return {ea, eb};
}

[[nodiscard]] std::array<odai::math::Vector3, 2> edgeVertices(
    const Navmesh::Triangle& triangle,
    std::uint8_t edgeIndex
) {
    if (edgeIndex == 0u) {
        return {triangle.p0, triangle.p1};
    }
    if (edgeIndex == 1u) {
        return {triangle.p1, triangle.p2};
    }
    return {triangle.p2, triangle.p0};
}

[[nodiscard]] bool canConnectEdges(
    const Navmesh::Triangle& lhs,
    std::uint8_t lhsEdge,
    const Navmesh::Triangle& rhs,
    std::uint8_t rhsEdge,
    float maxClimb
) {
    const std::array<odai::math::Vector3, 2> lhsEdgeVertices = edgeVertices(lhs, lhsEdge);
    const std::array<odai::math::Vector3, 2> rhsEdgeVertices = edgeVertices(rhs, rhsEdge);

    float maxHeightDelta = 0.0f;
    for (const odai::math::Vector3& lhsPoint : lhsEdgeVertices) {
        float bestDelta = std::numeric_limits<float>::max();
        for (const odai::math::Vector3& rhsPoint : rhsEdgeVertices) {
            const float dx = lhsPoint.x - rhsPoint.x;
            const float dz = lhsPoint.z - rhsPoint.z;
            if ((dx * dx) + (dz * dz) <= 0.25f) {
                bestDelta = std::min(bestDelta, std::fabs(lhsPoint.y - rhsPoint.y));
            }
        }
        if (bestDelta == std::numeric_limits<float>::max()) {
            bestDelta = std::fabs(lhs.center.y - rhs.center.y);
        }
        maxHeightDelta = std::max(maxHeightDelta, bestDelta);
    }
    return maxHeightDelta <= maxClimb;
}

[[nodiscard]] float orient2d(
    const odai::math::Vector3& a,
    const odai::math::Vector3& b,
    const odai::math::Vector3& c
) {
    return ((b.x - a.x) * (c.z - a.z)) - ((b.z - a.z) * (c.x - a.x));
}

[[nodiscard]] bool rangesOverlap(float minA, float maxA, float minB, float maxB) {
    return maxA >= minB && minA <= maxB;
}

[[nodiscard]] bool pointOnSegmentXz(
    const odai::math::Vector3& point,
    const odai::math::Vector3& a,
    const odai::math::Vector3& b
) {
    if (std::fabs(orient2d(a, b, point)) > 0.01f) {
        return false;
    }
    return
        point.x >= std::min(a.x, b.x) - 0.01f &&
        point.x <= std::max(a.x, b.x) + 0.01f &&
        point.z >= std::min(a.z, b.z) - 0.01f &&
        point.z <= std::max(a.z, b.z) + 0.01f;
}

[[nodiscard]] bool segmentsIntersectXz(
    const odai::math::Vector3& a0,
    const odai::math::Vector3& a1,
    const odai::math::Vector3& b0,
    const odai::math::Vector3& b1
) {
    const float o0 = orient2d(a0, a1, b0);
    const float o1 = orient2d(a0, a1, b1);
    const float o2 = orient2d(b0, b1, a0);
    const float o3 = orient2d(b0, b1, a1);

    if (((o0 > 0.0f && o1 < 0.0f) || (o0 < 0.0f && o1 > 0.0f)) &&
        ((o2 > 0.0f && o3 < 0.0f) || (o2 < 0.0f && o3 > 0.0f))) {
        return true;
    }
    return
        pointOnSegmentXz(b0, a0, a1) ||
        pointOnSegmentXz(b1, a0, a1) ||
        pointOnSegmentXz(a0, b0, b1) ||
        pointOnSegmentXz(a1, b0, b1);
}

} // namespace

void Navmesh::clear() {
    m_walkableTriangles.clear();
    m_obstacleTriangles.clear();
    m_linkCount = 0;
}

bool Navmesh::build(std::span<const NavmeshBuildTriangle> triangles, const NavmeshSettings& settings) {
    clear();
    m_settings = settings;

    const float walkableNormalY = minWalkableNormalY(settings.maxSlopeDegrees);
    for (const NavmeshBuildTriangle& source : triangles) {
        Triangle triangle = makeTriangle(source);
        if (odai::math::lengthSquared(odai::math::cross(triangle.p1 - triangle.p0, triangle.p2 - triangle.p0)) <=
            kGeometryEpsilon) {
            continue;
        }

        if (source.area != NavmeshArea::Obstacle && triangle.normal.y >= walkableNormalY) {
            m_walkableTriangles.push_back(std::move(triangle));
        } else {
            triangle.area = NavmeshArea::Obstacle;
            m_obstacleTriangles.push_back(std::move(triangle));
        }
    }

    std::unordered_map<EdgeKey, std::vector<EdgeRef>, EdgeKeyHash> edgeRefs;
    for (std::uint32_t triangleIndex = 0;
         triangleIndex < static_cast<std::uint32_t>(m_walkableTriangles.size());
         ++triangleIndex) {
        const Triangle& triangle = m_walkableTriangles[triangleIndex];
        for (std::uint8_t edgeIndex = 0; edgeIndex < 3u; ++edgeIndex) {
            const std::array<odai::math::Vector3, 2> edge = edgeVertices(triangle, edgeIndex);
            edgeRefs[edgeKey(edge[0], edge[1], settings.edgeMergeEpsilon)].push_back({triangleIndex, edgeIndex});
        }
    }

    for (const auto& [unusedKey, refs] : edgeRefs) {
        (void)unusedKey;
        if (refs.size() < 2u) {
            continue;
        }
        for (std::size_t i = 0; i < refs.size(); ++i) {
            for (std::size_t j = i + 1u; j < refs.size(); ++j) {
                const EdgeRef lhs = refs[i];
                const EdgeRef rhs = refs[j];
                if (lhs.triangleIndex == rhs.triangleIndex) {
                    continue;
                }
                if (!canConnectEdges(
                        m_walkableTriangles[lhs.triangleIndex],
                        lhs.edgeIndex,
                        m_walkableTriangles[rhs.triangleIndex],
                        rhs.edgeIndex,
                        settings.maxClimb)) {
                    continue;
                }
                m_walkableTriangles[lhs.triangleIndex].neighbors.push_back(rhs.triangleIndex);
                m_walkableTriangles[rhs.triangleIndex].neighbors.push_back(lhs.triangleIndex);
                ++m_linkCount;
            }
        }
    }

    return !m_walkableTriangles.empty();
}

bool Navmesh::buildFromGpuSceneAsset(
    const odai::importer::GpuSceneAsset& scene,
    const NavmeshSettings& settings
) {
    std::vector<NavmeshBuildTriangle> triangles;

    auto appendTriangle = [&triangles](const odai::math::Vector3& p0,
                                       const odai::math::Vector3& p1,
                                       const odai::math::Vector3& p2) {
        NavmeshBuildTriangle triangle{};
        triangle.p0 = p0;
        triangle.p1 = p1;
        triangle.p2 = p2;
        triangles.push_back(triangle);
    };

    auto appendIndexRange = [&](const odai::importer::GpuSceneMeshAsset& mesh,
                                const std::array<float, 16>& transform,
                                std::uint32_t firstIndex,
                                std::uint32_t indexCount) {
        const std::uint32_t lastIndex = std::min<std::uint32_t>(
            firstIndex + indexCount,
            static_cast<std::uint32_t>(scene.indices.size()));
        for (std::uint32_t index = firstIndex; index + 2u < lastIndex; index += 3u) {
            const std::uint32_t i0 = scene.indices[index];
            const std::uint32_t i1 = scene.indices[index + 1u];
            const std::uint32_t i2 = scene.indices[index + 2u];
            if (i0 < mesh.firstVertex ||
                i1 < mesh.firstVertex ||
                i2 < mesh.firstVertex ||
                i0 >= mesh.firstVertex + mesh.vertexCount ||
                i1 >= mesh.firstVertex + mesh.vertexCount ||
                i2 >= mesh.firstVertex + mesh.vertexCount ||
                i0 >= scene.vertices.size() ||
                i1 >= scene.vertices.size() ||
                i2 >= scene.vertices.size()) {
                continue;
            }

            appendTriangle(
                transformPoint(transform, scene.vertices[i0].position),
                transformPoint(transform, scene.vertices[i1].position),
                transformPoint(transform, scene.vertices[i2].position));
        }
    };

    for (std::uint32_t instanceIndex = 0;
         instanceIndex < scene.instances.objectIndices.size();
         ++instanceIndex) {
        const std::uint32_t meshAssetIndex = scene.instances.meshAssetIndices[instanceIndex];
        const std::uint32_t objectIndex = scene.instances.objectIndices[instanceIndex];
        if (meshAssetIndex >= scene.meshAssets.size() ||
            objectIndex >= scene.objects.appliedTransforms.size()) {
            continue;
        }

        const odai::importer::GpuSceneMeshAsset& mesh = scene.meshAssets[meshAssetIndex];
        const std::array<float, 16>& transform = scene.objects.appliedTransforms[objectIndex];
        if (mesh.partCount == 0u) {
            appendIndexRange(mesh, transform, mesh.firstIndex, mesh.indexCount);
            continue;
        }

        for (std::uint32_t partIndex = 0; partIndex < mesh.partCount; ++partIndex) {
            const std::uint32_t scenePartIndex = mesh.firstPart + partIndex;
            if (scenePartIndex >= scene.meshParts.size()) {
                continue;
            }
            const odai::importer::GpuSceneMeshAssetPart& part = scene.meshParts[scenePartIndex];
            if ((part.flags & kGpuSceneComponentFlagAlphaTest) != 0u) {
                continue;
            }
            appendIndexRange(mesh, transform, part.firstIndex, part.indexCount);
        }
    }

    return build(triangles, settings);
}

bool Navmesh::empty() const {
    return m_walkableTriangles.empty();
}

Navmesh::Stats Navmesh::stats() const {
    Stats result{};
    result.walkableTriangleCount = static_cast<std::uint32_t>(m_walkableTriangles.size());
    result.obstacleTriangleCount = static_cast<std::uint32_t>(m_obstacleTriangles.size());
    result.linkCount = m_linkCount;
    return result;
}

bool Navmesh::findNearestPoint(
    const odai::math::Vector3& position,
    odai::math::Vector3& outPoint
) const {
    std::uint32_t triangleIndex = 0;
    return findContainingOrNearestTriangle(position, triangleIndex, outPoint);
}

bool Navmesh::findPath(
    const odai::math::Vector3& start,
    const odai::math::Vector3& end,
    std::vector<NavmeshPathPoint>& outPath
) const {
    outPath.clear();
    std::uint32_t startTriangle = 0;
    std::uint32_t endTriangle = 0;
    odai::math::Vector3 startPoint{};
    odai::math::Vector3 endPoint{};
    if (!findContainingOrNearestTriangle(start, startTriangle, startPoint) ||
        !findContainingOrNearestTriangle(end, endTriangle, endPoint)) {
        return false;
    }

    if (startTriangle == endTriangle) {
        outPath.push_back({startPoint});
        if (distanceSquaredXz(startPoint, endPoint) > 1.0f ||
            std::fabs(startPoint.y - endPoint.y) > 0.01f) {
            outPath.push_back({endPoint});
        }
        return true;
    }

    struct QueueNode {
        std::uint32_t triangleIndex = 0;
        float cost = 0.0f;
    };
    struct QueueNodeGreater {
        bool operator()(const QueueNode& lhs, const QueueNode& rhs) const {
            return lhs.cost > rhs.cost;
        }
    };

    const std::size_t triangleCount = m_walkableTriangles.size();
    std::vector<float> bestCost(triangleCount, std::numeric_limits<float>::infinity());
    std::vector<std::uint32_t> previous(triangleCount, std::numeric_limits<std::uint32_t>::max());
    std::priority_queue<QueueNode, std::vector<QueueNode>, QueueNodeGreater> open;
    bestCost[startTriangle] = 0.0f;
    open.push({startTriangle, std::sqrt(distanceSquaredXz(m_walkableTriangles[startTriangle].center, endPoint))});

    while (!open.empty()) {
        const QueueNode current = open.top();
        open.pop();
        if (current.triangleIndex == endTriangle) {
            break;
        }

        const Triangle& triangle = m_walkableTriangles[current.triangleIndex];
        for (const std::uint32_t neighborIndex : triangle.neighbors) {
            if (neighborIndex >= m_walkableTriangles.size()) {
                continue;
            }
            const Triangle& neighbor = m_walkableTriangles[neighborIndex];
            if (segmentBlocked(triangle.center, neighbor.center)) {
                continue;
            }
            const float stepCost = std::sqrt(distanceSquaredXz(triangle.center, neighbor.center)) +
                std::fabs(triangle.center.y - neighbor.center.y);
            const float nextCost = bestCost[current.triangleIndex] + stepCost;
            if (nextCost >= bestCost[neighborIndex]) {
                continue;
            }
            bestCost[neighborIndex] = nextCost;
            previous[neighborIndex] = current.triangleIndex;
            const float heuristic = std::sqrt(distanceSquaredXz(neighbor.center, endPoint));
            open.push({neighborIndex, nextCost + heuristic});
        }
    }

    if (previous[endTriangle] == std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }

    std::vector<std::uint32_t> reversed;
    for (std::uint32_t at = endTriangle;
         at != std::numeric_limits<std::uint32_t>::max();
         at = previous[at]) {
        reversed.push_back(at);
        if (at == startTriangle) {
            break;
        }
    }
    if (reversed.empty() || reversed.back() != startTriangle) {
        return false;
    }

    std::reverse(reversed.begin(), reversed.end());
    outPath.push_back({startPoint});
    for (std::size_t i = 1; i + 1u < reversed.size(); ++i) {
        outPath.push_back({m_walkableTriangles[reversed[i]].center});
    }
    outPath.push_back({endPoint});
    return true;
}

bool Navmesh::raycast(
    const odai::math::Vector3& start,
    const odai::math::Vector3& end,
    odai::math::Vector3& outFarthestReachable
) const {
    std::vector<NavmeshPathPoint> path;
    if (findPath(start, end, path)) {
        outFarthestReachable = path.empty() ? end : path.back().position;
        return true;
    }

    odai::math::Vector3 startPoint{};
    if (!findNearestPoint(start, startPoint)) {
        return false;
    }
    outFarthestReachable = startPoint;
    return true;
}

std::vector<NavmeshDebugTriangle> Navmesh::debugTriangles() const {
    std::vector<NavmeshDebugTriangle> result;
    result.reserve(m_walkableTriangles.size() + m_obstacleTriangles.size());
    const auto append = [&result](const Triangle& triangle) {
        result.push_back({triangle.p0, triangle.p1, triangle.p2, triangle.area});
    };
    for (const Triangle& triangle : m_walkableTriangles) {
        append(triangle);
    }
    for (const Triangle& triangle : m_obstacleTriangles) {
        append(triangle);
    }
    return result;
}

bool Navmesh::findContainingOrNearestTriangle(
    const odai::math::Vector3& position,
    std::uint32_t& outTriangleIndex,
    odai::math::Vector3& outPoint
) const {
    bool found = false;
    float bestScore = std::numeric_limits<float>::max();
    odai::math::Vector3 bestPoint{};
    std::uint32_t bestIndex = 0;
    const float maxDistanceSq = m_settings.nearestPointMaxDistance * m_settings.nearestPointMaxDistance;

    for (std::uint32_t triangleIndex = 0;
         triangleIndex < static_cast<std::uint32_t>(m_walkableTriangles.size());
         ++triangleIndex) {
        const Triangle& triangle = m_walkableTriangles[triangleIndex];
        const odai::math::Vector3 candidate = closestPointOnTriangleXz(triangle, position);
        const float xzDistanceSq = distanceSquaredXz(candidate, position);
        const float yDistance = std::fabs(candidate.y - position.y);
        if (xzDistanceSq > maxDistanceSq && yDistance > m_settings.nearestPointMaxDistance) {
            continue;
        }

        const float score = xzDistanceSq + (yDistance * yDistance * 0.25f);
        if (!found || score < bestScore) {
            found = true;
            bestScore = score;
            bestPoint = candidate;
            bestIndex = triangleIndex;
        }
    }

    if (!found) {
        return false;
    }
    outTriangleIndex = bestIndex;
    outPoint = bestPoint;
    return true;
}

bool Navmesh::segmentBlocked(
    const odai::math::Vector3& start,
    const odai::math::Vector3& end
) const {
    const QueryAabb segmentBounds{
        std::min(start.x, end.x) - m_settings.agentRadius,
        std::max(start.x, end.x) + m_settings.agentRadius,
        std::min(start.y, end.y),
        std::max(start.y, end.y) + m_settings.agentHeight,
        std::min(start.z, end.z) - m_settings.agentRadius,
        std::max(start.z, end.z) + m_settings.agentRadius
    };

    for (const Triangle& obstacle : m_obstacleTriangles) {
        const QueryAabb obstacleBounds = makeBounds(obstacle.p0, obstacle.p1, obstacle.p2);
        if (!aabbOverlaps(segmentBounds, obstacleBounds)) {
            continue;
        }

        const std::array<std::array<odai::math::Vector3, 2>, 3> edges{{
            {obstacle.p0, obstacle.p1},
            {obstacle.p1, obstacle.p2},
            {obstacle.p2, obstacle.p0}
        }};
        for (const std::array<odai::math::Vector3, 2>& edge : edges) {
            if (!rangesOverlap(
                    std::min(start.y, end.y),
                    std::max(start.y, end.y) + m_settings.agentHeight,
                    std::min(edge[0].y, edge[1].y),
                    std::max(edge[0].y, edge[1].y))) {
                continue;
            }
            if (segmentsIntersectXz(start, end, edge[0], edge[1])) {
                return true;
            }
        }
    }
    return false;
}

} // namespace odai::world
