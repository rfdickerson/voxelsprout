#include "world/imported_scene_collision.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace odai::world {

namespace {

constexpr float kTileSize = 512.0f;
constexpr float kGeometryEpsilon = 1e-4f;
constexpr float kCollisionSkin = 0.05f;

[[nodiscard]] odai::math::Vector3 transformPoint(const std::array<float, 16>& transform, const float point[3]) {
    return {
        (transform[0] * point[0]) + (transform[1] * point[1]) + (transform[2] * point[2]) + transform[3],
        (transform[4] * point[0]) + (transform[5] * point[1]) + (transform[6] * point[2]) + transform[7],
        (transform[8] * point[0]) + (transform[9] * point[1]) + (transform[10] * point[2]) + transform[11]
    };
}

[[nodiscard]] ImportedSceneCollision::QueryAabb makeTriangleBounds(
    const odai::math::Vector3& p0,
    const odai::math::Vector3& p1,
    const odai::math::Vector3& p2
) {
    ImportedSceneCollision::QueryAabb bounds{};
    bounds.minX = std::min({p0.x, p1.x, p2.x});
    bounds.maxX = std::max({p0.x, p1.x, p2.x});
    bounds.minY = std::min({p0.y, p1.y, p2.y});
    bounds.maxY = std::max({p0.y, p1.y, p2.y});
    bounds.minZ = std::min({p0.z, p1.z, p2.z});
    bounds.maxZ = std::max({p0.z, p1.z, p2.z});
    return bounds;
}

[[nodiscard]] bool aabbOverlaps(
    const ImportedSceneCollision::QueryAabb& lhs,
    const ImportedSceneCollision::QueryAabb& rhs
) {
    return
        lhs.maxX >= rhs.minX &&
        lhs.minX <= rhs.maxX &&
        lhs.maxY >= rhs.minY &&
        lhs.minY <= rhs.maxY &&
        lhs.maxZ >= rhs.minZ &&
        lhs.minZ <= rhs.maxZ;
}

[[nodiscard]] bool pointInsideTriangleXz(
    const ImportedSceneCollision::Triangle& triangle,
    float x,
    float z
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
    const float u = ((v2x * v1z) - (v1x * v2z)) * invDenom;
    const float v = ((v0x * v2z) - (v2x * v0z)) * invDenom;
    return u >= -0.001f && v >= -0.001f && (u + v) <= 1.001f;
}

[[nodiscard]] bool triangleHeightAtXz(
    const ImportedSceneCollision::Triangle& triangle,
    float x,
    float z,
    float& outY
) {
    if (!pointInsideTriangleXz(triangle, x, z)) {
        return false;
    }
    if (std::fabs(triangle.normal.y) <= kGeometryEpsilon) {
        return false;
    }

    outY = triangle.p0.y -
        ((triangle.normal.x * (x - triangle.p0.x)) + (triangle.normal.z * (z - triangle.p0.z))) /
            triangle.normal.y;
    return true;
}

[[nodiscard]] odai::math::Vector3 orientedUpNormal(odai::math::Vector3 normal) {
    if (normal.y < 0.0f) {
        normal *= -1.0f;
    }
    return normal;
}

[[nodiscard]] odai::math::Vector3 closestPointOnSegmentXz(
    float pointX,
    float pointZ,
    const odai::math::Vector3& a,
    const odai::math::Vector3& b
) {
    const float abX = b.x - a.x;
    const float abZ = b.z - a.z;
    const float lengthSq = (abX * abX) + (abZ * abZ);
    if (lengthSq <= kGeometryEpsilon) {
        return {a.x, 0.0f, a.z};
    }

    const float t = std::clamp(((pointX - a.x) * abX + (pointZ - a.z) * abZ) / lengthSq, 0.0f, 1.0f);
    return {a.x + (abX * t), 0.0f, a.z + (abZ * t)};
}

[[nodiscard]] odai::math::Vector3 closestPointOnTriangleXz(
    float pointX,
    float pointZ,
    const ImportedSceneCollision::Triangle& triangle
) {
    if (pointInsideTriangleXz(triangle, pointX, pointZ)) {
        return {pointX, 0.0f, pointZ};
    }

    const odai::math::Vector3 edge01 = closestPointOnSegmentXz(pointX, pointZ, triangle.p0, triangle.p1);
    const odai::math::Vector3 edge12 = closestPointOnSegmentXz(pointX, pointZ, triangle.p1, triangle.p2);
    const odai::math::Vector3 edge20 = closestPointOnSegmentXz(pointX, pointZ, triangle.p2, triangle.p0);

    const auto distanceSq = [pointX, pointZ](const odai::math::Vector3& point) {
        const float dx = pointX - point.x;
        const float dz = pointZ - point.z;
        return (dx * dx) + (dz * dz);
    };

    odai::math::Vector3 closest = edge01;
    float bestDistanceSq = distanceSq(edge01);
    const float edge12DistanceSq = distanceSq(edge12);
    if (edge12DistanceSq < bestDistanceSq) {
        closest = edge12;
        bestDistanceSq = edge12DistanceSq;
    }
    if (distanceSq(edge20) < bestDistanceSq) {
        closest = edge20;
    }
    return closest;
}

} // namespace

void ImportedSceneCollision::clear() {
    m_triangles.clear();
    m_tileTriangleIndices.clear();
    m_queryScratch.clear();
    m_queryStamps.clear();
    m_queryStamp = 1;
}

bool ImportedSceneCollision::build(const odai::importer::GpuSceneAsset& scene) {
    clear();

    auto appendTriangle = [this](const odai::math::Vector3& p0,
                                const odai::math::Vector3& p1,
                                const odai::math::Vector3& p2) {
        const odai::math::Vector3 normal = odai::math::cross(p1 - p0, p2 - p0);
        const float normalLength = odai::math::length(normal);
        if (normalLength <= kGeometryEpsilon) {
            return;
        }

        Triangle triangle{};
        triangle.p0 = p0;
        triangle.p1 = p1;
        triangle.p2 = p2;
        triangle.normal = normal / normalLength;
        triangle.bounds = makeTriangleBounds(p0, p1, p2);

        const std::uint32_t triangleIndex = static_cast<std::uint32_t>(m_triangles.size());
        m_triangles.push_back(triangle);

        const int minTileX = tileCoord(triangle.bounds.minX);
        const int maxTileX = tileCoord(triangle.bounds.maxX);
        const int minTileZ = tileCoord(triangle.bounds.minZ);
        const int maxTileZ = tileCoord(triangle.bounds.maxZ);
        for (int tileZ = minTileZ; tileZ <= maxTileZ; ++tileZ) {
            for (int tileX = minTileX; tileX <= maxTileX; ++tileX) {
                m_tileTriangleIndices[tileKey(tileX, tileZ)].push_back(triangleIndex);
            }
        }
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
            appendIndexRange(mesh, transform, part.firstIndex, part.indexCount);
        }
    }

    m_queryStamps.assign(m_triangles.size(), 0u);
    return !m_triangles.empty();
}

bool ImportedSceneCollision::buildFromPackedScene(const odai::importer::ImportedScene& scene) {
    clear();

    auto appendTriangle = [this](const odai::math::Vector3& p0,
                                const odai::math::Vector3& p1,
                                const odai::math::Vector3& p2) {
        const odai::math::Vector3 normal = odai::math::cross(p1 - p0, p2 - p0);
        const float normalLength = odai::math::length(normal);
        if (normalLength <= kGeometryEpsilon) {
            return;
        }

        Triangle triangle{};
        triangle.p0 = p0;
        triangle.p1 = p1;
        triangle.p2 = p2;
        triangle.normal = normal / normalLength;
        triangle.bounds = makeTriangleBounds(p0, p1, p2);

        const std::uint32_t triangleIndex = static_cast<std::uint32_t>(m_triangles.size());
        m_triangles.push_back(triangle);

        const int minTileX = tileCoord(triangle.bounds.minX);
        const int maxTileX = tileCoord(triangle.bounds.maxX);
        const int minTileZ = tileCoord(triangle.bounds.minZ);
        const int maxTileZ = tileCoord(triangle.bounds.maxZ);
        for (int tileZ = minTileZ; tileZ <= maxTileZ; ++tileZ) {
            for (int tileX = minTileX; tileX <= maxTileX; ++tileX) {
                m_tileTriangleIndices[tileKey(tileX, tileZ)].push_back(triangleIndex);
            }
        }
    };

    auto packedPosition = [](const odai::importer::ImportedScenePackedVertex& vertex) {
        return odai::math::Vector3{vertex.position[0], vertex.position[1], vertex.position[2]};
    };

    for (const odai::importer::ImportedScenePackedDraw& draw : scene.packedDraws) {
        const std::uint32_t lastIndex = std::min<std::uint32_t>(
            draw.firstIndex + draw.indexCount,
            static_cast<std::uint32_t>(scene.packedIndices.size()));
        for (std::uint32_t index = draw.firstIndex; index + 2u < lastIndex; index += 3u) {
            const std::uint32_t i0 = scene.packedIndices[index];
            const std::uint32_t i1 = scene.packedIndices[index + 1u];
            const std::uint32_t i2 = scene.packedIndices[index + 2u];
            if (i0 >= scene.packedVertices.size() ||
                i1 >= scene.packedVertices.size() ||
                i2 >= scene.packedVertices.size()) {
                continue;
            }
            appendTriangle(
                packedPosition(scene.packedVertices[i0]),
                packedPosition(scene.packedVertices[i1]),
                packedPosition(scene.packedVertices[i2]));
        }
    }

    m_queryStamps.assign(m_triangles.size(), 0u);
    return !m_triangles.empty();
}

bool ImportedSceneCollision::empty() const {
    return m_triangles.empty();
}

ImportedSceneCollision::BuildStats ImportedSceneCollision::stats() const {
    BuildStats result{};
    result.triangleCount = static_cast<std::uint32_t>(m_triangles.size());
    result.tileCount = static_cast<std::uint32_t>(m_tileTriangleIndices.size());
    return result;
}

bool ImportedSceneCollision::findGroundSupport(
    float centerX,
    float feetY,
    float centerZ,
    float radius,
    float maxDrop,
    float maxStepUp,
    float minWalkableNormalY,
    GroundHit& outHit
) const {
    const QueryAabb queryBounds{
        centerX - radius,
        centerX + radius,
        feetY - maxDrop,
        feetY + maxStepUp,
        centerZ - radius,
        centerZ + radius
    };
    gatherCandidateTriangles(queryBounds);

    const float sampleRadius = radius * 0.55f;
    const float diagonalRadius = radius * 0.38f;
    const std::array<std::array<float, 2>, 9> samples{{
        {0.0f, 0.0f},
        {sampleRadius, 0.0f},
        {-sampleRadius, 0.0f},
        {0.0f, sampleRadius},
        {0.0f, -sampleRadius},
        {diagonalRadius, diagonalRadius},
        {-diagonalRadius, diagonalRadius},
        {diagonalRadius, -diagonalRadius},
        {-diagonalRadius, -diagonalRadius}
    }};

    bool found = false;
    float bestY = -std::numeric_limits<float>::infinity();
    odai::math::Vector3 bestNormal{0.0f, 1.0f, 0.0f};

    auto considerCandidate = [&](const Triangle& triangle, float sampleX, float sampleZ) {
        float candidateY = 0.0f;
        if (!triangleHeightAtXz(triangle, sampleX, sampleZ, candidateY)) {
            return;
        }
        if (candidateY < feetY - maxDrop || candidateY > feetY + maxStepUp) {
            return;
        }
        if (!found || candidateY > bestY) {
            found = true;
            bestY = candidateY;
            bestNormal = orientedUpNormal(triangle.normal);
        }
    };

    for (const std::uint32_t triangleIndex : m_queryScratch) {
        const Triangle& triangle = m_triangles[triangleIndex];
        if (!aabbOverlaps(triangle.bounds, queryBounds)) {
            continue;
        }
        if (std::fabs(triangle.normal.y) < minWalkableNormalY) {
            continue;
        }

        const odai::math::Vector3 closest = closestPointOnTriangleXz(centerX, centerZ, triangle);
        const float closestDx = closest.x - centerX;
        const float closestDz = closest.z - centerZ;
        if ((closestDx * closestDx) + (closestDz * closestDz) <= radius * radius) {
            considerCandidate(triangle, closest.x, closest.z);
        }

        for (const std::array<float, 2>& sample : samples) {
            considerCandidate(triangle, centerX + sample[0], centerZ + sample[1]);
        }
    }

    if (!found) {
        return false;
    }

    outHit.y = bestY;
    outHit.normal = bestNormal;
    return true;
}

bool ImportedSceneCollision::findCeiling(
    float centerX,
    float topY,
    float centerZ,
    float radius,
    float maxPenetration,
    CeilingHit& outHit
) const {
    const QueryAabb queryBounds{
        centerX - radius,
        centerX + radius,
        topY - maxPenetration,
        topY + kCollisionSkin,
        centerZ - radius,
        centerZ + radius
    };
    gatherCandidateTriangles(queryBounds);

    const float sampleRadius = radius * 0.55f;
    const std::array<std::array<float, 2>, 5> samples{{
        {0.0f, 0.0f},
        {sampleRadius, 0.0f},
        {-sampleRadius, 0.0f},
        {0.0f, sampleRadius},
        {0.0f, -sampleRadius}
    }};

    bool found = false;
    float bestY = std::numeric_limits<float>::infinity();
    odai::math::Vector3 bestNormal{0.0f, -1.0f, 0.0f};

    for (const std::uint32_t triangleIndex : m_queryScratch) {
        const Triangle& triangle = m_triangles[triangleIndex];
        if (!aabbOverlaps(triangle.bounds, queryBounds)) {
            continue;
        }
        if (std::fabs(triangle.normal.y) <= kGeometryEpsilon) {
            continue;
        }

        for (const std::array<float, 2>& sample : samples) {
            float candidateY = 0.0f;
            if (!triangleHeightAtXz(triangle, centerX + sample[0], centerZ + sample[1], candidateY)) {
                continue;
            }
            if (candidateY < topY - maxPenetration || candidateY > topY + kCollisionSkin) {
                continue;
            }
            if (!found || candidateY < bestY) {
                found = true;
                bestY = candidateY;
                bestNormal = triangle.normal.y > 0.0f ? -triangle.normal : triangle.normal;
            }
        }
    }

    if (!found) {
        return false;
    }

    outHit.y = bestY;
    outHit.normal = bestNormal;
    return true;
}

bool ImportedSceneCollision::resolveHorizontalCylinder(
    float centerX,
    float feetY,
    float centerZ,
    float radius,
    float height,
    float minWalkableNormalY,
    odai::math::Vector3& outCorrection
) const {
    outCorrection = {};
    if (m_triangles.empty()) {
        return false;
    }

    float resolvedX = centerX;
    float resolvedZ = centerZ;
    bool corrected = false;

    for (int iteration = 0; iteration < 4; ++iteration) {
        const QueryAabb queryBounds{
            resolvedX - radius,
            resolvedX + radius,
            feetY + kCollisionSkin,
            feetY + height - kCollisionSkin,
            resolvedZ - radius,
            resolvedZ + radius
        };
        gatherCandidateTriangles(queryBounds);

        odai::math::Vector3 push{};
        for (const std::uint32_t triangleIndex : m_queryScratch) {
            const Triangle& triangle = m_triangles[triangleIndex];
            if (!aabbOverlaps(triangle.bounds, queryBounds)) {
                continue;
            }
            if (std::fabs(triangle.normal.y) >= minWalkableNormalY) {
                continue;
            }

            const odai::math::Vector3 closest = closestPointOnTriangleXz(resolvedX, resolvedZ, triangle);
            float directionX = resolvedX - closest.x;
            float directionZ = resolvedZ - closest.z;
            float distanceSq = (directionX * directionX) + (directionZ * directionZ);
            bool directionFromTriangleNormal = false;
            if (distanceSq <= kGeometryEpsilon) {
                directionX = triangle.normal.x;
                directionZ = triangle.normal.z;
                const float normalLengthSq = (directionX * directionX) + (directionZ * directionZ);
                if (normalLengthSq <= kGeometryEpsilon) {
                    continue;
                }
                const float invNormalLength = 1.0f / std::sqrt(normalLengthSq);
                directionX *= invNormalLength;
                directionZ *= invNormalLength;
                if (((resolvedX - triangle.p0.x) * directionX + (resolvedZ - triangle.p0.z) * directionZ) < 0.0f) {
                    directionX = -directionX;
                    directionZ = -directionZ;
                }
                directionFromTriangleNormal = true;
                distanceSq = 0.0f;
            }

            const float distance = std::sqrt(distanceSq);
            if (distance >= radius) {
                continue;
            }

            const float pushDistance = (radius - distance) + kCollisionSkin;
            if (directionFromTriangleNormal) {
                push.x += directionX * pushDistance;
                push.z += directionZ * pushDistance;
            } else if (distance > kGeometryEpsilon) {
                push.x += (directionX / distance) * pushDistance;
                push.z += (directionZ / distance) * pushDistance;
            }
        }

        const float pushLengthSq = (push.x * push.x) + (push.z * push.z);
        if (pushLengthSq <= 1e-6f) {
            break;
        }

        const float pushLength = std::sqrt(pushLengthSq);
        const float maxPush = radius * 0.75f;
        if (pushLength > maxPush) {
            push *= (maxPush / pushLength);
        }
        resolvedX += push.x;
        resolvedZ += push.z;
        corrected = true;
    }

    if (!corrected) {
        return false;
    }

    outCorrection.x = resolvedX - centerX;
    outCorrection.z = resolvedZ - centerZ;
    return odai::math::lengthSquared(outCorrection) > 1e-6f;
}

int ImportedSceneCollision::tileCoord(float value) {
    return static_cast<int>(std::floor(value / kTileSize));
}

std::int64_t ImportedSceneCollision::tileKey(int tileX, int tileZ) {
    const std::uint64_t x = static_cast<std::uint32_t>(tileX);
    const std::uint64_t z = static_cast<std::uint32_t>(tileZ);
    return static_cast<std::int64_t>((x << 32u) | z);
}

void ImportedSceneCollision::gatherCandidateTriangles(const QueryAabb& bounds) const {
    m_queryScratch.clear();
    if (m_triangles.empty()) {
        return;
    }

    ++m_queryStamp;
    if (m_queryStamp == 0u) {
        std::fill(m_queryStamps.begin(), m_queryStamps.end(), 0u);
        m_queryStamp = 1u;
    }

    const int minTileX = tileCoord(bounds.minX);
    const int maxTileX = tileCoord(bounds.maxX);
    const int minTileZ = tileCoord(bounds.minZ);
    const int maxTileZ = tileCoord(bounds.maxZ);
    for (int tileZ = minTileZ; tileZ <= maxTileZ; ++tileZ) {
        for (int tileX = minTileX; tileX <= maxTileX; ++tileX) {
            const auto it = m_tileTriangleIndices.find(tileKey(tileX, tileZ));
            if (it == m_tileTriangleIndices.end()) {
                continue;
            }

            for (const std::uint32_t triangleIndex : it->second) {
                if (triangleIndex >= m_queryStamps.size() ||
                    m_queryStamps[triangleIndex] == m_queryStamp) {
                    continue;
                }
                m_queryStamps[triangleIndex] = m_queryStamp;
                m_queryScratch.push_back(triangleIndex);
            }
        }
    }
}

} // namespace odai::world
