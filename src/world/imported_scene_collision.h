#pragma once

#include "import/gpu_scene.h"
#include "math/math.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace odai::world {

class ImportedSceneCollision {
public:
    struct BuildStats {
        std::uint32_t triangleCount = 0;
        std::uint32_t tileCount = 0;
    };

    struct GroundHit {
        float y = 0.0f;
        odai::math::Vector3 normal{0.0f, 1.0f, 0.0f};
    };

    struct CeilingHit {
        float y = 0.0f;
        odai::math::Vector3 normal{0.0f, -1.0f, 0.0f};
    };

    struct QueryAabb {
        float minX = 0.0f;
        float maxX = 0.0f;
        float minY = 0.0f;
        float maxY = 0.0f;
        float minZ = 0.0f;
        float maxZ = 0.0f;
    };

    struct Triangle {
        odai::math::Vector3 p0;
        odai::math::Vector3 p1;
        odai::math::Vector3 p2;
        odai::math::Vector3 normal;
        QueryAabb bounds;
    };

    void clear();
    bool build(const odai::importer::GpuSceneAsset& scene);

    [[nodiscard]] bool empty() const;
    [[nodiscard]] BuildStats stats() const;

    [[nodiscard]] bool findGroundSupport(
        float centerX,
        float feetY,
        float centerZ,
        float radius,
        float maxDrop,
        float maxStepUp,
        float minWalkableNormalY,
        GroundHit& outHit
    ) const;

    [[nodiscard]] bool findCeiling(
        float centerX,
        float topY,
        float centerZ,
        float radius,
        float maxPenetration,
        CeilingHit& outHit
    ) const;

    [[nodiscard]] bool resolveHorizontalCylinder(
        float centerX,
        float feetY,
        float centerZ,
        float radius,
        float height,
        float minWalkableNormalY,
        odai::math::Vector3& outCorrection
    ) const;

private:
    [[nodiscard]] static int tileCoord(float value);
    [[nodiscard]] static std::int64_t tileKey(int tileX, int tileZ);
    void gatherCandidateTriangles(const QueryAabb& bounds) const;

    std::vector<Triangle> m_triangles;
    std::unordered_map<std::int64_t, std::vector<std::uint32_t>> m_tileTriangleIndices;

    mutable std::vector<std::uint32_t> m_queryScratch;
    mutable std::vector<std::uint32_t> m_queryStamps;
    mutable std::uint32_t m_queryStamp = 1;
};

} // namespace odai::world
