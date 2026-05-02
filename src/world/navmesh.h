#pragma once

#include "import/gpu_scene.h"
#include "math/math.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace odai::world {

enum class NavmeshArea : std::uint8_t {
    Ground,
    Water,
    Obstacle
};

struct NavmeshSettings {
    float agentRadius = 28.0f;
    float agentHeight = 128.0f;
    float maxClimb = 28.0f;
    float maxSlopeDegrees = 49.0f;
    float nearestPointMaxDistance = 256.0f;
    float edgeMergeEpsilon = 0.25f;
};

struct NavmeshBuildTriangle {
    odai::math::Vector3 p0{};
    odai::math::Vector3 p1{};
    odai::math::Vector3 p2{};
    NavmeshArea area = NavmeshArea::Ground;
};

struct NavmeshPathPoint {
    odai::math::Vector3 position{};
};

struct NavmeshDebugTriangle {
    odai::math::Vector3 p0{};
    odai::math::Vector3 p1{};
    odai::math::Vector3 p2{};
    NavmeshArea area = NavmeshArea::Ground;
};

class Navmesh {
public:
    struct Stats {
        std::uint32_t walkableTriangleCount = 0;
        std::uint32_t obstacleTriangleCount = 0;
        std::uint32_t linkCount = 0;
    };

    void clear();
    bool build(std::span<const NavmeshBuildTriangle> triangles, const NavmeshSettings& settings);
    bool buildFromGpuSceneAsset(
        const odai::importer::GpuSceneAsset& scene,
        const NavmeshSettings& settings
    );

    [[nodiscard]] bool empty() const;
    [[nodiscard]] Stats stats() const;
    [[nodiscard]] bool saveBinary(const std::filesystem::path& path) const;
    [[nodiscard]] bool loadBinary(const std::filesystem::path& path, const NavmeshSettings& expectedSettings);

    [[nodiscard]] bool findNearestPoint(
        const odai::math::Vector3& position,
        odai::math::Vector3& outPoint
    ) const;

    [[nodiscard]] bool findPath(
        const odai::math::Vector3& start,
        const odai::math::Vector3& end,
        std::vector<NavmeshPathPoint>& outPath
    ) const;

    [[nodiscard]] bool raycast(
        const odai::math::Vector3& start,
        const odai::math::Vector3& end,
        odai::math::Vector3& outFarthestReachable
    ) const;

    [[nodiscard]] std::vector<NavmeshDebugTriangle> debugTriangles() const;

    struct Triangle {
        odai::math::Vector3 p0{};
        odai::math::Vector3 p1{};
        odai::math::Vector3 p2{};
        odai::math::Vector3 normal{0.0f, 1.0f, 0.0f};
        odai::math::Vector3 center{};
        NavmeshArea area = NavmeshArea::Ground;
        std::vector<std::uint32_t> neighbors;
    };

private:
    [[nodiscard]] bool findContainingOrNearestTriangle(
        const odai::math::Vector3& position,
        std::uint32_t& outTriangleIndex,
        odai::math::Vector3& outPoint
    ) const;

    [[nodiscard]] bool segmentBlocked(
        const odai::math::Vector3& start,
        const odai::math::Vector3& end
    ) const;

    NavmeshSettings m_settings{};
    std::vector<Triangle> m_walkableTriangles;
    std::vector<Triangle> m_obstacleTriangles;
    std::uint32_t m_linkCount = 0;
};

} // namespace odai::world
