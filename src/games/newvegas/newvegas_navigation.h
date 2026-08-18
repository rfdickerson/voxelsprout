#pragma once

// Resident authored navigation for streamed Bethesda worlds.
//
// NAVM vertices arrive in Bethesda's Z-up space and are converted here once,
// when their cell becomes resident. Actor movement then deals only in engine
// space (Y-up), like collision and rendering. Meshes remain separated because
// their triangle neighbour indices are local to one NAVM record.

#include "import/cell_residency_planner.h"
#include "import/fnv/fallout_records.h"
#include "math/math.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace odai::games::newvegas {

class ActorNavigationWorld {
public:
    void addCell(
        const importer::CellCoord& cell,
        const std::vector<importer::fnv::FalloutNavMeshRecord>& records);
    void removeCell(const importer::CellCoord& cell);
    void clear();

    // Closest walkable point to an authored/spawned actor position. Horizontal
    // and vertical limits prevent a ground-floor mesh from stealing an actor
    // authored on a battlement or balcony.
    [[nodiscard]] bool projectPoint(
        float worldX,
        float worldY,
        float worldZ,
        float maxHorizontalDistance,
        float maxVerticalDistance,
        odai::math::Vector3& outPoint) const;

    // Chooses a reachable triangle near origin and returns an edge-by-edge
    // route from start. Every segment remains inside connected nav triangles;
    // it cannot cut across a rock, wall or unmeshed decoration.
    [[nodiscard]] bool buildWanderPath(
        const odai::math::Vector3& start,
        const odai::math::Vector3& origin,
        float radius,
        std::uint32_t randomValue,
        std::vector<odai::math::Vector3>& outWaypoints) const;

    [[nodiscard]] std::size_t meshCount() const;
    [[nodiscard]] std::size_t triangleCount() const;

private:
    struct Triangle {
        std::uint16_t vertex[3] = {};
        std::uint16_t neighbour[3] = {
            importer::fnv::kNavMeshNoNeighbour,
            importer::fnv::kNavMeshNoNeighbour,
            importer::fnv::kNavMeshNoNeighbour};
    };
    struct Mesh {
        std::uint32_t formId = 0u;
        std::vector<odai::math::Vector3> vertices;
        std::vector<Triangle> triangles;
    };
    struct Location {
        const Mesh* mesh = nullptr;
        std::size_t triangle = 0u;
        odai::math::Vector3 point{};
        float score = 0.0f;
    };

    [[nodiscard]] bool findNearest(
        const odai::math::Vector3& point,
        float maxHorizontalDistance,
        float maxVerticalDistance,
        Location& outLocation) const;

    std::unordered_map<importer::CellCoord, std::vector<Mesh>, importer::CellCoordHash> m_cells;
};

}  // namespace odai::games::newvegas
