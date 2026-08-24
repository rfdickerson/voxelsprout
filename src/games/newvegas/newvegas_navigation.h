#pragma once

// Resident authored navigation for streamed Bethesda worlds.
//
// NAVM vertices arrive in Bethesda's Z-up space and are converted here once,
// when their cell becomes resident. Actor movement then deals only in engine
// space (Y-up), like collision and rendering. Triangle neighbour indices stay
// local to one NAVM record; resident records are stitched only where authored
// border edges coincide.

#include "import/cell_residency_planner.h"
#include "import/fnv/fallout_records.h"
#include "import/imported_scene.h"
#include "math/math.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace odai::games::newvegas {

enum class ActorNavigationStepKind : std::uint8_t {
    Walk,
    ActivateDoor,
};

// A route is not only a polyline. Teleport doors are authored off-mesh links:
// the actor walks to position, activates the stable source reference, and is
// relocated to arrivalPosition before consuming the following step.
struct ActorNavigationStep {
    ActorNavigationStepKind kind = ActorNavigationStepKind::Walk;
    odai::math::Vector3 position{};
    odai::math::Vector3 arrivalPosition{};
    std::uint32_t doorReferenceFormId = 0u;
    friend bool operator==(const ActorNavigationStep& left,
                           const ActorNavigationStep& right) {
        return left.kind == right.kind &&
            left.position.x == right.position.x &&
            left.position.y == right.position.y &&
            left.position.z == right.position.z &&
            left.arrivalPosition.x == right.arrivalPosition.x &&
            left.arrivalPosition.y == right.arrivalPosition.y &&
            left.arrivalPosition.z == right.arrivalPosition.z &&
            left.doorReferenceFormId == right.doorReferenceFormId;
    }
};

class ActorNavigationWorld {
public:
    void addCell(
        const importer::CellCoord& cell,
        const std::vector<importer::fnv::FalloutNavMeshRecord>& records);
    void removeCell(const importer::CellCoord& cell);
    void clear();

    // Replaces the currently usable resident-space teleport links. Callers
    // must omit doors whose destination space is not resident; coordinates in
    // two interiors are unrelated even when their numeric values overlap.
    void setResidentDoors(const std::vector<importer::ImportedSceneDoor>& doors);

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
        std::vector<ActorNavigationStep>& outWaypoints) const;

    // Builds a deterministic point-to-point route on the currently resident
    // authored NAVM. Coincident border edges stitch adjacent NAVM records.
    // Resident teleport doors become explicit ActivateDoor actions. A door to
    // an unloaded space is never represented as a straight-line waypoint.
    [[nodiscard]] bool buildPath(
        const odai::math::Vector3& start,
        const odai::math::Vector3& goal,
        std::vector<ActorNavigationStep>& outWaypoints) const;

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
        importer::CellCoord cell;
        std::vector<odai::math::Vector3> vertices;
        std::vector<Triangle> triangles;
        std::vector<importer::fnv::FalloutNavMeshDoorPortal> doorPortals;
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
    std::vector<importer::ImportedSceneDoor> m_residentDoors;
};

}  // namespace odai::games::newvegas
