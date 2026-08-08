#pragma once

// Collision for the streamed Fallout world: terrain the player stands on, and
// solid geometry they cannot walk through.
//
// Fed per streamed cell and derived entirely from the cell's built
// ImportedScene, so it works identically whether that scene came from the
// plugin or from the on-disk cache -- there is no second source of truth.
//
// Everything here is ENGINE space (Y-up); the Fallout Z-up convention stops at
// the scene builder.
//
// Two representations, chosen for what each is good at:
//   * terrain as the per-cell 33x33 height lattice it already is, sampled
//     bilinearly -- exact and far cheaper than raycasting a mesh;
//   * static geometry as TRIANGLES, bucketed into a uniform XZ grid.
//
// The triangles matter. One oriented box per mesh was tried first and is the
// wrong shape for this world: a rock's box covers the walkable ground around
// it, so the player stops dead with open terrain visibly ahead. Triangles let
// "walk up and over this" and "do not pass through this" be different answers,
// which a single box per mesh cannot express -- it has one surface and no
// interior.

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "import/cell_residency_planner.h"
#include "import/imported_scene.h"

namespace odai::games::newvegas {

struct CollisionTuning {
    // Player capsule, in world units (~70 per metre).
    float radius = 34.0f;
    float eyeHeight = 120.0f;
    // Surfaces within this of the player's feet are stepped onto rather than
    // blocking: kerbs, road edges, rubble, the lip of a rock.
    float stepHeight = 55.0f;
    // A triangle whose upward normal component is at least this is a floor to
    // stand on; below it, a wall to be stopped by. cos(50 degrees) ~ 0.64, so
    // slopes up to about 50 degrees are walkable.
    float minWalkableNormalY = 0.64f;
};

class CollisionWorld {
public:
    void setTuning(const CollisionTuning& tuning) { m_tuning = tuning; }
    [[nodiscard]] const CollisionTuning& tuning() const { return m_tuning; }

    // Builds this cell's terrain lattice and collision triangles from its scene,
    // replacing anything already held for the same cell.
    void addCell(const importer::CellCoord& cell, const importer::ImportedScene& scene);
    void removeCell(const importer::CellCoord& cell);
    void clear();

    // Height of the surface the player stands on: terrain, raised to any
    // walkable geometry above it and at or below head height. `referenceY` is
    // the player's current foot height, used to reject ceilings and upper
    // storeys. False when no resident cell covers the position, which the caller
    // must treat as "do not clamp" rather than "height zero".
    [[nodiscard]] bool groundHeight(
        float worldX, float worldZ, float referenceY, float& outHeight) const;

    // Terrain only, ignoring geometry. Mostly for diagnostics.
    [[nodiscard]] bool terrainHeight(float worldX, float worldZ, float& outHeight) const;

    // Slides a horizontal move out of any wall it would end up inside.
    void resolveHorizontal(float& worldX, float eyeY, float& worldZ) const;

    [[nodiscard]] std::size_t residentCellCount() const { return m_cells.size(); }
    [[nodiscard]] std::size_t triangleCount() const;

    struct Triangle {
        // Three world-space vertices, xyz each.
        float v[9] = {};
        // Y component of the unit normal: the one number both the walkable test
        // and the wall test need, so the rest is not worth storing.
        float normalY = 0.0f;
    };

    // Visits every triangle bucketed near a position. Public for diagnostics.
    void forEachNearbyTriangle(
        float worldX, float worldZ, const std::function<void(const Triangle&)>& visit) const;

private:
    // Per cell, an 8x8 grid of 512-unit buckets. A query touches at most a 3x3
    // ring of these, so cost follows what is near the player rather than how
    // much is resident.
    static constexpr int kBucketGrid = 8;
    static constexpr float kBucketSize = 4096.0f / static_cast<float>(kBucketGrid);

    struct CellCollision {
        // 33x33 terrain posts, engine Y, row-major (gridZ * 33 + gridX). Empty
        // when the cell has no terrain.
        std::vector<float> heights;
        float originX = 0.0f;  // engine-space corner of the cell
        float originZ = 0.0f;
        std::vector<Triangle> triangles;
        std::vector<std::vector<std::uint32_t>> buckets;
    };

    CollisionTuning m_tuning{};
    std::unordered_map<importer::CellCoord, CellCollision, importer::CellCoordHash> m_cells;
};

}  // namespace odai::games::newvegas
