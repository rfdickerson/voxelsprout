#pragma once

#include <cstdint>
#include <vector>

#include "import/imported_scene.h"
#include "procgen/csg.h"

namespace odai::procgen {

// Flat-shaded triangle mesh in local space — the cacheable unit between the
// CSG evaluation and per-tile scene emission. textureIndex stays 0xffffffff so
// the renderer uses the per-vertex color path.
struct TriMesh {
    std::vector<odai::importer::ImportedScenePackedVertex> vertices;
    std::vector<std::uint32_t> indices;
    Vector3 boundsMin;
    Vector3 boundsMax;
};

// Fan-triangulates each (convex) polygon; vertices are duplicated per face so
// every triangle carries its polygon's flat normal and color.
TriMesh triangulate(const CsgMesh& mesh);

// Appends a cached TriMesh into the scene's packed streams at a world offset,
// multiplying vertex colors by colorMul (used for the brown-out tint) and
// expanding the scene bounds. The caller owns draw-range bookkeeping.
void appendTriMesh(const TriMesh& mesh, const Vector3& offset, const Color3& colorMul,
                   odai::importer::ImportedScene& scene);

// Same, but first rotates the mesh by quarterTurns * 90 degrees about the
// vertical axis through `pivot` (in the mesh's local space). Lets one cached
// building face whichever street its lot fronts. quarterTurns 0..3; exact
// axis-aligned rotation, so plane snapping is preserved.
void appendTriMeshRotated(const TriMesh& mesh, const Vector3& offset, int quarterTurns,
                          const Vector3& pivot, const Color3& colorMul,
                          odai::importer::ImportedScene& scene);

}  // namespace odai::procgen
