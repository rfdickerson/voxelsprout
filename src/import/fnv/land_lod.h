#pragma once

// Distant-landscape LOD tiles, built into an ImportedScene.
//
// The tile GRID -- naming, tiers, which sets exist -- is documented on
// fallout_records.h beside landLodTilePath(). This header is only about turning
// those tiles into geometry, and exists so the offline cooker and the runtime
// streamer build them the SAME way. They previously could not: the cooker owned
// the only copy, inside a function that also parsed argv and wrote a file.
//
// Resolution is passed in as callbacks rather than as an asset-source type,
// because the two callers hold different ones (the cooker's AssetResolver, the
// runtime's FalloutAssetSource) and neither is worth teaching to the other.

#include "import/fnv/fallout_records.h"
#include "import/imported_scene.h"

#include <cstdint>
#include <functional>
#include <string>

namespace odai::importer::fnv {

struct LandLodTierStats {
    std::size_t tilesResolved = 0;  // found in the archives
    std::size_t tilesParsed = 0;    // ...and produced geometry
    std::size_t tilesMissing = 0;   // absent, which is NORMAL: the grid is sparse
    std::size_t triangles = 0;
    std::size_t textures = 0;
};

// Byte suppliers for one mesh or texture path, in the archive-relative form
// resolveMesh/resolveTexture take. Returning false means "not present", which
// for a tile is an ordinary sparse-grid hole and not an error.
using LandLodByteResolver =
    std::function<bool(const std::string& path, std::vector<std::uint8_t>& outBytes)>;

// Appends every tile of one tier covering the inclusive CELL range into `out`.
//
// `sinkUnits` lowers every vertex by that many world units. It is not cosmetic:
// distant LOD is a coarse resampling of the same ground the streamed cells
// render at full detail, so where the two overlap they interpenetrate and
// z-fight. Sinking the LOD guarantees the detailed terrain wins wherever it
// exists, which is what lets the caller skip any exclusion logic and simply
// draw both. 0 disables it.
//
// Tile coordinates are derived internally; pass raw cell bounds.
bool appendLandLodTier(
    const LandLodByteResolver& resolveMesh,
    const LandLodByteResolver& resolveTexture,
    const std::string& worldspaceEditorId,
    LandLodSet set,
    std::int32_t tierCells,
    std::int32_t cellX0, std::int32_t cellZ0, std::int32_t cellX1, std::int32_t cellZ1,
    float sinkUnits,
    ImportedScene& out,
    LandLodTierStats& outStats,
    std::string& outError);

}  // namespace odai::importer::fnv
