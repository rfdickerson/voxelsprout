#pragma once

// Typed extraction of the record types needed to cook a Fallout: New Vegas
// exterior worldspace region or interior cell into an ImportedScene: TES4
// (plugin header/masters), STAT (static base model), CELL, WRLD, LAND
// (heightmap), and REFR (placed static instances).
//
// This intentionally covers a narrow slice of the ~700 Fallout record types —
// enough for terrain + static-mesh cooking, matching the scope of this
// project's (Windows-only, not in this repo) Morrowind balmora cooker. It is
// not a general-purpose ESM editor.
//
// Coordinate/scale notes carried over from public Gamebryo landscape
// documentation, NOT validated against a real .esm in this environment (no
// Fallout: New Vegas Data Files available here — see README for the local
// path convention once a user tests this against real data):
//   - Exterior cells are 4096 world units square.
//   - LAND VHGT posts form a 33x33 grid at 128-unit spacing (32 gaps * 128 = 4096).
//   - VHGT height deltas are accumulated in a fixed unit scale (kLandHeightScale);
//     this constant should be double-checked against a known in-game elevation
//     the first time this cooker runs against real assets.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "import/fnv/esm_reader.h"

namespace odai::importer::fnv {

// Height-post grid dimensions for one LAND record (one exterior cell).
constexpr int kLandGridSize = 33;
constexpr int kLandVertexCount = kLandGridSize * kLandGridSize;
constexpr float kLandPostSpacing = 128.0f;
// VHGT delta-byte -> world-unit height scale. See the file-level comment:
// this is a best-effort constant from public Gamebryo landscape format
// documentation, not verified against real Fallout: New Vegas data.
constexpr float kLandHeightScale = 8.0f;
// How many times a landscape texture repeats across one exterior cell. FNV
// tiles its base layer per quadrant, and a cell is 2x2 quadrants.
constexpr float kLandTextureTilesPerCell = 8.0f;
constexpr float kExteriorCellSize = kLandPostSpacing * static_cast<float>(kLandGridSize - 1);  // 4096

// One quadrant's alpha map is a 17x17 grid of posts: a cell's 33x33 posts split
// into 2x2 quadrants that share their middle row and column, so 16 gaps + 1.
// VTXT addresses these directly, and because the cooked terrain mesh keeps one
// vertex per post the alpha data maps onto vertices with no resampling.
constexpr int kLandQuadrantGridSize = (kLandGridSize / 2) + 1;  // 17
constexpr int kLandQuadrantVertexCount = kLandQuadrantGridSize * kLandQuadrantGridSize;  // 289

struct FalloutStaticRecord {
    std::uint32_t formId = 0;
    std::string editorId;
    std::string modelPath;  // relative to Data\Meshes, backslashes as stored in the plugin
};

struct FalloutPlacedReference {
    std::uint32_t formId = 0;
    std::uint32_t baseFormId = 0;  // NAME: the STAT (or other base record) this instance places
    float position[3] = {};        // DATA, world units
    float rotationRadians[3] = {};  // DATA
    float scale = 1.0f;             // XSCL, defaults to 1 when absent
    // XTEL: this reference is a teleport door. The target is another REFR (the
    // door on the far side), and the position/rotation are where the player
    // arrives -- expressed in the TARGET cell's space, not this one. Resolving
    // which cell that is means looking the target reference up globally, which
    // is what FalloutSceneData::cellIndexByReferenceFormId exists for.
    bool hasTeleport = false;
    std::uint32_t teleportTargetRefFormId = 0;
    float teleportPosition[3] = {};
    float teleportRotationRadians[3] = {};
};

// One ATXT/VTXT pair: a landscape texture blended over a quadrant's base, with
// a per-post opacity map. ATXT names the LTEX and the quadrant it covers; the
// VTXT that follows it lists (post, opacity) for the posts it touches.
struct FalloutLandTextureLayer {
    std::uint32_t textureFormId = 0;
    std::uint8_t quadrant = 0;    // 0=SW, 1=SE, 2=NW, 3=NE, same as BTXT
    std::uint16_t layerIndex = 0;  // ATXT's own stacking order within the quadrant
    // Opacity per quadrant post, row-major over kLandQuadrantGridSize. Zero
    // where VTXT said nothing, which is most of it for a typical layer.
    float opacity[kLandQuadrantVertexCount] = {};
};

struct FalloutLandRecord {
    std::uint32_t cellFormId = 0;
    bool hasHeights = false;
    float heights[kLandVertexCount] = {};   // row-major [row * 33 + col], world Z units
    bool hasNormals = false;
    float normals[kLandVertexCount * 3] = {};  // row-major, normalized
    // VCLR: per-post terrain tint, row-major RGB in [0,1]. This is the colour
    // the game actually shades landscape with -- baked ambient and the regional
    // palette that makes the Mojave read as sunbleached tan rather than the
    // generic ramp a height-based fallback produces. Absent on many cells, in
    // which case the neutral 1,1,1 default leaves the diffuse texture unmodified,
    // which is exactly what the game does too.
    bool hasColors = false;
    float colors[kLandVertexCount * 3] = {};
    // Base texture per quadrant (0=SW,1=SE... layout mirrors BTXT's own
    // quadrant index), 0 when the quadrant has no explicit BTXT record.
    std::uint32_t quadrantBaseTextureFormId[4] = {};
    // Additional texture layers blended over the base, from ATXT/VTXT pairs.
    // This is what carries roads, gravel and the transitions between ground
    // types; with only BTXT the terrain is one flat texture per quadrant with a
    // hard seam at every quadrant boundary.
    //
    // Ordered by the layer index ATXT declares, so blending them in vector order
    // reproduces the game's own stacking. Layers are sparse by nature: VTXT only
    // lists the posts where the layer is actually present, and everything else
    // stays at zero opacity.
    std::vector<FalloutLandTextureLayer> textureLayers;
};

struct FalloutCellRecord {
    std::uint32_t formId = 0;
    std::string editorId;
    bool isInterior = false;
    bool hasGridCoords = false;
    std::int32_t gridX = 0;
    std::int32_t gridZ = 0;
    std::uint32_t worldspaceFormId = 0;  // 0 for interior cells
    std::vector<FalloutPlacedReference> references;
    // Heap-allocated rather than stored inline. FalloutLandRecord is ~17 KB
    // (1089 heights + 3267 normal components), and holding it by value made
    // every FalloutCellRecord 17.5 KB whether or not it had terrain — 510 MB
    // across FalloutNV.esm's 30497 cells, including the 388 interiors that
    // have no LAND at all. It also made `cells` vector growth memcpy 17.5 KB
    // per element instead of moving a pointer. Null when the cell has no LAND.
    std::unique_ptr<FalloutLandRecord> land;
};

// A landscape texture. LAND's BTXT/ATXT subrecords name one of these by
// formID; it in turn points (TNAM) at a texture set whose first slot is the
// diffuse map. So the full chain from a terrain quadrant to a .dds is
// LAND.BTXT -> LTEX -> LTEX.TNAM -> TXST -> TXST.TX00.
struct FalloutLandTextureRecord {
    std::uint32_t formId = 0;
    std::string editorId;
    std::uint32_t textureSetFormId = 0;  // TNAM -> TXST
    std::string diffuseTexturePath;      // resolved from that TXST's TX00
};

struct FalloutWorldspaceRecord {
    std::uint32_t formId = 0;
    std::string editorId;
};

// Everything extracted from one plugin pass. Populated by extractFalloutScene
// as a flat pass over the whole file — the caller is expected to filter down
// to the cells/worldspace it actually wants to cook.
struct FalloutSceneData {
    std::vector<FalloutStaticRecord> statics;
    std::vector<FalloutLandTextureRecord> landTextures;  // LTEX, already resolved through TXST
    // Every placed reference's owning cell, by the reference's own formID. A
    // door's XTEL names its counterpart reference and nothing else -- the cell
    // it stands in is only discoverable by looking it up here.
    std::unordered_map<std::uint32_t, std::size_t> cellIndexByReferenceFormId;
    std::vector<FalloutWorldspaceRecord> worldspaces;
    std::vector<FalloutCellRecord> cells;
};

// Narrows what a pass actually materializes. A cook targeting one interior
// cell needs neither the other 30496 cells' references nor any of the 29363
// LAND records, and skipping them before they are decompressed is the
// difference between a ~1.6 s / 750 MB pass and a fraction of it.
struct FalloutExtractFilter {
    // Called once per cell, after its own header/EDID/grid fields are parsed
    // but before any of its LAND or REFR children are touched. Return false to
    // skip that cell's contents entirely; the cell itself is still recorded,
    // so editor-ID and grid lookups keep working. Null means "want everything".
    //
    // Worldspace-keyed predicates are safe here: a WRLD record is always
    // parsed before the cells of its world-children group.
    std::function<bool(const FalloutCellRecord&)> wantCellContents;

    // Called on entering a worldspace's world-children group, with that
    // worldspace's formID. Return false to skip the entire group — every cell,
    // reference and LAND record under it — without reading a byte of it.
    //
    // This is a much bigger saving than wantCellContents, because the walk
    // seeks past the whole group rather than scanning its record headers:
    // nearly all of FalloutNV.esm's 234 MB lives under world-children groups,
    // so a cook targeting an interior cell can skip past most of the file. Null
    // means "want every worldspace".
    std::function<bool(std::uint32_t worldspaceFormId)> wantWorldspace;
};

// Runs a single forward pass over the plugin, extracting TES4/STAT/WRLD/CELL/
// LAND/REFR records. Cell attribution for LAND/REFR records uses simple
// "most recently seen CELL record" tracking rather than fully validating the
// GRUP hierarchy — correct for well-formed retail plugins.
bool extractFalloutScene(const std::filesystem::path& esmPath, FalloutSceneData& outScene, std::string& outError);

// As above, but materializes only the cell contents `filter` asks for.
bool extractFalloutScene(
    const std::filesystem::path& esmPath,
    const FalloutExtractFilter& filter,
    FalloutSceneData& outScene,
    std::string& outError);

}  // namespace odai::importer::fnv
