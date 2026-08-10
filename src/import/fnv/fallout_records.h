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

// Distant-landscape LOD layout, MEASURED from retail archives with
// `odai_newvegas_probe --find`, not taken from documentation.
//
// There are TWO separate LOD sets under meshes\landscape\lod\<ws>\, and an
// earlier reading of this directory conflated them. They are:
//
//   TERRAIN LOD -- a four-tier pyramid, all four tiers directly under the
//   worldspace directory, no subdirectory anywhere:
//
//     meshes\landscape\lod\<ws>\<ws>.level<N>.x<X>.y<Y>.nif   N = 4, 8, 16, 32
//
//   WastelandNV counts are exactly 1024 / 256 / 64 / 16 -- a clean 4x per
//   step, because each tier's tile covers 4x the area of the one below.
//
//   OBJECT LOD -- merged distant buildings, level4 ONLY, under "blocks\":
//
//     meshes\landscape\lod\<ws>\blocks\<ws>.level4.x<X>.y<Y>.nif
//
//   301 tiles for WastelandNV, and nothing at all for level8/16/32.
//
// The mistake worth not repeating: reading "blocks\" as the level4 TERRAIN
// tier. Both locations answer to <ws>.level4.x<X>.y<Y>.nif and both parse into
// geometry, so the substitution is silent. Tell them apart by their texture --
// a terrain tile names a per-tile diffuse and spans its cells exactly:
//
//   ...\wastelandnv.level4.x24.y-12.nif   ->  Diffuse\WastelandNV.n.Level4.X24.Y-12.dds
//                                             uv [0,1]x[0,1], bounds = cells 24..28, -12..-8
//   ...\blocks\wastelandnv.level4.x24.y-12.nif -> Blocks\WastelandNV.Buildings.dds
//                                             uv [0.75,1]x[0.75,1]  (an atlas sub-rect)
//
// So terrain tiles have their OWN diffuse, one per tile, plus a matching
// per-tile normal map:
//
//   textures\landscape\lod\<ws>\diffuse\<ws>.n.level<N>.x<X>.y<Y>.dds
//   textures\landscape\lod\<ws>\normals\<ws>.n.level<N>.x<X>.y<Y>.dds
//
// It is the OBJECT tier that uses a shared per-worldspace atlas
// (textures\landscape\lod\<ws>\blocks\<ws>.buildings.dds), which is why its
// UVs address a sub-rect.
//
// The number in the name is the tile's width in CELLS, and the coordinate is
// the grid coordinate of the tile's corner cell, stepping by that same number.
// Both grids are sparse: tiles exist only where there is something to draw.
//
// Each tile NIF is a BSMultiBoundNode wrapping BSSegmentedTriShapes (see
// nif_scene.cc -- that block type had to be added before any of these parsed)
// whose vertices are already in WORLD units, unlike static models. Some tiles
// carry junk shapes with no UVs and no texture alongside the real one --
// level32.x0.y-32 has two, 89 verts each. They must be dropped: a part with no
// texture index shades white in imported_static.frag.slang, i.e. a white slab
// across the horizon.
//
// Texture paths inside these NIFs are rooted at the Data directory
// ("Data\Textures\Landscape\LOD\..."), which normalizeTexturePath has to strip.
constexpr int kLandLodTierCellCounts[4] = {4, 8, 16, 32};
constexpr int kLandLodBlockCells = 4;  // the finest tier
constexpr float kLandLodBlockSize = kExteriorCellSize * static_cast<float>(kLandLodBlockCells);

// The tile containing a cell, in the coordinates the LOD file names use.
// Floors toward negative infinity: at tier 4, cell -1 belongs to tile -4, not
// tile 0. Truncating instead would make the tiles straddling zero twice as wide
// as every other one and silently map cells to the wrong distant tile.
constexpr std::int32_t landLodTileOrigin(std::int32_t cellCoord, std::int32_t tierCells) {
    const std::int32_t tile = (cellCoord >= 0)
        ? (cellCoord / tierCells)
        : ((cellCoord - (tierCells - 1)) / tierCells);
    return tile * tierCells;
}

// The width of one tile of the given tier, in world units.
constexpr float landLodTileSize(std::int32_t tierCells) {
    return kExteriorCellSize * static_cast<float>(tierCells);
}

constexpr std::int32_t landLodBlockOrigin(std::int32_t cellCoord) {
    return landLodTileOrigin(cellCoord, kLandLodBlockCells);
}

// Which of the two LOD sets documented above a tile path names. This is an
// explicit choice and NOT derivable from the tier number, which is the mistake
// worth stopping: the object set exists only at level4, so "tier == 4 means
// blocks\" reads as a rule when it is really a coincidence of what shipped. Ask
// for terrain level4 under that rule and you silently get buildings.
enum class LandLodSet {
    Terrain,  // <ws>\<ws>.level<N>.x<X>.y<Y>.nif   -- tiers 4, 8, 16, 32
    Objects,  // <ws>\blocks\<ws>.level4.x<X>.y<Y>.nif -- tier 4 only
};

// Whether the given set actually ships tiles at the given tier. Only the
// terrain set is a full pyramid; asking the object set for anything coarser
// than level4 resolves nothing at all, which is indistinguishable from a sparse
// grid unless it is rejected up front.
constexpr bool landLodTierExists(LandLodSet set, std::int32_t tierCells) {
    if (set == LandLodSet::Objects) {
        return tierCells == kLandLodBlockCells;
    }
    return tierCells == 4 || tierCells == 8 || tierCells == 16 || tierCells == 32;
}

// The archive-relative mesh path of one LOD tile, in the form resolveMesh
// takes: lowercase, backslash-separated, rooted at meshes\.
//
// Tile coordinates are the tile's CORNER cell and must be a multiple of
// tierCells -- pass them through landLodTileOrigin first. Building the name
// from an arbitrary cell coordinate produces a path that resolves to nothing,
// which looks exactly like the sparse-grid hole it is not.
inline std::string landLodTilePath(
    const std::string& loweredWorldspace, LandLodSet set, std::int32_t tierCells,
    std::int32_t tileX, std::int32_t tileY) {
    std::string path = "landscape\\lod\\";
    path += loweredWorldspace;
    path += (set == LandLodSet::Objects) ? "\\blocks\\" : "\\";
    path += loweredWorldspace;
    path += ".level";
    path += std::to_string(tierCells);
    path += ".x";
    path += std::to_string(tileX);
    path += ".y";
    path += std::to_string(tileY);
    path += ".nif";
    return path;
}

// One quadrant's alpha map is a 17x17 grid of posts: a cell's 33x33 posts split
// into 2x2 quadrants that share their middle row and column, so 16 gaps + 1.
// VTXT addresses these directly, and because the cooked terrain mesh keeps one
// vertex per post the alpha data maps onto vertices with no resampling.
constexpr int kLandQuadrantGridSize = (kLandGridSize / 2) + 1;  // 17
constexpr int kLandQuadrantVertexCount = kLandQuadrantGridSize * kLandQuadrantGridSize;  // 289

struct FalloutStaticRecord {
    std::uint32_t formId = 0;
    // The record type this came from: STAT, MSTT, ACTI, DOOR, ... Kept so a
    // caller can tell which kinds of base record a cell's geometry came from,
    // which is the only way to attribute a rendering artifact back to a type.
    std::string recordType;
    std::string editorId;
    std::string modelPath;  // relative to Data\Meshes, backslashes as stored in the plugin
};

// A LIGH base record's light parameters.
//
// LIGH.DATA is 32 bytes in every one of the 501 records in FalloutNV.esm (the
// FO3/FNV split on whether falloff and FOV are present does NOT bite here).
// Layout read off the file with `odai_newvegas_probe --record FalloutNV.esm
// LIGH`, then cross-checked against editor IDs rather than assumed:
//
//   @0   i32  time              -1 everywhere (unlimited)
//   @4   u32  radius            world units; range 0..3000, median 200
//   @8   u8[4] colour           R, G, B, unused
//   @12  u32  flags             see below
//   @16  f32  falloffExponent   1.0 in 473 of 501
//   @20  f32  FOV               90 in 499 of 501
//   @24  u32  value
//   @28  f32  weight
//
// Colour byte order is red-first, established semantically: "LightFungusStalk
// YellowAMB" reads (218,217,165), a pale yellow, and both "NVTopsWarm*" lights
// read warm oranges ((240,148,79) and (241,167,131)). Reading it the other way
// round makes the yellow light blue. The fourth byte is 0 in all 501 records,
// which is what rules out an RGBA misread rather than an assumption about it.
//
// The flag census is the useful part: every value across all 501 records is one
// of 0 (419), flicker 0x08 (50), flickerSlow 0x40 (12), pulseSlow 0x100 (11),
// pulse 0x80 (5), dynamic 0x01 (4). There is NOT ONE spotlight (0x200),
// negative light (0x004) or off-by-default (0x020) in the base game, so this
// importer does not need to handle, approximate, or skip those kinds -- the
// flags that do occur are all animation, and a steady light is the correct
// still-frame of every one of them.
struct FalloutLightRecord {
    std::uint32_t formId = 0;
    std::string editorId;
    std::string modelPath;   // MODL, present on only 29 of 501 -- most lights have no visible mesh
    float color[3] = {1.0f, 1.0f, 1.0f};  // 0..1, decoded from the RGB bytes
    float radius = 0.0f;                  // world units
    float falloffExponent = 1.0f;
    // FNAM, a 4-byte float present on all 501 records: the GECK's per-light
    // "fade value", i.e. a brightness multiplier. Observed 0.75 / 0.9 / 1.2, so
    // it straddles 1.0 and is a real authored quantity rather than a constant.
    // This is the closest thing FNV has to an intensity, which otherwise has no
    // physical unit anywhere in the format.
    float fadeValue = 1.0f;
    std::uint32_t flags = 0u;
};

// A REGN (region) record: the named areas Fallout announces when you walk into
// them ("Mojave Outpost", "Quarry Junction", "Nipton").
//
// A region carries a lot this importer does not want -- weather tables, sound
// tables, grass/object scattering, map colours. Only the identity is read,
// because the one thing being built on it is discovery notification.
//
// The name to SHOW is RDMP, not EDID. Measured over FalloutNV.esm's 276 REGN
// records: all 276 carry an EDID, but only 55 carry an RDMP. That is not a gap
// in the data, it is the distinction itself -- RDMP is the "map name", and a
// region without one is deliberately not surfaced to the player (audio-only
// ambience zones, weather regions, encounter zones). So a region with no RDMP
// must be treated as undiscoverable rather than falling back to its editor ID,
// which would announce "NVDefaultRegion" at the player.
struct FalloutRegionRecord {
    std::uint32_t formId = 0;
    std::string editorId;
    // RDMP. Empty for the 221 regions that are not player-facing.
    std::string mapName;

    [[nodiscard]] bool isDiscoverable() const { return !mapName.empty(); }
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

// One navmesh triangle: three vertex indices, then the index of the triangle
// sharing each edge (kNavMeshNoNeighbour where the edge is a border). The
// adjacency IS the pathfinding graph -- no neighbour search is needed, which is
// the whole reason to use the authored navmesh rather than deriving one.
constexpr std::uint16_t kNavMeshNoNeighbour = 0xffffu;

struct FalloutNavMeshTriangle {
    std::uint16_t vertex[3] = {};
    std::uint16_t neighbour[3] = {kNavMeshNoNeighbour, kNavMeshNoNeighbour, kNavMeshNoNeighbour};
    std::uint16_t flags = 0;
    std::uint16_t coverFlags = 0;
};

// A door this navmesh connects to, by the door reference's formID. Pairs with
// the teleport doors the cooker already emits: this is which triangle an NPC
// must stand on to use one.
struct FalloutNavMeshDoorPortal {
    std::uint32_t doorRefFormId = 0;
    std::uint16_t triangleIndex = 0;
};

// NAVM: Fallout's authored navigation mesh for one cell.
//
// The layout below was derived from real records rather than documentation --
// every format in this importer that was reasoned out instead of measured has
// been wrong at least once. DATA's second and third words are the vertex and
// triangle counts, and the check is exact: NVVX is always 12 bytes per vertex
// and NVTR always 16 per triangle, verified against records of 687/746 (8244
// and 11936 bytes) and 28/26 (336 and 416).
struct FalloutNavMeshRecord {
    std::uint32_t formId = 0;
    std::uint32_t cellFormId = 0;
    // Vertices in Bethesda space (Z-up), like every other position this reader
    // produces. Converting is the consumer's job.
    std::vector<float> vertices;  // xyz per vertex
    std::vector<FalloutNavMeshTriangle> triangles;
    std::vector<FalloutNavMeshDoorPortal> doorPortals;
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
    // XCLR: the regions this cell belongs to, by REGN formID. A cell can be in
    // several at once (measured: up to 6), and 4363 of FalloutNV.esm's 30497
    // cells carry the subrecord at all -- so "no regions" is the common case
    // and not an error.
    std::vector<std::uint32_t> regionFormIds;
    // Heap-allocated rather than stored inline. FalloutLandRecord is ~17 KB
    // (1089 heights + 3267 normal components), and holding it by value made
    // every FalloutCellRecord 17.5 KB whether or not it had terrain — 510 MB
    // across FalloutNV.esm's 30497 cells, including the 388 interiors that
    // have no LAND at all. It also made `cells` vector growth memcpy 17.5 KB
    // per element instead of moving a pointer. Null when the cell has no LAND.
    std::unique_ptr<FalloutLandRecord> land;
    // NAVM records for this cell. Usually one, but a cell can carry several.
    std::vector<FalloutNavMeshRecord> navMeshes;
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
    std::vector<FalloutRegionRecord> regions;  // REGN, for discovery notification
    std::vector<FalloutLightRecord> lights;  // LIGH base records, placed by REFR like any other base
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

// Where one cell's records live in the plugin, so it can be materialized later
// without re-reading the file.
//
// This is the streaming index. A full extractFalloutScene pass over
// FalloutNV.esm materializes every cell's contents and costs ~750 MB of heap;
// the Mojave has 16,397 exterior cells and no machine holds them all. At 48
// bytes each this index is under 800 KB for the entire worldspace, and because
// EsmReader memory-maps the plugin, going from an entry to that cell's actual
// LAND/REFR/NAVM records is a pointer walk plus that one cell's decompression.
struct FalloutCellIndexEntry {
    std::uint32_t cellFormId = 0;
    // EDID, when the cell has one. Interiors are named ("GSDocMitchellHouse");
    // most exterior cells are not. This is what lets a caller ask for a place by
    // name instead of by grid coordinate.
    std::string editorId;
    std::uint32_t worldspaceFormId = 0;  // 0 for interior cells
    std::int32_t gridX = 0;
    std::int32_t gridZ = 0;
    bool hasGridCoords = false;
    bool isInterior = false;
    // XCLR, carried through from the cell header so region lookup costs the
    // streamer nothing at runtime -- the index pass already walks these
    // subrecords for EDID and XCLC.
    std::vector<std::uint32_t> regionFormIds;
    // Byte offset of the CELL record's own header.
    std::uint64_t cellRecordOffset = 0;
    // The cell-children GRUP that holds this cell's REFR/LAND/NAVM records.
    // Zero size means the cell has no children group at all (no contents).
    std::uint64_t childrenGroupOffset = 0;
    std::uint32_t childrenGroupSize = 0;
};

struct FalloutCellIndex {
    std::vector<FalloutCellIndexEntry> cells;
    std::vector<FalloutWorldspaceRecord> worldspaces;
    // Every placed reference's owning cell, by the reference's own formID --
    // built from record headers alone, exactly as extractFalloutScene does.
    // Doors need this to resolve an XTEL target to the cell it stands in.
    std::unordered_map<std::uint32_t, std::size_t> cellIndexByReferenceFormId;
};

// One pass that records where every cell's records are without materializing
// any of them. Reads record headers and group headers only: no LAND
// decompression, no subrecord walking except for the CELL records themselves
// (needed for EDID and the XCLC grid coordinates the streamer ranks by).
bool buildFalloutCellIndex(
    const std::filesystem::path& esmPath, FalloutCellIndex& outIndex, std::string& outError);

// Materializes exactly one cell from an already-open reader, using an entry
// from buildFalloutCellIndex. The reader must be open on the same plugin the
// index was built from -- offsets are file positions, and there is no check
// that they belong to this file.
//
// outCell is fully replaced. Returns false only on a malformed walk; a cell
// with no contents succeeds and yields an empty cell.
bool extractFalloutCellAt(
    EsmReader& reader,
    const FalloutCellIndexEntry& entry,
    FalloutCellRecord& outCell,
    std::string& outError);

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
