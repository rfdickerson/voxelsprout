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
#include "import/fnv/plugin_load_order.h"

namespace odai::importer::fnv {

inline constexpr std::uint16_t kCellFlagInterior = 0x0001u;
inline constexpr std::uint16_t kCellFlagShowSky = 0x0040u;
inline constexpr std::uint16_t kCellFlagUseSkyLighting = 0x0080u;

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
// `odai_bethesda_probe --find`, not taken from documentation.
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
    // Skyrim's generated terrain uses the same 4/8/16/32-cell pyramid, but
    // moved beside its object LOD and changed the extension. Unlike BTO object
    // vertices, BTR terrain vertices are local to the named tile.
    SkyrimTerrain,  // terrain\<ws>\<ws>.<N>.<X>.<Y>.btr
    // Skyrim moved generated object LOD to a different tree and extension.
    // The payload is still a NIF on the same four-cell tile lattice.
    SkyrimObjects,  // terrain\<ws>\objects\<ws>.4.<X>.<Y>.bto
};

// Whether the given set actually ships tiles at the given tier. Only the
// terrain set is a full pyramid; asking the object set for anything coarser
// than level4 resolves nothing at all, which is indistinguishable from a sparse
// grid unless it is rejected up front.
constexpr bool landLodTierExists(LandLodSet set, std::int32_t tierCells) {
    if (set == LandLodSet::Objects || set == LandLodSet::SkyrimObjects) {
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
    if (set == LandLodSet::SkyrimObjects || set == LandLodSet::SkyrimTerrain) {
        std::string path = "terrain\\";
        path += loweredWorldspace;
        path += set == LandLodSet::SkyrimObjects ? "\\objects\\" : "\\";
        path += loweredWorldspace;
        path += "." + std::to_string(tierCells);
        path += "." + std::to_string(tileX);
        path += "." + std::to_string(tileY);
        path += set == LandLodSet::SkyrimObjects ? ".bto" : ".btr";
        return path;
    }
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
    // TES4 TREE metadata. SPT paths are rooted at Data rather than under
    // Data\Meshes; these fields remain empty for every other record type.
    std::string treeLeafTexturePath;
    std::uint32_t treeSeed = 0u;
    float treeBillboardWidth = 0.0f;
    float treeBillboardHeight = 0.0f;
    float treeWind[8] = {};
};

// A LIGH base record's light parameters.
//
// LIGH.DATA is 32 bytes in every one of the 501 records in FalloutNV.esm (the
// FO3/FNV split on whether falloff and FOV are present does NOT bite here).
// Layout read off the file with `odai_bethesda_probe --record FalloutNV.esm
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
    //
    // ON A LOCALIZED PLUGIN THIS IS NOT THE NAME. Skyrim stores RDMP as a
    // four-byte string ID (see strings_table.h) and reading it as a zstring
    // yields its low byte as a character -- Whiterun's region announces "h".
    // Both fields are filled unconditionally because the parser does not see
    // the TES4 header; whoever has the plugin's localized flag decides which to
    // believe. FalloutWorldTables carries both for the same reason.
    std::string mapName;
    // RDMP read as a string ID. Zero unless RDMP was exactly four bytes, which
    // is what a localized plugin always writes and what a real map name never
    // is (the shortest in FalloutNV.esm is "Goodsprings").
    std::uint32_t mapNameStringId = 0;
    std::uint32_t worldspaceFormId = 0;
    bool deleted = false;

    struct Polygon {
        // Bethesda/plugin space: X/Y ground plane, pairs in authored order.
        std::vector<float> points;
    };
    struct Sound {
        std::uint32_t descriptorFormId = 0;
        std::uint32_t weatherFlags = 0;
        float chance = 0.0f;
    };
    std::vector<Polygon> polygons;
    std::vector<Sound> sounds;

    [[nodiscard]] bool isDiscoverable() const {
        return !mapName.empty() || mapNameStringId != 0u;
    }
};

struct FalloutSoundOutputModelRecord {
    std::uint32_t formId = 0;
    float minDistance = 150.0f;
    float maxDistance = 4000.0f;
    bool deleted = false;
};

struct FalloutSoundDescriptorRecord {
    std::uint32_t formId = 0;
    std::string editorId;
    std::vector<std::string> filePaths;
    std::uint32_t outputModelFormId = 0;
    std::uint32_t flags = 0;
    bool looping = false;
    bool deleted = false;
};

struct FalloutSoundBaseRecord {
    std::uint32_t formId = 0;
    std::uint32_t descriptorFormId = 0;
    bool deleted = false;
};

struct FalloutSoundEmitterRecord {
    std::uint32_t referenceFormId = 0;
    std::uint32_t descriptorFormId = 0;
    float position[3] = {};  // engine space, Y-up
};

struct FalloutPlacedReference {
    std::uint32_t formId = 0;
    std::uint32_t baseFormId = 0;  // NAME: the STAT (or other base record) this instance places
    // Winning contribution that supplied this placed record. VMAD object
    // properties are expressed in that plugin's local master-index space.
    std::size_t sourcePluginIndex = 0u;
    // MORROWIND NAMES ITS BASE BY STRING. TES3 has no formIDs at all, so a
    // reference's NAME subrecord is the base record's own id text. Empty for
    // every later generation; when it is set, baseFormId is 0 and the caller
    // resolves the name through FalloutWorldTables::baseFormIdsByEditorId.
    std::string baseEditorId;
    // The record header's own flags. Bit 0x0800 is "Initially Disabled": the
    // game does not render the reference until something enables it.
    std::uint32_t recordFlags = 0;
    // TES4 uses the record-header Deleted flag; TES3 uses a DELE subrecord
    // inside the FRMR block. Merged extraction retains the tombstone until all
    // contributions have been applied, then removes the placement.
    bool isDeleted = false;
    // XESP: enabled state follows another reference's, optionally inverted.
    bool hasEnableParent = false;
    std::uint32_t enableParentFormId = 0;
    bool enableParentOpposite = false;
    float position[3] = {};        // DATA, world units
    float rotationRadians[3] = {};  // DATA
    float scale = 1.0f;             // XSCL, defaults to 1 when absent
    // TES5 compiled script attachments. Gameplay adapters decode this outside
    // ImportedScene; retaining bytes avoids coupling import to Papyrus types.
    std::vector<std::uint8_t> vmadBytes;
    // XTEL: this reference is a teleport door. The target is another REFR (the
    // door on the far side), and the position/rotation are where the player
    // arrives -- expressed in the TARGET cell's space, not this one. Resolving
    // which cell that is means looking the target reference up globally, which
    // is what FalloutSceneData::cellIndexByReferenceFormId exists for.
    bool hasTeleport = false;
    std::uint32_t teleportTargetRefFormId = 0;
    // TES3 DODT/DNAM names the destination CELL directly instead of pointing
    // at a paired door reference. Empty DNAM means an exterior destination,
    // selected from the DODT world position.
    std::string teleportTargetCellEditorId;
    float teleportPosition[3] = {};
    float teleportRotationRadians[3] = {};
    // XLOC. Exploration mode exposes the authored lock but may deliberately
    // bypass it until keys and lockpicking exist.
    bool isLocked = false;
    std::uint8_t lockLevel = 0u;
    // XMRK plus its following marker fields. Skyrim stores FULL as a localized
    // string ID; keeping both representations lets the streamer's load-order
    // aware string-table pass resolve the correct source plugin.
    bool isMapMarker = false;
    std::string mapMarkerName;
    std::uint32_t mapMarkerNameStringId = 0u;
    std::uint8_t mapMarkerFlags = 0u;
    std::uint16_t mapMarkerType = 0u;
};

struct FalloutMapMarkerRecord {
    std::uint32_t referenceFormId = 0u;
    std::uint32_t cellFormId = 0u;
    std::uint32_t worldspaceFormId = 0u;
    std::string name;
    std::uint32_t nameStringId = 0u;
    float position[3] = {};
    std::uint8_t flags = 0u;
    std::uint16_t type = 0u;
    bool initiallyDisabled = false;
    bool deleted = false;
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
    // The TES3 plugin whose LTEX palette the raw VTEX indices address. Those
    // indices are local to the source file and cannot be remapped as formIDs.
    std::size_t sourcePluginIndex = 0;
    // Posts per side. 33 for TES4 onward, 65 for Morrowind -- and the arrays
    // below are sized from it rather than fixed, because at 65 they are four
    // times larger and a fixed worst case would cost every Fallout cell the
    // Morrowind price. The world a cell covers is (gridSize - 1) *
    // kLandPostSpacing, i.e. 4096 units at 33 posts and 8192 at 65: the SPACING
    // is the same 128 units in both, it is the cell that is bigger.
    int gridSize = kLandGridSize;
    [[nodiscard]] int vertexCount() const { return gridSize * gridSize; }
    [[nodiscard]] float cellWorldSize() const {
        return kLandPostSpacing * static_cast<float>(gridSize - 1);
    }
    bool hasHeights = false;
    std::vector<float> heights;   // row-major [row * gridSize + col], world Z units
    bool hasNormals = false;
    std::vector<float> normals;   // row-major, normalized
    // VCLR: per-post terrain tint, row-major RGB in [0,1]. This is the colour
    // the game actually shades landscape with -- baked ambient and the regional
    // palette that makes the Mojave read as sunbleached tan rather than the
    // generic ramp a height-based fallback produces. Absent on many cells, in
    // which case the neutral 1,1,1 default leaves the diffuse texture unmodified,
    // which is exactly what the game does too.
    bool hasColors = false;
    std::vector<float> colors;
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

    // MORROWIND HAS NO QUADRANTS AND NO LAYERS. Its VTEX is a 16x16 grid of
    // land-texture indices over the whole cell, each covering 4x4 quads, so the
    // terrain is textured by splatting whole blocks rather than by blending
    // per-post opacities. Empty for every other game.
    //
    // Values are the LTEX index PLUS ONE, exactly as stored: 0 means "the
    // worldspace default texture", which is why this is not simply an index.
    std::vector<std::uint16_t> morrowindTextureGrid;
};

// Side of Morrowind's VTEX grid, and how many terrain quads one of its entries
// covers. 16 * 4 = 64 quads = the 65-post grid.
// Morrowind's LAND grid: 65 posts a side over an 8192-unit cell. The post
// SPACING is the same 128 units every later game uses -- the cell is four times
// the area, not four times the sampling.
constexpr int kMorrowindLandGridSize = 65;
constexpr int kMorrowindTextureGridSize = 16;
constexpr int kMorrowindTextureBlockQuads = 4;

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

// NAVM: Bethesda's authored navigation mesh for one cell. Fallout/Oblivion use
// DATA + NVVX + NVTR subrecords; Skyrim stores the same vertex/triangle core in
// one packed NVNM subrecord. Both decode into this common representation.
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
    // Complete CELL DATA flags. Skyrim interiors use Show Sky (0x40) and Use
    // Sky Lighting (0x80) independently; collapsing DATA to isInterior loses
    // the distinction that decides both the background and ambient policy.
    std::uint16_t cellFlags = 0u;
    bool hasGridCoords = false;
    std::int32_t gridX = 0;
    std::int32_t gridZ = 0;
    std::uint32_t worldspaceFormId = 0;  // 0 for interior cells
    // TES5 XLCN: authored LCTN owning this cell. Kept out of ImportedScene;
    // gameplay consumes it from the streamed record index.
    std::uint32_t locationFormId = 0;

    // XCLL: how an INTERIOR is lit. An interior has no sun and, in Fallout's
    // own data, usually no LIGH placements either -- Doc Mitchell's house has
    // none at all -- so this subrecord is the whole lighting rig for the room,
    // and a reader that skips it renders the interior pitch black or, worse,
    // lets the exterior sun through the walls.
    //
    // 40 bytes in FO3/FNV: ambient RGBA, directional RGBA, fog-near RGBA, then
    // fog near/far as floats. Measured on GSDocMitchellHouse: ambient
    // (47,70,69), directional (0,0,0) -- black, i.e. the room is ambient-only --
    // fog (77,62,32), near 100, far 1500.
    bool hasLighting = false;
    float ambientColor[3] = {};
    float directionalColor[3] = {};
    float fogColor[3] = {};
    float fogNear = 0.0f;
    float fogFar = 0.0f;

    // XCLW: the height of this cell's water surface, in Bethesda Z.
    //
    // A cell with no water does NOT omit the subrecord in Fallout -- all 30497
    // of FalloutNV.esm's cells carry one, and a dry cell writes the sentinel
    // 0xCF000000, which is -2^31 as a float. So "has water" is a value test,
    // not a presence test, and a reader that trusts presence floods the whole
    // Mojave two billion units below the ground.
    //
    // Oblivion is the other way round: only 751 of Oblivion.esm's 35494 cells
    // carry XCLW at all, and no WRLD record in the file has a DNAM (censused:
    // 84 worldspaces, 0 DNAM), so there is no authored per-worldspace default
    // to fall back to. Tamriel's sea is simply at Z=0, which is what the
    // absent case resolves to.
    bool hasWater = false;
    float waterHeight = 0.0f;

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
// formID. How it reaches a .dds from there depends on the generation:
//   Fallout 3 / New Vegas: LAND.BTXT -> LTEX -> LTEX.TNAM -> TXST -> TXST.TX00
//   Oblivion:              LAND.BTXT -> LTEX -> LTEX.ICON  (a path, no TXST)
// Both land in diffuseTexturePath, so nothing downstream has to know which.
struct FalloutLandTextureRecord {
    std::uint32_t formId = 0;
    std::string editorId;
    std::uint32_t textureSetFormId = 0;  // TNAM -> TXST; 0 on Oblivion
    // From that TXST's TX00, or from Oblivion's own ICON with the
    // "landscape\" folder it is relative to already prepended.
    std::string diffuseTexturePath;
};

struct FalloutWorldspaceRecord {
    std::uint32_t formId = 0;
    std::string editorId;
    // DNAM: the height the ground sits at in any cell of this worldspace that
    // carries NO LAND record, and the height its water sits at when a cell
    // states none.
    //
    // A cell without LAND is not a hole in the world -- it is FLAT GROUND at
    // this height, which is how Bethesda avoids authoring a heightfield for a
    // region that is entirely covered by architecture. Megaton is the case that
    // makes it matter: cell (-2,-7) places 107 references and has no LAND, so an
    // importer that draws nothing there hangs a third of the town over open sky.
    //
    // Present on 28 of Fallout 3's 32 worldspaces and absent from every Oblivion
    // one (censused: 84 worldspaces, 0 DNAM), which is why this is optional
    // rather than assumed.
    bool hasDefaultHeights = false;
    float defaultLandHeight = 0.0f;
    float defaultWaterHeight = 0.0f;

    // WNAM: the worldspace this one hangs off. A WALLED CITY INHERITS NEARLY
    // EVERYTHING FROM ITS PARENT, and Skyrim leans on that far harder than the
    // earlier games do. WhiterunWorld's whole record is an EDID and this one
    // field: no CNAM, so it names no climate and nothing publishes a sky or a
    // cloud layer for it, and no DNAM, so its implied water height falls back to
    // ZERO -- which for a city standing at engine y -3120 is a full-cell water
    // quad slicing through the houses. Tamriel, the parent, declares both
    // (climate 0x812, default water -14000).
    //
    // So resolving this is not a nicety: unresolved, a Skyrim city renders with
    // no sky and underwater. 0 when the record names no parent.
    std::uint32_t parentWorldspaceFormId = 0;
};

// Everything extracted from one plugin pass. Populated by extractFalloutScene
// as a flat pass over the whole file — the caller is expected to filter down
// to the cells/worldspace it actually wants to cook.
struct FalloutSceneData {
    std::vector<FalloutStaticRecord> statics;
    std::vector<FalloutRegionRecord> regions;  // REGN, for discovery notification
    std::vector<FalloutSoundOutputModelRecord> soundOutputModels;
    std::vector<FalloutSoundDescriptorRecord> soundDescriptors;
    std::vector<FalloutSoundBaseRecord> soundBases;
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
// One plugin's contribution to a cell: where its children group sits in ITS
// file. Offsets are positions in that plugin, so the plugin index is not
// decoration -- reading a range against the wrong file is undefined.
struct FalloutCellContribution {
    std::size_t pluginIndex = 0;
    std::uint64_t childrenGroupOffset = 0;
    std::uint32_t childrenGroupSize = 0;
    // TES3 LAND is a top-level sibling rather than part of the CELL range.
    // Later LAND contributions replace earlier terrain independently of CELL.
    std::uint64_t landRecordOffset = 0;
    std::uint32_t landRecordSize = 0;
};

struct FalloutCellIndexEntry {
    std::uint32_t cellFormId = 0;
    // XCLL, carried from the CELL record so extractFalloutCellAt can hand it
    // back: that function rebuilds a cell from this entry plus the children
    // GRUP and never re-reads the CELL's own subrecords, so anything living
    // only on the record is invisible to every streaming caller. See the same
    // fields on FalloutCellRecord for what they mean.
    bool hasLighting = false;
    float ambientColor[3] = {};
    float directionalColor[3] = {};
    float fogColor[3] = {};
    float fogNear = 0.0f;
    float fogFar = 0.0f;
    // XCLW, carried for the same reason as the lighting above. Until it was,
    // EVERY streamed cell reported no water -- rivers, lakes and the sea
    // existed only in cooked scenes, because the cooker parses the CELL record
    // in full while the streamer rebuilds it from this entry.
    bool hasWater = false;
    float waterHeight = 0.0f;
    // EDID, when the cell has one. Interiors are named ("GSDocMitchellHouse");
    // most exterior cells are not. This is what lets a caller ask for a place by
    // name instead of by grid coordinate.
    std::string editorId;
    std::uint32_t worldspaceFormId = 0;  // 0 for interior cells
    std::uint32_t locationFormId = 0;
    std::int32_t gridX = 0;
    std::int32_t gridZ = 0;
    bool hasGridCoords = false;
    bool isInterior = false;
    std::uint16_t cellFlags = 0u;
    // XCLR, carried through from the cell header so region lookup costs the
    // streamer nothing at runtime -- the index pass already walks these
    // subrecords for EDID and XCLC.
    std::vector<std::uint32_t> regionFormIds;
    // Byte offset of the CELL record's own header.
    std::uint64_t cellRecordOffset = 0;
    // Every plugin that has something to say about this cell, in load order.
    // A cell's contents are not owned by one file: an override plugin ships a
    // children group holding ONLY the references it changes or adds, and the
    // rest still come from the master. So the contents are the merge of these,
    // later plugins winning per reference formID -- see extractFalloutCellMerged.
    //
    // The single-plugin builder fills exactly one of these, so both paths read
    // the same way.
    std::vector<FalloutCellContribution> contributions;
    // The cell-children GRUP that holds this cell's REFR/LAND/NAVM records.
    // Zero size means the cell has no children group at all (no contents).
    std::uint64_t childrenGroupOffset = 0;
    std::uint32_t childrenGroupSize = 0;
    // MORROWIND KEEPS ITS TERRAIN IN A SIBLING RECORD. TES3 has no children
    // group at all -- a CELL carries its references inline and its LAND is a
    // separate top-level record joined by grid coordinate -- so the index has to
    // remember where that record was. Zero size means the cell has no terrain.
    std::uint64_t landRecordOffset = 0;
    std::uint32_t landRecordSize = 0;
};

struct FalloutCellIndex {
    // World units one exterior cell covers. 4096 from Oblivion onward, 8192 in
    // Morrowind -- the post spacing is 128 in both, the cell is four times the
    // area. Carried here because the streamer's residency grid is expressed in
    // cells and would otherwise load a quarter of the world it thinks it is.
    float cellWorldSize = kExteriorCellSize;
    std::vector<FalloutCellIndexEntry> cells;
    // Indexed by FalloutCellContribution::pluginIndex, so a contribution can be
    // read without carrying the load order alongside the index everywhere.
    std::vector<std::filesystem::path> pluginPaths;
    std::vector<FalloutWorldspaceRecord> worldspaces;
    // Every placed reference's owning cell, by the reference's own formID --
    // built from record headers alone, exactly as extractFalloutScene does.
    // Doors need this to resolve an XTEL target to the cell it stands in.
    std::unordered_map<std::uint32_t, std::size_t> cellIndexByReferenceFormId;
    // XMRK references, extracted during the same header/index pass. These are
    // tiny (397 in Skyrim.esm) and drive compass/location discovery without
    // retaining all 693k placed references.
    std::vector<FalloutMapMarkerRecord> mapMarkers;
};

// One pass that records where every cell's records are without materializing
// any of them. Reads record headers and group headers only: no LAND
// decompression, no subrecord walking except for the CELL records themselves
// (needed for EDID and the XCLC grid coordinates the streamer ranks by).
bool buildFalloutCellIndex(
    const std::filesystem::path& esmPath, FalloutCellIndex& outIndex, std::string& outError);

// As above, across a whole load order. TES4+ cells are merged by their remapped
// formID; TES3 exteriors merge by grid and interiors by case-insensitive name.
// The same logical cell described by several plugins becomes one entry with
// several contributions, and a later plugin's CELL record replaces the earlier
// one's metadata (editor ID, grid, regions). Every formID the index carries is
// in the order's global space.
bool buildFalloutCellIndex(
    const FalloutLoadOrder& order, FalloutCellIndex& outIndex, std::string& outError);

// Materializes one cell by merging every plugin that contributes to it, in load
// order. References are keyed by formID with the later plugin winning, which is
// how an override patch moves or deletes a placement; LAND and navmeshes are
// replaced wholesale by the last plugin that supplies one. All formIDs come out
// remapped into the order's global space, matching the world tables.
bool extractFalloutCellMerged(
    const FalloutCellIndex& index,
    const FalloutLoadOrder& order,
    const FalloutCellIndexEntry& entry,
    FalloutCellRecord& outCell,
    std::string& outError);

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
