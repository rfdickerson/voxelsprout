#pragma once

// Streams Fallout exterior cells in and out around the player.
//
// Wires together the pieces built for it:
//   * CellResidencyPlanner  -- what to load and evict (prediction, hysteresis,
//                              in-flight budget, wasted-work accounting)
//   * core::JobSystem       -- cell scenes are read and deserialized off the
//                              main thread
//   * Renderer chunk API    -- addImportedSceneChunk / removeImportedSceneChunk,
//                              which suballocate from the shared geometry arenas
//                              and reference count textures by source path, so a
//                              texture shared between cells uploads once
//
// MAIN THREAD ONLY, matching world::ChunkMeshScheduler's contract: update() is
// the single entry point, called once per frame. Only file reading and
// deserialization run on workers; every renderer call happens on the main thread.
//
// Source data is the GAME'S OWN FILES: FalloutNV.esm plus the BSA archives. No
// cooking step, and no cooked scenes on disk. A worker seeks straight to a
// cell's records through the offset index built at open() (~0.5 ms), builds its
// geometry with CellSceneBuilder -- the same code the cooker uses, verified to
// produce identical output -- and the main thread hands the result to the
// renderer.
//
// Each job gets its OWN EsmReader and CellSceneBuilder, because neither is
// thread safe: EsmReader records walk errors in members, and the builder owns
// mutable texture and mesh caches. Only FalloutAssetSource is shared, which is
// safe by construction. The cost of that isolation is that two cells wanting
// the same texture each decode it; the renderer still uploads it once, so the
// waste is CPU-side, and it is measured rather than assumed (see
// CellStreamerStats::worstBuildMs).

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "import/cell_residency_planner.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/cell_builder.h"
#include "import/fnv/plugin_load_order.h"
#include "import/fnv/decoded_texture_cache.h"
#include "import/fnv/fallout_records.h"

namespace odai {
namespace core {
class JobSystem;
}
namespace render {
class Renderer;
}
namespace importer::fnv {

struct CellStreamerStats {
    CellResidencyStats residency{};
    std::uint64_t scenesLoaded = 0;
    // Cells served from the on-disk cache rather than rebuilt.
    std::uint64_t cacheHits = 0;
    std::uint64_t cacheMisses = 0;
    std::uint64_t cacheWriteFailures = 0;
    float lastCacheLoadMs = 0.0f;
    std::uint64_t loadFailures = 0;
    // Worst and most recent time a worker spent turning one cell's records into
    // geometry. Dominated by DDS decode on a cold texture set.
    float worstBuildMs = 0.0f;
    float lastBuildMs = 0.0f;
    // Cells that produced no renderable geometry. Legitimate for empty desert,
    // but a large count means the per-cell cook is not producing what it should.
    std::uint64_t emptyScenes = 0;
    // Worst and most recent single-frame cost of applying finished loads on the
    // main thread. This is the number that decides whether streaming is felt as
    // a hitch, so it is measured rather than assumed.
    float worstApplyMs = 0.0f;
    float lastApplyMs = 0.0f;
    std::size_t residentChunks = 0;
    // Effect-only meshes (\effects\, fx*) dropped at build time. These are the
    // dust sheets and glow cards Fallout scatters as ordinary statics.
    std::uint64_t effectMeshesSkipped = 0;
    // Nodes whose field walk was rejected, summed over cells built this run.
    // Only cache MISSES contribute: a cell served from disk never parses a NIF,
    // so this reads zero on a warm cache no matter what the meshes contain.
    std::uint64_t nodeParseFailures = 0;
    // Per-vertex ATXT layer contributions beyond the four a vertex can hold,
    // summed over cells built this run. Nonzero means a quadrant lost its
    // weakest painted layers, which renders as a hard straight edge at the
    // quadrant boundary rather than as anything that looks like an error. Same
    // cache caveat as nodeParseFailures: only misses contribute.
    std::uint64_t droppedTerrainLayers = 0;
    // Cells that contributed a water surface, i.e. coast, lake or river.
    std::uint64_t waterPatchesLoaded = 0;
    // Draws carrying kImportedSceneMaterialFlagAlphaBlend, i.e. those replayed
    // through the blended pipeline instead of the opaque one.
    std::uint64_t blendedPartsLoaded = 0;
};

class CellStreamer {
public:
    CellStreamer();
    ~CellStreamer();

    CellStreamer(const CellStreamer&) = delete;
    CellStreamer& operator=(const CellStreamer&) = delete;

    // Opens the game's data directory and plugin, builds the plugin-wide tables
    // and the cell offset index, and selects a worldspace to stream by editor ID
    // (e.g. "WastelandNV"). Grid coordinates are only unique within one
    // worldspace, so this is required rather than optional.
    //
    // Costs roughly 300 ms: ~90 ms for the world tables and ~215 ms for the
    // offset index, neither of which materializes any cell contents.
    bool open(
        const std::filesystem::path& dataFilesPath,
        const std::filesystem::path& pluginFileName,
        const std::string& worldspaceEditorId,
        core::JobSystem& jobs,
        std::string& outError);

    // Directory for the on-disk cache of built cells. A cell built once is
    // written here and loaded straight back on later visits and later runs,
    // skipping record extraction, NIF parsing and texture decode entirely.
    //
    // The plugin's size and modification time are folded into a subdirectory
    // name, so a changed or different plugin simply misses rather than reading
    // stale geometry -- nothing is ever deleted to invalidate. Empty disables it.
    // Stream with an explicit load order instead of one plugin. The order's
    // FIRST entry must be the plugin the worldspace lives in; the rest override
    // it in ascending priority. Call before open(). Without this the streamer
    // reads exactly one plugin, which is what it always did.
    void setLoadOrder(FalloutLoadOrder order) {
        m_loadOrder = std::move(order);
        m_useLoadOrder = !m_loadOrder.empty();
    }

    void setCacheDirectory(std::filesystem::path directory) {
        m_cacheDirectory = std::move(directory);
    }

    // Mod roots searched ahead of the game's own files, in load order. Must be
    // called before open(), which is where they are applied and where any
    // warning about them is logged.
    void addModDirectory(std::filesystem::path directory) {
        m_modDirectories.push_back(std::move(directory));
    }

    // Mip-drop ceiling for every texture a cell brings in, longest side, in
    // pixels. 0 disables the drop and keeps whatever the file ships.
    //
    // This is the single knob that decides whether a high-resolution texture
    // pack is visible at all: the default clamps Fallout's own 1024-square
    // diffuse maps to 512, and a 2048 mod texture would be dropped straight past
    // its own resolution to the same 512. It costs memory quadratically -- a
    // BC1 1024-square with mips is 683 KB against 171 KB at 512 -- so raising it
    // on an integrated GPU is a real decision, not a free win.
    //
    // Folded into the cache key, because cells bake their textures in.
    void setMaxTextureSize(std::uint32_t maxSize) { m_maxTextureSize = maxSize; }
    [[nodiscard]] std::uint32_t maxTextureSize() const { return m_maxTextureSize; }

    // The resolved asset source, valid after open(). Exposed so a caller can
    // pull assets the streamer itself has no interest in -- weather cloud
    // textures, which live in a mod's BSA -- without opening a second search
    // path and duplicating the precedence rules.
    [[nodiscard]] const FalloutAssetSource& assets() const { return m_assets; }

    void setConfig(const CellResidencyConfig& config) { m_planner.setConfig(config); }
    [[nodiscard]] const CellResidencyConfig& config() const { return m_planner.config(); }

    // Once per frame. Starts loads for cells coming into range, applies whatever
    // finished, and evicts what has left. World units, Fallout space (Z-up, so
    // the grid's second axis is world Y).
    void update(render::Renderer& renderer, const float position[3], const float velocity[3]);

    // Called on the main thread as cells become resident and are evicted, so a
    // consumer (collision, navigation) can mirror the resident set without the
    // streamer knowing what any of them are for. The scene reference is valid
    // only for the duration of the call.
    using CellResidentCallback =
        std::function<void(const CellCoord&, const ImportedScene&)>;
    using CellEvictedCallback = std::function<void(const CellCoord&)>;
    void setCellCallbacks(CellResidentCallback onResident, CellEvictedCallback onEvicted) {
        m_onCellResident = std::move(onResident);
        m_onCellEvicted = std::move(onEvicted);
    }

    // Blocks until in-flight loads finish and are drained. Required before
    // destroying the streamer or its job system.
    void waitIdle();

    // A spawn position in ENGINE space (Y-up), standing above the terrain at
    // the centre of the streamable cells. The world origin is usually outside
    // the worldspace entirely, and the height has to come from the actual LAND
    // record or the camera starts underground.
    //
    // Extracts the centre cell's records synchronously (~0.5 ms) to read its
    // terrain heights. Returns false when no cells are available.
    bool suggestedSpawnEngineSpace(float outPosition[3]) const;

    // True when no cell build is in flight and none is waiting to be applied.
    // A capture uses this to know the world has stopped arriving. Counting warm-up
    // FRAMES is the obvious proxy and a bad one: how much streaming fits in a
    // fixed frame count depends entirely on how fast the renderer happens to be,
    // so speeding up capture silently shortened the warm-up and started recording
    // half-loaded towns.
    bool isStreamingIdle() const;

    // Everything a room needs that is not its geometry: how it is lit, and
    // somewhere inside it to stand.
    struct InteriorScene {
        // As authored, sRGB 0..1. See FalloutCellRecord's XCLL note: an interior
        // has no sun, so ambient is usually the whole rig.
        bool hasLighting = false;
        float ambientColor[3] = {};
        float directionalColor[3] = {};
        float fogColor[3] = {};
        float fogNear = 0.0f;
        float fogFar = 0.0f;
        // Engine space, on the floor. Taken from the room's own teleport door
        // and stepped inward, because a door is the one reference in an interior
        // guaranteed to stand somewhere a person can.
        bool hasSpawn = false;
        float spawnPosition[3] = {};
        float spawnYawDegrees = 0.0f;
    };

    // Builds one INTERIOR cell into a scene, synchronously -- interiors are one
    // room, not a streaming grid, and Doc Mitchell's house is ~1.6 s cold and
    // far less warm. Uses the same extract-and-build path the streaming jobs do,
    // so an interior cannot drift from how an exterior cell is made.
    //
    // Returns false if no interior with that EditorID exists.
    bool buildInteriorScene(
        const std::string& interiorEditorId,
        ImportedScene& outScene,
        InteriorScene& outInterior,
        std::string& outError);

    // Spawn on the doorstep of a named interior, in ENGINE space -- e.g.
    // "GSDocMitchellHouse" for where Fallout: New Vegas actually starts you.
    //
    // Uses the interior's own teleport door: its XTEL records the arrival
    // position in the TARGET cell, which for a door leading out is the exterior
    // doorstep. That is exact, unlike guessing from the interior's coordinates,
    // which are in an unrelated space.
    //
    // Returns false if no such interior exists or it has no teleport door.
    bool spawnAtInteriorDoorEngineSpace(
        const std::string& interiorEditorId, float outPosition[3]) const;

    // Fallout is Z-up, the engine is Y-up, and the conversion is
    // (x, y, z) -> (x, z, -y). Both directions live here because the streamer is
    // the boundary between the two: the planner ranks cells in Fallout space,
    // the camera moves in engine space, and mixing them puts the player
    // underground while streaming cells from somewhere else entirely.
    static void engineToFallout(const float enginePosition[3], float outFallout[3]);
    static void falloutToEngine(const float falloutPosition[3], float outEngine[3]);

    // The discoverable regions covering an engine-space position, by display
    // name ("Mojave Outpost").
    //
    // Lives here rather than in the game because the streamer already owns both
    // halves -- the cell index (which carries each cell's XCLR region list) and
    // the world tables (REGN formID -> map name). Answering it anywhere else
    // would mean a second copy of one of them.
    //
    // Returns nothing outside the loaded worldspace, for a cell with no region,
    // and for regions with no RDMP map name. That last case is the common one:
    // only 55 of FalloutNV.esm's 276 regions are player-facing, and the other
    // 221 are weather/audio zones nobody should be told they have entered.
    [[nodiscard]] std::vector<std::string> regionNamesAtEngineSpace(
        const float enginePosition[3]) const;

    [[nodiscard]] CellStreamerStats stats() const;
    [[nodiscard]] std::size_t availableCellCount() const { return m_availableCells.size(); }
    // Inclusive grid bounds of the cells this worldspace actually has, which is
    // what a caller needs to cover the world with anything -- distant LOD tiles,
    // a map, a preload sweep. Returns false when the worldspace is empty.
    //
    // Derived from the OCCUPIED cells rather than from the worldspace record's
    // own NAM0/NAM9 corners, which are not the same thing: Bravil's corners span
    // 52 cells of open bay and frame no city at all.
    [[nodiscard]] bool cellGridBounds(
        std::int32_t& outMinX, std::int32_t& outMinZ,
        std::int32_t& outMaxX, std::int32_t& outMaxZ) const {
        bool any = false;
        for (const auto& [coord, unused] : m_availableCells) {
            (void)unused;
            if (!any) {
                outMinX = outMaxX = coord.x;
                outMinZ = outMaxZ = coord.z;
                any = true;
                continue;
            }
            outMinX = std::min(outMinX, coord.x);
            outMaxX = std::max(outMaxX, coord.x);
            outMinZ = std::min(outMinZ, coord.z);
            outMaxZ = std::max(outMaxZ, coord.z);
        }
        return any;
    }
    [[nodiscard]] bool isOpen() const { return m_jobs != nullptr; }

private:
    struct Pending;

    void applyCompletedLoads(render::Renderer& renderer);

    CellResidencyPlanner m_planner;
    std::filesystem::path m_esmPath;
    // The whole load order, when the caller supplied extra plugins. Cells,
    // references and base records are then merged across it with later plugins
    // overriding earlier ones -- the only way a patch's fixes reach the scene.
    // Empty means the single-plugin path, which stays byte-identical.
    FalloutLoadOrder m_loadOrder;
    bool m_useLoadOrder = false;
    std::vector<std::filesystem::path> m_modDirectories;
    std::uint32_t m_maxTextureSize = 512u;
    std::filesystem::path m_cacheDirectory;
    // m_cacheDirectory / <plugin fingerprint>, created on open(). Empty when
    // caching is off or the directory could not be created.
    std::filesystem::path m_resolvedCacheDirectory;
    DecodedTextureCache m_textureCache;
    FalloutWorldTables m_worldTables;
    FalloutCellIndex m_cellIndex;

public:
    // World units one exterior cell covers in the plugin being streamed. The
    // caller sizes its residency grid from this rather than assuming 4096:
    // Morrowind's cells are 8192, and a grid built on the wrong figure loads a
    // quarter of the world it believes it is loading.
    [[nodiscard]] float cellWorldSize() const { return m_cellIndex.cellWorldSize; }

private:
    FalloutAssetSource m_assets;
    // Grid coordinate -> index into m_cellIndex.cells, for the chosen worldspace
    // only. The worldspace is not a rectangle, so this is also what stops the
    // planner re-requesting holes forever.
    std::unordered_map<CellCoord, std::size_t, CellCoordHash> m_availableCells;
    // Grid coordinate -> the renderer chunk index holding it.
    std::unordered_map<CellCoord, std::size_t, CellCoordHash> m_residentChunks;
    std::shared_ptr<Pending> m_pending;
    CellResidentCallback m_onCellResident;
    CellEvictedCallback m_onCellEvicted;
    core::JobSystem* m_jobs = nullptr;
    CellStreamerStats m_stats{};
};

}  // namespace importer::fnv
}  // namespace odai
