#include "import/fnv/cell_streamer.h"

#include <algorithm>
#include <limits>
#include <cctype>
#include <condition_variable>
#include <iterator>
#include <mutex>
#include <system_error>
#include <utility>

#include "core/job_system.h"
#include "import/fnv/esm_reader.h"
#include "import/fnv/strings_table.h"
#include "core/log.h"
#include "core/frame_profiler.h"
#include "import/imported_scene.h"
#include "render/renderer.h"

namespace odai::importer::fnv {

namespace {

std::string cellAxisToken(std::int32_t value) {
    // 'm' rather than '-' so the name is a plain identifier on every
    // filesystem and cannot be mistaken for a command-line flag.
    return (value < 0) ? ("m" + std::to_string(-static_cast<std::int64_t>(value)))
                       : std::to_string(value);
}

// Identifies the plugin a cache was built from. Folded into a directory name
// rather than checked and deleted: a different or updated plugin simply misses,
// so no cache invalidation ever removes a file.
// Bump whenever CellSceneBuilder's output changes shape or content. It joins
// the cache key, so an old cache directory is simply never consulted again
// rather than silently serving cells built by the previous rules. Cheap to
// bump; forgetting to bump it produces a bug that looks like the new code not
// running at all.
//
// 2 = alpha-blend material flag + effect-only meshes skipped
// 3 = the alpha-cutout heuristic no longer forces alpha test onto blended
//     surfaces (v2 cells have the test bit baked into their packed vertices,
//     and the load-time pass only ever adds flags, so they cannot be repaired)
// 4 = NiStencilProperty DRAW_BOTH read into kImportedSceneMaterialFlagTwoSided
// 5 = ...with the draw-mode bits actually right; v4 never set the flag
// 6 = alpha blend now means blend WITHOUT test; blend+test is a cutout
// 7 = per-surface alpha-test threshold carried on the packed draw
// 8 = LIGH references emitted as ImportedScene lights (cached cells built
//     before this carry none, and ImportedScene::lights is serialized)
// 9 = SCOL (static collection) references placed; their merged mesh is real
//     geometry that every cached cell before this is missing
// v9 -> v10: cell scenes set alphaFlagsAuthored, so the texture-content cutout
// guess no longer forces alpha test onto authored-opaque shapes (Doc Mitchell's
// boarded-up planks). The flag is honoured at load, but v9 caches were PACKED
// with the guessed flags already baked into their vertices, so they must not
// be served.
// v10 -> v11: NIF roots now come from the file's own footer instead of "every
// node nobody claims as a child", and node recognition covers all 31
// NiNode-derived types from nif.xml rather than a hand-written twelve. Subtrees
// under an unrecognized or unparsed parent used to be promoted to roots and
// walked from IDENTITY, which baked their ancestors' translation OUT of the
// vertices -- the meshes seen floating in the sky. Cached cells hold those
// wrong world-space positions baked in, so they cannot be repaired at load and
// must miss.
// 12: NIF properties now inherit down the scene graph (nif_scene.cc), so a
// shape whose NiAlphaProperty or NiStencilProperty sits on a parent NiNode
// finally imports with an alpha mode instead of as fully opaque. Alpha test,
// blend and two-sidedness are baked into a cached cell's packed vertex flags,
// so a cell built before this fix carries the wrong ones and cannot be repaired
// at load -- it has to miss.
// 14: two changes to what a cell contains, both of which a cached cell has
// baked in and neither of which can be repaired at load.
//
// Degenerate triangles are now dropped at NIF parse. Measured across all 20746
// retail FalloutNV meshes: 7163270 of 43435533 triangles -- 16.5% of the
// game's geometry -- name the same vertex twice and rasterize nothing. The
// strip expander had always dropped them, because that is how Bethesda stitches
// strips together; the explicit NiTriShapeData path had no filter at all, and
// every one of those triangles was being submitted to every pass, every frame.
// (Oblivion measures 0, which fits: it is 38372 strip blocks against 1956
// explicit lists.)
//
// And: cells carry their WATER SURFACE (XCLW -> ImportedSceneWaterPatch). A
// cached cell built before this has no water patch in it at all, and nothing at
// load time can invent one -- the height it needs is in the plugin record, not
// in the cooked scene. Without the bump every existing install keeps serving a
// coastline with no sea and the fix looks like it did nothing.
// (13 was this same set mid-flight: the water half landed first and wrote
// caches, and the triangle half then had no way to invalidate them. Bumping
// once per SHIPPED change is the rule; bumping once per editing session is not
// enough, and the symptom is a fix that measures as doing exactly nothing.)
// 15: Morrowind cells become cacheable, and NiAlphaProperty's NiObjectNET
// prefix is now read per generation (readNiObjectNetPrefix). It was open-coded
// as the Fallout spelling only, so an INLINE name had its length read as a name
// index and its bytes as the extra-data count; the parse failed, the shape kept
// its default of opaque, and an alpha-tested leaf rendered as a solid slab.
// Morrowind goes 0 -> 1013 alpha-tested shapes on that fix alone.
//
// Oblivion measured UNCHANGED across it -- 469 alpha-tested shapes before and
// after -- so this bump is not repairing anything there. Said explicitly
// because the obvious reading of "a shared prefix was wrong" is that every
// generation was affected, and that is not what the numbers say.
// 16: RootCollisionNode subtrees are no longer drawn. Morrowind has no Havok --
// a mesh's collision hull is ordinary geometry under a node whose TYPE NAME is
// the whole semantics -- and drawing it put untextured, UV-less slabs a few
// units outside every wall. Across the archive that is 7067 shapes, and it took
// shapes with no diffuse from 7217 to 150.
// 17: NIF 10.1.0.101-10.2.0.0 is read. Oblivion ships a MIX of Gamebryo
// generations and 580 of its meshes are the older one, including
// icpalacetower01.nif -- the White-Gold Tower. +374 meshes, +454494 triangles,
// and zero out-of-range triangles across all 10.2M, which is the check that
// says the field walk is actually right rather than merely terminating.
// 18: Morrowind terrain carries synthesized per-vertex layer weights. VTEX
// names one texture per 512-unit block and nothing else, so the blend between
// blocks has to be invented at build time -- and it lives in the packed vertex,
// which means cached cells keep the hard-edged version until this bumps.
// 19: EditorMarker subtrees are no longer drawn
// 20: BSTriShape (Skyrim) geometry, and build-machine texture paths resolve
// 21: no implied sea level in a worldspace with no LAND record
// 22: skinned BSTriShape geometry (NiSkinPartition) -- banners, cloth
// 23: the game's own sky meshes are no longer placed as world geometry
// 24: initially-disabled references (REFR flag 0x800) are no longer drawn
// 25: NiAVObject-hidden subtrees (flag bit 0) are no longer drawn -- particle
//     emitter source meshes were rendering as a plane over the landscape
// 26: ~760 more Oblivion meshes parse (Havok, skinning and animation blocks
//     the no-size-table walk could not size), so cells containing them stop
//     dropping that geometry
// 27: streamed cells carry their water again -- the cell index never held
//     XCLW, so every river, lake and sea existed only in cooked scenes
// 28: FLOR records place (Whiterun's garlic braids), and TES5 ARMO world
//     models come from MOD2 -- its MODL is a binary armature list that was
//     being read as a path made of formID bytes
// 29: distant-LOD shells (*LOD.nif) draw two-sided -- they are hollow
//     single-sided hulls, and culling ate half of Dragonsreach
// 30: authored VERTEX ALPHA is read and lives in the packed vertex. It is what
//     feathers a placed road, path or dirt patch into the ground under it:
//     Whiterun's WRMainRoadPlains lays an alpha-TESTED grass/moss overlay whose
//     texture alpha is uniform and whose vertex alpha ramps 0->1 across the
//     fringe (108 of 238 vertices on one shape, 269 of 613 on another). Both
//     NIF generations dropped the channel -- the classic reader seeked past it,
//     BSVertexData never looked at its colour nibble -- so the overlay rendered
//     as a hard-edged uniform sheet, which reads as a decal or z-fighting
//     problem rather than as a missing vertex attribute.
// 31: stationary fire-effect NIF placements become procedural emissive
//     particle emitters. Their effect source meshes remain suppressed instead
//     of returning as opaque sheets.
// 32: fire scale widens the emitter footprint without enlarging every lobe;
//     nearby authored LIGH records win, with a flickering clustered fallback
//     only where the game did not place one.
// 33: animated banner skin partitions are settled under Jolt soft-body gravity
//     with their authored top attachment pinned, instead of freezing in the
//     sideways wind-blown bind pose.
// 34: TES3 NiLODNode children are selected and flattened instead of dropping
//     the whole model when its classic NIF has no block-size table.
// 35: packed terrain vertices carry an explicit terrain marker so runtime PBR
//     presets can choose a different terrain roughness from placed objects.
// 40: terrain diffuse UVs are world-phased rather than restarting in every
//     4096-unit TES4 cell, and partial water cells follow the LAND shoreline
//     instead of covering boardwalks with a full-cell cyan rectangle.
// 41: Skyrim's embedded rigid machinery tracks survive cell packing/cache and
//     animate water wheels plus the sawmill work cycle at runtime.
// 42: Morrowind NiBSAnimationNode statics promote their direct keyframe
//     controllers into the same runtime rigid-animation path.
constexpr int kCellBuildVersion = 42;

// How long applyCompletedLoads may spend uploading finished cells in one frame,
// and how slow a single chunk add has to be before it logs itself.
//
// 6 ms leaves room inside a 16.7 ms frame for the rest of the CPU work while
// still draining the queue quickly when several small cells land together.
constexpr float kChunkApplyBudgetMs = 6.0f;
constexpr float kSlowChunkApplyLogMs = 8.0f;

// Counts blended packed draws by inspecting the first vertex of each, matching
// how the renderer decides which pipeline a draw goes through. Runs on the
// cache-hit path too, so the number is the same whether a cell was rebuilt or
// loaded.
std::uint64_t countBlendedDraws(const ImportedScene& scene) {
    std::uint64_t blended = 0;
    for (const ImportedScenePackedDraw& draw : scene.packedDraws) {
        if (draw.indexCount == 0u || draw.firstIndex >= scene.packedIndices.size()) {
            continue;
        }
        const std::uint32_t vertexIndex = scene.packedIndices[draw.firstIndex];
        if (vertexIndex < scene.packedVertices.size() &&
            (scene.packedVertices[vertexIndex].flags &
             kImportedSceneMaterialFlagAlphaBlend) != 0u) {
            ++blended;
        }
    }
    return blended;
}

// `modFingerprint` and `maxTextureSize` join the key for the same reason the
// build version does: a cached cell has its textures baked in, so installing a
// texture pack or changing the mip-drop ceiling has to miss rather than keep
// serving the art the cell was built with. Both are appended only when they are
// non-default, so an unmodded cache directory keeps the name it always had and
// existing caches stay valid.
std::string pluginFingerprint(
    const std::filesystem::path& esmPath,
    const std::string& modFingerprint,
    std::uint32_t maxTextureSize,
    const std::string& loadOrderFingerprint) {
    std::error_code sizeError;
    const auto size = std::filesystem::file_size(esmPath, sizeError);
    std::error_code timeError;
    const auto writeTime = std::filesystem::last_write_time(esmPath, timeError);
    const auto ticks = writeTime.time_since_epoch().count();
    std::string fingerprint =
        std::to_string(sizeError ? 0ull : static_cast<std::uint64_t>(size)) + "_" +
        std::to_string(timeError ? 0ll : static_cast<long long>(ticks)) + "_v" +
        std::to_string(kCellBuildVersion);
    if (maxTextureSize != 512u) {
        fingerprint += "_t" + std::to_string(maxTextureSize);
    }
    if (!modFingerprint.empty()) {
        fingerprint += "_m" + modFingerprint;
    }
    // A cell built against one load order has that order's overrides BAKED in --
    // moved references, replaced terrain. Serving it to a different order is
    // serving another mod list's world.
    if (!loadOrderFingerprint.empty()) {
        fingerprint += "_l" + loadOrderFingerprint;
    }
    return fingerprint;
}

}  // namespace

// Loads in flight and their results. Held by shared_ptr so a job still running
// when the streamer is destroyed writes somewhere valid; waitIdle() makes that
// impossible in normal use, but ownership makes it impossible by construction.
struct CellStreamer::Pending {
    struct Result {
        CellCoord cell;
        ImportedScene scene;
        bool succeeded = false;
        bool fromCache = false;
        bool cacheWriteFailed = false;
        std::string error;
        float buildMs = 0.0f;
        float cacheLoadMs = 0.0f;
        std::uint64_t effectMeshesSkipped = 0;
        std::uint64_t nodeParseFailures = 0;
        std::uint64_t droppedTerrainLayers = 0;
        std::uint64_t waterPatches = 0;
        std::uint64_t blendedParts = 0;
    };

    std::mutex mutex;
    std::condition_variable idle;
    std::vector<Result> completed;
    std::uint32_t inFlight = 0;
};

CellStreamer::CellStreamer() : m_pending(std::make_shared<Pending>()) {}

CellStreamer::~CellStreamer() {
    waitIdle();
}

bool CellStreamer::open(
    const std::filesystem::path& dataFilesPath,
    const std::filesystem::path& pluginFileName,
    const std::string& worldspaceEditorId,
    core::JobSystem& jobs,
    std::string& outError) {
    outError.clear();
    m_availableCells.clear();
    m_residentChunks.clear();
    m_planner.reset();
    m_jobs = nullptr;
    m_esmPath = dataFilesPath / pluginFileName;

    if (!m_assets.open(dataFilesPath)) {
        outError = "cannot read Fallout data directory " + dataFilesPath.string();
        return false;
    }
    // After open(), which clears the warning list, and before it is drained
    // below, so an unreadable mod directory actually gets reported.
    for (const std::filesystem::path& modDirectory : m_modDirectories) {
        m_assets.addModDirectory(modDirectory);
    }
    for (const std::string& warning : m_assets.warnings()) {
        VOX_LOGW("streamer") << warning;
    }
    if (m_assets.modDirectoryCount() != 0u) {
        // Archives are reported separately because a mod can ship nothing loose
        // at all -- Nevada Skies is one 330 MB BSA -- and "0 files" next to a
        // working mod is the most confusing thing this log could say.
        VOX_LOGI("streamer") << "mods: " << m_assets.modDirectoryCount() << " directories, "
                             << m_assets.modFileCount() << " loose files, "
                             << m_assets.modArchiveCount() << " archives override the base game";
    }

    // With a load order, base records and cell contents are merged across every
    // plugin in it, later winning. Without one this is the single-plugin path it
    // has always been, and the results are identical -- a load order of one
    // remaps every formID to itself.
    if (m_useLoadOrder) {
        if (!buildFalloutWorldTables(m_loadOrder, m_worldTables, outError)) {
            return false;
        }
        if (!buildFalloutCellIndex(m_loadOrder, m_cellIndex, outError)) {
            return false;
        }
    } else {
        if (!buildFalloutWorldTables(m_esmPath, m_worldTables, outError)) {
            return false;
        }
        if (!buildFalloutCellIndex(m_esmPath, m_cellIndex, outError)) {
            return false;
        }
    }
    resolveLocalizedRegionNames();

    std::string loweredWorldspace = worldspaceEditorId;
    for (char& c : loweredWorldspace) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    const auto worldIt = m_worldTables.worldspaceFormIdsByEditorId.find(loweredWorldspace);
    if (worldIt == m_worldTables.worldspaceFormIdsByEditorId.end()) {
        outError = "no worldspace named \"" + worldspaceEditorId + "\" in " + m_esmPath.string();
        return false;
    }
    const std::uint32_t worldspaceFormId = worldIt->second;

    for (std::size_t i = 0; i < m_cellIndex.cells.size(); ++i) {
        const FalloutCellIndexEntry& entry = m_cellIndex.cells[i];
        if (entry.isInterior || !entry.hasGridCoords ||
            entry.worldspaceFormId != worldspaceFormId || entry.childrenGroupSize == 0u) {
            continue;
        }
        m_availableCells.emplace(CellCoord{entry.gridX, entry.gridZ}, i);
    }

    // Resolve the cache directory now so the per-cell jobs only have to join a
    // filename onto it.
    m_resolvedCacheDirectory.clear();
    if (!m_cacheDirectory.empty()) {
        std::string loweredForPath = loweredWorldspace;
        const std::filesystem::path candidate =
            m_cacheDirectory /
            pluginFingerprint(m_esmPath, m_assets.modFingerprint(), m_maxTextureSize,
                              m_useLoadOrder ? m_loadOrder.fingerprint() : std::string()) /
            loweredForPath;
        std::error_code createError;
        std::filesystem::create_directories(candidate, createError);
        if (createError) {
            VOX_LOGW("streamer") << "cell cache disabled: cannot create " << candidate.string()
                                 << ": " << createError.message();
        } else {
            m_resolvedCacheDirectory = candidate;
            VOX_LOGI("streamer") << "cell cache at " << candidate.string();
        }
    }

    m_jobs = &jobs;
    VOX_LOGI("streamer") << "streaming " << worldspaceEditorId << " from " << m_esmPath.string()
                         << ": " << m_availableCells.size() << " exterior cells, "
                         << m_worldTables.staticModelPaths.size() << " statics, "
                         << m_assets.archiveCount() << " archives";
    return true;
}

void CellStreamer::update(
    render::Renderer& renderer, const float position[3], const float velocity[3]) {
    if (m_jobs == nullptr) {
        return;
    }

    // Applied first: a load that finished since the last frame should be counted
    // as resident before the planner decides what else to ask for, or the
    // in-flight budget stays occupied by work that is already done.
    applyCompletedLoads(renderer);

    m_planner.update(position, velocity);

    for (const CellCoord& cell : m_planner.cellsToEvict()) {
        const auto resident = m_residentChunks.find(cell);
        if (resident != m_residentChunks.end()) {
            renderer.removeImportedSceneChunk(resident->second);
            m_residentChunks.erase(resident);
            if (m_onCellEvicted) {
                m_onCellEvicted(cell);
            }
        }
        m_planner.markEvicted(cell);
    }

    for (const CellCoord& cell : m_planner.cellsToLoad()) {
        const auto available = m_availableCells.find(cell);
        if (available == m_availableCells.end()) {
            // No scene for this cell: the worldspace is not a rectangle. Record
            // it so the planner never proposes it again.
            m_planner.markUnavailable(cell);
            continue;
        }

        m_planner.markLoadStarted(cell);
        const FalloutCellIndexEntry entry = m_cellIndex.cells[available->second];
        const std::filesystem::path esmPath = m_esmPath;
        // Both are copied per job rather than referenced: the streamer can be
        // destroyed while jobs are still in flight, and these are small (a path
        // list and a per-plugin index remap table).
        const FalloutCellIndex* cellIndex = &m_cellIndex;
        const FalloutLoadOrder* loadOrder = m_useLoadOrder ? &m_loadOrder : nullptr;
        const FalloutAssetSource* assets = &m_assets;
        const FalloutWorldTables* tables = &m_worldTables;
        std::shared_ptr<Pending> pending = m_pending;
        {
            std::lock_guard<std::mutex> lock(pending->mutex);
            ++pending->inFlight;
        }
        std::filesystem::path cachePath;
        if (!m_resolvedCacheDirectory.empty()) {
            cachePath = m_resolvedCacheDirectory /
                        ("cell_" + cellAxisToken(cell.x) + "_" + cellAxisToken(cell.z) + ".bin");
        }
        DecodedTextureCache* textureCache = &m_textureCache;
        const std::uint32_t maxTextureSize = m_maxTextureSize;
        m_jobs->enqueue([pending, esmPath, entry, cell, assets, tables, cachePath, textureCache,
                         maxTextureSize, cellIndex, loadOrder]() {
            const core::Stopwatch buildTimer;
            Pending::Result result;
            result.cell = cell;

            // Cache hit: skip record extraction, NIF parsing and texture decode
            // entirely. This is the whole point -- a rebuilt cell costs ~270 ms,
            // most of it DDS decode, and reading one back costs a file read.
            if (!cachePath.empty()) {
                std::error_code existsError;
                if (std::filesystem::exists(cachePath, existsError) && !existsError) {
                    const core::Stopwatch cacheTimer;
                    if (loadImportedScene(cachePath, result.scene)) {
                        result.succeeded = true;
                        result.fromCache = true;
                        result.cacheLoadMs = cacheTimer.elapsedMs();
                        result.buildMs = buildTimer.elapsedMs();
                        result.blendedParts = countBlendedDraws(result.scene);
                        // Counted from the loaded scene, like blendedParts
                        // above: this stat is the log line's whole evidence
                        // that water exists, and counting it only on the build
                        // path made every warm-cache run report waterCells=0
                        // while the water rendered fine -- which reads as the
                        // fix having regressed, twenty minutes after it
                        // demonstrably worked.
                        result.waterPatches = result.scene.waterPatches.size();
                        std::lock_guard<std::mutex> lock(pending->mutex);
                        pending->completed.push_back(std::move(result));
                        if (pending->inFlight > 0u) {
                            --pending->inFlight;
                        }
                        if (pending->inFlight == 0u) {
                            pending->idle.notify_all();
                        }
                        return;
                    }
                    // A cache file that will not load is treated as a miss and
                    // overwritten below rather than reported as an error.
                    result.scene = ImportedScene{};
                }
            }

            // Each job owns its reader: EsmReader records walk state in members
            // and is not safe to share. Opening one is a memory map, not a read.
            EsmReader reader;
            const bool needBaseReader = loadOrder == nullptr;
            if (needBaseReader && !reader.open(esmPath)) {
                result.error = reader.lastError();
            } else {
                FalloutCellRecord record;
                const bool extracted =
                    (loadOrder != nullptr)
                        ? extractFalloutCellMerged(*cellIndex, *loadOrder, entry, record,
                                                   result.error)
                        : extractFalloutCellAt(reader, entry, record, result.error);
                if (!extracted) {
                    // result.error already set
                } else {
                    CellSceneBuilder builder(*assets, *tables, textureCache);
                    builder.setMaxTextureSize(maxTextureSize);
                    const std::vector<const FalloutCellRecord*> single{&record};
                    // A single cell is all the evidence there is for what the
                    // worldspace default ground texture should be; the cooker
                    // gets to survey a whole region for this.
                    builder.setFallbackLandTexture(builder.dominantLandTexture(single));
                    builder.addCellTerrain(record);
                    builder.addCellStatics(record);
                    builder.finish(result.scene);
                    result.succeeded = true;
                    result.effectMeshesSkipped = builder.stats().effectMeshesSkipped;
                    result.nodeParseFailures = builder.stats().nodeParseFailures;
                    result.droppedTerrainLayers = builder.stats().droppedTerrainLayers;
                    result.waterPatches = builder.stats().waterPatchesEmitted;
                    result.blendedParts = countBlendedDraws(result.scene);

                    if (!cachePath.empty()) {
                        // Write to a temporary and rename, so a crash or a second
                        // process mid-write cannot leave a truncated scene that a
                        // later run would load as real geometry.
                        const std::filesystem::path tempPath =
                            cachePath.string() + ".tmp" + std::to_string(
                                reinterpret_cast<std::uintptr_t>(&result));
                        if (saveImportedScene(result.scene, tempPath)) {
                            std::error_code renameError;
                            std::filesystem::rename(tempPath, cachePath, renameError);
                            if (renameError) {
                                std::error_code removeError;
                                std::filesystem::remove(tempPath, removeError);
                                result.cacheWriteFailed = true;
                            }
                        } else {
                            result.cacheWriteFailed = true;
                        }
                    }
                }
            }
            result.buildMs = buildTimer.elapsedMs();

            std::lock_guard<std::mutex> lock(pending->mutex);
            pending->completed.push_back(std::move(result));
            if (pending->inFlight > 0u) {
                --pending->inFlight;
            }
            if (pending->inFlight == 0u) {
                pending->idle.notify_all();
            }
        });
    }
}

void CellStreamer::applyCompletedLoads(render::Renderer& renderer) {
    // Take only as many as this frame's budget allows; the rest stay queued and
    // are applied over the following frames.
    //
    // The COUNT budget alone is not enough and measurement says so: with it
    // already pinned at 1, a single apply was still measured at 70 ms -- a
    // 69k-vertex, 52-texture cell uploads geometry and decodes textures
    // synchronously here, and no count can subdivide one item. The time budget
    // below is what stops a run of merely-expensive cells (0.7-10.8 ms each)
    // from stacking into one frame behind a count of 1 per frame; it cannot
    // help the single oversized cell, which needs the upload itself amortized.
    const std::size_t budget = std::max<std::size_t>(1u, m_planner.config().maxChunkAppliesPerFrame);
    std::vector<Pending::Result> drained;
    {
        std::lock_guard<std::mutex> lock(m_pending->mutex);
        const std::size_t take = std::min(budget, m_pending->completed.size());
        drained.insert(
            drained.end(),
            std::make_move_iterator(m_pending->completed.begin()),
            std::make_move_iterator(m_pending->completed.begin() + static_cast<std::ptrdiff_t>(take)));
        m_pending->completed.erase(
            m_pending->completed.begin(),
            m_pending->completed.begin() + static_cast<std::ptrdiff_t>(take));
    }
    if (drained.empty()) {
        m_stats.lastApplyMs = 0.0f;
        return;
    }

    const core::Stopwatch applyTimer;
    std::size_t appliedThisFrame = 0;
    for (Pending::Result& result : drained) {
        // Stop once the frame's apply budget is spent, and put the rest back.
        // Always apply at least one, or a machine where every cell exceeds the
        // budget would never make progress.
        if (appliedThisFrame > 0 && applyTimer.elapsedMs() > kChunkApplyBudgetMs) {
            std::lock_guard<std::mutex> lock(m_pending->mutex);
            m_pending->completed.insert(
                m_pending->completed.begin(),
                std::make_move_iterator(drained.begin() + static_cast<std::ptrdiff_t>(appliedThisFrame)),
                std::make_move_iterator(drained.end()));
            break;
        }
        ++appliedThisFrame;
        if (!result.succeeded) {
            ++m_stats.loadFailures;
            VOX_LOGW("streamer") << "cell " << result.cell.x << "," << result.cell.z
                                 << " failed to load: " << result.error;
            // Not markUnavailable: the file exists and a retry may succeed, but
            // the planner must stop counting it as in flight.
            m_planner.markUnavailable(result.cell);
            continue;
        }

        // The player may have walked out of range while this was reading. The
        // planner owns that decision and counts it as wasted work.
        if (!m_planner.markLoadFinished(result.cell)) {
            continue;
        }

        m_stats.lastBuildMs = result.buildMs;
        m_stats.worstBuildMs = std::max(m_stats.worstBuildMs, result.buildMs);
        if (result.fromCache) {
            ++m_stats.cacheHits;
            m_stats.lastCacheLoadMs = result.cacheLoadMs;
        } else {
            ++m_stats.cacheMisses;
        }
        if (result.cacheWriteFailed) {
            ++m_stats.cacheWriteFailures;
        }
        if (result.scene.packedIndices.empty()) {
            ++m_stats.emptyScenes;
            continue;  // legitimately empty cell; resident with no geometry
        }

        const core::Stopwatch chunkTimer;
        const std::size_t chunkIndex = renderer.addImportedSceneChunk(result.scene);
        const float chunkMs = chunkTimer.elapsedMs();
        // Per-chunk, not just aggregated: a single slow add is what a player
        // feels, and an average hides it.
        //
        // Gated, because this ran unconditionally on every chunk add -- one
        // formatted log line with six fields, on the main thread, inside the
        // frame it is trying to measure. Measuring a hitch should not be part
        // of the hitch.
        static const bool s_logChunkAdds = std::getenv("ODAI_FNV_LOG_CHUNK_ADDS") != nullptr;
        if (s_logChunkAdds || chunkMs > kSlowChunkApplyLogMs) {
            VOX_LOGI("streamer") << "cell " << result.cell.x << "," << result.cell.z
                                 << " chunk add took " << chunkMs << " ms ("
                                 << result.scene.packedVertices.size() << " verts, "
                                 << result.scene.textures.size() << " textures)";
        }
        if (chunkIndex == render::Renderer::kInvalidImportedChunkIndex) {
            VOX_LOGW("streamer") << "cell " << result.cell.x << "," << result.cell.z
                                 << " geometry upload failed";
            ++m_stats.loadFailures;
            m_planner.markEvicted(result.cell);
            continue;
        }
        m_residentChunks.emplace(result.cell, chunkIndex);
        ++m_stats.scenesLoaded;
        m_stats.effectMeshesSkipped += result.effectMeshesSkipped;
        m_stats.nodeParseFailures += result.nodeParseFailures;
        m_stats.droppedTerrainLayers += result.droppedTerrainLayers;
        m_stats.waterPatchesLoaded += result.waterPatches;
        m_stats.blendedPartsLoaded += result.blendedParts;
        if (m_onCellResident) {
            // Before result.scene is destroyed at the end of this loop.
            m_onCellResident(result.cell, result.scene);
        }
    }

    m_stats.lastApplyMs = applyTimer.elapsedMs();
    m_stats.worstApplyMs = std::max(m_stats.worstApplyMs, m_stats.lastApplyMs);
}

namespace {

// Case-insensitive lookup of a cell by EditorID. Interiors are the only cells
// that reliably have one.
const FalloutCellIndexEntry* findCellByEditorId(
    const FalloutCellIndex& index, const std::string& editorId
) {
    std::string wanted = editorId;
    for (char& c : wanted) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    for (const FalloutCellIndexEntry& entry : index.cells) {
        if (entry.editorId.empty()) {
            continue;
        }
        std::string lowered = entry.editorId;
        for (char& c : lowered) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (lowered == wanted) {
            return &entry;
        }
    }
    return nullptr;
}

}  // namespace

bool CellStreamer::buildInteriorScene(
    const std::string& interiorEditorId,
    ImportedScene& outScene,
    InteriorScene& outInterior,
    std::string& outError
) {
    outScene = ImportedScene{};
    outInterior = InteriorScene{};
    outError.clear();

    const FalloutCellIndexEntry* entry = findCellByEditorId(m_cellIndex, interiorEditorId);
    if (entry == nullptr) {
        outError = "no cell named \"" + interiorEditorId + "\"";
        return false;
    }

    EsmReader reader;
    if (!m_useLoadOrder && !reader.open(m_esmPath)) {
        outError = reader.lastError();
        return false;
    }
    FalloutCellRecord record;
    const bool extracted = m_useLoadOrder
        ? extractFalloutCellMerged(m_cellIndex, m_loadOrder, *entry, record, outError)
        : extractFalloutCellAt(reader, *entry, record, outError);
    if (!extracted) {
        return false;
    }

    // The same builder the streaming jobs use, so an interior cannot drift from
    // how an exterior cell is made. No terrain pass and no fallback land
    // texture: an interior has no LAND record at all.
    CellSceneBuilder builder(m_assets, m_worldTables, &m_textureCache);
    builder.setMaxTextureSize(m_maxTextureSize);
    builder.addCellStatics(record);
    builder.finish(outScene);

    outInterior.hasLighting = record.hasLighting;
    outInterior.cellFlags = record.cellFlags;
    outInterior.showSky = (record.cellFlags & kCellFlagShowSky) != 0u;
    outInterior.useSkyLighting = (record.cellFlags & kCellFlagUseSkyLighting) != 0u;
    for (int channel = 0; channel < 3; ++channel) {
        outInterior.ambientColor[channel] = record.ambientColor[channel];
        outInterior.directionalColor[channel] = record.directionalColor[channel];
        outInterior.fogColor[channel] = record.fogColor[channel];
    }
    outInterior.fogNear = record.fogNear;
    outInterior.fogFar = record.fogFar;

    // Somewhere inside to stand: THE BIGGEST TRIANGLE OF THE ROOM'S OWN NAVMESH.
    //
    // Fallout authored a navmesh for every interior, and it is the exact answer
    // to "where can a person stand in here" -- no guessing, no clearance test.
    // The largest triangle is open floor rather than the strip behind a door or
    // the gap beside a bed, which is what makes its centroid a sane place to put
    // somebody.
    //
    // The first attempt stepped inward from the teleport door toward the
    // centroid of every reference in the cell, and it put the player's face
    // against a wall: a house is several rooms, so the average of its contents
    // is not reliably inside any of them, and the line to it crosses walls.
    const FalloutNavMeshRecord* bestMesh = nullptr;
    std::size_t bestTriangle = 0;
    double bestArea = 0.0;
    for (const FalloutNavMeshRecord& mesh : record.navMeshes) {
        const std::size_t vertexCount = mesh.vertices.size() / 3u;
        for (std::size_t t = 0; t < mesh.triangles.size(); ++t) {
            const FalloutNavMeshTriangle& tri = mesh.triangles[t];
            if (tri.vertex[0] >= vertexCount || tri.vertex[1] >= vertexCount ||
                tri.vertex[2] >= vertexCount) {
                continue;
            }
            const float* a = &mesh.vertices[static_cast<std::size_t>(tri.vertex[0]) * 3u];
            const float* b = &mesh.vertices[static_cast<std::size_t>(tri.vertex[1]) * 3u];
            const float* c = &mesh.vertices[static_cast<std::size_t>(tri.vertex[2]) * 3u];
            // Twice the area in the ground plane, which is all the comparison
            // needs -- Fallout's floors are flat enough that the horizontal
            // projection ranks them the same way the true area would.
            const double area = std::abs(
                (static_cast<double>(b[0] - a[0]) * static_cast<double>(c[1] - a[1])) -
                (static_cast<double>(c[0] - a[0]) * static_cast<double>(b[1] - a[1])));
            if (area > bestArea) {
                bestArea = area;
                bestMesh = &mesh;
                bestTriangle = t;
            }
        }
    }
    if (bestMesh != nullptr) {
        const FalloutNavMeshTriangle& tri = bestMesh->triangles[bestTriangle];
        float fallout[3] = {0.0f, 0.0f, 0.0f};
        for (int corner = 0; corner < 3; ++corner) {
            const float* v = &bestMesh->vertices[static_cast<std::size_t>(tri.vertex[corner]) * 3u];
            for (int axis = 0; axis < 3; ++axis) {
                fallout[axis] += v[axis] / 3.0f;
            }
        }
        falloutToEngine(fallout, outInterior.spawnPosition);
        // Face the room's teleport door, so the way out is the first thing in
        // view. Failing that, keep the default heading rather than inventing one.
        for (const FalloutPlacedReference& ref : record.references) {
            if (!ref.hasTeleport) {
                continue;
            }
            const float toDoorX = ref.position[0] - fallout[0];
            const float toDoorY = ref.position[1] - fallout[1];
            if ((toDoorX * toDoorX) + (toDoorY * toDoorY) > 1.0f) {
                // Engine space is (x, y, z) -> (x, z, -y), so a Fallout +y step
                // is an engine -z one; the yaw is measured in engine space
                // because that is what the camera reads.
                outInterior.spawnYawDegrees =
                    std::atan2(-toDoorY, toDoorX) * (180.0f / 3.14159265358979323846f);
            }
            break;
        }
        outInterior.hasSpawn = true;
    }

    // NO NAVMESH IS THE NORMAL CASE FOR SKYRIM, not an error. The search above
    // wants the largest navmesh triangle -- the middle of the biggest walkable
    // floor -- but Skyrim's NAVM is a TES5-layout record this reader does not
    // parse, so `record.navMeshes` is empty for every Skyrim interior. With no
    // spawn the caller keeps whatever camera it had, which means "--interior
    // WhiterunDragonsreach" loads Dragonsreach and leaves you standing outside
    // it in the worldspace, looking at sky. That reads as "the interior did not
    // load" when in fact 1431 references and 588599 vertices did.
    //
    // The fallback is the built geometry's own bounds: horizontally centred,
    // and low in the vertical span so the camera starts near the ground floor
    // rather than up in the rafters. Gravity and collision are already running
    // on the interior cell, so the exact height only has to be inside the room
    // -- the player settles onto the floor on the first tick.
    if (!outInterior.hasSpawn && !outScene.packedVertices.empty()) {
        outInterior.spawnPosition[0] = (outScene.boundsMin[0] + outScene.boundsMax[0]) * 0.5f;
        outInterior.spawnPosition[2] = (outScene.boundsMin[2] + outScene.boundsMax[2]) * 0.5f;
        outInterior.spawnPosition[1] =
            outScene.boundsMin[1] + ((outScene.boundsMax[1] - outScene.boundsMin[1]) * 0.15f);
        outInterior.hasSpawn = true;
        outInterior.spawnFromBounds = true;
    }

    const CellBuildStats& buildStats = builder.stats();
    const std::size_t droppedReferences =
        buildStats.referencesDroppedBaseNotFound +
        buildStats.referencesDroppedBaseHasNoModel +
        buildStats.referencesDroppedMeshUnresolved +
        buildStats.referencesDroppedMeshUnreadable;
    VOX_LOGI("streamer") << "interior " << interiorEditorId << ": " << record.references.size()
                         << " refs, " << outScene.packedVertices.size() << " verts, ambient ("
                         << static_cast<int>(outInterior.ambientColor[0] * 255.0f) << ","
                         << static_cast<int>(outInterior.ambientColor[1] * 255.0f) << ","
                         << static_cast<int>(outInterior.ambientColor[2] * 255.0f) << ")"
                         << ", XCLL=" << (outInterior.hasLighting ? "applied" : "absent")
                         << ", showSky=" << (outInterior.showSky ? "yes" : "no")
                         << ", useSkyLighting=" << (outInterior.useSkyLighting ? "yes" : "no")
                         << ", droppedRefs=" << droppedReferences
                         << ", skippedShapes=" << buildStats.skippedGeometryShapes
                         << ", particleEmitters=" << buildStats.particleEmittersPlaced
                         << ", gravityCloth=" << buildStats.clothMeshesSettled
                         << ", nodeParseFailures=" << buildStats.nodeParseFailures
                         << ", unresolvedTextures=" << buildStats.unresolvedTexturePaths.size()
                         << (outInterior.hasSpawn
                                 ? (outInterior.spawnFromBounds
                                        ? ", spawn from geometry bounds (no navmesh)"
                                        : ", spawn from navmesh")
                                 : ", NO spawn -- camera left where it was")
                         << (outInterior.hasSpawn
                                 ? (" at engine (" + std::to_string(outInterior.spawnPosition[0]) +
                                    ", " + std::to_string(outInterior.spawnPosition[1]) + ", " +
                                    std::to_string(outInterior.spawnPosition[2]) + ")")
                                 : std::string())
                         << " bounds engine x[" << outScene.boundsMin[0] << ", "
                         << outScene.boundsMax[0] << "] y[" << outScene.boundsMin[1] << ", "
                         << outScene.boundsMax[1] << "] z[" << outScene.boundsMin[2] << ", "
                         << outScene.boundsMax[2] << "]";
    return true;
}

bool CellStreamer::spawnAtInteriorDoorEngineSpace(
    const std::string& interiorEditorId, float outPosition[3]) const {
    std::string wanted = interiorEditorId;
    for (char& c : wanted) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    const FalloutCellIndexEntry* interior = nullptr;
    for (const FalloutCellIndexEntry& entry : m_cellIndex.cells) {
        if (entry.editorId.empty()) {
            continue;
        }
        std::string lowered = entry.editorId;
        for (char& c : lowered) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (lowered == wanted) {
            interior = &entry;
            break;
        }
    }
    if (interior == nullptr) {
        VOX_LOGW("streamer") << "spawn: no cell named \"" << interiorEditorId << "\"";
        return false;
    }

    FalloutCellRecord record;
    std::string error;
    EsmReader reader;
    const bool extracted = m_useLoadOrder
        ? extractFalloutCellMerged(m_cellIndex, m_loadOrder, *interior, record, error)
        : (reader.open(m_esmPath) && extractFalloutCellAt(reader, *interior, record, error));
    if (!extracted) {
        VOX_LOGW("streamer") << "spawn: could not read " << interiorEditorId << ": " << error;
        return false;
    }

    for (const FalloutPlacedReference& ref : record.references) {
        if (!ref.hasTeleport) {
            continue;
        }
        // XTEL's position is where the player arrives on the FAR side, i.e.
        // outside. Lift it to eye height: the recorded point is floor level.
        constexpr float kEyeHeightUnits = 120.0f;
        const float fallout[3] = {
            ref.teleportPosition[0], ref.teleportPosition[1],
            ref.teleportPosition[2] + kEyeHeightUnits};
        falloutToEngine(fallout, outPosition);
        VOX_LOGI("streamer") << "spawn: doorstep of " << interiorEditorId << " at Fallout ("
                             << fallout[0] << ", " << fallout[1] << ", " << fallout[2] << ")";
        return true;
    }
    VOX_LOGW("streamer") << "spawn: " << interiorEditorId << " has no teleport door";
    return false;
}

void CellStreamer::engineToFallout(const float enginePosition[3], float outFallout[3]) {
    // Inverse of (x, y, z) -> (x, z, -y).
    outFallout[0] = enginePosition[0];
    outFallout[1] = -enginePosition[2];
    outFallout[2] = enginePosition[1];  // engine Y is the up axis
}

void CellStreamer::falloutToEngine(const float falloutPosition[3], float outEngine[3]) {
    outEngine[0] = falloutPosition[0];
    outEngine[1] = falloutPosition[2];  // Fallout Z is the up axis
    outEngine[2] = -falloutPosition[1];
}

bool CellStreamer::isStreamingIdle() const {
    if (!m_pending) {
        return true;
    }
    std::lock_guard<std::mutex> lock(m_pending->mutex);
    return m_pending->inFlight == 0u && m_pending->completed.empty();
}

bool CellStreamer::suggestedSpawnEngineSpace(float outPosition[3]) const {
    if (m_availableCells.empty()) {
        return false;
    }
    std::int64_t sumX = 0;
    std::int64_t sumZ = 0;
    for (const auto& [cell, entryIndex] : m_availableCells) {
        (void)entryIndex;
        sumX += cell.x;
        sumZ += cell.z;
    }
    const auto count = static_cast<std::int64_t>(m_availableCells.size());
    CellCoord centre{
        static_cast<std::int32_t>(sumX / count), static_cast<std::int32_t>(sumZ / count)};

    // Walk outward from the centroid for a cell that has BOTH a scene and a
    // LAND record. Neither is guaranteed: the streamable set is sparse, and
    // plenty of exterior cells carry references but no terrain at all. Spawning
    // on the centroid regardless is how the camera ended up under the map --
    // the height fell back to a guess while the surrounding cells were
    // thousands of units higher.
    EsmReader reader;
    const bool readerOpen = m_useLoadOrder || reader.open(m_esmPath);
    if (!readerOpen) {
        VOX_LOGW("streamer") << "spawn: cannot open plugin: " << reader.lastError();
    }

    constexpr std::int32_t kMaxSpawnSearchRings = 24;
    CellCoord chosen = centre;
    float terrainHeight = 0.0f;
    bool haveHeight = false;
    float contentHeight = 0.0f;
    bool haveContentHeight = false;
    // Keep looking until a cell with something PLACED in it turns up, keeping
    // the first cell that merely had terrain as a fallback. Stopping at the
    // first terrain is what put the Megaton spawn in an empty edge cell whose
    // ground is nine thousand units below the town.
    bool haveContent = false;
    for (std::int32_t ring = 0; ring <= kMaxSpawnSearchRings && !haveContent; ++ring) {
        for (std::int32_t dz = -ring; dz <= ring && !haveContent; ++dz) {
            for (std::int32_t dx = -ring; dx <= ring && !haveContent; ++dx) {
                // Only the ring's perimeter; the interior was covered already.
                if (ring != 0 && std::abs(dx) != ring && std::abs(dz) != ring) {
                    continue;
                }
                const CellCoord candidate{centre.x + dx, centre.z + dz};
                const auto found = m_availableCells.find(candidate);
                if (found == m_availableCells.end() || !readerOpen) {
                    continue;
                }
                FalloutCellRecord record;
                std::string error;
                const bool extracted = m_useLoadOrder
                    ? extractFalloutCellMerged(
                          m_cellIndex, m_loadOrder, m_cellIndex.cells[found->second], record, error)
                    : extractFalloutCellAt(
                          reader, m_cellIndex.cells[found->second], record, error);
                if (!extracted ||
                    record.land == nullptr || !record.land->hasHeights) {
                    continue;
                }
                const float candidatePeak = *std::max_element(
                    std::begin(record.land->heights), std::end(record.land->heights));
                // WHAT IS PLACED IN THE CELL, not just what the terrain does.
                // Megaton is a town built inside a crater out of scrap: its
                // LAND peaks at 2872 while the town itself sits at ~12900, so
                // spawning above the terrain put the camera nine thousand units
                // underneath everything and looking at a bare white hill.
                //
                // The MEDIAN reference height rather than the maximum, because
                // a worldspace is entitled to one marker parked in the sky and
                // the maximum would follow it. Where content sits on the ground
                // -- the Capital Wasteland, the Mojave -- the median lands at
                // ground level and the terrain peak still wins, so this changes
                // nothing there.
                std::vector<float> referenceHeights;
                referenceHeights.reserve(record.references.size());
                for (const FalloutPlacedReference& reference : record.references) {
                    referenceHeights.push_back(reference.position[2]);
                }
                if (referenceHeights.empty()) {
                    // Terrain but nothing on it: remember it and keep looking.
                    if (!haveHeight) {
                        chosen = candidate;
                        terrainHeight = candidatePeak;
                        haveHeight = true;
                    }
                    continue;
                }
                const std::size_t middle = referenceHeights.size() / 2u;
                std::nth_element(
                    referenceHeights.begin(), referenceHeights.begin() + middle,
                    referenceHeights.end());
                chosen = candidate;
                terrainHeight = candidatePeak;
                contentHeight = referenceHeights[middle];
                haveHeight = true;
                haveContentHeight = true;
                haveContent = true;
            }
        }
    }

    if (!haveHeight) {
        VOX_LOGW("streamer") << "spawn: no cell with terrain within " << kMaxSpawnSearchRings
                             << " rings of the centre; falling back to a guessed height";
    } else {
        VOX_LOGI("streamer") << "spawn: cell " << chosen.x << "," << chosen.z
                             << " peak terrain height " << terrainHeight
                             << (haveContentHeight
                                     ? (", median placed height " + std::to_string(contentHeight))
                                     : std::string(", nothing placed"));
    }

    const float cellSize = m_planner.config().cellSize;
    float fallout[3] = {
        (static_cast<float>(chosen.x) + 0.5f) * cellSize,
        (static_cast<float>(chosen.z) + 0.5f) * cellSize,
        0.0f};
    // Clear of the cell's HIGHEST post, not its average, so a ridge running
    // through the cell does not swallow the camera.
    constexpr float kSpawnClearanceUnits = 600.0f;
    float groundHeight = haveHeight ? terrainHeight : 12000.0f;
    if (haveContentHeight) {
        groundHeight = std::max(groundHeight, contentHeight);
    }
    fallout[2] = groundHeight + kSpawnClearanceUnits;

    falloutToEngine(fallout, outPosition);
    return true;
}

void CellStreamer::waitIdle() {
    if (m_pending == nullptr) {
        return;
    }
    std::unique_lock<std::mutex> lock(m_pending->mutex);
    m_pending->idle.wait(lock, [this]() { return m_pending->inFlight == 0u; });
}

void CellStreamer::resolveLocalizedRegionNames() {
    if (m_worldTables.regionNameStringIdsByFormId.empty()) {
        return;  // no plugin here stores its region names as string IDs
    }
    // Which plugin a region came from decides which table to look it up in:
    // string IDs are local to the file that stored them, exactly like a
    // formID's mod index. After remapping, the formID's high byte IS the global
    // load-order position, so it names the plugin directly.
    const auto pluginFileNameAt = [this](std::uint8_t globalIndex) -> std::string {
        if (!m_useLoadOrder) {
            return m_esmPath.filename().string();
        }
        const std::vector<FalloutLoadOrderEntry>& entries = m_loadOrder.entries();
        for (const FalloutLoadOrderEntry& entry : entries) {
            if (entry.globalIndex == globalIndex) {
                return entry.header.isLocalized ? entry.header.fileName : std::string();
            }
        }
        return {};
    };
    if (!m_useLoadOrder) {
        // The single-plugin path never reads a TES4 header, so ask for one.
        // Cheap: a few hundred bytes at the front of the file.
        FalloutPluginHeader header{};
        std::string headerError;
        if (!readFalloutPluginHeader(m_esmPath, header, headerError) || !header.isLocalized) {
            return;
        }
    }

    std::unordered_map<std::string, FalloutStringTable> tablesByPlugin;
    std::size_t resolved = 0;
    std::size_t missing = 0;
    for (const auto& [formId, stringId] : m_worldTables.regionNameStringIdsByFormId) {
        const std::string pluginFileName = pluginFileNameAt(static_cast<std::uint8_t>(formId >> 24u));
        if (pluginFileName.empty()) {
            continue;
        }
        auto table = tablesByPlugin.find(pluginFileName);
        if (table == tablesByPlugin.end()) {
            FalloutStringTable loaded;
            std::string tableError;
            if (!loadFalloutStringTable(
                    m_assets, pluginFileName, falloutStringLanguage(),
                    FalloutStringFileKind::Strings, loaded, tableError)) {
                VOX_LOGW("streamer")
                    << "localized plugin " << pluginFileName << " has no readable "
                    << falloutStringLanguage() << " string table (" << tableError
                    << "); region names will read as single characters";
                // Cached as empty so a missing table is looked for once, not
                // once per region.
                table = tablesByPlugin.emplace(pluginFileName, FalloutStringTable{}).first;
            } else {
                table = tablesByPlugin.emplace(pluginFileName, std::move(loaded)).first;
            }
        }
        const std::string* text = table->second.find(stringId);
        if (text == nullptr || text->empty()) {
            ++missing;
            continue;
        }
        VOX_LOGD("streamer") << "region " << std::hex << formId << std::dec << " string "
                             << stringId << " -> \"" << *text << "\"";
        m_worldTables.regionNamesByFormId[formId] = *text;
        ++resolved;
    }
    // A localized plugin whose table never loaded would otherwise announce "h"
    // at the player with nothing in the log to say why.
    VOX_LOGI("streamer") << "localized region names: " << resolved << " resolved, " << missing
                         << " unresolved across " << tablesByPlugin.size() << " plugin(s)";
}

std::vector<std::string> CellStreamer::regionNamesAtEngineSpace(
    const float enginePosition[3]) const {
    std::vector<std::string> names;
    float fallout[3] = {};
    engineToFallout(enginePosition, fallout);

    // Which cell the position is in. Floors toward negative infinity: cell -1
    // spans [-4096, 0), so truncating would put every position in the strip
    // between -4096 and 0 into cell 0 and report the wrong region there.
    const auto cellOf = [this](float world) {
        return static_cast<std::int32_t>(std::floor(world / m_cellIndex.cellWorldSize));
    };
    const CellCoord coord{cellOf(fallout[0]), cellOf(fallout[1])};

    const auto found = m_availableCells.find(coord);
    if (found == m_availableCells.end() || found->second >= m_cellIndex.cells.size()) {
        return names;
    }
    for (const std::uint32_t regionFormId : m_cellIndex.cells[found->second].regionFormIds) {
        const auto named = m_worldTables.regionNamesByFormId.find(regionFormId);
        if (named != m_worldTables.regionNamesByFormId.end()) {
            names.push_back(named->second);
        }
    }
    return names;
}

CellStreamerStats CellStreamer::stats() const {
    CellStreamerStats snapshot = m_stats;
    snapshot.residency = m_planner.stats();
    snapshot.residentChunks = m_residentChunks.size();
    return snapshot;
}

}  // namespace odai::importer::fnv
