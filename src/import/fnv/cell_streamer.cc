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
constexpr int kCellBuildVersion = 12;

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
    std::uint32_t maxTextureSize) {
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

    if (!buildFalloutWorldTables(m_esmPath, m_worldTables, outError)) {
        return false;
    }
    if (!buildFalloutCellIndex(m_esmPath, m_cellIndex, outError)) {
        return false;
    }

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
            pluginFingerprint(m_esmPath, m_assets.modFingerprint(), m_maxTextureSize) /
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
                         maxTextureSize]() {
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
            if (!reader.open(esmPath)) {
                result.error = reader.lastError();
            } else {
                FalloutCellRecord record;
                if (!extractFalloutCellAt(reader, entry, record, result.error)) {
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
        m_stats.blendedPartsLoaded += result.blendedParts;
        if (m_onCellResident) {
            // Before result.scene is destroyed at the end of this loop.
            m_onCellResident(result.cell, result.scene);
        }
    }

    m_stats.lastApplyMs = applyTimer.elapsedMs();
    m_stats.worstApplyMs = std::max(m_stats.worstApplyMs, m_stats.lastApplyMs);
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

    EsmReader reader;
    if (!reader.open(m_esmPath)) {
        return false;
    }
    FalloutCellRecord record;
    std::string error;
    if (!extractFalloutCellAt(reader, *interior, record, error)) {
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
    const bool readerOpen = reader.open(m_esmPath);
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
                if (!extractFalloutCellAt(reader, m_cellIndex.cells[found->second], record, error) ||
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

std::vector<std::string> CellStreamer::regionNamesAtEngineSpace(
    const float enginePosition[3]) const {
    std::vector<std::string> names;
    float fallout[3] = {};
    engineToFallout(enginePosition, fallout);

    // Which cell the position is in. Floors toward negative infinity: cell -1
    // spans [-4096, 0), so truncating would put every position in the strip
    // between -4096 and 0 into cell 0 and report the wrong region there.
    const auto cellOf = [](float world) {
        return static_cast<std::int32_t>(std::floor(world / kExteriorCellSize));
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
