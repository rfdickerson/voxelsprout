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
constexpr int kCellBuildVersion = 6;

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

std::string pluginFingerprint(const std::filesystem::path& esmPath) {
    std::error_code sizeError;
    const auto size = std::filesystem::file_size(esmPath, sizeError);
    std::error_code timeError;
    const auto writeTime = std::filesystem::last_write_time(esmPath, timeError);
    const auto ticks = writeTime.time_since_epoch().count();
    return std::to_string(sizeError ? 0ull : static_cast<std::uint64_t>(size)) + "_" +
           std::to_string(timeError ? 0ll : static_cast<long long>(ticks)) + "_v" +
           std::to_string(kCellBuildVersion);
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
    for (const std::string& warning : m_assets.warnings()) {
        VOX_LOGW("streamer") << warning;
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
            m_cacheDirectory / pluginFingerprint(m_esmPath) / loweredForPath;
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
        m_jobs->enqueue([pending, esmPath, entry, cell, assets, tables, cachePath, textureCache]() {
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
    for (Pending::Result& result : drained) {
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
        VOX_LOGI("streamer") << "cell " << result.cell.x << "," << result.cell.z
                             << " chunk add took " << chunkMs << " ms ("
                             << result.scene.packedVertices.size() << " verts, "
                             << result.scene.textures.size() << " textures)";
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
    for (std::int32_t ring = 0; ring <= kMaxSpawnSearchRings && !haveHeight; ++ring) {
        for (std::int32_t dz = -ring; dz <= ring && !haveHeight; ++dz) {
            for (std::int32_t dx = -ring; dx <= ring && !haveHeight; ++dx) {
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
                chosen = candidate;
                terrainHeight = *std::max_element(
                    std::begin(record.land->heights), std::end(record.land->heights));
                haveHeight = true;
            }
        }
    }

    if (!haveHeight) {
        VOX_LOGW("streamer") << "spawn: no cell with terrain within " << kMaxSpawnSearchRings
                             << " rings of the centre; falling back to a guessed height";
    } else {
        VOX_LOGI("streamer") << "spawn: cell " << chosen.x << "," << chosen.z
                             << " peak terrain height " << terrainHeight;
    }

    const float cellSize = m_planner.config().cellSize;
    float fallout[3] = {
        (static_cast<float>(chosen.x) + 0.5f) * cellSize,
        (static_cast<float>(chosen.z) + 0.5f) * cellSize,
        0.0f};
    // Clear of the cell's HIGHEST post, not its average, so a ridge running
    // through the cell does not swallow the camera.
    constexpr float kSpawnClearanceUnits = 600.0f;
    fallout[2] = (haveHeight ? terrainHeight : 12000.0f) + kSpawnClearanceUnits;

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

CellStreamerStats CellStreamer::stats() const {
    CellStreamerStats snapshot = m_stats;
    snapshot.residency = m_planner.stats();
    snapshot.residentChunks = m_residentChunks.size();
    return snapshot;
}

}  // namespace odai::importer::fnv
