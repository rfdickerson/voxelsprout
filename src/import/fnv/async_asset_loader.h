#pragma once

// Loads Fallout assets on background threads, never twice at once.
//
// Modelled on world::ChunkMeshScheduler's contract, deliberately: request() and
// drainCompleted() are main-thread only, the work happens on a core::JobSystem,
// and a generation counter lets results that are no longer wanted be discarded
// rather than applied. Reusing that shape rather than inventing a second async
// idiom is the point.
//
// What this is NOT: a cache of decoded assets. GPU texture residency is already
// reference counted by path inside the renderer (see render/bindless_slot_table.h),
// and duplicating that here would mean two refcounts to keep in step. This only
// guarantees that N cells wanting the same texture in the same frame produce ONE
// background load, and that its bytes arrive on the main thread exactly once.

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "import/fnv/asset_source.h"

namespace odai {
namespace core {
class JobSystem;
}
namespace importer::fnv {

enum class AssetKind : std::uint8_t {
    Mesh,
    Texture,
};

struct LoadedAsset {
    AssetKind kind = AssetKind::Mesh;
    // The normalized path this was requested under. Callers key their own
    // structures on this, and it matches what the renderer uses for texture
    // dedup after the same lowercasing.
    std::string key;
    std::vector<std::uint8_t> bytes;
    // Whatever the requester passed to request(), handed straight back. The
    // streamer uses it to attribute a completed load to the cell that wanted it
    // without keeping a side table.
    std::uint64_t userData = 0;
    // Generation the request was made in. Compare against currentGeneration()
    // to drop results for work that has since been abandoned.
    std::uint32_t generation = 0;
    bool succeeded = false;
    std::string error;
};

struct AsyncAssetLoaderStats {
    // Requests that found a load already in flight for the same key and did not
    // start a second one. On a real walk this should dominate: adjacent cells
    // share most of their meshes and textures.
    std::uint64_t deduplicatedRequests = 0;
    std::uint64_t startedLoads = 0;
    std::uint64_t completedLoads = 0;
    std::uint64_t failedLoads = 0;
    // Results dropped because their generation was stale by the time they
    // arrived -- wasted work, counted rather than hidden.
    std::uint64_t discardedResults = 0;
    std::uint32_t loadsInFlight = 0;
};

class AsyncAssetLoader {
public:
    // Neither pointer is owned; both must outlive this loader. `source` must
    // already be open, and is only ever read from worker threads.
    AsyncAssetLoader(const FalloutAssetSource& source, core::JobSystem& jobs);
    ~AsyncAssetLoader();

    AsyncAssetLoader(const AsyncAssetLoader&) = delete;
    AsyncAssetLoader& operator=(const AsyncAssetLoader&) = delete;

    // MAIN THREAD ONLY. Enqueues a load unless the loader already owes a result
    // for the same key. Returns true if a new job was started, false if the
    // request was folded into a pending one (the common, desirable case).
    //
    // "Pending" spans request until drainCompleted delivers the result, NOT just
    // while the job runs -- otherwise a load finishing mid-burst would let the
    // next request start a duplicate of an asset already in hand.
    bool request(AssetKind kind, const std::string& path, std::uint64_t userData = 0);

    // MAIN THREAD ONLY. Moves every finished load into `outAssets`, appending.
    // Results whose generation is older than currentGeneration() are dropped and
    // counted as discarded rather than returned.
    void drainCompleted(std::vector<LoadedAsset>& outAssets);

    // MAIN THREAD ONLY. Abandons interest in everything requested so far:
    // in-flight jobs still run to completion (they cannot be cancelled mid-read)
    // but their results are discarded on arrival. Used when the player teleports
    // and the entire pending set becomes irrelevant.
    void bumpGeneration() { ++m_generation; }
    [[nodiscard]] std::uint32_t currentGeneration() const { return m_generation; }

    [[nodiscard]] AsyncAssetLoaderStats stats() const;

    // MAIN THREAD ONLY. Blocks until every in-flight load has finished and been
    // drained away. Required before destroying the loader or the job system.
    void waitIdle();

private:
    struct Shared;

    const FalloutAssetSource& m_source;
    core::JobSystem& m_jobs;
    // Held by the loader and by every in-flight job, so a job that outlives a
    // drain still has somewhere valid to put its result.
    std::shared_ptr<Shared> m_shared;
    std::uint32_t m_generation = 1;
};

}  // namespace importer::fnv
}  // namespace odai
