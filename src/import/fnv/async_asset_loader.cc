#include "import/fnv/async_asset_loader.h"

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <utility>

#include "core/job_system.h"

namespace odai::importer::fnv {

namespace {

// Dedup key: the normalized Bethesda path, lowercased. Casing varies between
// the ESM, the NIF texture sets and the BSA index for the same file, so a
// raw-string key would start two loads for one asset and then hand the renderer
// two different names for it.
std::string loaderKey(AssetKind kind, const std::string& path) {
    std::string key = (kind == AssetKind::Mesh) ? normalizeModelPath(path) : normalizeTexturePath(path);
    for (char& c : key) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return key;
}

}  // namespace

// Shared between the loader and every in-flight job. Held by shared_ptr so a
// job that is still running when the loader is destroyed writes into a live
// object rather than a dangling one -- waitIdle() makes that impossible in
// normal use, but the ownership makes it impossible by construction.
struct AsyncAssetLoader::Shared {
    std::mutex mutex;
    std::condition_variable idle;
    std::vector<LoadedAsset> completed;
    // Keys the loader owes a result for: populated on request, erased only when
    // that result is DRAINED (or discarded), not when the job finishes.
    //
    // Erasing on job completion instead was a real bug, not just a flaky test:
    // if a load finished while further requests for the same asset were still
    // arriving, the next one saw an empty set and started a second identical
    // load. Under streaming that is a repeated BSA inflate of a texture the
    // loader already had in hand. Holding the key until delivery makes dedup
    // deterministic regardless of thread timing.
    std::unordered_map<std::string, std::uint32_t> pending;
    AsyncAssetLoaderStats stats;
};

AsyncAssetLoader::AsyncAssetLoader(const FalloutAssetSource& source, core::JobSystem& jobs)
    : m_source(source), m_jobs(jobs), m_shared(std::make_shared<Shared>()) {}

AsyncAssetLoader::~AsyncAssetLoader() {
    waitIdle();
}

bool AsyncAssetLoader::request(AssetKind kind, const std::string& path, std::uint64_t userData) {
    const std::string key = loaderKey(kind, path);

    {
        std::lock_guard<std::mutex> lock(m_shared->mutex);
        const auto existing = m_shared->pending.find(key);
        if (existing != m_shared->pending.end()) {
            ++m_shared->stats.deduplicatedRequests;
            return false;
        }
        m_shared->pending.emplace(key, m_generation);
        ++m_shared->stats.startedLoads;
        ++m_shared->stats.loadsInFlight;
    }

    // Captured by value: the job may outlive this call. `source` is a reference
    // to something the caller guarantees outlives the loader, and is only read.
    const FalloutAssetSource* source = &m_source;
    std::shared_ptr<Shared> shared = m_shared;
    const std::uint32_t generation = m_generation;
    m_jobs.enqueue([shared, source, kind, key, userData, generation]() {
        LoadedAsset asset;
        asset.kind = kind;
        asset.key = key;
        asset.userData = userData;
        asset.generation = generation;
        asset.succeeded = (kind == AssetKind::Mesh)
            ? source->resolveMesh(key, asset.bytes, asset.error)
            : source->resolveTexture(key, asset.bytes, asset.error);
        if (!asset.succeeded) {
            asset.bytes.clear();
        }

        std::lock_guard<std::mutex> lock(shared->mutex);
        // NOTE: shared->pending keeps this key until drainCompleted takes the
        // result. Only the in-flight *count* drops here.
        if (asset.succeeded) {
            ++shared->stats.completedLoads;
        } else {
            ++shared->stats.failedLoads;
        }
        if (shared->stats.loadsInFlight > 0u) {
            --shared->stats.loadsInFlight;
        }
        shared->completed.push_back(std::move(asset));
        if (shared->stats.loadsInFlight == 0u) {
            shared->idle.notify_all();
        }
    });
    return true;
}

void AsyncAssetLoader::drainCompleted(std::vector<LoadedAsset>& outAssets) {
    std::vector<LoadedAsset> drained;
    {
        std::lock_guard<std::mutex> lock(m_shared->mutex);
        drained.swap(m_shared->completed);
    }

    std::uint64_t discarded = 0;
    std::vector<std::string> deliveredKeys;
    deliveredKeys.reserve(drained.size());
    for (LoadedAsset& asset : drained) {
        deliveredKeys.push_back(asset.key);
        if (asset.generation != m_generation) {
            ++discarded;  // requested before the world moved on; not worth applying
            continue;
        }
        outAssets.push_back(std::move(asset));
    }
    if (!deliveredKeys.empty()) {
        std::lock_guard<std::mutex> lock(m_shared->mutex);
        // The loader no longer owes anything for these keys. A caller that
        // wants one again -- because it evicted the asset and came back -- will
        // correctly start a fresh load.
        for (const std::string& key : deliveredKeys) {
            m_shared->pending.erase(key);
        }
        m_shared->stats.discardedResults += discarded;
    }
}

AsyncAssetLoaderStats AsyncAssetLoader::stats() const {
    std::lock_guard<std::mutex> lock(m_shared->mutex);
    return m_shared->stats;
}

void AsyncAssetLoader::waitIdle() {
    std::unique_lock<std::mutex> lock(m_shared->mutex);
    m_shared->idle.wait(lock, [this]() { return m_shared->stats.loadsInFlight == 0u; });
}

}  // namespace odai::importer::fnv
