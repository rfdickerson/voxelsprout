#include "games/voxelcraft/voxelcraft_streaming.h"

#include "world/chunk_grid.h"

#include <cmath>
#include <utility>

namespace odai::games::voxelcraft {

namespace {

int floorDivInt(int value, int divisor) {
    int quotient = value / divisor;
    const int remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        --quotient;
    }
    return quotient;
}

} // namespace

std::vector<odai::world::World::ChunkKey> computeDesiredChunkKeys(
    const odai::world::World::ChunkStreamingConfig& config,
    float worldX,
    float worldZ
) {
    const int centerChunkX = floorDivInt(static_cast<int>(std::floor(worldX)), odai::world::Chunk::kSizeX);
    const int centerChunkZ = floorDivInt(static_cast<int>(std::floor(worldZ)), odai::world::Chunk::kSizeZ);
    const int radiusX = config.radiusChunksX + kChunkPrefetchMarginChunks;
    const int radiusZ = config.radiusChunksZ + kChunkPrefetchMarginChunks;

    std::vector<odai::world::World::ChunkKey> keys;
    keys.reserve(static_cast<std::size_t>(((radiusX * 2) + 1) * ((radiusZ * 2) + 1)));
    for (int chunkZ = centerChunkZ - radiusZ; chunkZ <= centerChunkZ + radiusZ; ++chunkZ) {
        for (int chunkX = centerChunkX - radiusX; chunkX <= centerChunkX + radiusX; ++chunkX) {
            keys.push_back(odai::world::World::ChunkKey{chunkX, 0, chunkZ});
        }
    }
    return keys;
}

ChunkStreamingPipeline::~ChunkStreamingPipeline() {
    stop();
}

void ChunkStreamingPipeline::start() {
    if (m_worker.joinable()) {
        return;
    }
    m_shuttingDown = false;
    m_worker = std::thread(&ChunkStreamingPipeline::workerLoop, this);
}

void ChunkStreamingPipeline::stop() {
    if (!m_worker.joinable()) {
        return;
    }
    m_shuttingDown = true;
    m_wake.notify_all();
    m_worker.join();
}

void ChunkStreamingPipeline::requestChunk(const odai::world::World::ChunkKey& key) {
    std::lock_guard<std::mutex> lock(m_requestMutex);
    m_cancelled.erase(key);
    if (!m_requested.insert(key).second) {
        return;
    }
    m_pendingRequests.push_back(key);
    m_wake.notify_one();
}

void ChunkStreamingPipeline::cancelChunk(const odai::world::World::ChunkKey& key) {
    std::lock_guard<std::mutex> lock(m_requestMutex);
    m_cancelled.insert(key);
}

std::vector<ChunkStreamingPipeline::CompletedChunk> ChunkStreamingPipeline::drainCompleted() {
    std::vector<CompletedChunk> drained;
    std::lock_guard<std::mutex> lock(m_completedMutex);
    drained.reserve(m_completed.size());
    for (auto& completed : m_completed) {
        drained.push_back(std::move(completed));
    }
    m_completed.clear();
    return drained;
}

void ChunkStreamingPipeline::workerLoop() {
    for (;;) {
        odai::world::World::ChunkKey key{};
        bool cancelled = false;
        {
            std::unique_lock<std::mutex> lock(m_requestMutex);
            m_wake.wait(lock, [this] { return m_shuttingDown.load() || !m_pendingRequests.empty(); });
            if (m_shuttingDown.load()) {
                return;
            }
            key = m_pendingRequests.front();
            m_pendingRequests.pop_front();
            m_requested.erase(key);
            cancelled = m_cancelled.erase(key) > 0;
        }
        if (cancelled) {
            continue;
        }

        odai::world::Chunk chunk = odai::world::buildProceduralChunk(key.chunkX, key.chunkY, key.chunkZ);
        odai::world::ChunkLodMeshes meshes = odai::world::buildChunkLodMeshes(chunk);

        std::lock_guard<std::mutex> lock(m_completedMutex);
        m_completed.push_back(CompletedChunk{key, std::move(chunk), std::move(meshes)});
    }
}

} // namespace odai::games::voxelcraft
