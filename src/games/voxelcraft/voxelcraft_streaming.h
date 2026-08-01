#pragma once

#include "world/chunk.h"
#include "world/chunk_mesher.h"
#include "world/world.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

// VoxelCraft async chunk streaming pipeline
// Responsible for: generating + meshing chunks off the main thread so
// World::updateStreamingWindowForWorldPosition never blocks on buildProceduralChunk /
// buildChunkLodMeshes while the player is moving.
// Should NOT do: touch World/ChunkClipmapIndex directly (both stay main-thread-only; the
// caller inserts completed chunks via World::insertGeneratedChunk after draining).
namespace odai::games::voxelcraft {

// One chunk beyond World's own resident streaming radius, so the async pipeline tends to
// finish generating a chunk before World's (hysteresis-smoothed) window actually needs it.
constexpr int kChunkPrefetchMarginChunks = 1;

// The set of chunk keys the streaming pipeline should have generated, given a world
// position and World's own streaming radius config, plus a prefetch margin. Pure function
// (no World/pipeline state) so it's independently testable -- see
// tests/foundation_tests.cc's VoxelCraftStreaming cases.
[[nodiscard]] std::vector<odai::world::World::ChunkKey> computeDesiredChunkKeys(
    const odai::world::World::ChunkStreamingConfig& config,
    float worldX,
    float worldZ
);

class ChunkStreamingPipeline {
public:
    struct CompletedChunk {
        odai::world::World::ChunkKey key{};
        odai::world::Chunk chunk;
        odai::world::ChunkLodMeshes meshes;
    };

    ChunkStreamingPipeline() = default;
    ~ChunkStreamingPipeline();
    ChunkStreamingPipeline(const ChunkStreamingPipeline&) = delete;
    ChunkStreamingPipeline& operator=(const ChunkStreamingPipeline&) = delete;

    void start();
    void stop();

    // No-op if this key is already requested/in-flight.
    void requestChunk(const odai::world::World::ChunkKey& key);
    // No-op if the job already started generating (the pipeline still hands back a result
    // the caller can simply discard as out-of-radius); otherwise drops the pending request.
    void cancelChunk(const odai::world::World::ChunkKey& key);

    // Non-blocking. Call once per frame on the main thread.
    [[nodiscard]] std::vector<CompletedChunk> drainCompleted();

private:
    void workerLoop();

    std::thread m_worker;
    std::atomic<bool> m_shuttingDown{false};

    std::mutex m_requestMutex;
    std::condition_variable m_wake;
    std::deque<odai::world::World::ChunkKey> m_pendingRequests;
    std::unordered_set<odai::world::World::ChunkKey, odai::world::World::ChunkKeyHash> m_requested;
    std::unordered_set<odai::world::World::ChunkKey, odai::world::World::ChunkKeyHash> m_cancelled;

    std::mutex m_completedMutex;
    std::deque<CompletedChunk> m_completed;
};

} // namespace odai::games::voxelcraft
