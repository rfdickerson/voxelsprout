#pragma once

#include "core/job_system.h"
#include "world/chunk_grid.h"
#include "world/chunk_mesher.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

// Off-thread chunk meshing with explicit lifecycle states.
//
// Each resident chunk moves through Clean -> Dirty -> Meshing -> Ready; the
// scheduler owns that state machine on the main thread and uses a
// core::JobSystem purely as an executor. Workers never see the live ChunkGrid:
// kickJobs() snapshots each chunk BY COPY before enqueueing, because the main
// thread keeps mutating voxels/macro cells (App::applyVoxelEdit) and streaming
// can reallocate the grid's chunk vector while a mesh is in flight.
//
// A per-chunk generation counter resolves edit-during-mesh races: markDirty()
// on a Meshing chunk bumps the generation, and drainReady() discards results
// whose generation no longer matches (the chunk stays Dirty and is re-kicked).
//
// Pure CPU, no Vulkan. Main-thread only, except the worker-side ready queue.
namespace odai::world {

enum class ChunkMeshState : std::uint8_t {
    Clean = 0,
    Dirty = 1,
    Meshing = 2
};

// Throughput and wasted-work counters, cumulative since construction or the
// last resetStats().
//
// The scheduler can finish meshing a chunk and then throw the result away: the
// chunk was edited mid-mesh (generation bumped), or it left the resident window
// before the worker finished. That is real worker time spent on output nobody
// ever sees, and without these counters it is invisible -- the frame just looks
// busy. A high wastedFraction() means the fix is scheduling policy (kick fewer
// jobs, kick them later, coalesce edits), not a faster mesher.
struct ChunkMeshSchedulerStats {
    std::uint64_t jobsLaunched = 0;
    std::uint64_t resultsAccepted = 0;
    // Edited, or meshing options changed, while the mesh was in flight.
    std::uint64_t discardedStale = 0;
    // Chunk left the resident window while the mesh was in flight.
    std::uint64_t discardedEvicted = 0;
    // Entry disappeared entirely before the result came back.
    std::uint64_t discardedUntracked = 0;
    // Evicted while still Dirty: dropped before a job was ever launched, so it
    // cost no worker time. Tracked separately for exactly that reason.
    std::uint64_t droppedDirtyBeforeKick = 0;

    float meshMsAccepted = 0.0f;
    float meshMsWasted = 0.0f;

    [[nodiscard]] std::uint64_t discardedTotal() const {
        return discardedStale + discardedEvicted + discardedUntracked;
    }

    // Share of completed worker time that produced nothing usable, in [0, 1].
    [[nodiscard]] float wastedFraction() const {
        const float total = meshMsAccepted + meshMsWasted;
        return total > 0.0f ? (meshMsWasted / total) : 0.0f;
    }
};

class ChunkMeshScheduler {
public:
    ChunkMeshScheduler(core::JobSystem& jobs, MeshingOptions options);

    // Dirties every tracked chunk and bumps all generations so in-flight
    // results built with the old options are discarded on drain.
    void setMeshingOptions(MeshingOptions options);
    [[nodiscard]] MeshingOptions meshingOptions() const { return m_options; }

    void markDirty(ChunkMeshKey key);

    // Snapshots up to maxJobs Dirty chunks (nearest to the camera first, same
    // Chebyshev ordering the renderer's remesh queue used) and enqueues mesh
    // jobs for them. Dirty chunks not resident in `grid` are dropped.
    // Returns the number of jobs launched. Main thread only.
    std::size_t kickJobs(
        const ChunkGrid& grid,
        int cameraChunkX,
        int cameraChunkZ,
        std::size_t maxJobs
    );

    // Moves out completed results that are still current: stale generations
    // are discarded (their chunk stays Dirty for the next kick) and results
    // for chunks no longer resident in `grid` are dropped. Main thread only.
    std::vector<ChunkMeshResult> drainReady(const ChunkGrid& grid);

    [[nodiscard]] std::size_t dirtyCount() const;
    [[nodiscard]] std::size_t meshingCount() const;

    // Main thread only, like the rest of the class: every counter is updated in
    // kickJobs()/drainReady(), never on a worker.
    [[nodiscard]] const ChunkMeshSchedulerStats& stats() const { return m_stats; }
    void resetStats() { m_stats = ChunkMeshSchedulerStats{}; }

private:
    struct Entry {
        ChunkMeshState state = ChunkMeshState::Clean;
        std::uint64_t generation = 0;
    };

    core::JobSystem& m_jobs;
    MeshingOptions m_options;
    ChunkMeshSchedulerStats m_stats{};
    // Main-thread only.
    std::unordered_map<ChunkMeshKey, Entry, ChunkMeshKeyHash> m_entries;
    // Worker -> main-thread handoff.
    std::mutex m_readyMutex;
    std::vector<ChunkMeshResult> m_readyQueue;
};

} // namespace odai::world
