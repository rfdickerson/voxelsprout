#include "core/job_system.h"
#include "world/chunk_grid.h"
#include "world/chunk_mesh_scheduler.h"
#include "world/chunk_mesher.h"
#include "world/grass_scatter.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        ++g_failures;
        std::cerr << "[chunk mesh scheduler test] FAILED: " << message << "\n";
    }
}

odai::world::Chunk makeSlabChunk(int chunkX, int chunkY, int chunkZ, int slabY) {
    odai::world::Chunk chunk(chunkX, chunkY, chunkZ);
    for (int z = 0; z < odai::world::Chunk::kSizeZ; ++z) {
        for (int x = 0; x < odai::world::Chunk::kSizeX; ++x) {
            chunk.setVoxel(x, slabY, z, odai::world::Voxel{odai::world::VoxelType::Solid});
        }
    }
    return chunk;
}

odai::world::ChunkGrid makeGrid(std::vector<odai::world::Chunk> chunks) {
    odai::world::ChunkGrid grid;
    grid.setChunks(std::move(chunks));
    return grid;
}

odai::world::ChunkMeshKey keyFor(const odai::world::Chunk& chunk) {
    return odai::world::ChunkMeshKey{chunk.chunkX(), chunk.chunkY(), chunk.chunkZ()};
}

bool meshesEqual(const odai::world::ChunkLodMeshes& a, const odai::world::ChunkLodMeshes& b) {
    for (std::uint32_t lod = 0; lod < odai::world::kChunkMeshLodCount; ++lod) {
        const odai::world::ChunkMeshData& meshA = a.lodMeshes[lod];
        const odai::world::ChunkMeshData& meshB = b.lodMeshes[lod];
        if (meshA.vertices.size() != meshB.vertices.size() || meshA.indices.size() != meshB.indices.size()) {
            return false;
        }
        for (std::size_t i = 0; i < meshA.vertices.size(); ++i) {
            if (meshA.vertices[i].bits != meshB.vertices[i].bits) {
                return false;
            }
        }
        for (std::size_t i = 0; i < meshA.indices.size(); ++i) {
            if (meshA.indices[i] != meshB.indices[i]) {
                return false;
            }
        }
    }
    return true;
}

// Synchronous JobSystem: kickJobs runs the mesh job inline, so results are
// deterministic and already in the ready queue when kickJobs returns.
void testLifecycleAndResultEquivalence() {
    odai::core::JobSystem jobs(0);
    odai::world::ChunkMeshScheduler scheduler(jobs, odai::world::MeshingOptions{});

    odai::world::ChunkGrid grid = makeGrid({makeSlabChunk(0, 0, 0, 7)});
    scheduler.markDirty(keyFor(grid.chunks()[0]));
    expectTrue(scheduler.dirtyCount() == 1, "markDirty leaves the chunk Dirty");

    const std::size_t launched = scheduler.kickJobs(grid, 0, 0, 8);
    expectTrue(launched == 1, "kickJobs launches one job for one dirty chunk");
    expectTrue(scheduler.dirtyCount() == 0, "kicked chunk left the Dirty state");

    std::vector<odai::world::ChunkMeshResult> results = scheduler.drainReady(grid);
    expectTrue(results.size() == 1, "drainReady returns the finished result");
    expectTrue(scheduler.meshingCount() == 0, "drained chunk returned to Clean");

    const odai::world::ChunkLodMeshes direct =
        odai::world::buildChunkLodMeshes(grid.chunks()[0], odai::world::MeshingOptions{});
    expectTrue(meshesEqual(results[0].meshes, direct), "scheduled mesh matches direct mesher output");
    expectTrue(
        results[0].stats.exposedFaceCount * 4u ==
            odai::world::buildChunkLodMeshes(grid.chunks()[0], odai::world::MeshingOptions{odai::world::MeshingMode::Naive})
                .lodMeshes[0]
                .vertices.size(),
        "result stats carry the naive-equivalent face count"
    );

    // Nothing dirty: no work.
    expectTrue(scheduler.kickJobs(grid, 0, 0, 8) == 0, "clean scheduler launches nothing");
    expectTrue(scheduler.drainReady(grid).empty(), "clean scheduler drains nothing");
}

void testStaleGenerationDiscarded() {
    odai::core::JobSystem jobs(0);
    odai::world::ChunkMeshScheduler scheduler(jobs, odai::world::MeshingOptions{});

    odai::world::ChunkGrid grid = makeGrid({makeSlabChunk(0, 0, 0, 7)});
    const odai::world::ChunkMeshKey key = keyFor(grid.chunks()[0]);
    scheduler.markDirty(key);
    scheduler.kickJobs(grid, 0, 0, 8);
    // Edit lands after the mesh finished but before the drain: the stored
    // result's generation is now stale and must be discarded.
    scheduler.markDirty(key);

    const std::vector<odai::world::ChunkMeshResult> results = scheduler.drainReady(grid);
    expectTrue(results.empty(), "stale-generation result is discarded");
    expectTrue(scheduler.dirtyCount() == 1, "chunk stays Dirty for the next kick");

    scheduler.kickJobs(grid, 0, 0, 8);
    expectTrue(scheduler.drainReady(grid).size() == 1, "re-kicked chunk delivers a fresh result");
}

void testMeshingOptionsChangeInvalidatesInFlight() {
    odai::core::JobSystem jobs(0);
    odai::world::ChunkMeshScheduler scheduler(jobs, odai::world::MeshingOptions{odai::world::MeshingMode::Greedy});

    odai::world::ChunkGrid grid = makeGrid({makeSlabChunk(0, 0, 0, 7)});
    scheduler.markDirty(keyFor(grid.chunks()[0]));
    scheduler.kickJobs(grid, 0, 0, 8);
    scheduler.setMeshingOptions(odai::world::MeshingOptions{odai::world::MeshingMode::Naive});

    expectTrue(scheduler.drainReady(grid).empty(), "old-mode result discarded after options change");
    scheduler.kickJobs(grid, 0, 0, 8);
    const std::vector<odai::world::ChunkMeshResult> results = scheduler.drainReady(grid);
    expectTrue(results.size() == 1, "new-mode result delivered");
    const odai::world::ChunkLodMeshes naiveDirect =
        odai::world::buildChunkLodMeshes(grid.chunks()[0], odai::world::MeshingOptions{odai::world::MeshingMode::Naive});
    expectTrue(meshesEqual(results[0].meshes, naiveDirect), "re-mesh used the new meshing options");
}

void testEvictedChunkDropped() {
    odai::core::JobSystem jobs(0);
    odai::world::ChunkMeshScheduler scheduler(jobs, odai::world::MeshingOptions{});

    odai::world::ChunkGrid grid = makeGrid({makeSlabChunk(0, 0, 0, 7), makeSlabChunk(1, 0, 0, 9)});
    scheduler.markDirty(keyFor(grid.chunks()[0]));
    scheduler.markDirty(keyFor(grid.chunks()[1]));
    scheduler.kickJobs(grid, 0, 0, 8);

    // Chunk (1,0,0) streams out before the drain.
    odai::world::ChunkGrid shrunkGrid = makeGrid({makeSlabChunk(0, 0, 0, 7)});
    const std::vector<odai::world::ChunkMeshResult> results = scheduler.drainReady(shrunkGrid);
    expectTrue(results.size() == 1, "only the still-resident chunk's result survives");
    expectTrue(results[0].key == odai::world::ChunkMeshKey{0, 0, 0}, "surviving result is the resident chunk");

    // Dirty chunks that stream out before their kick are forgotten too.
    scheduler.markDirty(odai::world::ChunkMeshKey{5, 0, 5});
    expectTrue(scheduler.kickJobs(shrunkGrid, 0, 0, 8) == 0, "evicted dirty chunk launches no job");
    expectTrue(scheduler.dirtyCount() == 0, "evicted dirty chunk is forgotten");
}

// The scheduler can finish a mesh and then throw it away. These counters are
// the only way that shows up as anything other than "the frame looks busy".
void testWastedWorkAccounting() {
    odai::core::JobSystem jobs(0);
    odai::world::ChunkMeshScheduler scheduler(jobs, odai::world::MeshingOptions{});

    odai::world::ChunkGrid grid = makeGrid({makeSlabChunk(0, 0, 0, 7)});
    const odai::world::ChunkMeshKey key = keyFor(grid.chunks()[0]);

    expectTrue(scheduler.stats().jobsLaunched == 0u, "a fresh scheduler has launched nothing");
    expectTrue(scheduler.stats().wastedFraction() == 0.0f, "no work means no wasted fraction");

    // A clean accept: launched, delivered, charged to accepted.
    scheduler.markDirty(key);
    scheduler.kickJobs(grid, 0, 0, 8);
    expectTrue(scheduler.stats().jobsLaunched == 1u, "kickJobs counts the launch");
    expectTrue(scheduler.drainReady(grid).size() == 1u, "the result is delivered");
    expectTrue(scheduler.stats().resultsAccepted == 1u, "an accepted result is counted");
    expectTrue(scheduler.stats().discardedTotal() == 0u, "nothing was discarded yet");

    // Edit-during-mesh: the finished mesh is thrown away and charged as waste.
    scheduler.markDirty(key);
    scheduler.kickJobs(grid, 0, 0, 8);
    scheduler.markDirty(key);
    expectTrue(scheduler.drainReady(grid).empty(), "the stale result is discarded");
    expectTrue(scheduler.stats().discardedStale == 1u, "a stale discard is attributed to staleness");
    expectTrue(scheduler.stats().discardedTotal() == 1u, "exactly one discard so far");
    expectTrue(scheduler.stats().resultsAccepted == 1u, "a discard is not counted as accepted");

    // Timings are wall-clock, so assert the accounting relationship rather than
    // any particular duration: every completed job lands in exactly one bucket.
    const auto& s = scheduler.stats();
    expectTrue(s.meshMsAccepted >= 0.0f && s.meshMsWasted >= 0.0f, "mesh timings are non-negative");
    expectTrue(s.wastedFraction() >= 0.0f && s.wastedFraction() <= 1.0f, "wasted fraction is a ratio");

    scheduler.resetStats();
    expectTrue(scheduler.stats().jobsLaunched == 0u, "resetStats clears the counters");
    expectTrue(scheduler.stats().discardedTotal() == 0u, "resetStats clears the discards");
}

// Eviction has two shapes and they must not be conflated: one burned worker
// time, the other never launched a job at all.
void testEvictionWasteIsSeparatedFromDroppedDirty() {
    odai::core::JobSystem jobs(0);
    odai::world::ChunkMeshScheduler scheduler(jobs, odai::world::MeshingOptions{});

    odai::world::ChunkGrid grid = makeGrid({makeSlabChunk(0, 0, 0, 7), makeSlabChunk(1, 0, 0, 9)});
    scheduler.markDirty(keyFor(grid.chunks()[0]));
    scheduler.markDirty(keyFor(grid.chunks()[1]));
    scheduler.kickJobs(grid, 0, 0, 8);

    // (1,0,0) meshed, then streamed out before the drain: worker time burned.
    odai::world::ChunkGrid shrunkGrid = makeGrid({makeSlabChunk(0, 0, 0, 7)});
    expectTrue(scheduler.drainReady(shrunkGrid).size() == 1u, "the resident result survives");
    expectTrue(scheduler.stats().discardedEvicted == 1u, "an evicted in-flight mesh is charged as waste");
    expectTrue(scheduler.stats().droppedDirtyBeforeKick == 0u, "nothing was dropped before a kick yet");

    // A chunk that streams out while still Dirty never reaches a worker, so it
    // must not inflate the waste numbers.
    scheduler.markDirty(odai::world::ChunkMeshKey{5, 0, 5});
    expectTrue(scheduler.kickJobs(shrunkGrid, 0, 0, 8) == 0, "evicted dirty chunk launches no job");
    expectTrue(scheduler.stats().droppedDirtyBeforeKick == 1u, "dropped-before-kick is counted separately");
    expectTrue(scheduler.stats().discardedEvicted == 1u, "dropping before a kick adds no eviction waste");
}

void testNearestFirstKickOrderAndBudget() {
    odai::core::JobSystem jobs(0);
    odai::world::ChunkMeshScheduler scheduler(jobs, odai::world::MeshingOptions{});

    std::vector<odai::world::Chunk> chunks;
    for (int x = 0; x < 5; ++x) {
        chunks.push_back(makeSlabChunk(x, 0, 0, 6));
    }
    odai::world::ChunkGrid grid = makeGrid(std::move(chunks));
    for (const odai::world::Chunk& chunk : grid.chunks()) {
        scheduler.markDirty(keyFor(chunk));
    }

    // Camera at chunk x=4: budget of 2 must pick x=4 then x=3.
    expectTrue(scheduler.kickJobs(grid, 4, 0, 2) == 2, "budget bounds launched jobs");
    std::vector<odai::world::ChunkMeshResult> results = scheduler.drainReady(grid);
    expectTrue(results.size() == 2, "both budgeted jobs completed");
    bool sawNearest = false;
    bool sawSecondNearest = false;
    for (const odai::world::ChunkMeshResult& result : results) {
        sawNearest = sawNearest || result.key == odai::world::ChunkMeshKey{4, 0, 0};
        sawSecondNearest = sawSecondNearest || result.key == odai::world::ChunkMeshKey{3, 0, 0};
    }
    expectTrue(sawNearest && sawSecondNearest, "nearest chunks to the camera are kicked first");
    expectTrue(scheduler.dirtyCount() == 3, "chunks past the budget stay Dirty");
}

void testGrassDeterminism() {
    odai::world::Chunk chunk(0, 0, 0);
    for (int z = 0; z < odai::world::Chunk::kSizeZ; ++z) {
        for (int x = 0; x < odai::world::Chunk::kSizeX; ++x) {
            chunk.setVoxel(x, 5, z, odai::world::Voxel{odai::world::VoxelType::Grass});
        }
    }
    odai::world::GrassScatterParams params{};
    const std::vector<odai::world::GrassInstance> a = odai::world::buildGrassInstances(chunk, params);
    const std::vector<odai::world::GrassInstance> b = odai::world::buildGrassInstances(chunk, params);
    expectTrue(!a.empty(), "grass scatter produced instances on a grass slab");
    bool identical = a.size() == b.size();
    for (std::size_t i = 0; identical && i < a.size(); ++i) {
        for (int component = 0; component < 4; ++component) {
            identical = identical &&
                a[i].worldPosYaw[component] == b[i].worldPosYaw[component] &&
                a[i].colorTint[component] == b[i].colorTint[component];
        }
    }
    expectTrue(identical, "grass scatter is deterministic for identical inputs");

    odai::world::GrassScatterParams farParams{};
    farParams.residentCenterChunkX = 10;
    farParams.residentCenterChunkZ = 10;
    expectTrue(
        odai::world::buildGrassInstances(chunk, farParams).empty(),
        "chunks outside the active radius grow no grass"
    );
}

void testThreadedStress() {
    odai::core::JobSystem jobs(4);
    odai::world::ChunkMeshScheduler scheduler(jobs, odai::world::MeshingOptions{});

    std::vector<odai::world::Chunk> chunks;
    for (int x = -3; x <= 3; ++x) {
        for (int z = -3; z <= 3; ++z) {
            chunks.push_back(makeSlabChunk(x, 0, z, 4 + ((x + z + 6) % 8)));
        }
    }
    odai::world::ChunkGrid grid = makeGrid(std::move(chunks));

    std::mt19937 rng(12345);
    std::uniform_int_distribution<std::size_t> pickChunk(0, grid.chunks().size() - 1);
    std::size_t deliveredResultCount = 0;
    for (int frame = 0; frame < 200; ++frame) {
        for (int dirtyIndex = 0; dirtyIndex < 3; ++dirtyIndex) {
            scheduler.markDirty(keyFor(grid.chunks()[pickChunk(rng)]));
        }
        scheduler.kickJobs(grid, 0, 0, 8);
        deliveredResultCount += scheduler.drainReady(grid).size();
    }
    jobs.waitIdle();
    deliveredResultCount += scheduler.drainReady(grid).size();
    expectTrue(deliveredResultCount > 0, "threaded stress delivered results");
    expectTrue(scheduler.meshingCount() == 0, "no chunk left stuck in Meshing after final drain");
}

} // namespace

int main() {
    testLifecycleAndResultEquivalence();
    testStaleGenerationDiscarded();
    testMeshingOptionsChangeInvalidatesInFlight();
    testEvictedChunkDropped();
    testWastedWorkAccounting();
    testEvictionWasteIsSeparatedFromDroppedDirty();
    testNearestFirstKickOrderAndBudget();
    testGrassDeterminism();
    testThreadedStress();

    if (g_failures != 0) {
        std::cerr << "[chunk mesh scheduler test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[chunk mesh scheduler test] all checks passed\n";
    return 0;
}
