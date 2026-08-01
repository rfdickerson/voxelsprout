#include "world/chunk_mesh_scheduler.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace odai::world {

namespace {

const Chunk* findResidentChunk(const ChunkGrid& grid, ChunkMeshKey key) {
    for (const Chunk& chunk : grid.chunks()) {
        if (chunk.chunkX() == key.x && chunk.chunkY() == key.y && chunk.chunkZ() == key.z) {
            return &chunk;
        }
    }
    return nullptr;
}

} // namespace

ChunkMeshScheduler::ChunkMeshScheduler(core::JobSystem& jobs, MeshingOptions options)
    : m_jobs(jobs),
      m_options(options) {}

void ChunkMeshScheduler::setMeshingOptions(MeshingOptions options) {
    m_options = options;
    for (auto& [key, entry] : m_entries) {
        ++entry.generation;
        entry.state = ChunkMeshState::Dirty;
    }
}

void ChunkMeshScheduler::markDirty(ChunkMeshKey key) {
    Entry& entry = m_entries[key];
    ++entry.generation;
    entry.state = ChunkMeshState::Dirty;
}

std::size_t ChunkMeshScheduler::kickJobs(
    const ChunkGrid& grid,
    int cameraChunkX,
    int cameraChunkZ,
    std::size_t maxJobs
) {
    std::vector<ChunkMeshKey> dirtyKeys;
    dirtyKeys.reserve(m_entries.size());
    for (const auto& [key, entry] : m_entries) {
        if (entry.state == ChunkMeshState::Dirty) {
            dirtyKeys.push_back(key);
        }
    }
    if (dirtyKeys.empty()) {
        return 0;
    }

    std::sort(dirtyKeys.begin(), dirtyKeys.end(), [&](const ChunkMeshKey& a, const ChunkMeshKey& b) {
        const auto chebyshevDistance = [&](const ChunkMeshKey& key) {
            const int dx = std::abs(key.x - cameraChunkX);
            const int dz = std::abs(key.z - cameraChunkZ);
            return std::max(dx, dz);
        };
        const int distanceA = chebyshevDistance(a);
        const int distanceB = chebyshevDistance(b);
        if (distanceA != distanceB) {
            return distanceA < distanceB;
        }
        if (a.x != b.x) {
            return a.x < b.x;
        }
        if (a.y != b.y) {
            return a.y < b.y;
        }
        return a.z < b.z;
    });

    std::size_t launchedJobCount = 0;
    for (const ChunkMeshKey& key : dirtyKeys) {
        if (launchedJobCount >= maxJobs) {
            break;
        }
        const Chunk* residentChunk = findResidentChunk(grid, key);
        if (residentChunk == nullptr) {
            // Evicted while dirty: forget it rather than meshing a ghost.
            m_entries.erase(key);
            continue;
        }

        Entry& entry = m_entries[key];
        entry.state = ChunkMeshState::Meshing;
        const std::uint64_t generation = entry.generation;
        const MeshingOptions options = m_options;

        // Snapshot by copy: the grid chunk stays mutable on the main thread.
        m_jobs.enqueue([this, key, generation, options, snapshot = *residentChunk]() {
            ChunkMeshResult result;
            result.key = key;
            result.generation = generation;
            result.meshes = buildChunkLodMeshes(snapshot, options, &result.stats);
            {
                std::lock_guard<std::mutex> lock(m_readyMutex);
                m_readyQueue.push_back(std::move(result));
            }
        });
        ++launchedJobCount;
    }
    return launchedJobCount;
}

std::vector<ChunkMeshResult> ChunkMeshScheduler::drainReady(const ChunkGrid& grid) {
    std::vector<ChunkMeshResult> completed;
    {
        std::lock_guard<std::mutex> lock(m_readyMutex);
        completed.swap(m_readyQueue);
    }
    if (completed.empty()) {
        return completed;
    }

    std::vector<ChunkMeshResult> current;
    current.reserve(completed.size());
    for (ChunkMeshResult& result : completed) {
        const auto entryIt = m_entries.find(result.key);
        if (entryIt == m_entries.end()) {
            continue;
        }
        if (entryIt->second.generation != result.generation) {
            // Edited (or options changed) while meshing: markDirty already set
            // the state back to Dirty, so the chunk is re-kicked next frame.
            continue;
        }
        if (findResidentChunk(grid, result.key) == nullptr) {
            m_entries.erase(entryIt);
            continue;
        }
        entryIt->second.state = ChunkMeshState::Clean;
        current.push_back(std::move(result));
    }
    return current;
}

std::size_t ChunkMeshScheduler::dirtyCount() const {
    std::size_t count = 0;
    for (const auto& [key, entry] : m_entries) {
        if (entry.state == ChunkMeshState::Dirty) {
            ++count;
        }
    }
    return count;
}

std::size_t ChunkMeshScheduler::meshingCount() const {
    std::size_t count = 0;
    for (const auto& [key, entry] : m_entries) {
        if (entry.state == ChunkMeshState::Meshing) {
            ++count;
        }
    }
    return count;
}

} // namespace odai::world
