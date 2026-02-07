#pragma once

#include <cstddef>
#include <vector>

#include "world/Chunk.hpp"
#include "world/Voxel.hpp"

// World ChunkGrid subsystem
// Responsible for: owning a collection of chunks that represent world space.
// Should NOT do: pathfinding, factory simulation, or rendering API calls.
namespace world {

class ChunkGrid {
public:
    ChunkGrid() = default;
    void initializeFlatWorld();
    std::size_t chunkCount() const;
    const std::vector<Chunk>& chunks() const;

private:
    std::vector<Chunk> m_chunks;
};

inline void ChunkGrid::initializeFlatWorld() {
    m_chunks.clear();
    m_chunks.emplace_back();

    // Minimal flat world: one chunk with a solid ground layer at y=0.
    m_chunks[0].fillLayer(0, Voxel{VoxelType::Solid});
}

inline std::size_t ChunkGrid::chunkCount() const {
    return m_chunks.size();
}

inline const std::vector<Chunk>& ChunkGrid::chunks() const {
    return m_chunks;
}

} // namespace world
