#pragma once

#include <cstddef>
#include <cstdint>
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
    std::vector<Chunk>& chunks();
    const std::vector<Chunk>& chunks() const;

private:
    std::vector<Chunk> m_chunks;
};

inline void ChunkGrid::initializeFlatWorld() {
    auto hash3 = [](int x, int y, int z, std::uint32_t seed) -> std::uint32_t {
        std::uint32_t h = seed;
        h ^= static_cast<std::uint32_t>(x) * 0x9E3779B9u;
        h ^= static_cast<std::uint32_t>(y) * 0x85EBCA6Bu;
        h ^= static_cast<std::uint32_t>(z) * 0xC2B2AE35u;
        h ^= (h >> 16);
        h *= 0x7FEB352Du;
        h ^= (h >> 15);
        h *= 0x846CA68Bu;
        h ^= (h >> 16);
        return h;
    };

    m_chunks.clear();
    m_chunks.emplace_back();

    Chunk& chunk = m_chunks[0];

    // Ground + sky setup:
    // - One flat ground layer at y=0.
    // - A few deterministic voxels above ground for AO and depth testing.
    for (int z = 0; z < Chunk::kSizeZ; ++z) {
        for (int x = 0; x < Chunk::kSizeX; ++x) {
            chunk.setVoxel(x, 0, z, Voxel{VoxelType::Solid});

            const std::uint32_t clutterNoise = hash3(x, 1, z, 0x1B56C4E9u);
            if ((clutterNoise & 0xFFu) < 18u) {
                const int height = 1 + static_cast<int>((clutterNoise >> 8) % 3u); // 1..3
                for (int y = 1; y <= height; ++y) {
                    chunk.setVoxel(x, y, z, Voxel{VoxelType::Solid});
                }
            }
        }
    }
}

inline std::size_t ChunkGrid::chunkCount() const {
    return m_chunks.size();
}

inline std::vector<Chunk>& ChunkGrid::chunks() {
    return m_chunks;
}

inline const std::vector<Chunk>& ChunkGrid::chunks() const {
    return m_chunks;
}

} // namespace world
