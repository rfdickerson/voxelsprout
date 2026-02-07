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

    // Procedural one-chunk terrain:
    // - Base ground uses a deterministic height variation.
    // - Sparse voxels above ground create corners/overhangs so AO is visible.
    for (int z = 0; z < Chunk::kSizeZ; ++z) {
        for (int x = 0; x < Chunk::kSizeX; ++x) {
            const std::uint32_t groundNoise = hash3(x, 0, z, 0xA341316Cu);
            const int groundHeight = 2 + static_cast<int>(groundNoise % 4u); // 2..5

            for (int y = 0; y <= groundHeight; ++y) {
                chunk.setVoxel(x, y, z, Voxel{VoxelType::Solid});
            }

            // Random rocks/voxels above the ground.
            for (int y = groundHeight + 1; y < Chunk::kSizeY && y <= groundHeight + 4; ++y) {
                const std::uint32_t clutterNoise = hash3(x, y, z, 0x1B56C4E9u);
                if ((clutterNoise & 0xFFu) < 20u) {
                    chunk.setVoxel(x, y, z, Voxel{VoxelType::Solid});
                }
            }

            // Occasional short pillar to increase AO contrast.
            const std::uint32_t pillarNoise = hash3(x, 7, z, 0xC8013EA4u);
            if ((pillarNoise & 0x3Fu) == 0u) {
                const int pillarTop = std::min(Chunk::kSizeY - 1, groundHeight + 3);
                for (int y = groundHeight + 1; y <= pillarTop; ++y) {
                    chunk.setVoxel(x, y, z, Voxel{VoxelType::Solid});
                }
            }
        }
    }
}

inline std::size_t ChunkGrid::chunkCount() const {
    return m_chunks.size();
}

inline const std::vector<Chunk>& ChunkGrid::chunks() const {
    return m_chunks;
}

} // namespace world
