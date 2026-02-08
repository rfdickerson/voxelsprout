#pragma once

#include <algorithm>
#include <array>
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

    auto chunkHeightAt = [&](int worldX, int worldZ) -> int {
        const std::uint32_t broad = hash3(worldX >> 2, 0, worldZ >> 2, 0x36A17C4Du);
        const std::uint32_t fine = hash3(worldX, 1, worldZ, 0xA45F23B1u);
        int height = 1;
        height += static_cast<int>((broad >> 8) & 0x3u);
        height += static_cast<int>((fine >> 11) & 0x1u);
        return std::clamp(height, 1, Chunk::kSizeY - 2);
    };

    m_chunks.clear();

    // Keep the center chunk first so app-side interaction logic remains valid.
    constexpr std::array<std::array<int, 2>, 9> kChunkCoords = {{
        {{0, 0}},
        {{-1, 0}}, {{1, 0}}, {{0, -1}}, {{0, 1}},
        {{-1, -1}}, {{1, -1}}, {{-1, 1}}, {{1, 1}}
    }};

    m_chunks.reserve(kChunkCoords.size());
    for (const std::array<int, 2>& coord : kChunkCoords) {
        const int chunkX = coord[0];
        const int chunkZ = coord[1];
        m_chunks.emplace_back(chunkX, 0, chunkZ);
        Chunk& chunk = m_chunks.back();

        for (int z = 0; z < Chunk::kSizeZ; ++z) {
            for (int x = 0; x < Chunk::kSizeX; ++x) {
                const int worldX = (chunkX * Chunk::kSizeX) + x;
                const int worldZ = (chunkZ * Chunk::kSizeZ) + z;
                const int terrainHeight = chunkHeightAt(worldX, worldZ);

                for (int y = 0; y <= terrainHeight; ++y) {
                    chunk.setVoxel(x, y, z, Voxel{VoxelType::Solid});
                }

                const std::uint32_t clutterNoise = hash3(worldX, 2, worldZ, 0x1B56C4E9u);
                if ((clutterNoise & 0xFFu) < 20u) {
                    const int clutterHeight = 1 + static_cast<int>((clutterNoise >> 10) % 4u);
                    const int top = std::min(terrainHeight + clutterHeight, Chunk::kSizeY - 1);
                    for (int y = terrainHeight + 1; y <= top; ++y) {
                        chunk.setVoxel(x, y, z, Voxel{VoxelType::Solid});
                    }
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
