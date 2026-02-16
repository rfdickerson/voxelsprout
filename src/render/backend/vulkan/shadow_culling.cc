#include "render/backend/vulkan/renderer_backend.h"

#include <tuple>
#include <unordered_map>

namespace voxelsprout::render {

namespace {
struct ChunkCoordHash {
    std::size_t operator()(const std::tuple<int, int, int>& key) const noexcept {
        const std::size_t hx = std::hash<int>{}(std::get<0>(key));
        const std::size_t hy = std::hash<int>{}(std::get<1>(key));
        const std::size_t hz = std::hash<int>{}(std::get<2>(key));
        return hx ^ (hy << 1u) ^ (hz << 2u);
    }
};
} // namespace

std::vector<std::uint8_t> RendererBackend::buildShadowCandidateMask(
    std::span<const voxelsprout::world::Chunk> chunks,
    std::span<const std::size_t> visibleChunkIndices
) const {
    std::vector<std::uint8_t> shadowCandidateMask;
    if (!m_shadowDebugSettings.enableOccluderCulling || visibleChunkIndices.empty()) {
        return shadowCandidateMask;
    }

    shadowCandidateMask.assign(chunks.size(), 0u);
    std::unordered_map<std::tuple<int, int, int>, std::size_t, ChunkCoordHash> chunkIndexByCoord;
    chunkIndexByCoord.reserve(chunks.size() * 2u);
    for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
        const voxelsprout::world::Chunk& chunk = chunks[chunkArrayIndex];
        chunkIndexByCoord.emplace(
            std::tuple<int, int, int>{chunk.chunkX(), chunk.chunkY(), chunk.chunkZ()},
            chunkArrayIndex
        );
    }

    const auto markCandidateChunk = [&](int chunkX, int chunkY, int chunkZ) {
        const auto it = chunkIndexByCoord.find(std::tuple<int, int, int>{chunkX, chunkY, chunkZ});
        if (it != chunkIndexByCoord.end()) {
            shadowCandidateMask[it->second] = 1u;
        }
    };

    for (const std::size_t visibleChunkIndex : visibleChunkIndices) {
        if (visibleChunkIndex >= chunks.size()) {
            continue;
        }
        const voxelsprout::world::Chunk& chunk = chunks[visibleChunkIndex];
        shadowCandidateMask[visibleChunkIndex] = 1u;
        const int baseChunkX = chunk.chunkX();
        const int baseChunkY = chunk.chunkY();
        const int baseChunkZ = chunk.chunkZ();

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    markCandidateChunk(baseChunkX + dx, baseChunkY + dy, baseChunkZ + dz);
                }
            }
        }
    }

    return shadowCandidateMask;
}

} // namespace voxelsprout::render
