#include "render/backend/vulkan/shadow_culling_utils.h"

#include <unordered_map>

#include "core/hash.h"

namespace odai::render {

std::vector<std::uint8_t> buildShadowCandidateMask(
    std::span<const odai::world::Chunk> chunks,
    std::span<const std::size_t> visibleChunkIndices,
    bool enableOccluderCulling
) {
    std::vector<std::uint8_t> shadowCandidateMask;
    if (!enableOccluderCulling || visibleChunkIndices.empty()) {
        return shadowCandidateMask;
    }

    shadowCandidateMask.assign(chunks.size(), 0u);
    std::unordered_map<odai::core::Cell3i, std::size_t, odai::core::Cell3Hash> chunkIndexByCoord;
    chunkIndexByCoord.reserve(chunks.size() * 2u);
    for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
        const odai::world::Chunk& chunk = chunks[chunkArrayIndex];
        chunkIndexByCoord.emplace(
            odai::core::Cell3i{chunk.chunkX(), chunk.chunkY(), chunk.chunkZ()},
            chunkArrayIndex
        );
    }

    const auto markCandidateChunk = [&](int chunkX, int chunkY, int chunkZ) {
        const auto it = chunkIndexByCoord.find(odai::core::Cell3i{chunkX, chunkY, chunkZ});
        if (it != chunkIndexByCoord.end()) {
            shadowCandidateMask[it->second] = 1u;
        }
    };

    for (const std::size_t visibleChunkIndex : visibleChunkIndices) {
        if (visibleChunkIndex >= chunks.size()) {
            continue;
        }
        const odai::world::Chunk& chunk = chunks[visibleChunkIndex];
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

}  // namespace odai::render
