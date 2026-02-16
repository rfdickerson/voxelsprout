#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "world/chunk.h"

namespace voxelsprout::render {

std::vector<std::uint8_t> buildShadowCandidateMask(
    std::span<const voxelsprout::world::Chunk> chunks,
    std::span<const std::size_t> visibleChunkIndices,
    bool enableOccluderCulling
);

}  // namespace voxelsprout::render
