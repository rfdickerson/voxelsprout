#pragma once

#include <cstdint>

// World spatial query stats shared by spatial backends (clipmap, legacy octree).
namespace odai::world {

struct SpatialQueryStats {
    std::uint32_t visitedNodeCount = 0;
    std::uint32_t candidateChunkCount = 0;
    std::uint32_t visibleChunkCount = 0;
    std::uint32_t retainedChunkCount = 0;
    std::uint32_t newlyVisibleChunkCount = 0;
    std::uint32_t evictedChunkCount = 0;
    std::uint32_t clipmapActiveLevelCount = 0;
    std::uint32_t clipmapUpdatedLevelCount = 0;
    std::uint32_t clipmapUpdatedSlabCount = 0;
    std::uint32_t clipmapUpdatedBrickCount = 0;
    std::uint32_t clipmapResidentBrickCount = 0;
};

} // namespace odai::world
