#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/Grid3.hpp"
#include "world/ChunkGrid.hpp"

// World SpatialIndex subsystem
// Responsible for: deterministic broad-phase spatial lookup over chunk bounds.
// Should NOT do: meshing, rendering, or simulation stepping.
namespace world {

struct SpatialQueryStats {
    std::uint32_t visitedNodeCount = 0;
    std::uint32_t candidateChunkCount = 0;
    std::uint32_t visibleChunkCount = 0;
};

class ChunkSpatialIndex {
public:
    void clear();
    void rebuild(const ChunkGrid& chunkGrid);

    [[nodiscard]] bool valid() const;
    [[nodiscard]] std::size_t chunkCount() const;
    [[nodiscard]] const core::CellAabb& worldBounds() const;

    // Query chunks whose chunk AABBs intersect the given bounds.
    // Returned chunk indices point into ChunkGrid::chunks().
    [[nodiscard]] std::vector<std::size_t> queryChunksIntersecting(
        const core::CellAabb& bounds,
        SpatialQueryStats* outStats = nullptr
    ) const;

    // Fallback path for systems not yet integrated with spatial queries.
    [[nodiscard]] const std::vector<std::size_t>& allChunkIndices() const;

private:
    struct Node {
        core::CellAabb bounds{};
        std::uint32_t childA = 0;
        std::uint32_t childB = 0;
        std::uint32_t firstItem = 0;
        std::uint16_t itemCount = 0;
        bool leaf = false;
    };

    static constexpr std::size_t kMaxLeafItems = 8;

    std::uint32_t buildNode(
        const std::vector<core::CellAabb>& chunkBounds,
        std::vector<std::size_t>& sortedChunkIndices,
        std::size_t begin,
        std::size_t count
    );

    std::vector<Node> m_nodes;
    std::vector<std::size_t> m_sortedChunkIndices;
    std::vector<std::size_t> m_allChunkIndices;
    std::vector<core::CellAabb> m_chunkBounds;
    core::CellAabb m_worldBounds{};
    bool m_valid = false;
};

} // namespace world
