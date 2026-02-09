#include "world/SpatialIndex.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace world {
namespace {

bool aabbIntersects(const core::CellAabb& lhs, const core::CellAabb& rhs) {
    if (!lhs.valid || lhs.empty() || !rhs.valid || rhs.empty()) {
        return false;
    }
    return lhs.minInclusive.x < rhs.maxExclusive.x && lhs.maxExclusive.x > rhs.minInclusive.x &&
           lhs.minInclusive.y < rhs.maxExclusive.y && lhs.maxExclusive.y > rhs.minInclusive.y &&
           lhs.minInclusive.z < rhs.maxExclusive.z && lhs.maxExclusive.z > rhs.minInclusive.z;
}

core::CellAabb mergeAabbs(const core::CellAabb& lhs, const core::CellAabb& rhs) {
    if (!lhs.valid || lhs.empty()) {
        return rhs;
    }
    if (!rhs.valid || rhs.empty()) {
        return lhs;
    }
    core::CellAabb merged{};
    merged.valid = true;
    merged.minInclusive.x = std::min(lhs.minInclusive.x, rhs.minInclusive.x);
    merged.minInclusive.y = std::min(lhs.minInclusive.y, rhs.minInclusive.y);
    merged.minInclusive.z = std::min(lhs.minInclusive.z, rhs.minInclusive.z);
    merged.maxExclusive.x = std::max(lhs.maxExclusive.x, rhs.maxExclusive.x);
    merged.maxExclusive.y = std::max(lhs.maxExclusive.y, rhs.maxExclusive.y);
    merged.maxExclusive.z = std::max(lhs.maxExclusive.z, rhs.maxExclusive.z);
    return merged;
}

int axisExtent(const core::CellAabb& bounds, int axis) {
    switch (axis) {
    case 0:
        return bounds.maxExclusive.x - bounds.minInclusive.x;
    case 1:
        return bounds.maxExclusive.y - bounds.minInclusive.y;
    case 2:
    default:
        return bounds.maxExclusive.z - bounds.minInclusive.z;
    }
}

int axisCenter2(const core::CellAabb& bounds, int axis) {
    switch (axis) {
    case 0:
        return bounds.minInclusive.x + bounds.maxExclusive.x;
    case 1:
        return bounds.minInclusive.y + bounds.maxExclusive.y;
    case 2:
    default:
        return bounds.minInclusive.z + bounds.maxExclusive.z;
    }
}

} // namespace

void ChunkSpatialIndex::clear() {
    m_nodes.clear();
    m_sortedChunkIndices.clear();
    m_allChunkIndices.clear();
    m_chunkBounds.clear();
    m_worldBounds = {};
    m_valid = false;
}

void ChunkSpatialIndex::rebuild(const ChunkGrid& chunkGrid) {
    clear();
    const std::vector<Chunk>& chunks = chunkGrid.chunks();
    if (chunks.empty()) {
        return;
    }

    m_chunkBounds.resize(chunks.size());
    m_allChunkIndices.resize(chunks.size());
    m_sortedChunkIndices.resize(chunks.size());
    for (std::size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        const Chunk& chunk = chunks[chunkIndex];
        const int minX = chunk.chunkX() * Chunk::kSizeX;
        const int minY = chunk.chunkY() * Chunk::kSizeY;
        const int minZ = chunk.chunkZ() * Chunk::kSizeZ;

        core::CellAabb bounds{};
        bounds.valid = true;
        bounds.minInclusive = core::Cell3i{minX, minY, minZ};
        bounds.maxExclusive = core::Cell3i{minX + Chunk::kSizeX, minY + Chunk::kSizeY, minZ + Chunk::kSizeZ};
        m_chunkBounds[chunkIndex] = bounds;
        m_allChunkIndices[chunkIndex] = chunkIndex;
        m_sortedChunkIndices[chunkIndex] = chunkIndex;
        m_worldBounds = mergeAabbs(m_worldBounds, bounds);
    }

    buildNode(m_chunkBounds, m_sortedChunkIndices, 0, m_sortedChunkIndices.size());
    m_valid = !m_nodes.empty();
}

bool ChunkSpatialIndex::valid() const {
    return m_valid;
}

std::size_t ChunkSpatialIndex::chunkCount() const {
    return m_allChunkIndices.size();
}

const core::CellAabb& ChunkSpatialIndex::worldBounds() const {
    return m_worldBounds;
}

std::vector<std::size_t> ChunkSpatialIndex::queryChunksIntersecting(
    const core::CellAabb& bounds,
    SpatialQueryStats* outStats
) const {
    std::vector<std::size_t> result;
    if (outStats != nullptr) {
        *outStats = SpatialQueryStats{};
    }
    if (!m_valid || m_nodes.empty() || !aabbIntersects(m_worldBounds, bounds)) {
        return result;
    }

    std::vector<std::uint32_t> stack;
    stack.push_back(0u);
    while (!stack.empty()) {
        const std::uint32_t nodeIndex = stack.back();
        stack.pop_back();
        if (nodeIndex >= m_nodes.size()) {
            continue;
        }
        if (outStats != nullptr) {
            ++outStats->visitedNodeCount;
        }

        const Node& node = m_nodes[nodeIndex];
        if (!aabbIntersects(node.bounds, bounds)) {
            continue;
        }
        if (node.leaf) {
            const std::size_t start = static_cast<std::size_t>(node.firstItem);
            const std::size_t count = static_cast<std::size_t>(node.itemCount);
            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t sortedIndex = start + i;
                if (sortedIndex >= m_sortedChunkIndices.size()) {
                    continue;
                }
                const std::size_t chunkIndex = m_sortedChunkIndices[sortedIndex];
                if (chunkIndex >= m_chunkBounds.size()) {
                    continue;
                }
                if (outStats != nullptr) {
                    ++outStats->candidateChunkCount;
                }
                if (aabbIntersects(m_chunkBounds[chunkIndex], bounds)) {
                    result.push_back(chunkIndex);
                    if (outStats != nullptr) {
                        ++outStats->visibleChunkCount;
                    }
                }
            }
        } else {
            if (node.childA < m_nodes.size()) {
                stack.push_back(node.childA);
            }
            if (node.childB < m_nodes.size()) {
                stack.push_back(node.childB);
            }
        }
    }

    return result;
}

const std::vector<std::size_t>& ChunkSpatialIndex::allChunkIndices() const {
    return m_allChunkIndices;
}

std::uint32_t ChunkSpatialIndex::buildNode(
    const std::vector<core::CellAabb>& chunkBounds,
    std::vector<std::size_t>& sortedChunkIndices,
    std::size_t begin,
    std::size_t count
) {
    const std::uint32_t nodeIndex = static_cast<std::uint32_t>(m_nodes.size());
    m_nodes.push_back(Node{});

    core::CellAabb nodeBounds{};
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t chunkIndex = sortedChunkIndices[begin + i];
        if (chunkIndex < chunkBounds.size()) {
            nodeBounds = mergeAabbs(nodeBounds, chunkBounds[chunkIndex]);
        }
    }
    m_nodes[nodeIndex].bounds = nodeBounds;

    if (count <= kMaxLeafItems) {
        m_nodes[nodeIndex].leaf = true;
        m_nodes[nodeIndex].firstItem = static_cast<std::uint32_t>(begin);
        m_nodes[nodeIndex].itemCount = static_cast<std::uint16_t>(count);
        return nodeIndex;
    }

    int splitAxis = 0;
    int bestExtent = axisExtent(nodeBounds, 0);
    for (int axis = 1; axis < 3; ++axis) {
        const int extent = axisExtent(nodeBounds, axis);
        if (extent > bestExtent) {
            bestExtent = extent;
            splitAxis = axis;
        }
    }

    auto startIt = sortedChunkIndices.begin() + static_cast<std::ptrdiff_t>(begin);
    auto endIt = startIt + static_cast<std::ptrdiff_t>(count);
    std::stable_sort(
        startIt,
        endIt,
        [&](std::size_t lhsChunkIndex, std::size_t rhsChunkIndex) {
            const core::CellAabb& lhs = chunkBounds[lhsChunkIndex];
            const core::CellAabb& rhs = chunkBounds[rhsChunkIndex];
            const int lhsCenter = axisCenter2(lhs, splitAxis);
            const int rhsCenter = axisCenter2(rhs, splitAxis);
            if (lhsCenter != rhsCenter) {
                return lhsCenter < rhsCenter;
            }
            return lhsChunkIndex < rhsChunkIndex;
        }
    );

    const std::size_t leftCount = count / 2u;
    const std::size_t rightCount = count - leftCount;
    if (leftCount == 0 || rightCount == 0) {
        m_nodes[nodeIndex].leaf = true;
        m_nodes[nodeIndex].firstItem = static_cast<std::uint32_t>(begin);
        m_nodes[nodeIndex].itemCount = static_cast<std::uint16_t>(count);
        return nodeIndex;
    }

    m_nodes[nodeIndex].leaf = false;
    m_nodes[nodeIndex].childA = buildNode(chunkBounds, sortedChunkIndices, begin, leftCount);
    m_nodes[nodeIndex].childB = buildNode(chunkBounds, sortedChunkIndices, begin + leftCount, rightCount);
    return nodeIndex;
}

} // namespace world
