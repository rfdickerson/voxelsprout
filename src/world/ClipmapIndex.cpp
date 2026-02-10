#include "world/ClipmapIndex.hpp"

#include <algorithm>
#include <cmath>

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

core::CellAabb chunkBoundsFromChunk(const Chunk& chunk) {
    const std::int32_t minX = chunk.chunkX() * Chunk::kSizeX;
    const std::int32_t minY = chunk.chunkY() * Chunk::kSizeY;
    const std::int32_t minZ = chunk.chunkZ() * Chunk::kSizeZ;
    core::CellAabb bounds{};
    bounds.valid = true;
    bounds.minInclusive = core::Cell3i{minX, minY, minZ};
    bounds.maxExclusive = core::Cell3i{minX + Chunk::kSizeX, minY + Chunk::kSizeY, minZ + Chunk::kSizeZ};
    return bounds;
}

std::int32_t floorToCell(float value) {
    return static_cast<std::int32_t>(std::floor(value));
}

} // namespace

void ChunkClipmapIndex::clear() {
    m_chunkBounds.clear();
    m_allChunkIndices.clear();
    m_worldBounds = {};
    m_levels.clear();
    m_valid = false;
    m_levelsInitialized = false;
    m_lastUpdatedLevelCount = 0;
    m_lastUpdatedSlabCount = 0;
}

void ChunkClipmapIndex::rebuild(const ChunkGrid& chunkGrid) {
    clear();
    const std::vector<Chunk>& chunks = chunkGrid.chunks();
    if (chunks.empty()) {
        return;
    }

    m_chunkBounds.resize(chunks.size());
    m_allChunkIndices.resize(chunks.size());
    for (std::size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        const core::CellAabb chunkBounds = chunkBoundsFromChunk(chunks[chunkIndex]);
        m_chunkBounds[chunkIndex] = chunkBounds;
        m_allChunkIndices[chunkIndex] = chunkIndex;
        m_worldBounds.includeAabb(chunkBounds);
    }

    rebuildLevels();
    m_valid = !m_levels.empty() && m_worldBounds.valid && !m_worldBounds.empty();
}

void ChunkClipmapIndex::setConfig(const ClipmapConfig& config) {
    ClipmapConfig clamped = config;
    clamped.levelCount = std::clamp<std::uint32_t>(clamped.levelCount, 1u, 10u);
    clamped.gridResolution = std::clamp<std::int32_t>(clamped.gridResolution, 16, 512);
    clamped.baseVoxelSize = std::clamp<std::int32_t>(clamped.baseVoxelSize, 1, 64);
    if (m_config.levelCount == clamped.levelCount &&
        m_config.gridResolution == clamped.gridResolution &&
        m_config.baseVoxelSize == clamped.baseVoxelSize) {
        return;
    }
    m_config = clamped;
    rebuildLevels();
    m_levelsInitialized = false;
}

const ClipmapConfig& ChunkClipmapIndex::config() const {
    return m_config;
}

bool ChunkClipmapIndex::valid() const {
    return m_valid;
}

std::size_t ChunkClipmapIndex::chunkCount() const {
    return m_allChunkIndices.size();
}

const core::CellAabb& ChunkClipmapIndex::worldBounds() const {
    return m_worldBounds;
}

const std::vector<std::size_t>& ChunkClipmapIndex::allChunkIndices() const {
    return m_allChunkIndices;
}

void ChunkClipmapIndex::updateCamera(float cameraX, float cameraY, float cameraZ, SpatialQueryStats* outStats) {
    if (outStats != nullptr) {
        *outStats = SpatialQueryStats{};
    }
    if (!m_valid || m_levels.empty()) {
        return;
    }

    const std::int32_t cameraCellX = floorToCell(cameraX);
    const std::int32_t cameraCellY = floorToCell(cameraY);
    const std::int32_t cameraCellZ = floorToCell(cameraZ);

    std::uint32_t updatedLevels = 0;
    std::uint32_t updatedSlabs = 0;
    for (ClipmapLevel& level : m_levels) {
        const std::int32_t snappedX = snapDownToMultiple(cameraCellX, level.voxelSize);
        const std::int32_t snappedY = snapDownToMultiple(cameraCellY, level.voxelSize);
        const std::int32_t snappedZ = snapDownToMultiple(cameraCellZ, level.voxelSize);
        const std::int32_t halfCoverage = (level.gridResolution * level.voxelSize) / 2;
        const core::Cell3i newOrigin{
            snappedX - halfCoverage,
            snappedY - halfCoverage,
            snappedZ - halfCoverage
        };

        if (!m_levelsInitialized || newOrigin != level.originMin) {
            ++updatedLevels;
            if (m_levelsInitialized) {
                const std::int32_t deltaCellsX = std::abs((newOrigin.x - level.originMin.x) / level.voxelSize);
                const std::int32_t deltaCellsY = std::abs((newOrigin.y - level.originMin.y) / level.voxelSize);
                const std::int32_t deltaCellsZ = std::abs((newOrigin.z - level.originMin.z) / level.voxelSize);
                updatedSlabs += static_cast<std::uint32_t>(deltaCellsX + deltaCellsY + deltaCellsZ);
            } else {
                updatedSlabs += 3u;
            }
            level.originMin = newOrigin;
            level.bounds = makeLevelBounds(newOrigin, level.gridResolution, level.voxelSize);
        }
    }

    m_levelsInitialized = true;
    m_lastUpdatedLevelCount = updatedLevels;
    m_lastUpdatedSlabCount = updatedSlabs;

    if (outStats != nullptr) {
        outStats->visitedNodeCount = static_cast<std::uint32_t>(m_levels.size());
        outStats->clipmapActiveLevelCount = static_cast<std::uint32_t>(m_levels.size());
        outStats->clipmapUpdatedLevelCount = m_lastUpdatedLevelCount;
        outStats->clipmapUpdatedSlabCount = m_lastUpdatedSlabCount;
    }
}

std::vector<std::size_t> ChunkClipmapIndex::queryChunksIntersecting(
    const core::CellAabb& bounds,
    SpatialQueryStats* outStats
) const {
    std::vector<std::size_t> result;
    if (outStats != nullptr) {
        *outStats = SpatialQueryStats{};
    }
    if (!m_valid || m_levels.empty() || !bounds.valid || bounds.empty()) {
        return result;
    }

    const core::CellAabb clipmapBounds = m_levels.back().bounds;
    if (!aabbIntersects(clipmapBounds, bounds)) {
        return result;
    }
    const core::CellAabb effectiveBounds = core::intersectAabb(clipmapBounds, bounds);
    if (!effectiveBounds.valid || effectiveBounds.empty()) {
        return result;
    }

    result.reserve(m_chunkBounds.size());
    std::uint32_t candidateCount = 0;
    for (std::size_t chunkIndex = 0; chunkIndex < m_chunkBounds.size(); ++chunkIndex) {
        if (!aabbIntersects(m_chunkBounds[chunkIndex], effectiveBounds)) {
            continue;
        }
        ++candidateCount;
        result.push_back(chunkIndex);
    }

    if (outStats != nullptr) {
        outStats->visitedNodeCount = static_cast<std::uint32_t>(m_levels.size());
        outStats->candidateChunkCount = candidateCount;
        outStats->visibleChunkCount = static_cast<std::uint32_t>(result.size());
        outStats->clipmapActiveLevelCount = static_cast<std::uint32_t>(m_levels.size());
        outStats->clipmapUpdatedLevelCount = m_lastUpdatedLevelCount;
        outStats->clipmapUpdatedSlabCount = m_lastUpdatedSlabCount;
    }
    return result;
}

void ChunkClipmapIndex::rebuildLevels() {
    m_levels.clear();
    m_levels.reserve(m_config.levelCount);
    for (std::uint32_t levelIndex = 0; levelIndex < m_config.levelCount; ++levelIndex) {
        ClipmapLevel level{};
        level.voxelSize = m_config.baseVoxelSize << levelIndex;
        level.gridResolution = m_config.gridResolution;
        level.originMin = core::Cell3i{};
        level.bounds = makeLevelBounds(level.originMin, level.gridResolution, level.voxelSize);
        m_levels.push_back(level);
    }
}

std::int32_t ChunkClipmapIndex::snapDownToMultiple(std::int32_t value, std::int32_t multiple) {
    if (multiple <= 1) {
        return value;
    }
    std::int32_t remainder = value % multiple;
    if (remainder < 0) {
        remainder += multiple;
    }
    return value - remainder;
}

core::CellAabb ChunkClipmapIndex::makeLevelBounds(
    const core::Cell3i& originMin,
    std::int32_t gridResolution,
    std::int32_t voxelSize
) {
    core::CellAabb bounds{};
    bounds.valid = true;
    bounds.minInclusive = originMin;
    const std::int32_t extent = gridResolution * voxelSize;
    bounds.maxExclusive = core::Cell3i{
        originMin.x + extent,
        originMin.y + extent,
        originMin.z + extent
    };
    return bounds;
}

} // namespace world

