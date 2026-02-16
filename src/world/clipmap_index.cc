#include "world/clipmap_index.h"

#include <algorithm>
#include <cmath>

namespace voxelsprout::world {
namespace {

bool aabbIntersects(const voxelsprout::core::CellAabb& lhs, const voxelsprout::core::CellAabb& rhs) {
    if (!lhs.valid || lhs.empty() || !rhs.valid || rhs.empty()) {
        return false;
    }
    return lhs.minInclusive.x < rhs.maxExclusive.x && lhs.maxExclusive.x > rhs.minInclusive.x &&
           lhs.minInclusive.y < rhs.maxExclusive.y && lhs.maxExclusive.y > rhs.minInclusive.y &&
           lhs.minInclusive.z < rhs.maxExclusive.z && lhs.maxExclusive.z > rhs.minInclusive.z;
}

voxelsprout::core::CellAabb chunkBoundsFromChunk(const Chunk& chunk) {
    const std::int32_t minX = chunk.chunkX() * Chunk::kSizeX;
    const std::int32_t minY = chunk.chunkY() * Chunk::kSizeY;
    const std::int32_t minZ = chunk.chunkZ() * Chunk::kSizeZ;
    voxelsprout::core::CellAabb bounds{};
    bounds.valid = true;
    bounds.minInclusive = voxelsprout::core::Cell3i{minX, minY, minZ};
    bounds.maxExclusive = voxelsprout::core::Cell3i{minX + Chunk::kSizeX, minY + Chunk::kSizeY, minZ + Chunk::kSizeZ};
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
    m_lastUpdatedBrickCount = 0;
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
        const voxelsprout::core::CellAabb chunkBounds = chunkBoundsFromChunk(chunks[chunkIndex]);
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
    clamped.brickResolution = std::clamp<std::int32_t>(clamped.brickResolution, 2, 32);
    if (m_config.levelCount == clamped.levelCount &&
        m_config.gridResolution == clamped.gridResolution &&
        m_config.baseVoxelSize == clamped.baseVoxelSize &&
        m_config.brickResolution == clamped.brickResolution) {
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

const voxelsprout::core::CellAabb& ChunkClipmapIndex::worldBounds() const {
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
    std::uint32_t updatedBricks = 0;
    std::uint32_t residentBricks = 0;

    for (ClipmapLevel& level : m_levels) {
        const std::int32_t snappedX = snapDownToMultiple(cameraCellX, level.voxelSize);
        const std::int32_t snappedY = snapDownToMultiple(cameraCellY, level.voxelSize);
        const std::int32_t snappedZ = snapDownToMultiple(cameraCellZ, level.voxelSize);
        const std::int32_t halfCoverage = (level.gridResolution * level.voxelSize) / 2;
        const voxelsprout::core::Cell3i newOrigin{
            snappedX - halfCoverage,
            snappedY - halfCoverage,
            snappedZ - halfCoverage
        };

        const std::int32_t brickWorldSize = level.voxelSize * level.brickResolution;
        const BrickCoord newOriginBrickMin = worldToBrickCoord(newOrigin, brickWorldSize);

        if (!m_levelsInitialized) {
            ++updatedLevels;
            updatedSlabs += 3u;
            level.originMin = newOrigin;
            level.originBrickMin = newOriginBrickMin;
            level.bounds = makeLevelBounds(newOrigin, level.gridResolution, level.voxelSize);
            markAllBricksDirty(level);
        } else if (newOrigin != level.originMin) {
            ++updatedLevels;
            const std::int32_t deltaBrickX = newOriginBrickMin.x - level.originBrickMin.x;
            const std::int32_t deltaBrickY = newOriginBrickMin.y - level.originBrickMin.y;
            const std::int32_t deltaBrickZ = newOriginBrickMin.z - level.originBrickMin.z;

            const std::int32_t maxShift = std::max({std::abs(deltaBrickX), std::abs(deltaBrickY), std::abs(deltaBrickZ)});
            if (maxShift >= level.brickGridResolution) {
                updatedSlabs += static_cast<std::uint32_t>(level.brickGridResolution * 3);
                markAllBricksDirty(level);
            } else {
                const auto markSlabX = [&](std::int32_t absoluteBrickX) {
                    for (std::int32_t by = 0; by < level.brickGridResolution; ++by) {
                        for (std::int32_t bz = 0; bz < level.brickGridResolution; ++bz) {
                            markBrickDirtyAbsolute(
                                level,
                                BrickCoord{
                                    absoluteBrickX,
                                    newOriginBrickMin.y + by,
                                    newOriginBrickMin.z + bz
                                }
                            );
                        }
                    }
                };
                const auto markSlabY = [&](std::int32_t absoluteBrickY) {
                    for (std::int32_t bx = 0; bx < level.brickGridResolution; ++bx) {
                        for (std::int32_t bz = 0; bz < level.brickGridResolution; ++bz) {
                            markBrickDirtyAbsolute(
                                level,
                                BrickCoord{
                                    newOriginBrickMin.x + bx,
                                    absoluteBrickY,
                                    newOriginBrickMin.z + bz
                                }
                            );
                        }
                    }
                };
                const auto markSlabZ = [&](std::int32_t absoluteBrickZ) {
                    for (std::int32_t bx = 0; bx < level.brickGridResolution; ++bx) {
                        for (std::int32_t by = 0; by < level.brickGridResolution; ++by) {
                            markBrickDirtyAbsolute(
                                level,
                                BrickCoord{
                                    newOriginBrickMin.x + bx,
                                    newOriginBrickMin.y + by,
                                    absoluteBrickZ
                                }
                            );
                        }
                    }
                };

                if (deltaBrickX > 0) {
                    for (std::int32_t s = 0; s < deltaBrickX; ++s) {
                        markSlabX(level.originBrickMin.x + level.brickGridResolution + s);
                        ++updatedSlabs;
                    }
                } else if (deltaBrickX < 0) {
                    for (std::int32_t s = 0; s < -deltaBrickX; ++s) {
                        markSlabX(newOriginBrickMin.x + s);
                        ++updatedSlabs;
                    }
                }
                if (deltaBrickY > 0) {
                    for (std::int32_t s = 0; s < deltaBrickY; ++s) {
                        markSlabY(level.originBrickMin.y + level.brickGridResolution + s);
                        ++updatedSlabs;
                    }
                } else if (deltaBrickY < 0) {
                    for (std::int32_t s = 0; s < -deltaBrickY; ++s) {
                        markSlabY(newOriginBrickMin.y + s);
                        ++updatedSlabs;
                    }
                }
                if (deltaBrickZ > 0) {
                    for (std::int32_t s = 0; s < deltaBrickZ; ++s) {
                        markSlabZ(level.originBrickMin.z + level.brickGridResolution + s);
                        ++updatedSlabs;
                    }
                } else if (deltaBrickZ < 0) {
                    for (std::int32_t s = 0; s < -deltaBrickZ; ++s) {
                        markSlabZ(newOriginBrickMin.z + s);
                        ++updatedSlabs;
                    }
                }
            }

            level.originMin = newOrigin;
            level.originBrickMin = newOriginBrickMin;
            level.bounds = makeLevelBounds(newOrigin, level.gridResolution, level.voxelSize);
        }

        updatedBricks += processDirtyBricks(level);
        residentBricks += static_cast<std::uint32_t>(level.brickVersions.size());
    }

    m_levelsInitialized = true;
    m_lastUpdatedLevelCount = updatedLevels;
    m_lastUpdatedSlabCount = updatedSlabs;
    m_lastUpdatedBrickCount = updatedBricks;

    if (outStats != nullptr) {
        outStats->visitedNodeCount = static_cast<std::uint32_t>(m_levels.size());
        outStats->clipmapActiveLevelCount = static_cast<std::uint32_t>(m_levels.size());
        outStats->clipmapUpdatedLevelCount = m_lastUpdatedLevelCount;
        outStats->clipmapUpdatedSlabCount = m_lastUpdatedSlabCount;
        outStats->clipmapUpdatedBrickCount = m_lastUpdatedBrickCount;
        outStats->clipmapResidentBrickCount = residentBricks;
    }
}

std::vector<std::size_t> ChunkClipmapIndex::queryChunksIntersecting(
    const voxelsprout::core::CellAabb& bounds,
    SpatialQueryStats* outStats
) const {
    std::vector<std::size_t> result;
    if (outStats != nullptr) {
        *outStats = SpatialQueryStats{};
    }
    if (!m_valid || m_levels.empty() || !bounds.valid || bounds.empty()) {
        return result;
    }

    const voxelsprout::core::CellAabb clipmapBounds = m_levels.back().bounds;
    if (!aabbIntersects(clipmapBounds, bounds)) {
        return result;
    }
    const voxelsprout::core::CellAabb effectiveBounds = voxelsprout::core::intersectAabb(clipmapBounds, bounds);
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
        outStats->clipmapUpdatedBrickCount = m_lastUpdatedBrickCount;
        std::uint32_t residentBricks = 0;
        for (const ClipmapLevel& level : m_levels) {
            residentBricks += static_cast<std::uint32_t>(level.brickVersions.size());
        }
        outStats->clipmapResidentBrickCount = residentBricks;
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
        level.brickResolution = std::clamp(m_config.brickResolution, 2, level.gridResolution);
        level.brickGridResolution = std::max(
            1,
            (level.gridResolution + level.brickResolution - 1) / level.brickResolution
        );
        level.originMin = voxelsprout::core::Cell3i{};
        level.originBrickMin = BrickCoord{};
        level.bounds = makeLevelBounds(level.originMin, level.gridResolution, level.voxelSize);
        const std::size_t brickCount = static_cast<std::size_t>(level.brickGridResolution) *
                                       static_cast<std::size_t>(level.brickGridResolution) *
                                       static_cast<std::size_t>(level.brickGridResolution);
        level.brickVersions.assign(brickCount, 0u);
        level.brickDirtyMask.assign(brickCount, 0u);
        level.dirtyBrickRingQueue.clear();
        m_levels.push_back(std::move(level));
    }
}

std::int32_t ChunkClipmapIndex::positiveModulo(std::int32_t value, std::int32_t modulus) {
    if (modulus <= 0) {
        return 0;
    }
    std::int32_t result = value % modulus;
    if (result < 0) {
        result += modulus;
    }
    return result;
}

std::size_t ChunkClipmapIndex::brickLinearIndex(
    std::int32_t x,
    std::int32_t y,
    std::int32_t z,
    std::int32_t brickGridResolution
) {
    return static_cast<std::size_t>(
        x + (brickGridResolution * (z + (brickGridResolution * y)))
    );
}

ChunkClipmapIndex::BrickCoord ChunkClipmapIndex::worldToBrickCoord(
    const voxelsprout::core::Cell3i& worldCell,
    std::int32_t brickWorldSize
) {
    return BrickCoord{
        snapDownToMultiple(worldCell.x, brickWorldSize) / brickWorldSize,
        snapDownToMultiple(worldCell.y, brickWorldSize) / brickWorldSize,
        snapDownToMultiple(worldCell.z, brickWorldSize) / brickWorldSize
    };
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

voxelsprout::core::CellAabb ChunkClipmapIndex::makeLevelBounds(
    const voxelsprout::core::Cell3i& originMin,
    std::int32_t gridResolution,
    std::int32_t voxelSize
) {
    voxelsprout::core::CellAabb bounds{};
    bounds.valid = true;
    bounds.minInclusive = originMin;
    const std::int32_t extent = gridResolution * voxelSize;
    bounds.maxExclusive = voxelsprout::core::Cell3i{
        originMin.x + extent,
        originMin.y + extent,
        originMin.z + extent
    };
    return bounds;
}

void ChunkClipmapIndex::markAllBricksDirty(ClipmapLevel& level) {
    level.dirtyBrickRingQueue.clear();
    for (std::int32_t by = 0; by < level.brickGridResolution; ++by) {
        for (std::int32_t bz = 0; bz < level.brickGridResolution; ++bz) {
            for (std::int32_t bx = 0; bx < level.brickGridResolution; ++bx) {
                const std::size_t index = brickLinearIndex(bx, by, bz, level.brickGridResolution);
                if (index >= level.brickDirtyMask.size()) {
                    continue;
                }
                level.brickDirtyMask[index] = 1u;
                level.dirtyBrickRingQueue.push_back(voxelsprout::core::Cell3i{bx, by, bz});
            }
        }
    }
}

void ChunkClipmapIndex::markBrickDirtyAbsolute(ClipmapLevel& level, const BrickCoord& absoluteBrickCoord) {
    const std::int32_t ringX = positiveModulo(absoluteBrickCoord.x, level.brickGridResolution);
    const std::int32_t ringY = positiveModulo(absoluteBrickCoord.y, level.brickGridResolution);
    const std::int32_t ringZ = positiveModulo(absoluteBrickCoord.z, level.brickGridResolution);
    const std::size_t index = brickLinearIndex(ringX, ringY, ringZ, level.brickGridResolution);
    if (index >= level.brickDirtyMask.size()) {
        return;
    }
    if (level.brickDirtyMask[index] != 0u) {
        return;
    }
    level.brickDirtyMask[index] = 1u;
    level.dirtyBrickRingQueue.push_back(voxelsprout::core::Cell3i{ringX, ringY, ringZ});
}

std::uint32_t ChunkClipmapIndex::processDirtyBricks(ClipmapLevel& level) {
    std::uint32_t updatedBrickCount = 0;
    for (const voxelsprout::core::Cell3i& ringCoord : level.dirtyBrickRingQueue) {
        const std::size_t index = brickLinearIndex(ringCoord.x, ringCoord.y, ringCoord.z, level.brickGridResolution);
        if (index >= level.brickDirtyMask.size() || index >= level.brickVersions.size()) {
            continue;
        }
        level.brickDirtyMask[index] = 0u;
        ++level.brickVersions[index];
        ++updatedBrickCount;
    }
    level.dirtyBrickRingQueue.clear();
    return updatedBrickCount;
}

} // namespace voxelsprout::world
