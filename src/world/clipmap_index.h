#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/grid3.h"
#include "world/chunk_grid.h"
#include "world/spatial_index.h"

// World ClipmapIndex subsystem
// Responsible for: deterministic camera-centered clipmap bounds + chunk broad-phase lookup.
// Should NOT do: voxel storage, meshing, rendering, or simulation stepping.
namespace voxelsprout::world {

struct ClipmapConfig {
    std::uint32_t levelCount = 5;
    std::int32_t gridResolution = 128;
    std::int32_t baseVoxelSize = 1;
    std::int32_t brickResolution = 8;
};

class ChunkClipmapIndex {
public:
    void clear();
    void rebuild(const ChunkGrid& chunkGrid);
    void setConfig(const ClipmapConfig& config);

    [[nodiscard]] const ClipmapConfig& config() const;
    [[nodiscard]] bool valid() const;
    [[nodiscard]] std::size_t chunkCount() const;
    [[nodiscard]] const voxelsprout::core::CellAabb& worldBounds() const;
    [[nodiscard]] const std::vector<std::size_t>& allChunkIndices() const;

    void updateCamera(float cameraX, float cameraY, float cameraZ, SpatialQueryStats* outStats = nullptr);

    // Query chunks intersecting the frustum broad-phase bounds and inside the active clipmap extents.
    [[nodiscard]] std::vector<std::size_t> queryChunksIntersecting(
        const voxelsprout::core::CellAabb& bounds,
        SpatialQueryStats* outStats = nullptr
    ) const;

private:
    struct BrickCoord {
        std::int32_t x = 0;
        std::int32_t y = 0;
        std::int32_t z = 0;
    };

    struct ClipmapLevel {
        std::int32_t voxelSize = 1;
        std::int32_t gridResolution = 1;
        std::int32_t brickResolution = 1;
        std::int32_t brickGridResolution = 1;
        voxelsprout::core::Cell3i originMin{};
        BrickCoord originBrickMin{};
        voxelsprout::core::CellAabb bounds{};
        std::vector<std::uint32_t> brickVersions;
        std::vector<std::uint8_t> brickDirtyMask;
        std::vector<voxelsprout::core::Cell3i> dirtyBrickRingQueue;
    };

    void rebuildLevels();
    static std::int32_t positiveModulo(std::int32_t value, std::int32_t modulus);
    static std::size_t brickLinearIndex(std::int32_t x, std::int32_t y, std::int32_t z, std::int32_t brickGridResolution);
    static BrickCoord worldToBrickCoord(const voxelsprout::core::Cell3i& worldCell, std::int32_t brickWorldSize);
    static std::int32_t snapDownToMultiple(std::int32_t value, std::int32_t multiple);
    static voxelsprout::core::CellAabb makeLevelBounds(const voxelsprout::core::Cell3i& originMin, std::int32_t gridResolution, std::int32_t voxelSize);
    void markAllBricksDirty(ClipmapLevel& level);
    void markBrickDirtyAbsolute(ClipmapLevel& level, const BrickCoord& absoluteBrickCoord);
    std::uint32_t processDirtyBricks(ClipmapLevel& level);

    std::vector<voxelsprout::core::CellAabb> m_chunkBounds;
    std::vector<std::size_t> m_allChunkIndices;
    voxelsprout::core::CellAabb m_worldBounds{};
    ClipmapConfig m_config{};
    std::vector<ClipmapLevel> m_levels;
    bool m_valid = false;
    bool m_levelsInitialized = false;
    std::uint32_t m_lastUpdatedLevelCount = 0;
    std::uint32_t m_lastUpdatedSlabCount = 0;
    std::uint32_t m_lastUpdatedBrickCount = 0;
};

} // namespace voxelsprout::world
