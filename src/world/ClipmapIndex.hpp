#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/Grid3.hpp"
#include "world/ChunkGrid.hpp"
#include "world/SpatialIndex.hpp"

// World ClipmapIndex subsystem
// Responsible for: deterministic camera-centered clipmap bounds + chunk broad-phase lookup.
// Should NOT do: voxel storage, meshing, rendering, or simulation stepping.
namespace world {

struct ClipmapConfig {
    std::uint32_t levelCount = 5;
    std::int32_t gridResolution = 128;
    std::int32_t baseVoxelSize = 1;
};

class ChunkClipmapIndex {
public:
    void clear();
    void rebuild(const ChunkGrid& chunkGrid);
    void setConfig(const ClipmapConfig& config);

    [[nodiscard]] const ClipmapConfig& config() const;
    [[nodiscard]] bool valid() const;
    [[nodiscard]] std::size_t chunkCount() const;
    [[nodiscard]] const core::CellAabb& worldBounds() const;
    [[nodiscard]] const std::vector<std::size_t>& allChunkIndices() const;

    void updateCamera(float cameraX, float cameraY, float cameraZ, SpatialQueryStats* outStats = nullptr);

    // Query chunks intersecting the frustum broad-phase bounds and inside the active clipmap extents.
    [[nodiscard]] std::vector<std::size_t> queryChunksIntersecting(
        const core::CellAabb& bounds,
        SpatialQueryStats* outStats = nullptr
    ) const;

private:
    struct ClipmapLevel {
        std::int32_t voxelSize = 1;
        std::int32_t gridResolution = 1;
        core::Cell3i originMin{};
        core::CellAabb bounds{};
    };

    void rebuildLevels();
    static std::int32_t snapDownToMultiple(std::int32_t value, std::int32_t multiple);
    static core::CellAabb makeLevelBounds(const core::Cell3i& originMin, std::int32_t gridResolution, std::int32_t voxelSize);

    std::vector<core::CellAabb> m_chunkBounds;
    std::vector<std::size_t> m_allChunkIndices;
    core::CellAabb m_worldBounds{};
    ClipmapConfig m_config{};
    std::vector<ClipmapLevel> m_levels;
    bool m_valid = false;
    bool m_levelsInitialized = false;
    std::uint32_t m_lastUpdatedLevelCount = 0;
    std::uint32_t m_lastUpdatedSlabCount = 0;
};

} // namespace world

