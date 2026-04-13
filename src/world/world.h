#pragma once

#include "world/chunk_grid.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>

namespace voxelsprout::world {

class World {
public:
    struct ChunkKey {
        int chunkX = 0;
        int chunkY = 0;
        int chunkZ = 0;

        [[nodiscard]] bool operator==(const ChunkKey& other) const = default;
    };

    struct ChunkStreamingConfig {
        int radiusChunksX = 2;
        int radiusChunksZ = 2;
    };

    struct ChunkStreamingStats {
        int centerChunkX = 0;
        int centerChunkZ = 0;
        std::uint32_t residentChunkCount = 0;
        std::uint32_t storedChunkCount = 0;
        std::uint32_t enteredChunkCount = 0;
        std::uint32_t exitedChunkCount = 0;
        bool changed = false;
    };

    struct ChunkStreamingUpdate {
        ChunkStreamingStats stats{};
        std::vector<ChunkKey> generatedChunkKeys;
        std::vector<ChunkKey> enteredChunkKeys;
        std::vector<ChunkKey> exitedChunkKeys;
        std::vector<std::size_t> residentChunkIndicesNeedingUpload;
        bool requiresFullMeshUpload = false;
    };

    struct LoadResult {
        bool loadedFromFile = false;
        bool initializedFallback = false;
    };

    struct MagicaStampSpec {
        const char* relativePath = nullptr;
        float placementX = 0.0f;
        float placementY = 0.0f;
        float placementZ = 0.0f;
        float uniformScale = 1.0f;
    };

    struct MagicaStampResult {
        std::uint32_t stampedResourceCount = 0;
        std::uint64_t stampedVoxelCount = 0;
        std::uint64_t clippedVoxelCount = 0;
        std::array<std::uint32_t, 16> baseColorPalette{};
        std::uint8_t baseColorPaletteCount = 0;
    };

    bool loadOrInitialize(const std::filesystem::path& worldPath, LoadResult* outResult = nullptr);
    bool save(const std::filesystem::path& worldPath) const;
    void regenerateFlatWorld();
    void setStreamingConfig(const ChunkStreamingConfig& config);
    [[nodiscard]] ChunkStreamingConfig streamingConfig() const;
    [[nodiscard]] ChunkStreamingUpdate updateStreamingWindowForWorldPosition(float worldX, float worldZ);
    [[nodiscard]] const ChunkStreamingStats& streamingStats() const;
    bool setVoxelAtWorld(int worldX, int worldY, int worldZ, Voxel voxel);

    MagicaStampResult stampMagicaResources(std::span<const MagicaStampSpec> specs);

    ChunkGrid& chunkGrid();
    const ChunkGrid& chunkGrid() const;

private:
    static std::filesystem::path resolveAssetPath(const std::filesystem::path& relativePath);
    [[nodiscard]] ChunkStreamingUpdate syncResidentChunkGrid(int centerChunkX, int centerChunkZ);
    bool worldToChunkLocal(
        int worldX,
        int worldY,
        int worldZ,
        std::size_t& outChunkIndex,
        int& outLocalX,
        int& outLocalY,
        int& outLocalZ
    ) const;

    ChunkGrid m_chunkGrid;
    std::vector<Chunk> m_chunkStorage;
    ChunkStreamingConfig m_streamingConfig{};
    ChunkStreamingStats m_streamingStats{};
};

} // namespace voxelsprout::world
