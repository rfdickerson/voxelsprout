#pragma once

#include "world/chunk_grid.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>

namespace world {

class World {
public:
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

    MagicaStampResult stampMagicaResources(std::span<const MagicaStampSpec> specs);

    ChunkGrid& chunkGrid();
    const ChunkGrid& chunkGrid() const;

private:
    static std::filesystem::path resolveAssetPath(const std::filesystem::path& relativePath);
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
};

} // namespace world
