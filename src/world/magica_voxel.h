#pragma once

#include "world/chunk_mesher.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace voxelsprout::world {

struct MagicaVoxel {
    std::uint8_t x = 0;
    std::uint8_t y = 0;
    std::uint8_t z = 0;
    std::uint8_t paletteIndex = 0;
};

struct MagicaVoxelModel {
    int sizeX = 0;
    int sizeY = 0;
    int sizeZ = 0;
    std::vector<MagicaVoxel> voxels;
    // Palette indexed by voxel color index (0 = empty).
    std::array<std::uint32_t, 256> paletteRgba{};
    bool hasPalette = false;
};

struct MagicaVoxelMeshChunk {
    int originX = 0;
    int originY = 0;
    int originZ = 0;
    ChunkMeshData mesh{};
};

bool loadMagicaVoxelModel(const std::filesystem::path& path, MagicaVoxelModel& outModel);
std::vector<MagicaVoxelMeshChunk> buildMagicaVoxelMeshChunks(const MagicaVoxelModel& model);
ChunkMeshData buildMagicaVoxelMesh(const MagicaVoxelModel& model);

} // namespace voxelsprout::world
