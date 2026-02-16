#pragma once

#include "world/chunk_grid.h"

#include <array>
#include <cstdint>
#include <vector>

namespace voxelsprout::world {

constexpr std::uint32_t kChunkMeshLodCount = 3;

enum class MeshingMode : std::uint8_t {
    Naive = 0,
    Greedy = 1
};

struct MeshingOptions {
    MeshingMode mode = MeshingMode::Greedy;
};

// Packed voxel vertex used by the Vulkan vertex fetch stage.
// We keep this tightly packed so GPU bandwidth stays low when meshes get denser.
//
// Bit layout in PackedVoxelVertex::bits (LSB -> MSB):
// - bits  0.. 4: local x in chunk (0..31)
// - bits  5.. 9: local y in chunk (0..31)
// - bits 10..14: local z in chunk (0..31)
// - bits 15..17: face id (0..5 for +/-X, +/-Y, +/-Z)
// - bits 18..19: corner id (0..3)
// - bits 20..21: AO level (0 darkest .. 3 brightest)
// - bits 22..25: material id (0..15)
// - bits 26..29: base color index (0..15)
// - bits 30..31: lod level (0=8x, 1=4x, 2=1x)
//
// This format also supports future greedy meshing and instancing:
// - Greedy meshing can add width/height in reserved or expanded fields.
// - Instancing can move xyz to an instance buffer and keep face/corner/material per-vertex.
struct PackedVoxelVertex {
    std::uint32_t bits = 0;

    static constexpr std::uint32_t kShiftX = 0;
    static constexpr std::uint32_t kShiftY = 5;
    static constexpr std::uint32_t kShiftZ = 10;
    static constexpr std::uint32_t kShiftFace = 15;
    static constexpr std::uint32_t kShiftCorner = 18;
    static constexpr std::uint32_t kShiftAo = 20;
    static constexpr std::uint32_t kShiftMaterial = 22;
    static constexpr std::uint32_t kShiftBaseColor = 26;
    static constexpr std::uint32_t kShiftLodLevel = 30;

    static constexpr std::uint32_t kMask5 = 0x1Fu;
    static constexpr std::uint32_t kMask3 = 0x7u;
    static constexpr std::uint32_t kMask4 = 0xFu;
    static constexpr std::uint32_t kMask2 = 0x3u;

    static std::uint32_t pack(
        std::uint32_t x,
        std::uint32_t y,
        std::uint32_t z,
        std::uint32_t face,
        std::uint32_t corner,
        std::uint32_t ao,
        std::uint32_t material,
        std::uint32_t baseColorIndex,
        std::uint32_t lodLevel
    );
};

struct ChunkMeshData {
    std::vector<PackedVoxelVertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct ChunkLodMeshes {
    std::array<ChunkMeshData, kChunkMeshLodCount> lodMeshes;
};

ChunkLodMeshes buildChunkLodMeshes(const Chunk& chunk, MeshingOptions options = {});
ChunkMeshData buildChunkMesh(const Chunk& chunk, MeshingOptions options = {});

// Builds a mesh for the first chunk in the grid.
// This intentionally targets one chunk only for debug clarity.
ChunkMeshData buildSingleChunkMesh(const ChunkGrid& chunkGrid, MeshingOptions options = {});

} // namespace voxelsprout::world
