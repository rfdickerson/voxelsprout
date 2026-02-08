#include "world/ChunkMesher.hpp"

#include "world/Chunk.hpp"
#include "world/Voxel.hpp"

#include <array>
#include <cstddef>

namespace world {

namespace {

struct FaceNeighbor {
    int nx;
    int ny;
    int nz;
    std::uint32_t faceId;
};

constexpr std::array<FaceNeighbor, 6> kFaceNeighbors = {
    FaceNeighbor{+1, 0, 0, 0u},
    FaceNeighbor{-1, 0, 0, 1u},
    FaceNeighbor{0, +1, 0, 2u},
    FaceNeighbor{0, -1, 0, 3u},
    FaceNeighbor{0, 0, +1, 4u},
    FaceNeighbor{0, 0, -1, 5u},
};

std::uint8_t materialForVoxelType(VoxelType type) {
    switch (type) {
    case VoxelType::Solid:
        return 1;
    case VoxelType::Empty:
    default:
        return 0;
    }
}

bool isSolidVoxel(const Chunk& chunk, int x, int y, int z) {
    return chunk.voxelAt(x, y, z).type != VoxelType::Empty;
}

void appendVoxelFace(
    ChunkMeshData& mesh,
    int x,
    int y,
    int z,
    std::uint32_t faceId,
    std::uint32_t material,
    std::uint32_t lodLevel
) {
    constexpr std::uint32_t kAo = 3u;
    const std::uint32_t baseVertex = static_cast<std::uint32_t>(mesh.vertices.size());
    for (std::uint32_t corner = 0; corner < 4; ++corner) {
        PackedVoxelVertex vertex{};
        vertex.bits = PackedVoxelVertex::pack(
            static_cast<std::uint32_t>(x),
            static_cast<std::uint32_t>(y),
            static_cast<std::uint32_t>(z),
            faceId,
            corner,
            kAo,
            material,
            lodLevel
        );
        mesh.vertices.push_back(vertex);
    }

    mesh.indices.push_back(baseVertex + 0);
    mesh.indices.push_back(baseVertex + 1);
    mesh.indices.push_back(baseVertex + 2);
    mesh.indices.push_back(baseVertex + 0);
    mesh.indices.push_back(baseVertex + 2);
    mesh.indices.push_back(baseVertex + 3);
}

} // namespace

std::uint32_t PackedVoxelVertex::pack(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t z,
    std::uint32_t face,
    std::uint32_t corner,
    std::uint32_t ao,
    std::uint32_t material,
    std::uint32_t lodLevel
) {
    return ((x & kMask5) << kShiftX) |
           ((y & kMask5) << kShiftY) |
           ((z & kMask5) << kShiftZ) |
           ((face & kMask3) << kShiftFace) |
           ((corner & kMask2) << kShiftCorner) |
           ((ao & kMask2) << kShiftAo) |
           ((material & kMask8) << kShiftMaterial) |
           ((lodLevel & kMask2) << kShiftLodLevel);
}

ChunkLodMeshes buildChunkLodMeshes(const Chunk& chunk) {
    ChunkLodMeshes meshes{};
    static_assert(Chunk::kSizeX <= 32 && Chunk::kSizeY <= 32 && Chunk::kSizeZ <= 32, "Packed position fields are 5-bit");

    ChunkMeshData& baseMesh = meshes.lodMeshes[0];
    baseMesh.vertices.reserve(static_cast<std::size_t>(Chunk::kSizeX * Chunk::kSizeY * Chunk::kSizeZ * 6 * 4));
    baseMesh.indices.reserve(static_cast<std::size_t>(Chunk::kSizeX * Chunk::kSizeY * Chunk::kSizeZ * 6 * 6));

    for (int y = 0; y < Chunk::kSizeY; ++y) {
        for (int z = 0; z < Chunk::kSizeZ; ++z) {
            for (int x = 0; x < Chunk::kSizeX; ++x) {
                const Voxel voxel = chunk.voxelAt(x, y, z);
                if (voxel.type == VoxelType::Empty) {
                    continue;
                }

                const std::uint32_t material = materialForVoxelType(voxel.type);
                for (const FaceNeighbor& face : kFaceNeighbors) {
                    if (isSolidVoxel(chunk, x + face.nx, y + face.ny, z + face.nz)) {
                        continue;
                    }
                    appendVoxelFace(baseMesh, x, y, z, face.faceId, material, 0u);
                }
            }
        }
    }

    return meshes;
}

ChunkMeshData buildChunkMesh(const Chunk& chunk) {
    const ChunkLodMeshes lodMeshes = buildChunkLodMeshes(chunk);
    ChunkMeshData merged{};
    std::size_t vertexTotal = 0;
    std::size_t indexTotal = 0;
    for (const ChunkMeshData& mesh : lodMeshes.lodMeshes) {
        vertexTotal += mesh.vertices.size();
        indexTotal += mesh.indices.size();
    }
    merged.vertices.reserve(vertexTotal);
    merged.indices.reserve(indexTotal);

    for (const ChunkMeshData& mesh : lodMeshes.lodMeshes) {
        const std::uint32_t baseVertex = static_cast<std::uint32_t>(merged.vertices.size());
        merged.vertices.insert(merged.vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
        for (const std::uint32_t index : mesh.indices) {
            merged.indices.push_back(baseVertex + index);
        }
    }

    return merged;
}

ChunkMeshData buildSingleChunkMesh(const ChunkGrid& chunkGrid) {
    if (chunkGrid.chunks().empty()) {
        return ChunkMeshData{};
    }
    return buildChunkMesh(chunkGrid.chunks().front());
}

} // namespace world
