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

struct CornerAxes {
    int x;
    int y;
    int z;
};

constexpr std::array<std::array<CornerAxes, 4>, 6> kFaceCornerAxes = {{
    // +X
    {{{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1}}},
    // -X
    {{{0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {0, 0, 0}}},
    // +Y
    {{{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}}},
    // -Y
    {{{0, 0, 1}, {0, 0, 0}, {1, 0, 0}, {1, 0, 1}}},
    // +Z
    {{{1, 0, 1}, {1, 1, 1}, {0, 1, 1}, {0, 0, 1}}},
    // -Z
    {{{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}},
}};

void faceAoAxes(std::uint32_t faceId, int& ux, int& uy, int& uz, int& vx, int& vy, int& vz) {
    switch (faceId) {
    case 0u:
    case 1u:
        ux = 0; uy = 1; uz = 0;
        vx = 0; vy = 0; vz = 1;
        break;
    case 2u:
    case 3u:
        ux = 1; uy = 0; uz = 0;
        vx = 0; vy = 0; vz = 1;
        break;
    case 4u:
    case 5u:
    default:
        ux = 1; uy = 0; uz = 0;
        vx = 0; vy = 1; vz = 0;
        break;
    }
}

std::uint32_t cornerAoLevel(
    const Chunk& chunk,
    int x,
    int y,
    int z,
    std::uint32_t faceId,
    std::uint32_t corner
) {
    if (faceId >= kFaceNeighbors.size() || corner >= 4u) {
        return 3u;
    }

    const FaceNeighbor& face = kFaceNeighbors[faceId];
    const CornerAxes& cornerAxes = kFaceCornerAxes[faceId][corner];

    int ux = 0;
    int uy = 0;
    int uz = 0;
    int vx = 0;
    int vy = 0;
    int vz = 0;
    faceAoAxes(faceId, ux, uy, uz, vx, vy, vz);

    const int uSign = ((ux != 0 ? cornerAxes.x : (uy != 0 ? cornerAxes.y : cornerAxes.z)) != 0) ? +1 : -1;
    const int vSign = ((vx != 0 ? cornerAxes.x : (vy != 0 ? cornerAxes.y : cornerAxes.z)) != 0) ? +1 : -1;

    const int baseX = x + face.nx;
    const int baseY = y + face.ny;
    const int baseZ = z + face.nz;

    const bool sideA = isSolidVoxel(chunk, baseX + (ux * uSign), baseY + (uy * uSign), baseZ + (uz * uSign));
    const bool sideB = isSolidVoxel(chunk, baseX + (vx * vSign), baseY + (vy * vSign), baseZ + (vz * vSign));
    const bool cornerSolid = isSolidVoxel(
        chunk,
        baseX + (ux * uSign) + (vx * vSign),
        baseY + (uy * uSign) + (vy * vSign),
        baseZ + (uz * uSign) + (vz * vSign)
    );

    int occlusion = 0;
    if (sideA && sideB) {
        occlusion = 3;
    } else {
        occlusion = (sideA ? 1 : 0) + (sideB ? 1 : 0) + (cornerSolid ? 1 : 0);
    }
    return static_cast<std::uint32_t>(3 - occlusion);
}

void appendVoxelFace(
    const Chunk& chunk,
    ChunkMeshData& mesh,
    int x,
    int y,
    int z,
    std::uint32_t faceId,
    std::uint32_t material,
    std::uint32_t lodLevel
) {
    const std::uint32_t baseVertex = static_cast<std::uint32_t>(mesh.vertices.size());
    for (std::uint32_t corner = 0; corner < 4; ++corner) {
        const std::uint32_t ao = cornerAoLevel(chunk, x, y, z, faceId, corner);
        PackedVoxelVertex vertex{};
        vertex.bits = PackedVoxelVertex::pack(
            static_cast<std::uint32_t>(x),
            static_cast<std::uint32_t>(y),
            static_cast<std::uint32_t>(z),
            faceId,
            corner,
            ao,
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
                    appendVoxelFace(chunk, baseMesh, x, y, z, face.faceId, material, 0u);
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
