#include "world/ChunkMesher.hpp"

#include "world/Chunk.hpp"
#include "world/Voxel.hpp"

#include <array>

namespace world {

namespace {

struct FaceInfo {
    int nx;
    int ny;
    int nz;

    int ux;
    int uy;
    int uz;

    int vx;
    int vy;
    int vz;

    // Corner signs along u/v for AO sampling.
    std::array<int, 4> cornerSignU;
    std::array<int, 4> cornerSignV;
};

constexpr std::array<FaceInfo, 6> kFaces = {
    FaceInfo{
        +1, 0, 0,
        0, +1, 0,
        0, 0, +1,
        {{-1, +1, +1, -1}},
        {{-1, -1, +1, +1}}
    },
    FaceInfo{
        -1, 0, 0,
        0, +1, 0,
        0, 0, +1,
        {{-1, +1, +1, -1}},
        {{+1, +1, -1, -1}}
    },
    FaceInfo{
        0, +1, 0,
        +1, 0, 0,
        0, 0, +1,
        {{-1, -1, +1, +1}},
        {{-1, +1, +1, -1}}
    },
    FaceInfo{
        0, -1, 0,
        +1, 0, 0,
        0, 0, +1,
        {{-1, -1, +1, +1}},
        {{+1, -1, -1, +1}}
    },
    FaceInfo{
        0, 0, +1,
        +1, 0, 0,
        0, +1, 0,
        {{+1, +1, -1, -1}},
        {{-1, +1, +1, -1}}
    },
    FaceInfo{
        0, 0, -1,
        +1, 0, 0,
        0, +1, 0,
        {{-1, -1, +1, +1}},
        {{-1, +1, +1, -1}}
    },
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

std::uint8_t computeCornerAo(const Chunk& chunk, int x, int y, int z, const FaceInfo& face, int cornerIndex) {
    const int su = face.cornerSignU[cornerIndex];
    const int sv = face.cornerSignV[cornerIndex];

    // AO probes three voxels around this corner on the exposed side of the face.
    // sideA and sideB are axis-adjacent blockers; corner is the diagonal blocker.
    const bool sideA = chunk.isSolid(
        x + face.nx + (su * face.ux),
        y + face.ny + (su * face.uy),
        z + face.nz + (su * face.uz)
    );
    const bool sideB = chunk.isSolid(
        x + face.nx + (sv * face.vx),
        y + face.ny + (sv * face.vy),
        z + face.nz + (sv * face.vz)
    );
    const bool corner = chunk.isSolid(
        x + face.nx + (su * face.ux) + (sv * face.vx),
        y + face.ny + (su * face.uy) + (sv * face.vy),
        z + face.nz + (su * face.uz) + (sv * face.vz)
    );

    if (sideA && sideB) {
        return 0;
    }

    const std::uint32_t occlusionCount =
        (sideA ? 1u : 0u) +
        (sideB ? 1u : 0u) +
        (corner ? 1u : 0u);
    return static_cast<std::uint8_t>(3u - occlusionCount);
}

} // namespace

std::uint32_t PackedVoxelVertex::pack(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t z,
    std::uint32_t face,
    std::uint32_t corner,
    std::uint32_t ao,
    std::uint32_t material
) {
    return ((x & kMask5) << kShiftX) |
           ((y & kMask5) << kShiftY) |
           ((z & kMask5) << kShiftZ) |
           ((face & kMask3) << kShiftFace) |
           ((corner & kMask2) << kShiftCorner) |
           ((ao & kMask2) << kShiftAo) |
           ((material & kMask8) << kShiftMaterial);
}

ChunkMeshData buildChunkMesh(const Chunk& chunk) {
    ChunkMeshData mesh{};
    static_assert(Chunk::kSizeX <= 32 && Chunk::kSizeY <= 32 && Chunk::kSizeZ <= 32, "Packed position fields are 5-bit");

    constexpr std::size_t kMaxFaces = static_cast<std::size_t>(Chunk::kSizeX * Chunk::kSizeY * Chunk::kSizeZ * 6);
    mesh.vertices.reserve(kMaxFaces * 4);
    mesh.indices.reserve(kMaxFaces * 6);

    for (int y = 0; y < Chunk::kSizeY; ++y) {
        for (int z = 0; z < Chunk::kSizeZ; ++z) {
            for (int x = 0; x < Chunk::kSizeX; ++x) {
                const Voxel voxel = chunk.voxelAt(x, y, z);
                if (voxel.type == VoxelType::Empty) {
                    continue;
                }

                const std::uint32_t material = materialForVoxelType(voxel.type);

                for (std::uint32_t faceId = 0; faceId < kFaces.size(); ++faceId) {
                    const FaceInfo& face = kFaces[faceId];
                    if (chunk.isSolid(x + face.nx, y + face.ny, z + face.nz)) {
                        continue;
                    }

                    const std::uint32_t baseVertex = static_cast<std::uint32_t>(mesh.vertices.size());
                    for (std::uint32_t corner = 0; corner < 4; ++corner) {
                        const std::uint8_t ao = computeCornerAo(chunk, x, y, z, face, static_cast<int>(corner));
                        PackedVoxelVertex vertex{};
                        vertex.bits = PackedVoxelVertex::pack(
                            static_cast<std::uint32_t>(x),
                            static_cast<std::uint32_t>(y),
                            static_cast<std::uint32_t>(z),
                            faceId,
                            corner,
                            ao,
                            material
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
            }
        }
    }

    return mesh;
}

ChunkMeshData buildSingleChunkMesh(const ChunkGrid& chunkGrid) {
    if (chunkGrid.chunks().empty()) {
        return ChunkMeshData{};
    }
    return buildChunkMesh(chunkGrid.chunks().front());
}

} // namespace world
