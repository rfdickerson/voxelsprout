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
    std::uint32_t material
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

void appendUniformMacroCellFace(
    ChunkMeshData& mesh,
    const Chunk& chunk,
    int baseX,
    int baseY,
    int baseZ,
    const FaceNeighbor& face,
    std::uint32_t material
) {
    for (int a = 0; a < Chunk::kMacroVoxelSize; ++a) {
        for (int b = 0; b < Chunk::kMacroVoxelSize; ++b) {
            int x = baseX;
            int y = baseY;
            int z = baseZ;

            switch (face.faceId) {
            case 0u: // +X
                x = baseX + (Chunk::kMacroVoxelSize - 1);
                y = baseY + a;
                z = baseZ + b;
                break;
            case 1u: // -X
                x = baseX;
                y = baseY + a;
                z = baseZ + b;
                break;
            case 2u: // +Y
                x = baseX + a;
                y = baseY + (Chunk::kMacroVoxelSize - 1);
                z = baseZ + b;
                break;
            case 3u: // -Y
                x = baseX + a;
                y = baseY;
                z = baseZ + b;
                break;
            case 4u: // +Z
                x = baseX + a;
                y = baseY + b;
                z = baseZ + (Chunk::kMacroVoxelSize - 1);
                break;
            case 5u: // -Z
            default:
                x = baseX + a;
                y = baseY + b;
                z = baseZ;
                break;
            }

            if (isSolidVoxel(chunk, x + face.nx, y + face.ny, z + face.nz)) {
                continue;
            }
            appendVoxelFace(mesh, x, y, z, face.faceId, material);
        }
    }
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

    constexpr std::size_t kMaxVoxelFaces =
        static_cast<std::size_t>(Chunk::kSizeX * Chunk::kSizeY * Chunk::kSizeZ * 6);
    mesh.vertices.reserve(kMaxVoxelFaces * 4);
    mesh.indices.reserve(kMaxVoxelFaces * 6);

    for (int my = 0; my < Chunk::kMacroSizeY; ++my) {
        for (int mz = 0; mz < Chunk::kMacroSizeZ; ++mz) {
            for (int mx = 0; mx < Chunk::kMacroSizeX; ++mx) {
                const Chunk::MacroCell cell = chunk.macroCellAt(mx, my, mz);
                const int baseX = mx * Chunk::kMacroVoxelSize;
                const int baseY = my * Chunk::kMacroVoxelSize;
                const int baseZ = mz * Chunk::kMacroVoxelSize;

                if (cell.resolution == Chunk::CellResolution::Uniform) {
                    if (cell.voxel.type == VoxelType::Empty) {
                        continue;
                    }
                    const std::uint32_t material = materialForVoxelType(cell.voxel.type);
                    for (const FaceNeighbor& face : kFaceNeighbors) {
                        appendUniformMacroCellFace(mesh, chunk, baseX, baseY, baseZ, face, material);
                    }
                    continue;
                }

                // Phase 2: refined macro cells emit true per-voxel surfaces.
                for (int localY = 0; localY < Chunk::kMacroVoxelSize; ++localY) {
                    for (int localZ = 0; localZ < Chunk::kMacroVoxelSize; ++localZ) {
                        for (int localX = 0; localX < Chunk::kMacroVoxelSize; ++localX) {
                            const int x = baseX + localX;
                            const int y = baseY + localY;
                            const int z = baseZ + localZ;
                            const Voxel voxel = chunk.voxelAt(x, y, z);
                            if (voxel.type == VoxelType::Empty) {
                                continue;
                            }

                            const std::uint32_t material = materialForVoxelType(voxel.type);
                            for (const FaceNeighbor& face : kFaceNeighbors) {
                                if (isSolidVoxel(chunk, x + face.nx, y + face.ny, z + face.nz)) {
                                    continue;
                                }
                                appendVoxelFace(mesh, x, y, z, face.faceId, material);
                            }
                        }
                    }
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
