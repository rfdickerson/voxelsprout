#include "world/chunk_mesher.h"

#include "world/chunk.h"
#include "world/voxel.h"

#include <array>
#include <cstddef>

namespace voxelsprout::world {

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

constexpr std::uint8_t kMaterialStone = 1u;
constexpr std::uint8_t kMaterialDirt = 2u;
constexpr std::uint8_t kMaterialGrass = 3u;
constexpr std::uint8_t kMaterialWood = 4u;
constexpr std::uint8_t kMaterialRed = 5u;
constexpr std::uint8_t kMaterialPalette = 6u;

std::uint8_t materialForVoxelType(VoxelType type) {
    switch (type) {
    case VoxelType::Stone:
        return kMaterialStone;
    case VoxelType::Dirt:
        return kMaterialDirt;
    case VoxelType::Grass:
        return kMaterialGrass;
    case VoxelType::Wood:
        return kMaterialWood;
    case VoxelType::SolidRed:
        return kMaterialRed;
    case VoxelType::Empty:
    default:
        return 0;
    }
}

std::uint8_t packedBaseColorIndexForVoxel(const Voxel& voxel) {
    return (voxel.baseColorIndex <= PackedVoxelVertex::kMask4) ? voxel.baseColorIndex : 0u;
}

std::uint8_t materialForVoxel(const Voxel& voxel) {
    if (voxel.type == VoxelType::Empty) {
        return 0u;
    }
    if (voxel.baseColorIndex <= PackedVoxelVertex::kMask4) {
        return kMaterialPalette;
    }
    return materialForVoxelType(voxel.type);
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
    std::uint32_t baseColorIndex,
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
            baseColorIndex,
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

constexpr std::uint16_t kEmptyMaskKey = 0xFFFFu;

void faceSliceDimensions(std::uint32_t faceId, int& outSliceCount, int& outUCount, int& outVCount) {
    switch (faceId) {
    case 0u:
    case 1u:
        outSliceCount = Chunk::kSizeX;
        outUCount = Chunk::kSizeY;
        outVCount = Chunk::kSizeZ;
        break;
    case 2u:
    case 3u:
        outSliceCount = Chunk::kSizeY;
        outUCount = Chunk::kSizeX;
        outVCount = Chunk::kSizeZ;
        break;
    case 4u:
    case 5u:
    default:
        outSliceCount = Chunk::kSizeZ;
        outUCount = Chunk::kSizeX;
        outVCount = Chunk::kSizeY;
        break;
    }
}

void faceSliceCellToVoxel(std::uint32_t faceId, int slice, int u, int v, int& outX, int& outY, int& outZ) {
    switch (faceId) {
    case 0u:
    case 1u:
        outX = slice;
        outY = u;
        outZ = v;
        break;
    case 2u:
    case 3u:
        outX = u;
        outY = slice;
        outZ = v;
        break;
    case 4u:
    case 5u:
    default:
        outX = u;
        outY = v;
        outZ = slice;
        break;
    }
}

void faceRectCornerGrid(
    std::uint32_t faceId,
    int slice,
    int u,
    int v,
    int width,
    int height,
    std::uint32_t corner,
    int& outX,
    int& outY,
    int& outZ
) {
    switch (faceId) {
    case 0u: // +X
        if (corner == 0u) { outX = slice + 1; outY = u; outZ = v; return; }
        if (corner == 1u) { outX = slice + 1; outY = u + width; outZ = v; return; }
        if (corner == 2u) { outX = slice + 1; outY = u + width; outZ = v + height; return; }
        outX = slice + 1; outY = u; outZ = v + height; return;
    case 1u: // -X
        if (corner == 0u) { outX = slice; outY = u; outZ = v + height; return; }
        if (corner == 1u) { outX = slice; outY = u + width; outZ = v + height; return; }
        if (corner == 2u) { outX = slice; outY = u + width; outZ = v; return; }
        outX = slice; outY = u; outZ = v; return;
    case 2u: // +Y
        if (corner == 0u) { outX = u; outY = slice + 1; outZ = v; return; }
        if (corner == 1u) { outX = u; outY = slice + 1; outZ = v + height; return; }
        if (corner == 2u) { outX = u + width; outY = slice + 1; outZ = v + height; return; }
        outX = u + width; outY = slice + 1; outZ = v; return;
    case 3u: // -Y
        if (corner == 0u) { outX = u; outY = slice; outZ = v + height; return; }
        if (corner == 1u) { outX = u; outY = slice; outZ = v; return; }
        if (corner == 2u) { outX = u + width; outY = slice; outZ = v; return; }
        outX = u + width; outY = slice; outZ = v + height; return;
    case 4u: // +Z
        if (corner == 0u) { outX = u + width; outY = v; outZ = slice + 1; return; }
        if (corner == 1u) { outX = u + width; outY = v + height; outZ = slice + 1; return; }
        if (corner == 2u) { outX = u; outY = v + height; outZ = slice + 1; return; }
        outX = u; outY = v; outZ = slice + 1; return;
    case 5u: // -Z
    default:
        if (corner == 0u) { outX = u; outY = v; outZ = slice; return; }
        if (corner == 1u) { outX = u; outY = v + height; outZ = slice; return; }
        if (corner == 2u) { outX = u + width; outY = v + height; outZ = slice; return; }
        outX = u + width; outY = v; outZ = slice; return;
    }
}

std::uint8_t faceCornerAoSignature(
    const Chunk& chunk,
    int x,
    int y,
    int z,
    std::uint32_t faceId
) {
    std::uint8_t signature = 0;
    for (std::uint32_t corner = 0; corner < 4u; ++corner) {
        const std::uint32_t ao = cornerAoLevel(chunk, x, y, z, faceId, corner) & 0x3u;
        signature |= static_cast<std::uint8_t>(ao << (corner * 2u));
    }
    return signature;
}

std::uint16_t makeMaskKey(std::uint8_t material, std::uint8_t aoSignature, std::uint8_t baseColorIndex) {
    // 16-bit greedy mask key layout:
    // bits 12..15: material (4 bits)
    // bits  4..11: AO signature (8 bits; 4 corners x 2 bits)
    // bits  0.. 3: base color index (4 bits)
    return static_cast<std::uint16_t>(
        (static_cast<std::uint16_t>(material & PackedVoxelVertex::kMask4) << 12u) |
        (static_cast<std::uint16_t>(aoSignature) << 4u) |
        static_cast<std::uint16_t>(baseColorIndex & PackedVoxelVertex::kMask4)
    );
}

bool appendGreedyFaceQuad(
    ChunkMeshData& mesh,
    std::uint32_t faceId,
    int slice,
    int u,
    int v,
    int width,
    int height,
    std::uint8_t material,
    std::uint8_t aoSignature,
    std::uint32_t baseColorIndex,
    std::uint32_t lodLevel
) {
    const std::uint32_t baseVertex = static_cast<std::uint32_t>(mesh.vertices.size());
    for (std::uint32_t corner = 0; corner < 4u; ++corner) {
        int gridX = 0;
        int gridY = 0;
        int gridZ = 0;
        faceRectCornerGrid(faceId, slice, u, v, width, height, corner, gridX, gridY, gridZ);
        const CornerAxes& offset = kFaceCornerAxes[faceId][corner];
        const int baseX = gridX - offset.x;
        const int baseY = gridY - offset.y;
        const int baseZ = gridZ - offset.z;
        if (baseX < 0 || baseX >= Chunk::kSizeX ||
            baseY < 0 || baseY >= Chunk::kSizeY ||
            baseZ < 0 || baseZ >= Chunk::kSizeZ) {
            return false;
        }
        const std::uint32_t ao = (aoSignature >> (corner * 2u)) & 0x3u;
        PackedVoxelVertex vertex{};
        vertex.bits = PackedVoxelVertex::pack(
            static_cast<std::uint32_t>(baseX),
            static_cast<std::uint32_t>(baseY),
            static_cast<std::uint32_t>(baseZ),
            faceId,
            corner,
            ao,
            material,
            baseColorIndex,
            lodLevel
        );
        mesh.vertices.push_back(vertex);
    }

    mesh.indices.push_back(baseVertex + 0u);
    mesh.indices.push_back(baseVertex + 1u);
    mesh.indices.push_back(baseVertex + 2u);
    mesh.indices.push_back(baseVertex + 0u);
    mesh.indices.push_back(baseVertex + 2u);
    mesh.indices.push_back(baseVertex + 3u);
    return true;
}

ChunkLodMeshes buildChunkLodMeshesGreedy(const Chunk& chunk) {
    ChunkLodMeshes meshes{};
    static_assert(Chunk::kSizeX <= 32 && Chunk::kSizeY <= 32 && Chunk::kSizeZ <= 32, "Packed position fields are 5-bit");
    ChunkMeshData& baseMesh = meshes.lodMeshes[0];

    for (std::uint32_t faceId = 0; faceId < kFaceNeighbors.size(); ++faceId) {
        int sliceCount = 0;
        int uCount = 0;
        int vCount = 0;
        faceSliceDimensions(faceId, sliceCount, uCount, vCount);
        std::vector<std::uint16_t> mask(static_cast<std::size_t>(uCount * vCount), kEmptyMaskKey);

        for (int slice = 0; slice < sliceCount; ++slice) {
            std::fill(mask.begin(), mask.end(), kEmptyMaskKey);

            for (int v = 0; v < vCount; ++v) {
                for (int u = 0; u < uCount; ++u) {
                    int x = 0;
                    int y = 0;
                    int z = 0;
                    faceSliceCellToVoxel(faceId, slice, u, v, x, y, z);

                    const Voxel voxel = chunk.voxelAt(x, y, z);
                    if (voxel.type == VoxelType::Empty) {
                        continue;
                    }

                    const FaceNeighbor& face = kFaceNeighbors[faceId];
                    if (isSolidVoxel(chunk, x + face.nx, y + face.ny, z + face.nz)) {
                        continue;
                    }

                    const std::uint8_t material = materialForVoxel(voxel);
                    const std::uint8_t baseColorIndex = packedBaseColorIndexForVoxel(voxel);
                    const std::uint8_t aoSignature = faceCornerAoSignature(chunk, x, y, z, faceId);
                    const std::size_t maskIndex = static_cast<std::size_t>(u + (v * uCount));
                    mask[maskIndex] = makeMaskKey(material, aoSignature, baseColorIndex);
                }
            }

            for (int v = 0; v < vCount; ++v) {
                for (int u = 0; u < uCount;) {
                    const std::size_t startIndex = static_cast<std::size_t>(u + (v * uCount));
                    const std::uint16_t key = mask[startIndex];
                    if (key == kEmptyMaskKey) {
                        ++u;
                        continue;
                    }

                    int width = 1;
                    while ((u + width) < uCount) {
                        const std::size_t widthIndex = static_cast<std::size_t>((u + width) + (v * uCount));
                        if (mask[widthIndex] != key) {
                            break;
                        }
                        ++width;
                    }

                    int height = 1;
                    bool canGrow = true;
                    while ((v + height) < vCount && canGrow) {
                        for (int offsetU = 0; offsetU < width; ++offsetU) {
                            const std::size_t growIndex =
                                static_cast<std::size_t>((u + offsetU) + ((v + height) * uCount));
                            if (mask[growIndex] != key) {
                                canGrow = false;
                                break;
                            }
                        }
                        if (canGrow) {
                            ++height;
                        }
                    }

                    const std::uint8_t material =
                        static_cast<std::uint8_t>((key >> 12u) & PackedVoxelVertex::kMask4);
                    const std::uint8_t aoSignature = static_cast<std::uint8_t>((key >> 4u) & 0xFFu);
                    const std::uint8_t baseColorIndex =
                        static_cast<std::uint8_t>(key & PackedVoxelVertex::kMask4);
                    const bool mergedQuadAppended =
                        appendGreedyFaceQuad(
                            baseMesh,
                            faceId,
                            slice,
                            u,
                            v,
                            width,
                            height,
                            material,
                            aoSignature,
                            baseColorIndex,
                            0u
                        );
                    if (!mergedQuadAppended) {
                        // Fallback path: preserve correctness if a merged quad cannot be encoded.
                        for (int emitV = 0; emitV < height; ++emitV) {
                            for (int emitU = 0; emitU < width; ++emitU) {
                                int x = 0;
                                int y = 0;
                                int z = 0;
                                faceSliceCellToVoxel(faceId, slice, u + emitU, v + emitV, x, y, z);
                                const Voxel voxel = chunk.voxelAt(x, y, z);
                                const std::uint8_t fallbackMaterial = materialForVoxel(voxel);
                                const std::uint8_t fallbackBaseColorIndex = packedBaseColorIndexForVoxel(voxel);
                                appendVoxelFace(
                                    chunk,
                                    baseMesh,
                                    x,
                                    y,
                                    z,
                                    faceId,
                                    fallbackMaterial,
                                    fallbackBaseColorIndex,
                                    0u
                                );
                            }
                        }
                    }

                    for (int clearV = 0; clearV < height; ++clearV) {
                        for (int clearU = 0; clearU < width; ++clearU) {
                            const std::size_t clearIndex =
                                static_cast<std::size_t>((u + clearU) + ((v + clearV) * uCount));
                            mask[clearIndex] = kEmptyMaskKey;
                        }
                    }

                    u += width;
                }
            }
        }
    }

    return meshes;
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
    std::uint32_t baseColorIndex,
    std::uint32_t lodLevel
) {
    return ((x & kMask5) << kShiftX) |
           ((y & kMask5) << kShiftY) |
           ((z & kMask5) << kShiftZ) |
           ((face & kMask3) << kShiftFace) |
           ((corner & kMask2) << kShiftCorner) |
           ((ao & kMask2) << kShiftAo) |
           ((material & kMask4) << kShiftMaterial) |
           ((baseColorIndex & kMask4) << kShiftBaseColor) |
           ((lodLevel & kMask2) << kShiftLodLevel);
}

ChunkLodMeshes buildChunkLodMeshesNaive(const Chunk& chunk) {
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

                const std::uint8_t material = materialForVoxel(voxel);
                const std::uint8_t baseColorIndex = packedBaseColorIndexForVoxel(voxel);
                for (const FaceNeighbor& face : kFaceNeighbors) {
                    if (isSolidVoxel(chunk, x + face.nx, y + face.ny, z + face.nz)) {
                        continue;
                    }
                    appendVoxelFace(chunk, baseMesh, x, y, z, face.faceId, material, baseColorIndex, 0u);
                }
            }
        }
    }

    return meshes;
}

ChunkLodMeshes buildChunkLodMeshes(const Chunk& chunk, MeshingOptions options) {
    switch (options.mode) {
    case MeshingMode::Greedy:
        return buildChunkLodMeshesGreedy(chunk);
    case MeshingMode::Naive:
    default:
        return buildChunkLodMeshesNaive(chunk);
    }
}

ChunkMeshData buildChunkMesh(const Chunk& chunk, MeshingOptions options) {
    const ChunkLodMeshes lodMeshes = buildChunkLodMeshes(chunk, options);
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

ChunkMeshData buildSingleChunkMesh(const ChunkGrid& chunkGrid, MeshingOptions options) {
    if (chunkGrid.chunks().empty()) {
        return ChunkMeshData{};
    }
    return buildChunkMesh(chunkGrid.chunks().front(), options);
}

} // namespace voxelsprout::world
