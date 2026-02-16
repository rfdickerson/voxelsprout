#include "world/magica_voxel.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>

namespace voxelsprout::world {

namespace {

constexpr std::uint32_t fourCc(const char a, const char b, const char c, const char d) {
    return
        static_cast<std::uint32_t>(static_cast<std::uint8_t>(a)) |
        (static_cast<std::uint32_t>(static_cast<std::uint8_t>(b)) << 8u) |
        (static_cast<std::uint32_t>(static_cast<std::uint8_t>(c)) << 16u) |
        (static_cast<std::uint32_t>(static_cast<std::uint8_t>(d)) << 24u);
}

constexpr std::uint32_t kChunkMain = fourCc('M', 'A', 'I', 'N');
constexpr std::uint32_t kChunkSize = fourCc('S', 'I', 'Z', 'E');
constexpr std::uint32_t kChunkXyzi = fourCc('X', 'Y', 'Z', 'I');
constexpr std::uint32_t kChunkRgba = fourCc('R', 'G', 'B', 'A');

std::uint32_t readU32Le(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return
        static_cast<std::uint32_t>(bytes[offset + 0]) |
        (static_cast<std::uint32_t>(bytes[offset + 1]) << 8u) |
        (static_cast<std::uint32_t>(bytes[offset + 2]) << 16u) |
        (static_cast<std::uint32_t>(bytes[offset + 3]) << 24u);
}

std::int32_t readI32Le(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return static_cast<std::int32_t>(readU32Le(bytes, offset));
}

std::uint32_t packRgba(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
    return
        static_cast<std::uint32_t>(r) |
        (static_cast<std::uint32_t>(g) << 8u) |
        (static_cast<std::uint32_t>(b) << 16u) |
        (static_cast<std::uint32_t>(a) << 24u);
}

std::array<std::uint32_t, 256> makeFallbackPalette() {
    std::array<std::uint32_t, 256> palette{};
    palette[0] = 0u;
    for (std::uint32_t index = 1; index < palette.size(); ++index) {
        const std::uint8_t shade = static_cast<std::uint8_t>(index);
        palette[index] = packRgba(shade, shade, shade, 255u);
    }
    return palette;
}

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

std::size_t denseIndex(int x, int y, int z, int sizeX, int sizeY) {
    return static_cast<std::size_t>(x + (y * sizeX) + (z * sizeX * sizeY));
}

bool isSolid(
    const std::vector<std::uint8_t>& densePalette,
    int sizeX,
    int sizeY,
    int sizeZ,
    int x,
    int y,
    int z
) {
    if (x < 0 || x >= sizeX || y < 0 || y >= sizeY || z < 0 || z >= sizeZ) {
        return false;
    }
    return densePalette[denseIndex(x, y, z, sizeX, sizeY)] != 0u;
}

std::uint32_t cornerAoLevel(
    const std::vector<std::uint8_t>& densePalette,
    int sizeX,
    int sizeY,
    int sizeZ,
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

    const bool sideA = isSolid(
        densePalette,
        sizeX,
        sizeY,
        sizeZ,
        baseX + (ux * uSign),
        baseY + (uy * uSign),
        baseZ + (uz * uSign)
    );
    const bool sideB = isSolid(
        densePalette,
        sizeX,
        sizeY,
        sizeZ,
        baseX + (vx * vSign),
        baseY + (vy * vSign),
        baseZ + (vz * vSign)
    );
    const bool cornerSolid = isSolid(
        densePalette,
        sizeX,
        sizeY,
        sizeZ,
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

constexpr std::uint8_t kMaterialPalette = 6u;

std::uint8_t quantizeBaseColorIndex(
    std::uint32_t rgba,
    std::array<std::uint32_t, 16>& paletteSlots,
    std::uint8_t& paletteSlotCount
) {
    for (std::uint8_t i = 0; i < paletteSlotCount; ++i) {
        if (paletteSlots[i] == rgba) {
            return i;
        }
    }

    if (paletteSlotCount < static_cast<std::uint8_t>(paletteSlots.size())) {
        const std::uint8_t slot = paletteSlotCount;
        paletteSlots[slot] = rgba;
        ++paletteSlotCount;
        return slot;
    }

    const int r = static_cast<int>(rgba & 0xFFu);
    const int g = static_cast<int>((rgba >> 8u) & 0xFFu);
    const int b = static_cast<int>((rgba >> 16u) & 0xFFu);

    std::uint8_t nearest = 0u;
    int bestDistance = std::numeric_limits<int>::max();
    for (std::uint8_t i = 0; i < static_cast<std::uint8_t>(paletteSlots.size()); ++i) {
        const std::uint32_t candidate = paletteSlots[i];
        const int cr = static_cast<int>(candidate & 0xFFu);
        const int cg = static_cast<int>((candidate >> 8u) & 0xFFu);
        const int cb = static_cast<int>((candidate >> 16u) & 0xFFu);
        const int dr = r - cr;
        const int dg = g - cg;
        const int db = b - cb;
        const int distance = (dr * dr) + (dg * dg) + (db * db);
        if (distance < bestDistance) {
            bestDistance = distance;
            nearest = i;
        }
    }
    return nearest;
}

constexpr std::uint16_t kEmptyMaskKey = 0xFFFFu;

void faceSliceDimensionsForVolume(
    std::uint32_t faceId,
    int sizeX,
    int sizeY,
    int sizeZ,
    int& outSliceCount,
    int& outUCount,
    int& outVCount
) {
    switch (faceId) {
    case 0u:
    case 1u:
        outSliceCount = sizeX;
        outUCount = sizeY;
        outVCount = sizeZ;
        break;
    case 2u:
    case 3u:
        outSliceCount = sizeY;
        outUCount = sizeX;
        outVCount = sizeZ;
        break;
    case 4u:
    case 5u:
    default:
        outSliceCount = sizeZ;
        outUCount = sizeX;
        outVCount = sizeY;
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

std::uint8_t faceCornerAoSignatureDense(
    const std::vector<std::uint8_t>& densePalette,
    int sizeX,
    int sizeY,
    int sizeZ,
    int x,
    int y,
    int z,
    std::uint32_t faceId
) {
    std::uint8_t signature = 0;
    for (std::uint32_t corner = 0; corner < 4u; ++corner) {
        const std::uint32_t ao = cornerAoLevel(densePalette, sizeX, sizeY, sizeZ, x, y, z, faceId, corner) & 0x3u;
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

bool appendGreedyFaceQuadLocal(
    ChunkMeshData& mesh,
    std::uint32_t faceId,
    int slice,
    int u,
    int v,
    int width,
    int height,
    std::uint8_t material,
    std::uint8_t aoSignature,
    std::uint8_t baseColorIndex,
    std::uint32_t lodLevel,
    int sizeX,
    int sizeY,
    int sizeZ
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
        if (baseX < 0 || baseX >= sizeX ||
            baseY < 0 || baseY >= sizeY ||
            baseZ < 0 || baseZ >= sizeZ) {
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

void appendDenseVoxelFaceLocal(
    ChunkMeshData& mesh,
    const std::vector<std::uint8_t>& densePalette,
    int sizeX,
    int sizeY,
    int sizeZ,
    int globalX,
    int globalY,
    int globalZ,
    int localX,
    int localY,
    int localZ,
    std::uint32_t faceId,
    std::uint8_t material,
    std::uint8_t baseColorIndex
) {
    const std::uint32_t baseVertex = static_cast<std::uint32_t>(mesh.vertices.size());
    for (std::uint32_t corner = 0; corner < 4u; ++corner) {
        const std::uint32_t ao = cornerAoLevel(
            densePalette,
            sizeX,
            sizeY,
            sizeZ,
            globalX,
            globalY,
            globalZ,
            faceId,
            corner
        );
        PackedVoxelVertex vertex{};
        vertex.bits = PackedVoxelVertex::pack(
            static_cast<std::uint32_t>(localX),
            static_cast<std::uint32_t>(localY),
            static_cast<std::uint32_t>(localZ),
            faceId,
            corner,
            ao,
            material,
            baseColorIndex,
            0u
        );
        mesh.vertices.push_back(vertex);
    }
    mesh.indices.push_back(baseVertex + 0u);
    mesh.indices.push_back(baseVertex + 1u);
    mesh.indices.push_back(baseVertex + 2u);
    mesh.indices.push_back(baseVertex + 0u);
    mesh.indices.push_back(baseVertex + 2u);
    mesh.indices.push_back(baseVertex + 3u);
}

} // namespace

bool loadMagicaVoxelModel(const std::filesystem::path& path, MagicaVoxelModel& outModel) {
    outModel = MagicaVoxelModel{};

    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return false;
    }

    const std::streamsize fileSize = stream.tellg();
    if (fileSize <= 0) {
        return false;
    }
    stream.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(fileSize));
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), fileSize)) {
        return false;
    }

    if (bytes.size() < 20u) {
        return false;
    }
    if (std::memcmp(bytes.data(), "VOX ", 4) != 0) {
        return false;
    }

    std::size_t offset = 4u;
    (void)readU32Le(bytes, offset); // version
    offset += 4u;

    if ((offset + 12u) > bytes.size()) {
        return false;
    }
    const std::uint32_t mainId = readU32Le(bytes, offset + 0u);
    const std::uint32_t mainContentSize = readU32Le(bytes, offset + 4u);
    const std::uint32_t mainChildrenSize = readU32Le(bytes, offset + 8u);
    offset += 12u;

    if (mainId != kChunkMain) {
        return false;
    }
    const std::size_t mainContentEnd = offset + static_cast<std::size_t>(mainContentSize);
    const std::size_t mainChildrenEnd = mainContentEnd + static_cast<std::size_t>(mainChildrenSize);
    if (mainChildrenEnd > bytes.size()) {
        return false;
    }

    std::array<std::uint32_t, 256> palette = makeFallbackPalette();
    bool hasPalette = false;

    bool havePendingSize = false;
    int pendingSizeX = 0;
    int pendingSizeY = 0;
    int pendingSizeZ = 0;
    bool loadedFirstModel = false;

    std::size_t cursor = mainContentEnd;
    while ((cursor + 12u) <= mainChildrenEnd) {
        const std::uint32_t chunkId = readU32Le(bytes, cursor + 0u);
        const std::uint32_t contentSize = readU32Le(bytes, cursor + 4u);
        const std::uint32_t childrenSize = readU32Le(bytes, cursor + 8u);
        cursor += 12u;

        const std::size_t contentBegin = cursor;
        const std::size_t contentEnd = contentBegin + static_cast<std::size_t>(contentSize);
        const std::size_t childrenEnd = contentEnd + static_cast<std::size_t>(childrenSize);
        if (childrenEnd > mainChildrenEnd) {
            return false;
        }

        if (chunkId == kChunkSize) {
            if (contentSize >= 12u) {
                const int sx = readI32Le(bytes, contentBegin + 0u);
                const int sy = readI32Le(bytes, contentBegin + 4u);
                const int sz = readI32Le(bytes, contentBegin + 8u);
                if (sx > 0 && sy > 0 && sz > 0) {
                    pendingSizeX = sx;
                    pendingSizeY = sy;
                    pendingSizeZ = sz;
                    havePendingSize = true;
                }
            }
        } else if (chunkId == kChunkXyzi) {
            if (havePendingSize && contentSize >= 4u) {
                const std::uint32_t voxelCount = readU32Le(bytes, contentBegin);
                const std::size_t voxelBytes = static_cast<std::size_t>(voxelCount) * 4u;
                if ((4u + voxelBytes) <= static_cast<std::size_t>(contentSize) && !loadedFirstModel) {
                    outModel.sizeX = pendingSizeX;
                    outModel.sizeY = pendingSizeY;
                    outModel.sizeZ = pendingSizeZ;
                    outModel.voxels.clear();
                    outModel.voxels.reserve(voxelCount);

                    std::size_t voxelCursor = contentBegin + 4u;
                    for (std::uint32_t i = 0; i < voxelCount; ++i) {
                        const std::uint8_t x = bytes[voxelCursor + 0u];
                        const std::uint8_t y = bytes[voxelCursor + 1u];
                        const std::uint8_t z = bytes[voxelCursor + 2u];
                        const std::uint8_t paletteIndex = bytes[voxelCursor + 3u];
                        voxelCursor += 4u;

                        if (paletteIndex == 0u) {
                            continue;
                        }
                        if (static_cast<int>(x) >= pendingSizeX ||
                            static_cast<int>(y) >= pendingSizeY ||
                            static_cast<int>(z) >= pendingSizeZ) {
                            continue;
                        }

                        outModel.voxels.push_back(MagicaVoxel{x, y, z, paletteIndex});
                    }
                    loadedFirstModel = true;
                }
            }
        } else if (chunkId == kChunkRgba) {
            if (contentSize >= 1024u) {
                std::array<std::uint32_t, 256> parsedPalette{};
                parsedPalette[0] = 0u;
                for (std::uint32_t paletteIndex = 1u; paletteIndex < parsedPalette.size(); ++paletteIndex) {
                    const std::size_t paletteByteOffset = contentBegin + ((paletteIndex - 1u) * 4u);
                    const std::uint8_t r = bytes[paletteByteOffset + 0u];
                    const std::uint8_t g = bytes[paletteByteOffset + 1u];
                    const std::uint8_t b = bytes[paletteByteOffset + 2u];
                    const std::uint8_t a = bytes[paletteByteOffset + 3u];
                    parsedPalette[paletteIndex] = packRgba(r, g, b, a);
                }
                palette = parsedPalette;
                hasPalette = true;
            }
        }

        cursor = childrenEnd;
    }

    if (!loadedFirstModel || outModel.voxels.empty()) {
        return false;
    }

    outModel.paletteRgba = palette;
    outModel.hasPalette = hasPalette;
    return true;
}

std::vector<MagicaVoxelMeshChunk> buildMagicaVoxelMeshChunks(const MagicaVoxelModel& model) {
    std::vector<MagicaVoxelMeshChunk> chunks{};

    if (model.sizeX <= 0 || model.sizeY <= 0 || model.sizeZ <= 0 || model.voxels.empty()) {
        return chunks;
    }

    const int transformedSizeX = model.sizeX;
    const int transformedSizeY = model.sizeZ;
    const int transformedSizeZ = model.sizeY;

    std::vector<std::uint8_t> densePalette(
        static_cast<std::size_t>(transformedSizeX * transformedSizeY * transformedSizeZ),
        0u
    );
    std::array<std::uint32_t, 16> baseColorPaletteSlots{};
    std::uint8_t baseColorPaletteSlotCount = 0u;

    for (const MagicaVoxel& voxel : model.voxels) {
        const int transformedX = static_cast<int>(voxel.x);
        const int transformedY = static_cast<int>(voxel.z);
        const int transformedZ = static_cast<int>(voxel.y);
        if (transformedX < 0 || transformedX >= transformedSizeX ||
            transformedY < 0 || transformedY >= transformedSizeY ||
            transformedZ < 0 || transformedZ >= transformedSizeZ) {
            continue;
        }

        densePalette[denseIndex(transformedX, transformedY, transformedZ, transformedSizeX, transformedSizeY)] =
            voxel.paletteIndex;
    }

    constexpr int kTileExtent = 32;
    const int tileCountX = (transformedSizeX + (kTileExtent - 1)) / kTileExtent;
    const int tileCountY = (transformedSizeY + (kTileExtent - 1)) / kTileExtent;
    const int tileCountZ = (transformedSizeZ + (kTileExtent - 1)) / kTileExtent;
    chunks.reserve(static_cast<std::size_t>(tileCountX * tileCountY * tileCountZ));

    for (int tileZ = 0; tileZ < transformedSizeZ; tileZ += kTileExtent) {
        const int tileEndZ = std::min(tileZ + kTileExtent, transformedSizeZ);
        for (int tileY = 0; tileY < transformedSizeY; tileY += kTileExtent) {
            const int tileEndY = std::min(tileY + kTileExtent, transformedSizeY);
            for (int tileX = 0; tileX < transformedSizeX; tileX += kTileExtent) {
                const int tileEndX = std::min(tileX + kTileExtent, transformedSizeX);

                MagicaVoxelMeshChunk chunk{};
                chunk.originX = tileX;
                chunk.originY = tileY;
                chunk.originZ = tileZ;
                ChunkMeshData& mesh = chunk.mesh;
                const int localSizeX = tileEndX - tileX;
                const int localSizeY = tileEndY - tileY;
                const int localSizeZ = tileEndZ - tileZ;
                for (std::uint32_t faceId = 0; faceId < kFaceNeighbors.size(); ++faceId) {
                    int sliceCount = 0;
                    int uCount = 0;
                    int vCount = 0;
                    faceSliceDimensionsForVolume(
                        faceId,
                        localSizeX,
                        localSizeY,
                        localSizeZ,
                        sliceCount,
                        uCount,
                        vCount
                    );
                    std::vector<std::uint16_t> mask(static_cast<std::size_t>(uCount * vCount), kEmptyMaskKey);

                    for (int slice = 0; slice < sliceCount; ++slice) {
                        std::fill(mask.begin(), mask.end(), kEmptyMaskKey);

                        for (int v = 0; v < vCount; ++v) {
                            for (int u = 0; u < uCount; ++u) {
                                int localX = 0;
                                int localY = 0;
                                int localZ = 0;
                                faceSliceCellToVoxel(faceId, slice, u, v, localX, localY, localZ);
                                const int globalX = tileX + localX;
                                const int globalY = tileY + localY;
                                const int globalZ = tileZ + localZ;

                                const std::uint8_t paletteIndex = densePalette[denseIndex(
                                    globalX,
                                    globalY,
                                    globalZ,
                                    transformedSizeX,
                                    transformedSizeY
                                )];
                                if (paletteIndex == 0u) {
                                    continue;
                                }

                                const FaceNeighbor& face = kFaceNeighbors[faceId];
                                if (isSolid(
                                        densePalette,
                                        transformedSizeX,
                                        transformedSizeY,
                                        transformedSizeZ,
                                        globalX + face.nx,
                                        globalY + face.ny,
                                        globalZ + face.nz
                                    )) {
                                    continue;
                                }

                                const std::uint32_t baseColorRgba = model.paletteRgba[paletteIndex];
                                const std::uint8_t baseColorIndex = quantizeBaseColorIndex(
                                    baseColorRgba,
                                    baseColorPaletteSlots,
                                    baseColorPaletteSlotCount
                                );
                                const std::uint8_t material = kMaterialPalette;
                                const std::uint8_t aoSignature = faceCornerAoSignatureDense(
                                    densePalette,
                                    transformedSizeX,
                                    transformedSizeY,
                                    transformedSizeZ,
                                    globalX,
                                    globalY,
                                    globalZ,
                                    faceId
                                );
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
                                const bool mergedQuadAppended = appendGreedyFaceQuadLocal(
                                    mesh,
                                    faceId,
                                    slice,
                                    u,
                                    v,
                                    width,
                                    height,
                                    material,
                                    aoSignature,
                                    baseColorIndex,
                                    0u,
                                    localSizeX,
                                    localSizeY,
                                    localSizeZ
                                );
                                if (!mergedQuadAppended) {
                                    for (int emitV = 0; emitV < height; ++emitV) {
                                        for (int emitU = 0; emitU < width; ++emitU) {
                                            int localX = 0;
                                            int localY = 0;
                                            int localZ = 0;
                                            faceSliceCellToVoxel(faceId, slice, u + emitU, v + emitV, localX, localY, localZ);
                                            const std::uint8_t paletteIndex = densePalette[denseIndex(
                                                tileX + localX,
                                                tileY + localY,
                                                tileZ + localZ,
                                                transformedSizeX,
                                                transformedSizeY
                                            )];
                                            std::uint8_t fallbackBaseColorIndex = 0u;
                                            if (paletteIndex != 0u) {
                                                fallbackBaseColorIndex = quantizeBaseColorIndex(
                                                    model.paletteRgba[paletteIndex],
                                                    baseColorPaletteSlots,
                                                    baseColorPaletteSlotCount
                                                );
                                            }
                                            appendDenseVoxelFaceLocal(
                                                mesh,
                                                densePalette,
                                                transformedSizeX,
                                                transformedSizeY,
                                                transformedSizeZ,
                                                tileX + localX,
                                                tileY + localY,
                                                tileZ + localZ,
                                                localX,
                                                localY,
                                                localZ,
                                                faceId,
                                                material,
                                                fallbackBaseColorIndex
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

                if (!mesh.indices.empty()) {
                    chunks.push_back(std::move(chunk));
                }
            }
        }
    }

    return chunks;
}

ChunkMeshData buildMagicaVoxelMesh(const MagicaVoxelModel& model) {
    ChunkMeshData mesh{};
    if (model.sizeX <= 0 || model.sizeY <= 0 || model.sizeZ <= 0 || model.voxels.empty()) {
        return mesh;
    }

    const int transformedSizeX = model.sizeX;
    const int transformedSizeY = model.sizeZ;
    const int transformedSizeZ = model.sizeY;
    if (transformedSizeX > 32 || transformedSizeY > 32 || transformedSizeZ > 32) {
        return mesh;
    }

    std::vector<MagicaVoxelMeshChunk> chunks = buildMagicaVoxelMeshChunks(model);
    if (!chunks.empty()) {
        return std::move(chunks.front().mesh);
    }
    return mesh;
}

} // namespace voxelsprout::world
