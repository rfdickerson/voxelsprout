#include "world/world.h"

#include "core/log.h"
#include "world/magica_voxel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace voxelsprout::world {

namespace {

MagicaVoxelModel downscaleMagicaModel(const MagicaVoxelModel& source, float scale) {
    if (scale <= 0.0f || scale >= 0.999f) {
        return source;
    }

    MagicaVoxelModel scaled = source;
    scaled.voxels.clear();

    const int scaledSizeX = std::max(1, static_cast<int>(std::ceil(static_cast<float>(source.sizeX) * scale)));
    const int scaledSizeY = std::max(1, static_cast<int>(std::ceil(static_cast<float>(source.sizeY) * scale)));
    const int scaledSizeZ = std::max(1, static_cast<int>(std::ceil(static_cast<float>(source.sizeZ) * scale)));
    scaled.sizeX = scaledSizeX;
    scaled.sizeY = scaledSizeY;
    scaled.sizeZ = scaledSizeZ;

    const std::size_t cellCount =
        static_cast<std::size_t>(scaledSizeX) * static_cast<std::size_t>(scaledSizeY) * static_cast<std::size_t>(scaledSizeZ);
    std::vector<std::uint8_t> densePalette(cellCount, 0u);
    auto denseIndex = [&](int x, int y, int z) -> std::size_t {
        return static_cast<std::size_t>(x + (y * scaledSizeX) + (z * scaledSizeX * scaledSizeY));
    };

    for (const MagicaVoxel& voxel : source.voxels) {
        const int scaledX = std::clamp(static_cast<int>(std::floor(static_cast<float>(voxel.x) * scale)), 0, scaledSizeX - 1);
        const int scaledY = std::clamp(static_cast<int>(std::floor(static_cast<float>(voxel.y) * scale)), 0, scaledSizeY - 1);
        const int scaledZ = std::clamp(static_cast<int>(std::floor(static_cast<float>(voxel.z) * scale)), 0, scaledSizeZ - 1);
        const std::size_t index = denseIndex(scaledX, scaledY, scaledZ);
        if (densePalette[index] == 0u) {
            densePalette[index] = voxel.paletteIndex;
        }
    }

    for (int z = 0; z < scaledSizeZ; ++z) {
        for (int y = 0; y < scaledSizeY; ++y) {
            for (int x = 0; x < scaledSizeX; ++x) {
                const std::uint8_t paletteIndex = densePalette[denseIndex(x, y, z)];
                if (paletteIndex == 0u) {
                    continue;
                }
                scaled.voxels.push_back(MagicaVoxel{
                    static_cast<std::uint8_t>(x),
                    static_cast<std::uint8_t>(y),
                    static_cast<std::uint8_t>(z),
                    paletteIndex
                });
            }
        }
    }

    return scaled;
}

VoxelType voxelTypeForMagicaRgba(std::uint32_t rgba) {
    const int r = static_cast<int>(rgba & 0xFFu);
    const int g = static_cast<int>((rgba >> 8u) & 0xFFu);
    const int b = static_cast<int>((rgba >> 16u) & 0xFFu);
    const int a = static_cast<int>((rgba >> 24u) & 0xFFu);
    if (a <= 8) {
        return VoxelType::Empty;
    }

    struct VoxelRef {
        VoxelType type = VoxelType::Empty;
        int r = 0;
        int g = 0;
        int b = 0;
    };
    constexpr std::array<VoxelRef, 5> kVoxelRefs = {
        VoxelRef{VoxelType::Stone, 168, 168, 168},
        VoxelRef{VoxelType::Dirt, 134, 93, 52},
        VoxelRef{VoxelType::Grass, 96, 164, 80},
        VoxelRef{VoxelType::Wood, 154, 121, 84},
        VoxelRef{VoxelType::SolidRed, 228, 84, 66},
    };

    VoxelType closest = kVoxelRefs.front().type;
    int bestDistance = std::numeric_limits<int>::max();
    for (const VoxelRef& reference : kVoxelRefs) {
        const int dr = r - reference.r;
        const int dg = g - reference.g;
        const int db = b - reference.b;
        const int distance = (dr * dr) + (dg * dg) + (db * db);
        if (distance < bestDistance) {
            bestDistance = distance;
            closest = reference.type;
        }
    }
    return closest;
}

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

    std::uint8_t nearestSlot = 0u;
    int bestDistance = std::numeric_limits<int>::max();
    for (std::uint8_t i = 0; i < static_cast<std::uint8_t>(paletteSlots.size()); ++i) {
        const std::uint32_t slotRgba = paletteSlots[i];
        const int slotR = static_cast<int>(slotRgba & 0xFFu);
        const int slotG = static_cast<int>((slotRgba >> 8u) & 0xFFu);
        const int slotB = static_cast<int>((slotRgba >> 16u) & 0xFFu);
        const int dr = r - slotR;
        const int dg = g - slotG;
        const int db = b - slotB;
        const int distance = (dr * dr) + (dg * dg) + (db * db);
        if (distance < bestDistance) {
            bestDistance = distance;
            nearestSlot = i;
        }
    }
    return nearestSlot;
}

} // namespace

bool World::loadOrInitialize(const std::filesystem::path& worldPath, LoadResult* outResult) {
    LoadResult result{};
    if (m_chunkGrid.loadFromBinaryFile(worldPath)) {
        result.loadedFromFile = true;
        if (outResult != nullptr) {
            *outResult = result;
        }
        return true;
    }

    m_chunkGrid.initializeEmptyWorld();
    result.initializedFallback = true;
    if (outResult != nullptr) {
        *outResult = result;
    }
    return false;
}

bool World::save(const std::filesystem::path& worldPath) const {
    return m_chunkGrid.saveToBinaryFile(worldPath);
}

void World::regenerateFlatWorld() {
    m_chunkGrid.initializeFlatWorld();
}

World::MagicaStampResult World::stampMagicaResources(std::span<const MagicaStampSpec> specs) {
    MagicaStampResult result{};

    for (const MagicaStampSpec& spec : specs) {
        if (spec.relativePath == nullptr || spec.relativePath[0] == '\0') {
            continue;
        }

        const std::filesystem::path magicaPath = resolveAssetPath(std::filesystem::path{spec.relativePath});
        MagicaVoxelModel loadedModel{};
        if (!loadMagicaVoxelModel(magicaPath, loadedModel)) {
            std::error_code cwdError;
            const std::filesystem::path cwd = std::filesystem::current_path(cwdError);
            VOX_LOGW("world") << "failed to load magica resource at "
                               << std::filesystem::absolute(magicaPath).string()
                               << " (cwd=" << (cwdError ? std::string{"<unavailable>"} : cwd.string()) << ")";
            continue;
        }

        const MagicaVoxelModel magicaModel = downscaleMagicaModel(loadedModel, spec.uniformScale);
        const int transformedSizeX = magicaModel.sizeX;
        const int transformedSizeZ = magicaModel.sizeY;
        const int worldOriginX =
            static_cast<int>(std::lround(spec.placementX - (0.5f * static_cast<float>(transformedSizeX))));
        const int worldOriginY = static_cast<int>(std::lround(spec.placementY));
        const int worldOriginZ =
            static_cast<int>(std::lround(spec.placementZ - (0.5f * static_cast<float>(transformedSizeZ))));

        std::uint64_t resourceStamped = 0;
        std::uint64_t resourceClipped = 0;

        for (const MagicaVoxel& voxel : magicaModel.voxels) {
            const std::uint32_t paletteRgba = magicaModel.paletteRgba[voxel.paletteIndex];
            const VoxelType voxelType = voxelTypeForMagicaRgba(paletteRgba);
            if (voxelType == VoxelType::Empty) {
                continue;
            }

            const int worldX = worldOriginX + static_cast<int>(voxel.x);
            const int worldY = worldOriginY + static_cast<int>(voxel.z);
            const int worldZ = worldOriginZ + static_cast<int>(voxel.y);

            std::size_t chunkIndex = 0;
            int localX = 0;
            int localY = 0;
            int localZ = 0;
            if (!worldToChunkLocal(worldX, worldY, worldZ, chunkIndex, localX, localY, localZ)) {
                ++resourceClipped;
                continue;
            }

            Chunk& chunk = m_chunkGrid.chunks()[chunkIndex];
            const std::uint8_t baseColorIndex = quantizeBaseColorIndex(
                paletteRgba,
                result.baseColorPalette,
                result.baseColorPaletteCount
            );
            chunk.setVoxel(localX, localY, localZ, Voxel{voxelType, baseColorIndex});
            ++resourceStamped;
        }

        if (resourceStamped == 0) {
            VOX_LOGW("world") << "magica resource stamped no world voxels: "
                               << std::filesystem::absolute(magicaPath).string()
                               << " (clipped=" << resourceClipped << ")";
            continue;
        }

        ++result.stampedResourceCount;
        result.stampedVoxelCount += resourceStamped;
        result.clippedVoxelCount += resourceClipped;
        VOX_LOGI("world") << "stamped magica resource " << std::filesystem::absolute(magicaPath).string()
                           << " (" << resourceStamped << " voxels, clipped=" << resourceClipped
                           << ", scale=" << spec.uniformScale << ")";
    }

    return result;
}

ChunkGrid& World::chunkGrid() {
    return m_chunkGrid;
}

const ChunkGrid& World::chunkGrid() const {
    return m_chunkGrid;
}

std::filesystem::path World::resolveAssetPath(const std::filesystem::path& relativePath) {
    std::vector<std::filesystem::path> baseCandidates;
    baseCandidates.reserve(6);

#if defined(VOXEL_PROJECT_SOURCE_DIR)
    baseCandidates.emplace_back(std::filesystem::path{VOXEL_PROJECT_SOURCE_DIR});
#endif

    std::error_code cwdError;
    const std::filesystem::path cwd = std::filesystem::current_path(cwdError);
    if (!cwdError) {
        baseCandidates.push_back(cwd);
        baseCandidates.push_back(cwd / "..");
        baseCandidates.push_back(cwd / ".." / "..");
        baseCandidates.push_back(cwd / ".." / ".." / "..");
    }

    for (const std::filesystem::path& base : baseCandidates) {
        const std::filesystem::path candidate = base / relativePath;
        std::error_code existsError;
        if (!std::filesystem::exists(candidate, existsError) || existsError) {
            continue;
        }

        std::error_code canonicalError;
        const std::filesystem::path canonicalPath = std::filesystem::weakly_canonical(candidate, canonicalError);
        if (!canonicalError) {
            return canonicalPath;
        }
        return candidate;
    }

    return relativePath;
}

bool World::worldToChunkLocal(
    int worldX,
    int worldY,
    int worldZ,
    std::size_t& outChunkIndex,
    int& outLocalX,
    int& outLocalY,
    int& outLocalZ
) const {
    const std::vector<Chunk>& chunks = m_chunkGrid.chunks();
    for (std::size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
        const Chunk& chunk = chunks[chunkIndex];
        const int chunkMinX = chunk.chunkX() * Chunk::kSizeX;
        const int chunkMinY = chunk.chunkY() * Chunk::kSizeY;
        const int chunkMinZ = chunk.chunkZ() * Chunk::kSizeZ;
        const int localX = worldX - chunkMinX;
        const int localY = worldY - chunkMinY;
        const int localZ = worldZ - chunkMinZ;

        const bool inBounds =
            localX >= 0 && localX < Chunk::kSizeX &&
            localY >= 0 && localY < Chunk::kSizeY &&
            localZ >= 0 && localZ < Chunk::kSizeZ;
        if (!inBounds) {
            continue;
        }

        outChunkIndex = chunkIndex;
        outLocalX = localX;
        outLocalY = localY;
        outLocalZ = localZ;
        return true;
    }

    return false;
}

} // namespace voxelsprout::world
