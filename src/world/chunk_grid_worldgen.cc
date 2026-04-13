#include "world/chunk_grid.h"

#include "core/log.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace voxelsprout::world {

namespace {

enum class BiomeType : std::uint8_t {
    Plains = 0,
    Forest = 1,
    RockyHighlands = 2,
    DryScrub = 3,
};

struct ColumnSample {
    int terrainHeight = 1;
    BiomeType biome = BiomeType::Plains;
    float moisture = 0.0f;
    float temperature = 0.0f;
    float ruggedness = 0.0f;
    float slope = 0.0f;
    bool supportsGrass = false;
    bool supportsDenseVegetation = false;
};

struct WorldGenerationStats {
    std::array<std::uint32_t, 4> biomeColumnCounts{};
    std::uint32_t grassyColumnCount = 0;
    std::uint32_t denseVegetationColumnCount = 0;
    std::uint32_t treeCount = 0;
    std::uint32_t settlementColumnCount = 0;
};

struct SettlementSite {
    int x = 0;
    int z = 0;
    int radius = 10;
};

constexpr std::array<std::uint8_t, 256> kPerlinPermutation = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
    140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,
    247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,
    57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
    74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,
    60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,
    65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,
    200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,
    52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,
    207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,
    119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
    129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,
    218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,
    81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,
    184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,
    222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

float fade(float t) {
    return t * t * t * (t * ((t * 6.0f) - 15.0f) + 10.0f);
}

float lerp(float a, float b, float t) {
    return a + (t * (b - a));
}

float grad2(std::uint8_t hash, float x, float z) {
    switch (hash & 0x7u) {
    case 0: return x + z;
    case 1: return -x + z;
    case 2: return x - z;
    case 3: return -x - z;
    case 4: return x;
    case 5: return -x;
    case 6: return z;
    default: return -z;
    }
}

std::uint8_t permAt(int index) {
    return kPerlinPermutation[static_cast<std::size_t>(index & 255)];
}

float perlin2(float x, float z) {
    const int xi0 = static_cast<int>(std::floor(x)) & 255;
    const int zi0 = static_cast<int>(std::floor(z)) & 255;
    const int xi1 = (xi0 + 1) & 255;
    const int zi1 = (zi0 + 1) & 255;

    const float xf = x - std::floor(x);
    const float zf = z - std::floor(z);
    const float u = fade(xf);
    const float v = fade(zf);

    const std::uint8_t aa = permAt(static_cast<int>(permAt(xi0)) + zi0);
    const std::uint8_t ab = permAt(static_cast<int>(permAt(xi0)) + zi1);
    const std::uint8_t ba = permAt(static_cast<int>(permAt(xi1)) + zi0);
    const std::uint8_t bb = permAt(static_cast<int>(permAt(xi1)) + zi1);

    const float x0 = lerp(grad2(aa, xf, zf), grad2(ba, xf - 1.0f, zf), u);
    const float x1 = lerp(grad2(ab, xf, zf - 1.0f), grad2(bb, xf - 1.0f, zf - 1.0f), u);
    return lerp(x0, x1, v);
}

std::uint32_t hashCoords(int x, int z, std::uint32_t salt = 0u) {
    std::uint32_t hash = static_cast<std::uint32_t>(x) * 0x9E3779B9u;
    hash ^= static_cast<std::uint32_t>(z) * 0x85EBCA6Bu;
    hash ^= salt * 0xC2B2AE35u;
    hash ^= hash >> 16u;
    hash *= 0x7FEB352Du;
    hash ^= hash >> 15u;
    hash *= 0x846CA68Bu;
    hash ^= hash >> 16u;
    return hash;
}

int floorDiv(int value, int divisor) {
    int quotient = value / divisor;
    const int remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        --quotient;
    }
    return quotient;
}

int positiveModulo(int value, int divisor) {
    const int remainder = value % divisor;
    return remainder < 0 ? remainder + divisor : remainder;
}

const char* biomeName(BiomeType biome) {
    switch (biome) {
    case BiomeType::Plains: return "plains";
    case BiomeType::Forest: return "forest";
    case BiomeType::RockyHighlands: return "rocky";
    case BiomeType::DryScrub: return "dry";
    }
    return "plains";
}

BiomeType classifyBiome(float moisture, float temperature, float ruggedness) {
    if (ruggedness > 0.64f) {
        return BiomeType::RockyHighlands;
    }
    if (moisture < 0.34f && temperature > 0.50f) {
        return BiomeType::DryScrub;
    }
    if (moisture > 0.55f) {
        return BiomeType::Forest;
    }
    return BiomeType::Plains;
}

std::vector<SettlementSite> settlementSitesNearWorld(int worldX, int worldZ) {
    constexpr int kSettlementCellSize = 160;
    std::vector<SettlementSite> sites;
    const int cellX = floorDiv(worldX, kSettlementCellSize);
    const int cellZ = floorDiv(worldZ, kSettlementCellSize);
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const int sampleCellX = cellX + offsetX;
            const int sampleCellZ = cellZ + offsetZ;
            const std::uint32_t siteHash = hashCoords(sampleCellX, sampleCellZ, 71u);
            if ((siteHash & 0xFFu) < 160u) {
                continue;
            }
            const int cellOriginX = sampleCellX * kSettlementCellSize;
            const int cellOriginZ = sampleCellZ * kSettlementCellSize;
            const int jitterX = static_cast<int>((siteHash >> 8u) % 64u) - 32;
            const int jitterZ = static_cast<int>((siteHash >> 16u) % 64u) - 32;
            const int radius = 9 + static_cast<int>((siteHash >> 24u) % 4u);
            sites.push_back(SettlementSite{
                cellOriginX + (kSettlementCellSize / 2) + jitterX,
                cellOriginZ + (kSettlementCellSize / 2) + jitterZ,
                radius
            });
        }
    }
    return sites;
}

ColumnSample sampleColumnAtWorld(int worldX, int worldZ, WorldGenerationStats* outStats = nullptr) {
    const float wx = static_cast<float>(worldX);
    const float wz = static_cast<float>(worldZ);
    const float warpX = perlin2((wx * 0.008f) + 13.7f, (wz * 0.008f) - 7.1f) * 18.0f;
    const float warpZ = perlin2((wx * 0.008f) - 19.4f, (wz * 0.008f) + 29.3f) * 18.0f;
    const float sampleX = wx + warpX;
    const float sampleZ = wz + warpZ;

    const float macroShape = perlin2(sampleX * 0.0085f, sampleZ * 0.0085f);
    const float rolling = perlin2(sampleX * 0.020f, sampleZ * 0.020f);
    const float detail = perlin2(sampleX * 0.046f, sampleZ * 0.046f);
    const float ruggednessField = (perlin2((sampleX * 0.011f) + 52.0f, (sampleZ * 0.011f) - 44.0f) * 0.5f) + 0.5f;
    const float moisture = (perlin2((sampleX * 0.006f) + 91.0f, (sampleZ * 0.006f) - 37.0f) * 0.5f) + 0.5f;
    const float temperature = (perlin2((sampleX * 0.005f) - 63.0f, (sampleZ * 0.005f) + 84.0f) * 0.5f) + 0.5f;
    const float basinNoise = (perlin2((sampleX * 0.004f) + 141.0f, (sampleZ * 0.004f) + 26.0f) * 0.5f) + 0.5f;
    const float basinDepth = std::pow(std::clamp((0.50f - basinNoise) / 0.50f, 0.0f, 1.0f), 1.7f) * 5.0f;

    ColumnSample sample{};
    sample.moisture = moisture;
    sample.temperature = temperature;
    sample.ruggedness = ruggednessField;
    sample.biome = classifyBiome(moisture, temperature, ruggednessField);

    float biomeHeightBias = 0.0f;
    float biomeReliefBias = 0.0f;
    switch (sample.biome) {
    case BiomeType::Forest:
        biomeHeightBias = 1.4f;
        biomeReliefBias = 0.9f;
        break;
    case BiomeType::RockyHighlands:
        biomeHeightBias = 4.8f;
        biomeReliefBias = 2.6f;
        break;
    case BiomeType::DryScrub:
        biomeHeightBias = -0.8f;
        biomeReliefBias = 0.4f;
        break;
    case BiomeType::Plains:
    default:
        biomeHeightBias = 0.0f;
        biomeReliefBias = 0.3f;
        break;
    }

    float terrainHeightF =
        9.0f +
        (macroShape * (5.0f + biomeReliefBias)) +
        (rolling * (2.7f + (ruggednessField * 1.8f))) +
        (detail * (0.8f + biomeReliefBias)) +
        biomeHeightBias -
        basinDepth;

    for (const SettlementSite& settlement : settlementSitesNearWorld(worldX, worldZ)) {
        const float dx = static_cast<float>(worldX - settlement.x);
        const float dz = static_cast<float>(worldZ - settlement.z);
        const float distance = std::sqrt((dx * dx) + (dz * dz));
        if (distance > static_cast<float>(settlement.radius)) {
            continue;
        }
        const float t = 1.0f - (distance / static_cast<float>(settlement.radius));
        const bool roadMask =
            positiveModulo(std::abs(worldX - settlement.x), 8) <= 1 ||
            positiveModulo(std::abs(worldZ - settlement.z), 8) <= 1;
        if (roadMask) {
            terrainHeightF = std::max(terrainHeightF, 10.0f + (t * 2.0f));
        } else {
            const float lotNoise = (perlin2((dx * 0.55f) + 17.0f, (dz * 0.55f) - 31.0f) * 0.5f) + 0.5f;
            if (lotNoise > 0.38f) {
                terrainHeightF = std::max(terrainHeightF, 12.0f + (t * (3.0f + (lotNoise * 5.0f))));
                if (outStats != nullptr) {
                    ++outStats->settlementColumnCount;
                }
            }
        }
    }

    sample.terrainHeight = std::clamp(static_cast<int>(std::round(terrainHeightF)), 1, Chunk::kSizeY - 6);
    return sample;
}

bool supportsGrassAtWorld(const ColumnSample& sample, int worldX, int worldZ) {
    const std::uint32_t surfaceHash = hashCoords(worldX, worldZ, 7u);
    const float grassPatch = static_cast<float>((surfaceHash >> 8u) & 0xFFu) / 255.0f;
    return sample.slope <= 2.0f &&
           ((sample.biome == BiomeType::Forest) ||
            (sample.biome == BiomeType::Plains && grassPatch > 0.20f) ||
            (sample.biome == BiomeType::DryScrub && grassPatch > 0.72f) ||
            (sample.biome == BiomeType::RockyHighlands && sample.ruggedness < 0.55f && grassPatch > 0.84f));
}

bool supportsDenseVegetationAtWorld(const ColumnSample& sample) {
    return sample.supportsGrass &&
           sample.slope <= 1.0f &&
           (sample.biome == BiomeType::Forest ||
            (sample.biome == BiomeType::Plains && sample.moisture > 0.45f));
}

void stampTreeVariant(Chunk& chunk, int originX, int originY, int originZ, int variantIndex) {
    const int trunkHeight = 4 + (variantIndex % 3);
    const int canopyRadius = (variantIndex == 0) ? 2 : 1;
    const int canopyBottom = originY + trunkHeight - 1;
    const int canopyTop = canopyBottom + 2 + (variantIndex == 2 ? 1 : 0);

    const int chunkMinX = chunk.chunkX() * Chunk::kSizeX;
    const int chunkMinY = chunk.chunkY() * Chunk::kSizeY;
    const int chunkMinZ = chunk.chunkZ() * Chunk::kSizeZ;
    auto setChunkWorldVoxel = [&](int worldX, int worldY, int worldZ, Voxel voxel, bool onlyIfEmpty = false) {
        const int localX = worldX - chunkMinX;
        const int localY = worldY - chunkMinY;
        const int localZ = worldZ - chunkMinZ;
        const bool inBounds =
            localX >= 0 && localX < Chunk::kSizeX &&
            localY >= 0 && localY < Chunk::kSizeY &&
            localZ >= 0 && localZ < Chunk::kSizeZ;
        if (!inBounds) {
            return;
        }
        if (onlyIfEmpty && chunk.voxelAt(localX, localY, localZ).type != VoxelType::Empty) {
            return;
        }
        chunk.setVoxel(localX, localY, localZ, voxel);
    };

    for (int y = 0; y < trunkHeight; ++y) {
        setChunkWorldVoxel(originX, originY + y, originZ, Voxel{VoxelType::Wood});
    }

    for (int y = canopyBottom; y <= canopyTop; ++y) {
        const int localY = y - canopyBottom;
        const int layerRadius = std::max(1, canopyRadius - (localY > 1 ? 1 : 0));
        for (int dz = -layerRadius; dz <= layerRadius; ++dz) {
            for (int dx = -layerRadius; dx <= layerRadius; ++dx) {
                const int manhattan = std::abs(dx) + std::abs(dz);
                if (manhattan > layerRadius + 1) {
                    continue;
                }
                if (dx == 0 && dz == 0 && y <= originY + trunkHeight) {
                    continue;
                }
                setChunkWorldVoxel(originX + dx, y, originZ + dz, Voxel{VoxelType::Leaves}, true);
            }
        }
    }

    if (variantIndex == 1) {
        setChunkWorldVoxel(originX, canopyTop + 1, originZ, Voxel{VoxelType::Leaves}, true);
    }
}

bool shouldPlaceTreeAtWorld(int worldX, int worldZ, const ColumnSample& sample) {
    if (!sample.supportsDenseVegetation || sample.terrainHeight >= Chunk::kSizeY - 8) {
        return false;
    }
    const int spacing = (sample.biome == BiomeType::Forest) ? 5 : 7;
    if (positiveModulo(worldX, spacing) != spacing / 2 || positiveModulo(worldZ, spacing) != spacing / 2) {
        return false;
    }

    const std::uint32_t treeHash = hashCoords(worldX, worldZ, 19u);
    const float treeChance = static_cast<float>(treeHash & 0xFFu) / 255.0f;
    const float threshold = (sample.biome == BiomeType::Forest) ? 0.34f : 0.20f;
    if (treeChance < threshold) {
        return false;
    }

    for (int dz = -spacing; dz <= spacing; dz += spacing) {
        for (int dx = -spacing; dx <= spacing; dx += spacing) {
            if (dx == 0 && dz == 0) {
                continue;
            }
            const int neighborX = worldX + dx;
            const int neighborZ = worldZ + dz;
            const ColumnSample neighborSample = sampleColumnAtWorld(neighborX, neighborZ);
            if (!neighborSample.supportsDenseVegetation) {
                continue;
            }
            const std::uint32_t neighborHash = hashCoords(neighborX, neighborZ, 19u);
            const float neighborChance = static_cast<float>(neighborHash & 0xFFu) / 255.0f;
            if (neighborChance > treeChance && neighborChance >= threshold) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

Chunk buildProceduralChunk(int chunkX, int chunkY, int chunkZ) {
    Chunk chunk(chunkX, chunkY, chunkZ);
    if (chunkY != 0) {
        return chunk;
    }

    WorldGenerationStats stats{};
    const int worldMinX = chunkX * Chunk::kSizeX;
    const int worldMinZ = chunkZ * Chunk::kSizeZ;
    const int worldMaxX = worldMinX + Chunk::kSizeX - 1;
    const int worldMaxZ = worldMinZ + Chunk::kSizeZ - 1;

    for (int localZ = 0; localZ < Chunk::kSizeZ; ++localZ) {
        for (int localX = 0; localX < Chunk::kSizeX; ++localX) {
            const int worldX = worldMinX + localX;
            const int worldZ = worldMinZ + localZ;
            ColumnSample sample = sampleColumnAtWorld(worldX, worldZ, &stats);

            const ColumnSample west = sampleColumnAtWorld(worldX - 1, worldZ);
            const ColumnSample east = sampleColumnAtWorld(worldX + 1, worldZ);
            const ColumnSample north = sampleColumnAtWorld(worldX, worldZ - 1);
            const ColumnSample south = sampleColumnAtWorld(worldX, worldZ + 1);
            sample.slope = static_cast<float>(
                std::max(
                    std::abs(east.terrainHeight - west.terrainHeight),
                    std::abs(south.terrainHeight - north.terrainHeight)
                )
            );
            sample.supportsGrass = supportsGrassAtWorld(sample, worldX, worldZ);
            sample.supportsDenseVegetation = supportsDenseVegetationAtWorld(sample);

            ++stats.biomeColumnCounts[static_cast<std::size_t>(sample.biome)];
            if (sample.supportsGrass) {
                ++stats.grassyColumnCount;
            }
            if (sample.supportsDenseVegetation) {
                ++stats.denseVegetationColumnCount;
            }

            for (int y = 0; y <= sample.terrainHeight; ++y) {
                VoxelType voxelType = VoxelType::Stone;
                if (y <= sample.terrainHeight - 4) {
                    voxelType = VoxelType::Stone;
                } else if (y < sample.terrainHeight) {
                    voxelType = (sample.biome == BiomeType::RockyHighlands && sample.slope > 2.0f)
                        ? VoxelType::Stone
                        : VoxelType::Dirt;
                } else if (sample.supportsGrass) {
                    voxelType = VoxelType::Grass;
                } else if (sample.biome == BiomeType::RockyHighlands || sample.slope > 2.0f) {
                    voxelType = VoxelType::Stone;
                } else {
                    voxelType = VoxelType::Dirt;
                }
                chunk.setVoxel(localX, y, localZ, Voxel{voxelType});
            }
        }
    }

    for (int worldZ = worldMinZ - 3; worldZ <= worldMaxZ + 3; ++worldZ) {
        for (int worldX = worldMinX - 3; worldX <= worldMaxX + 3; ++worldX) {
            const ColumnSample sample = sampleColumnAtWorld(worldX, worldZ);
            if (!shouldPlaceTreeAtWorld(worldX, worldZ, sample)) {
                continue;
            }
            const int baseY = sample.terrainHeight + 1;
            const std::uint32_t treeHash = hashCoords(worldX, worldZ, 19u);
            stampTreeVariant(chunk, worldX, baseY, worldZ, static_cast<int>((treeHash >> 9u) % 3u));
            ++stats.treeCount;
        }
    }

    VOX_LOGD("world") << "generated procedural chunk"
                      << " chunk=(" << chunkX << "," << chunkY << "," << chunkZ << ")"
                      << ", biomes={"
                      << biomeName(BiomeType::Plains) << ":" << stats.biomeColumnCounts[0]
                      << ", " << biomeName(BiomeType::Forest) << ":" << stats.biomeColumnCounts[1]
                      << ", " << biomeName(BiomeType::RockyHighlands) << ":" << stats.biomeColumnCounts[2]
                      << ", " << biomeName(BiomeType::DryScrub) << ":" << stats.biomeColumnCounts[3]
                      << "}, trees=" << stats.treeCount;
    return chunk;
}

void ChunkGrid::initializeFlatWorld() {
    initializeEmptyWorld();
    if (m_chunks.empty()) {
        return;
    }

    std::size_t treeChunkCount = 0;
    for (Chunk& chunk : m_chunks) {
        chunk = buildProceduralChunk(chunk.chunkX(), chunk.chunkY(), chunk.chunkZ());
        if (chunk.chunkY() == 0) {
            ++treeChunkCount;
        }
    }

    VOX_LOGI("world") << "generated procedural world chunks=" << treeChunkCount;
}

} // namespace voxelsprout::world
