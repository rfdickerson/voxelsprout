#include "world/chunk_grid.h"

#include "core/log.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace odai::world {

namespace {

enum class BiomeType : std::uint8_t {
    Plains = 0,
    Forest = 1,
    RockyHighlands = 2,
    DryScrub = 3,
};

struct ColumnSample {
    int terrainHeight = 1;
    float terrainHeightF = 1.0f;
    BiomeType biome = BiomeType::Plains;
    float moisture = 0.0f;
    float temperature = 0.0f;
    float ruggedness = 0.0f;
    float slope = 0.0f;
    float ridge = 0.0f;
    float flow = 0.0f;
    float cliffBias = 0.0f;
    float settlementMask = 0.0f;
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

struct WarpedPoint {
    float x = 0.0f;
    float z = 0.0f;
};

struct TerrainPatch {
    int minWorldX = 0;
    int minWorldZ = 0;
    int width = 0;
    int depth = 0;
    std::vector<ColumnSample> samples;

    [[nodiscard]] int index(int localX, int localZ) const {
        return localX + (localZ * width);
    }

    [[nodiscard]] bool containsWorld(int worldX, int worldZ) const {
        return worldX >= minWorldX &&
               worldZ >= minWorldZ &&
               worldX < minWorldX + width &&
               worldZ < minWorldZ + depth;
    }

    [[nodiscard]] ColumnSample& sampleAtLocal(int localX, int localZ) {
        return samples[static_cast<std::size_t>(index(localX, localZ))];
    }

    [[nodiscard]] const ColumnSample& sampleAtLocal(int localX, int localZ) const {
        return samples[static_cast<std::size_t>(index(localX, localZ))];
    }
};

constexpr int kTerrainPatchBorder = 12;
constexpr int kHydraulicErosionIterations = 10;

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

float smoothstep(float edge0, float edge1, float value) {
    if (edge0 == edge1) {
        return value < edge0 ? 0.0f : 1.0f;
    }
    const float t = std::clamp((value - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - (2.0f * t));
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

float ridgedPerlin2(float x, float z) {
    return 1.0f - std::abs(std::clamp(perlin2(x, z), -1.0f, 1.0f));
}

WarpedPoint warpTerrainDomain(float worldX, float worldZ) {
    const float broadWarpX =
        perlin2((worldX * 0.0065f) + 13.7f, (worldZ * 0.0065f) - 7.1f) * 22.0f;
    const float broadWarpZ =
        perlin2((worldX * 0.0065f) - 19.4f, (worldZ * 0.0065f) + 29.3f) * 22.0f;
    const float foldedX = worldX + broadWarpX;
    const float foldedZ = worldZ + broadWarpZ;
    const float localWarpX =
        perlin2((foldedX * 0.020f) + 101.0f, (foldedZ * 0.020f) - 61.0f) * 5.0f;
    const float localWarpZ =
        perlin2((foldedX * 0.020f) - 44.0f, (foldedZ * 0.020f) + 87.0f) * 5.0f;
    return WarpedPoint{foldedX + localWarpX, foldedZ + localWarpZ};
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

ColumnSample sampleBaseColumnAtWorld(int worldX, int worldZ) {
    const float wx = static_cast<float>(worldX);
    const float wz = static_cast<float>(worldZ);
    const WarpedPoint warped = warpTerrainDomain(wx, wz);
    const float sampleX = warped.x;
    const float sampleZ = warped.z;

    const float macroShape = perlin2(sampleX * 0.0085f, sampleZ * 0.0085f);
    const float rolling = perlin2(sampleX * 0.020f, sampleZ * 0.020f);
    const float detail = perlin2(sampleX * 0.046f, sampleZ * 0.046f);
    const float ridgePrimary = ridgedPerlin2((sampleX * 0.017f) - 24.0f, (sampleZ * 0.017f) + 18.0f);
    const float ridgeSecondary = ridgedPerlin2((sampleX * 0.041f) + 9.0f, (sampleZ * 0.041f) - 35.0f);
    const float ridge = std::pow(std::clamp((ridgePrimary * 0.78f) + (ridgeSecondary * 0.22f), 0.0f, 1.0f), 2.4f);
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
    sample.ridge = ridge;

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
        (ridge * ruggednessField * (2.2f + biomeReliefBias)) +
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
            sample.settlementMask = std::max(sample.settlementMask, t);
        } else {
            const float lotNoise = (perlin2((dx * 0.55f) + 17.0f, (dz * 0.55f) - 31.0f) * 0.5f) + 0.5f;
            if (lotNoise > 0.38f) {
                terrainHeightF = std::max(terrainHeightF, 12.0f + (t * (3.0f + (lotNoise * 5.0f))));
                sample.settlementMask = std::max(sample.settlementMask, t);
            }
        }
    }

    sample.terrainHeightF = std::clamp(terrainHeightF, 1.0f, static_cast<float>(Chunk::kSizeY - 6));
    sample.terrainHeight = std::clamp(static_cast<int>(std::round(sample.terrainHeightF)), 1, Chunk::kSizeY - 6);
    return sample;
}

float heightAtLocalClamped(const TerrainPatch& patch, int localX, int localZ) {
    const int x = std::clamp(localX, 0, patch.width - 1);
    const int z = std::clamp(localZ, 0, patch.depth - 1);
    return patch.sampleAtLocal(x, z).terrainHeightF;
}

float slopeAtLocal(const TerrainPatch& patch, int localX, int localZ) {
    const float west = heightAtLocalClamped(patch, localX - 1, localZ);
    const float east = heightAtLocalClamped(patch, localX + 1, localZ);
    const float north = heightAtLocalClamped(patch, localX, localZ - 1);
    const float south = heightAtLocalClamped(patch, localX, localZ + 1);
    const float dx = (east - west) * 0.5f;
    const float dz = (south - north) * 0.5f;
    return std::sqrt((dx * dx) + (dz * dz));
}

ColumnSample sampleTerrainAtWorld(const TerrainPatch& patch, int worldX, int worldZ) {
    if (!patch.containsWorld(worldX, worldZ)) {
        return sampleBaseColumnAtWorld(worldX, worldZ);
    }
    return patch.sampleAtLocal(worldX - patch.minWorldX, worldZ - patch.minWorldZ);
}

void clampPatchHeights(TerrainPatch& patch) {
    for (ColumnSample& sample : patch.samples) {
        sample.terrainHeightF = std::clamp(sample.terrainHeightF, 1.0f, static_cast<float>(Chunk::kSizeY - 6));
    }
}

void applyRidgeSharpening(TerrainPatch& patch) {
    for (int localZ = 0; localZ < patch.depth; ++localZ) {
        for (int localX = 0; localX < patch.width; ++localX) {
            ColumnSample& sample = patch.sampleAtLocal(localX, localZ);
            const float ridgeMask =
                sample.ridge * smoothstep(0.45f, 0.85f, sample.ruggedness) * (1.0f - (sample.settlementMask * 0.85f));
            if (ridgeMask <= 0.001f) {
                continue;
            }
            const float terraced = std::round(sample.terrainHeightF * 0.80f) / 0.80f;
            sample.terrainHeightF = lerp(sample.terrainHeightF, terraced + 0.35f, ridgeMask * 0.35f);
        }
    }
    clampPatchHeights(patch);
}

void runLightweightHydraulicErosion(TerrainPatch& patch) {
    const int cellCount = patch.width * patch.depth;
    std::vector<float> water(static_cast<std::size_t>(cellCount), 0.0f);
    std::vector<float> sediment(static_cast<std::size_t>(cellCount), 0.0f);
    std::vector<float> nextWater(static_cast<std::size_t>(cellCount), 0.0f);
    std::vector<float> nextSediment(static_cast<std::size_t>(cellCount), 0.0f);

    constexpr std::array<int, 4> kOffsetX = {-1, 1, 0, 0};
    constexpr std::array<int, 4> kOffsetZ = {0, 0, -1, 1};

    for (int iteration = 0; iteration < kHydraulicErosionIterations; ++iteration) {
        for (int localZ = 0; localZ < patch.depth; ++localZ) {
            for (int localX = 0; localX < patch.width; ++localX) {
                const int worldX = patch.minWorldX + localX;
                const int worldZ = patch.minWorldZ + localZ;
                const int index = patch.index(localX, localZ);
                const ColumnSample& sample = patch.sampleAtLocal(localX, localZ);
                const float rainNoise =
                    (perlin2((static_cast<float>(worldX) * 0.071f) + 11.0f,
                             (static_cast<float>(worldZ) * 0.071f) - 29.0f) * 0.5f) + 0.5f;
                const float settlementScale = 1.0f - (sample.settlementMask * 0.75f);
                water[static_cast<std::size_t>(index)] += (0.018f + (rainNoise * 0.012f)) * settlementScale;
            }
        }

        std::fill(nextWater.begin(), nextWater.end(), 0.0f);
        std::fill(nextSediment.begin(), nextSediment.end(), 0.0f);

        for (int localZ = 0; localZ < patch.depth; ++localZ) {
            for (int localX = 0; localX < patch.width; ++localX) {
                const int index = patch.index(localX, localZ);
                const std::size_t sampleIndex = static_cast<std::size_t>(index);
                const float currentSurface = patch.sampleAtLocal(localX, localZ).terrainHeightF + water[sampleIndex];

                std::array<int, 4> lowerIndices{};
                std::array<float, 4> lowerWeights{};
                int lowerCount = 0;
                float weightSum = 0.0f;
                for (std::size_t neighbor = 0; neighbor < kOffsetX.size(); ++neighbor) {
                    const int nx = localX + kOffsetX[neighbor];
                    const int nz = localZ + kOffsetZ[neighbor];
                    if (nx < 0 || nx >= patch.width || nz < 0 || nz >= patch.depth) {
                        continue;
                    }
                    const int neighborIndex = patch.index(nx, nz);
                    const float neighborSurface =
                        patch.sampleAtLocal(nx, nz).terrainHeightF + water[static_cast<std::size_t>(neighborIndex)];
                    const float drop = currentSurface - neighborSurface;
                    if (drop <= 0.001f) {
                        continue;
                    }
                    lowerIndices[static_cast<std::size_t>(lowerCount)] = neighborIndex;
                    lowerWeights[static_cast<std::size_t>(lowerCount)] = drop;
                    weightSum += drop;
                    ++lowerCount;
                }

                if (lowerCount == 0 || weightSum <= 0.001f) {
                    nextWater[sampleIndex] += water[sampleIndex];
                    nextSediment[sampleIndex] += sediment[sampleIndex];
                    continue;
                }

                const float moveFraction = 0.52f;
                const float movedWater = water[sampleIndex] * moveFraction;
                const float movedSediment = sediment[sampleIndex] * moveFraction;
                nextWater[sampleIndex] += water[sampleIndex] - movedWater;
                nextSediment[sampleIndex] += sediment[sampleIndex] - movedSediment;
                for (int lower = 0; lower < lowerCount; ++lower) {
                    const float share = lowerWeights[static_cast<std::size_t>(lower)] / weightSum;
                    const std::size_t lowerIndex = static_cast<std::size_t>(lowerIndices[static_cast<std::size_t>(lower)]);
                    nextWater[lowerIndex] += movedWater * share;
                    nextSediment[lowerIndex] += movedSediment * share;
                }
            }
        }

        water.swap(nextWater);
        sediment.swap(nextSediment);

        for (int localZ = 0; localZ < patch.depth; ++localZ) {
            for (int localX = 0; localX < patch.width; ++localX) {
                const int index = patch.index(localX, localZ);
                const std::size_t sampleIndex = static_cast<std::size_t>(index);
                ColumnSample& sample = patch.sampleAtLocal(localX, localZ);
                const float slope = slopeAtLocal(patch, localX, localZ);
                const float settlementScale = 1.0f - (sample.settlementMask * 0.90f);
                const float capacity = water[sampleIndex] * (0.18f + (slope * 0.28f));
                if (sediment[sampleIndex] > capacity) {
                    const float deposit = (sediment[sampleIndex] - capacity) * 0.18f * settlementScale;
                    sample.terrainHeightF += deposit;
                    sediment[sampleIndex] -= deposit;
                } else {
                    const float erosion =
                        std::min((capacity - sediment[sampleIndex]) * 0.060f * settlementScale, 0.16f);
                    sample.terrainHeightF -= erosion;
                    sediment[sampleIndex] += erosion;
                }
                water[sampleIndex] *= 0.62f;
            }
        }
        clampPatchHeights(patch);
    }
}

void generateFlowMapAndCarve(TerrainPatch& patch) {
    const int cellCount = patch.width * patch.depth;
    std::vector<float> flow(static_cast<std::size_t>(cellCount), 1.0f);
    std::vector<int> downhill(static_cast<std::size_t>(cellCount), 0);
    std::vector<int> order(static_cast<std::size_t>(cellCount), 0);
    std::iota(order.begin(), order.end(), 0);

    for (int localZ = 0; localZ < patch.depth; ++localZ) {
        for (int localX = 0; localX < patch.width; ++localX) {
            const int index = patch.index(localX, localZ);
            downhill[static_cast<std::size_t>(index)] = index;
            const int worldX = patch.minWorldX + localX;
            const int worldZ = patch.minWorldZ + localZ;
            const float rainfall =
                0.75f +
                (((perlin2((static_cast<float>(worldX) * 0.015f) + 73.0f,
                           (static_cast<float>(worldZ) * 0.015f) - 41.0f) * 0.5f) + 0.5f) * 0.50f);
            flow[static_cast<std::size_t>(index)] = rainfall;

            float bestDrop = 0.0f;
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dz == 0) {
                        continue;
                    }
                    const int nx = localX + dx;
                    const int nz = localZ + dz;
                    if (nx < 0 || nx >= patch.width || nz < 0 || nz >= patch.depth) {
                        continue;
                    }
                    const float distance = (dx != 0 && dz != 0) ? 1.41421356f : 1.0f;
                    const float drop =
                        (patch.sampleAtLocal(localX, localZ).terrainHeightF -
                         patch.sampleAtLocal(nx, nz).terrainHeightF) / distance;
                    if (drop > bestDrop) {
                        bestDrop = drop;
                        downhill[static_cast<std::size_t>(index)] = patch.index(nx, nz);
                    }
                }
            }
        }
    }

    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        const int lhsX = lhs % patch.width;
        const int lhsZ = lhs / patch.width;
        const int rhsX = rhs % patch.width;
        const int rhsZ = rhs / patch.width;
        return patch.sampleAtLocal(lhsX, lhsZ).terrainHeightF > patch.sampleAtLocal(rhsX, rhsZ).terrainHeightF;
    });

    for (int index : order) {
        const int receiver = downhill[static_cast<std::size_t>(index)];
        if (receiver != index) {
            flow[static_cast<std::size_t>(receiver)] += flow[static_cast<std::size_t>(index)] * 0.92f;
        }
    }

    const float maxFlow = *std::max_element(flow.begin(), flow.end());
    const float flowDenominator = std::log1p(std::max(maxFlow, 1.0f));
    for (int localZ = 0; localZ < patch.depth; ++localZ) {
        for (int localX = 0; localX < patch.width; ++localX) {
            const int index = patch.index(localX, localZ);
            ColumnSample& sample = patch.sampleAtLocal(localX, localZ);
            const float flow01 = std::clamp(std::log1p(flow[static_cast<std::size_t>(index)]) / flowDenominator, 0.0f, 1.0f);
            const float slope = slopeAtLocal(patch, localX, localZ);
            const float slopeMask = smoothstep(0.25f, 2.75f, slope);
            const float channelMask =
                smoothstep(0.48f, 0.92f, flow01) * slopeMask * (1.0f - (sample.settlementMask * 0.90f));
            sample.flow = flow01;
            sample.terrainHeightF -= channelMask * (0.08f + (0.42f * std::min(slope, 1.0f)));
        }
    }
    clampPatchHeights(patch);
}

void applySlopeBasedNoiseDetail(TerrainPatch& patch) {
    std::vector<float> originalHeights;
    originalHeights.reserve(patch.samples.size());
    for (const ColumnSample& sample : patch.samples) {
        originalHeights.push_back(sample.terrainHeightF);
    }

    auto originalHeightAt = [&](int localX, int localZ) {
        const int x = std::clamp(localX, 0, patch.width - 1);
        const int z = std::clamp(localZ, 0, patch.depth - 1);
        return originalHeights[static_cast<std::size_t>(patch.index(x, z))];
    };

    for (int localZ = 0; localZ < patch.depth; ++localZ) {
        for (int localX = 0; localX < patch.width; ++localX) {
            const int worldX = patch.minWorldX + localX;
            const int worldZ = patch.minWorldZ + localZ;
            ColumnSample& sample = patch.sampleAtLocal(localX, localZ);
            const float west = originalHeightAt(localX - 1, localZ);
            const float east = originalHeightAt(localX + 1, localZ);
            const float north = originalHeightAt(localX, localZ - 1);
            const float south = originalHeightAt(localX, localZ + 1);
            const float dx = (east - west) * 0.5f;
            const float dz = (south - north) * 0.5f;
            const float slope = std::sqrt((dx * dx) + (dz * dz));
            const float slopeMask = smoothstep(0.80f, 4.20f, slope);
            const float settlementScale = 1.0f - (sample.settlementMask * 0.85f);
            const float coarseDetail =
                perlin2((static_cast<float>(worldX) * 0.170f) + 19.0f,
                        (static_cast<float>(worldZ) * 0.170f) - 43.0f);
            const float fineDetail =
                perlin2((static_cast<float>(worldX) * 0.360f) - 83.0f,
                        (static_cast<float>(worldZ) * 0.360f) + 12.0f);
            const float detail = (coarseDetail * 0.34f) + (fineDetail * 0.12f);
            sample.terrainHeightF += detail * slopeMask * settlementScale;
        }
    }
    clampPatchHeights(patch);
}

void applyCliffBias(TerrainPatch& patch) {
    for (int localZ = 0; localZ < patch.depth; ++localZ) {
        for (int localX = 0; localX < patch.width; ++localX) {
            ColumnSample& sample = patch.sampleAtLocal(localX, localZ);
            const int worldX = patch.minWorldX + localX;
            const int worldZ = patch.minWorldZ + localZ;
            const float slope = slopeAtLocal(patch, localX, localZ);
            const float cliff =
                smoothstep(3.15f, 6.80f, slope) * (0.55f + (sample.ruggedness * 0.45f)) *
                (1.0f - (sample.settlementMask * 0.80f));
            const float ridgeCliff = sample.ridge * smoothstep(1.60f, 4.20f, slope) * 0.50f;
            sample.cliffBias = std::clamp(std::max(cliff, ridgeCliff), 0.0f, 1.0f);
            if (sample.cliffBias <= 0.001f) {
                continue;
            }
            const float shelfHeight = std::round(sample.terrainHeightF * 0.66f) / 0.66f;
            const float fracture =
                perlin2((static_cast<float>(worldX) * 0.105f) + 211.0f,
                        (static_cast<float>(worldZ) * 0.105f) - 97.0f);
            sample.terrainHeightF =
                lerp(sample.terrainHeightF, shelfHeight, sample.cliffBias * 0.28f) +
                (fracture * sample.cliffBias * 0.16f);
        }
    }
    clampPatchHeights(patch);
}

void finalizeTerrainPatch(TerrainPatch& patch) {
    clampPatchHeights(patch);
    for (int localZ = 0; localZ < patch.depth; ++localZ) {
        for (int localX = 0; localX < patch.width; ++localX) {
            ColumnSample& sample = patch.sampleAtLocal(localX, localZ);
            sample.slope = slopeAtLocal(patch, localX, localZ);
            sample.terrainHeight = std::clamp(
                static_cast<int>(std::round(sample.terrainHeightF)),
                1,
                Chunk::kSizeY - 6
            );
        }
    }
}

TerrainPatch buildTerrainPatch(int worldMinX, int worldMinZ, int width, int depth) {
    TerrainPatch patch{};
    patch.minWorldX = worldMinX - kTerrainPatchBorder;
    patch.minWorldZ = worldMinZ - kTerrainPatchBorder;
    patch.width = width + (kTerrainPatchBorder * 2);
    patch.depth = depth + (kTerrainPatchBorder * 2);
    patch.samples.resize(static_cast<std::size_t>(patch.width * patch.depth));

    for (int localZ = 0; localZ < patch.depth; ++localZ) {
        for (int localX = 0; localX < patch.width; ++localX) {
            const int worldX = patch.minWorldX + localX;
            const int worldZ = patch.minWorldZ + localZ;
            patch.sampleAtLocal(localX, localZ) = sampleBaseColumnAtWorld(worldX, worldZ);
        }
    }

    applyRidgeSharpening(patch);
    runLightweightHydraulicErosion(patch);
    generateFlowMapAndCarve(patch);
    applySlopeBasedNoiseDetail(patch);
    applyCliffBias(patch);
    finalizeTerrainPatch(patch);
    return patch;
}

bool supportsGrassAtWorld(const ColumnSample& sample, int worldX, int worldZ) {
    const std::uint32_t surfaceHash = hashCoords(worldX, worldZ, 7u);
    const float grassPatch = static_cast<float>((surfaceHash >> 8u) & 0xFFu) / 255.0f;
    const bool carvedChannel = sample.flow > 0.72f && sample.slope > 0.20f;
    return !carvedChannel &&
           sample.cliffBias < 0.42f &&
           sample.slope <= 2.0f &&
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

void refreshPatchVegetationSupport(TerrainPatch& patch) {
    for (int localZ = 0; localZ < patch.depth; ++localZ) {
        for (int localX = 0; localX < patch.width; ++localX) {
            const int worldX = patch.minWorldX + localX;
            const int worldZ = patch.minWorldZ + localZ;
            ColumnSample& sample = patch.sampleAtLocal(localX, localZ);
            sample.supportsGrass = supportsGrassAtWorld(sample, worldX, worldZ);
            sample.supportsDenseVegetation = supportsDenseVegetationAtWorld(sample);
        }
    }
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

bool shouldPlaceTreeAtWorld(const TerrainPatch& patch, int worldX, int worldZ) {
    const ColumnSample sample = sampleTerrainAtWorld(patch, worldX, worldZ);
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
            const ColumnSample neighborSample = sampleTerrainAtWorld(patch, neighborX, neighborZ);
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
    TerrainPatch terrain = buildTerrainPatch(worldMinX, worldMinZ, Chunk::kSizeX, Chunk::kSizeZ);
    refreshPatchVegetationSupport(terrain);

    for (int localZ = 0; localZ < Chunk::kSizeZ; ++localZ) {
        for (int localX = 0; localX < Chunk::kSizeX; ++localX) {
            const int worldX = worldMinX + localX;
            const int worldZ = worldMinZ + localZ;
            const ColumnSample sample = sampleTerrainAtWorld(terrain, worldX, worldZ);
            const bool cliffOrRock =
                sample.biome == BiomeType::RockyHighlands ||
                sample.slope > 2.25f ||
                sample.cliffBias > 0.34f;
            const bool carvedChannel = sample.flow > 0.70f && sample.slope > 0.20f;

            ++stats.biomeColumnCounts[static_cast<std::size_t>(sample.biome)];
            if (sample.supportsGrass) {
                ++stats.grassyColumnCount;
            }
            if (sample.supportsDenseVegetation) {
                ++stats.denseVegetationColumnCount;
            }
            if (sample.settlementMask > 0.0f) {
                ++stats.settlementColumnCount;
            }

            for (int y = 0; y <= sample.terrainHeight; ++y) {
                VoxelType voxelType = VoxelType::Stone;
                if (y <= sample.terrainHeight - 4) {
                    voxelType = VoxelType::Stone;
                } else if (y < sample.terrainHeight) {
                    voxelType = cliffOrRock && y >= sample.terrainHeight - 2
                        ? VoxelType::Stone
                        : VoxelType::Dirt;
                } else if (sample.supportsGrass) {
                    voxelType = VoxelType::Grass;
                } else if (cliffOrRock) {
                    voxelType = VoxelType::Stone;
                } else if (carvedChannel) {
                    voxelType = VoxelType::Dirt;
                } else {
                    voxelType = VoxelType::Dirt;
                }
                chunk.setVoxel(localX, y, localZ, Voxel{voxelType});
            }
        }
    }

    for (int worldZ = worldMinZ - 3; worldZ <= worldMaxZ + 3; ++worldZ) {
        for (int worldX = worldMinX - 3; worldX <= worldMaxX + 3; ++worldX) {
            const ColumnSample sample = sampleTerrainAtWorld(terrain, worldX, worldZ);
            if (!shouldPlaceTreeAtWorld(terrain, worldX, worldZ)) {
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
                      << "}, grass=" << stats.grassyColumnCount
                      << ", denseVegetation=" << stats.denseVegetationColumnCount
                      << ", settlements=" << stats.settlementColumnCount
                      << ", trees=" << stats.treeCount;
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

} // namespace odai::world
