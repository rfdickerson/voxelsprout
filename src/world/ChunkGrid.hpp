#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "world/Chunk.hpp"
#include "world/Voxel.hpp"

// World ChunkGrid subsystem
// Responsible for: owning a collection of chunks that represent world space.
// Should NOT do: pathfinding, factory simulation, or rendering API calls.
namespace world {

class ChunkGrid {
public:
    ChunkGrid() = default;
    void initializeEmptyWorld();
    void initializeFlatWorld();
    bool loadFromBinaryFile(const std::filesystem::path& path);
    bool saveToBinaryFile(const std::filesystem::path& path) const;
    std::size_t chunkCount() const;
    std::vector<Chunk>& chunks();
    const std::vector<Chunk>& chunks() const;

private:
    std::vector<Chunk> m_chunks;
};

inline void ChunkGrid::initializeFlatWorld() {
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

    auto fade = [](float t) -> float {
        return t * t * t * (t * ((t * 6.0f) - 15.0f) + 10.0f);
    };
    auto lerp = [](float a, float b, float t) -> float {
        return a + (t * (b - a));
    };
    auto grad2 = [](std::uint8_t hash, float x, float z) -> float {
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
    };
    auto permAt = [&](int index) -> std::uint8_t {
        return kPerlinPermutation[static_cast<std::size_t>(index & 255)];
    };
    auto perlin2 = [&](float x, float z) -> float {
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
    };

    auto chunkHeightAt = [&](int worldX, int worldZ) -> int {
        const float wx = static_cast<float>(worldX);
        const float wz = static_cast<float>(worldZ);

        // Domain warping avoids repetitive grid-aligned noise contours.
        const float warpX = perlin2((wx * 0.010f) + 13.7f, (wz * 0.010f) - 7.2f) * 22.0f;
        const float warpZ = perlin2((wx * 0.010f) - 29.1f, (wz * 0.010f) + 22.4f) * 22.0f;
        const float sampleX = wx + warpX;
        const float sampleZ = wz + warpZ;

        const float broadHills = perlin2(sampleX * 0.013f, sampleZ * 0.013f) * 7.2f;
        const float midHills = perlin2(sampleX * 0.031f, sampleZ * 0.031f) * 3.4f;

        // Valleys are carved where low-frequency noise dips negative.
        const float valleyNoise = perlin2((sampleX * 0.006f) - 20.0f, (sampleZ * 0.006f) + 42.0f);
        const float valleyMask = std::pow(std::max(0.0f, -valleyNoise), 1.9f);
        const float valleyCut = valleyMask * 15.0f;

        // Build a few strong monolithic massifs with steep cliff walls.
        const float massifSeed = (perlin2((sampleX * 0.004f) + 87.0f, (sampleZ * 0.004f) - 51.0f) * 0.5f) + 0.5f;
        const float massifMask = std::pow(std::clamp((massifSeed - 0.50f) * 2.0f, 0.0f, 1.0f), 2.2f);
        const float ridgeNoise = 1.0f - std::abs(perlin2((sampleX * 0.019f) - 11.0f, (sampleZ * 0.019f) + 5.0f));
        const float wallMask = std::pow(std::clamp((ridgeNoise - 0.46f) * 3.0f, 0.0f, 1.0f), 1.6f);
        const float cliffRamp = std::pow(wallMask, 1.35f) * massifMask;
        const float cliffLift = cliffRamp * 23.0f;

        // Carve deep canyons around cliff zones for dramatic relief.
        const float canyonNoise = perlin2((sampleX * 0.009f) + 33.0f, (sampleZ * 0.009f) - 71.0f);
        const float canyonMask = std::pow(std::max(0.0f, -canyonNoise), 1.7f) * (0.45f + (0.55f * (1.0f - massifMask)));
        const float canyonCut = canyonMask * 10.5f;

        const float heightFloat = 8.5f + broadHills + midHills + cliffLift - valleyCut - canyonCut;
        const int height = static_cast<int>(std::round(heightFloat));
        return std::clamp(height, 1, Chunk::kSizeY - 2);
    };

    initializeEmptyWorld();

    for (Chunk& chunk : m_chunks) {
        const int chunkX = chunk.chunkX();
        const int chunkZ = chunk.chunkZ();

        for (int z = 0; z < Chunk::kSizeZ; ++z) {
            for (int x = 0; x < Chunk::kSizeX; ++x) {
                const int worldX = (chunkX * Chunk::kSizeX) + x;
                const int worldZ = (chunkZ * Chunk::kSizeZ) + z;
                const int terrainHeight = chunkHeightAt(worldX, worldZ);

                for (int y = 0; y <= terrainHeight; ++y) {
                    chunk.setVoxel(x, y, z, Voxel{VoxelType::Solid});
                }
            }
        }
    }
}

inline void ChunkGrid::initializeEmptyWorld() {
    m_chunks.clear();

    // Keep the center chunk first so app-side interaction logic remains valid.
    constexpr int kChunkRadius = 3;
    constexpr int kChunkGridWidth = (kChunkRadius * 2) + 1;
    constexpr int kChunkCount = kChunkGridWidth * kChunkGridWidth;

    m_chunks.reserve(static_cast<std::size_t>(kChunkCount));
    m_chunks.emplace_back(0, 0, 0);
    for (int chunkZ = -kChunkRadius; chunkZ <= kChunkRadius; ++chunkZ) {
        for (int chunkX = -kChunkRadius; chunkX <= kChunkRadius; ++chunkX) {
            if (chunkX == 0 && chunkZ == 0) {
                continue;
            }
            m_chunks.emplace_back(chunkX, 0, chunkZ);
        }
    }
}

inline bool ChunkGrid::loadFromBinaryFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    char magic[4]{};
    std::uint32_t version = 0;
    std::uint32_t chunkCount = 0;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&chunkCount), sizeof(chunkCount));
    if (!in.good()) {
        return false;
    }

    constexpr char kExpectedMagic[4] = {'V', 'X', 'W', '1'};
    if (std::memcmp(magic, kExpectedMagic, sizeof(magic)) != 0 || version != 1u) {
        return false;
    }
    if (chunkCount == 0 || chunkCount > 4096u) {
        return false;
    }

    constexpr std::size_t kVoxelsPerChunk =
        static_cast<std::size_t>(Chunk::kSizeX * Chunk::kSizeY * Chunk::kSizeZ);
    constexpr std::size_t kBytesPerChunk = (kVoxelsPerChunk + 7u) / 8u;
    std::vector<std::uint8_t> packed(kBytesPerChunk, 0u);

    std::vector<Chunk> loadedChunks;
    loadedChunks.reserve(static_cast<std::size_t>(chunkCount));

    for (std::uint32_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
        std::int32_t chunkX = 0;
        std::int32_t chunkY = 0;
        std::int32_t chunkZ = 0;
        in.read(reinterpret_cast<char*>(&chunkX), sizeof(chunkX));
        in.read(reinterpret_cast<char*>(&chunkY), sizeof(chunkY));
        in.read(reinterpret_cast<char*>(&chunkZ), sizeof(chunkZ));
        in.read(reinterpret_cast<char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
        if (!in.good()) {
            return false;
        }

        Chunk chunk(chunkX, chunkY, chunkZ);
        chunk.setFromSolidBitfield(packed.data(), packed.size());
        loadedChunks.push_back(std::move(chunk));
    }

    m_chunks = std::move(loadedChunks);
    return true;
}

inline bool ChunkGrid::saveToBinaryFile(const std::filesystem::path& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }

    constexpr char kMagic[4] = {'V', 'X', 'W', '1'};
    constexpr std::uint32_t kVersion = 1u;
    const std::uint32_t chunkCount = static_cast<std::uint32_t>(m_chunks.size());
    out.write(kMagic, sizeof(kMagic));
    out.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
    out.write(reinterpret_cast<const char*>(&chunkCount), sizeof(chunkCount));
    if (!out.good()) {
        return false;
    }

    constexpr std::size_t kVoxelsPerChunk =
        static_cast<std::size_t>(Chunk::kSizeX * Chunk::kSizeY * Chunk::kSizeZ);
    constexpr std::size_t kBytesPerChunk = (kVoxelsPerChunk + 7u) / 8u;
    std::vector<std::uint8_t> packed(kBytesPerChunk, 0u);

    for (const Chunk& chunk : m_chunks) {
        const std::int32_t chunkX = static_cast<std::int32_t>(chunk.chunkX());
        const std::int32_t chunkY = static_cast<std::int32_t>(chunk.chunkY());
        const std::int32_t chunkZ = static_cast<std::int32_t>(chunk.chunkZ());
        out.write(reinterpret_cast<const char*>(&chunkX), sizeof(chunkX));
        out.write(reinterpret_cast<const char*>(&chunkY), sizeof(chunkY));
        out.write(reinterpret_cast<const char*>(&chunkZ), sizeof(chunkZ));

        std::fill(packed.begin(), packed.end(), std::uint8_t{0});
        std::size_t voxelIndex = 0;
        for (int y = 0; y < Chunk::kSizeY; ++y) {
            for (int z = 0; z < Chunk::kSizeZ; ++z) {
                for (int x = 0; x < Chunk::kSizeX; ++x, ++voxelIndex) {
                    if (!chunk.isSolid(x, y, z)) {
                        continue;
                    }
                    const std::size_t byteIndex = voxelIndex >> 3u;
                    const std::uint8_t bitMask = static_cast<std::uint8_t>(1u << (voxelIndex & 7u));
                    packed[byteIndex] = static_cast<std::uint8_t>(packed[byteIndex] | bitMask);
                }
            }
        }

        out.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
        if (!out.good()) {
            return false;
        }
    }

    return true;
}

inline std::size_t ChunkGrid::chunkCount() const {
    return m_chunks.size();
}

inline std::vector<Chunk>& ChunkGrid::chunks() {
    return m_chunks;
}

inline const std::vector<Chunk>& ChunkGrid::chunks() const {
    return m_chunks;
}

} // namespace world
