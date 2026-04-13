#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

#include "core/log.h"
#include "world/chunk.h"
#include "world/voxel.h"

// World ChunkGrid subsystem
// Responsible for: owning a collection of chunks that represent world space.
// Should NOT do: pathfinding, factory simulation, or rendering API calls.
namespace voxelsprout::world {

class ChunkGrid {
public:
    ChunkGrid() = default;
    void initializeEmptyWorld();
    void initializeFlatWorld();
    bool loadFromBinaryFile(const std::filesystem::path& path);
    bool saveToBinaryFile(const std::filesystem::path& path) const;
    std::size_t chunkCount() const;
    void setChunks(std::vector<Chunk> chunks);
    std::vector<Chunk>& chunks();
    const std::vector<Chunk>& chunks() const;

private:
    std::vector<Chunk> m_chunks;
};

Chunk buildProceduralChunk(int chunkX, int chunkY, int chunkZ);

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
    using Clock = std::chrono::steady_clock;
    const auto loadStart = Clock::now();
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
    if (std::memcmp(magic, kExpectedMagic, sizeof(magic)) != 0) {
        return false;
    }
    if (version != 1u && version != 2u && version != 3u) {
        return false;
    }
    if (chunkCount == 0 || chunkCount > 4096u) {
        return false;
    }

    constexpr std::size_t kVoxelsPerChunk =
        static_cast<std::size_t>(Chunk::kSizeX * Chunk::kSizeY * Chunk::kSizeZ);
    constexpr std::size_t kBytesPerChunk = (kVoxelsPerChunk + 7u) / 8u;
    std::vector<std::uint8_t> typeBytes(kVoxelsPerChunk, 0u);
    std::vector<std::uint8_t> baseColorBytes(kVoxelsPerChunk, 0xFFu);
    std::vector<std::uint8_t> packed(kBytesPerChunk, 0u);
    std::int64_t ioMicros = 0;
    std::int64_t decodeMicros = 0;

    std::vector<Chunk> loadedChunks;
    loadedChunks.reserve(static_cast<std::size_t>(chunkCount));

    for (std::uint32_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
        const auto ioStart = Clock::now();
        std::int32_t chunkX = 0;
        std::int32_t chunkY = 0;
        std::int32_t chunkZ = 0;
        in.read(reinterpret_cast<char*>(&chunkX), sizeof(chunkX));
        in.read(reinterpret_cast<char*>(&chunkY), sizeof(chunkY));
        in.read(reinterpret_cast<char*>(&chunkZ), sizeof(chunkZ));
        Chunk chunk(chunkX, chunkY, chunkZ);
        if (version == 1u) {
            in.read(reinterpret_cast<char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
            if (!in.good()) {
                return false;
            }
            ioMicros += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - ioStart).count();
            const auto decodeStart = Clock::now();
            chunk.setFromSolidBitfield(packed.data(), packed.size());
            decodeMicros += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - decodeStart).count();
        } else if (version == 2u) {
            in.read(reinterpret_cast<char*>(typeBytes.data()), static_cast<std::streamsize>(typeBytes.size()));
            if (!in.good()) {
                return false;
            }
            ioMicros += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - ioStart).count();
            const auto decodeStart = Clock::now();
            chunk.setFromTypedVoxelBytes(typeBytes.data(), typeBytes.size());
            decodeMicros += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - decodeStart).count();
        } else {
            in.read(reinterpret_cast<char*>(typeBytes.data()), static_cast<std::streamsize>(typeBytes.size()));
            in.read(reinterpret_cast<char*>(baseColorBytes.data()), static_cast<std::streamsize>(baseColorBytes.size()));
            if (!in.good()) {
                return false;
            }
            ioMicros += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - ioStart).count();
            const auto decodeStart = Clock::now();
            chunk.setFromTypedVoxelAndBaseColorBytes(typeBytes.data(), baseColorBytes.data(), typeBytes.size());
            decodeMicros += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - decodeStart).count();
        }
        loadedChunks.push_back(std::move(chunk));
    }

    m_chunks = std::move(loadedChunks);
    const auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - loadStart).count();
    const double ioMs = static_cast<double>(ioMicros) / 1000.0;
    const double decodeMs = static_cast<double>(decodeMicros) / 1000.0;
    VOX_LOGI("world") << "load binary '" << path.string() << "'"
                      << " version=" << version
                      << ", chunks=" << chunkCount
                      << ", ioMs=" << ioMs
                      << ", decodeMs=" << decodeMs
                      << ", totalMs=" << totalMs;
    return true;
}

inline bool ChunkGrid::saveToBinaryFile(const std::filesystem::path& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }

    constexpr char kMagic[4] = {'V', 'X', 'W', '1'};
    constexpr std::uint32_t kVersion = 3u;
    const std::uint32_t chunkCount = static_cast<std::uint32_t>(m_chunks.size());
    out.write(kMagic, sizeof(kMagic));
    out.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
    out.write(reinterpret_cast<const char*>(&chunkCount), sizeof(chunkCount));
    if (!out.good()) {
        return false;
    }

    constexpr std::size_t kVoxelsPerChunk =
        static_cast<std::size_t>(Chunk::kSizeX * Chunk::kSizeY * Chunk::kSizeZ);
    std::vector<std::uint8_t> typeBytes(kVoxelsPerChunk, 0u);
    std::vector<std::uint8_t> baseColorBytes(kVoxelsPerChunk, 0xFFu);

    for (const Chunk& chunk : m_chunks) {
        const std::int32_t chunkX = static_cast<std::int32_t>(chunk.chunkX());
        const std::int32_t chunkY = static_cast<std::int32_t>(chunk.chunkY());
        const std::int32_t chunkZ = static_cast<std::int32_t>(chunk.chunkZ());
        out.write(reinterpret_cast<const char*>(&chunkX), sizeof(chunkX));
        out.write(reinterpret_cast<const char*>(&chunkY), sizeof(chunkY));
        out.write(reinterpret_cast<const char*>(&chunkZ), sizeof(chunkZ));

        std::size_t voxelIndex = 0;
        for (int y = 0; y < Chunk::kSizeY; ++y) {
            for (int z = 0; z < Chunk::kSizeZ; ++z) {
                for (int x = 0; x < Chunk::kSizeX; ++x, ++voxelIndex) {
                    const Voxel voxel = chunk.voxelAt(x, y, z);
                    typeBytes[voxelIndex] = static_cast<std::uint8_t>(voxel.type);
                    baseColorBytes[voxelIndex] = voxel.baseColorIndex;
                }
            }
        }

        out.write(reinterpret_cast<const char*>(typeBytes.data()), static_cast<std::streamsize>(typeBytes.size()));
        out.write(reinterpret_cast<const char*>(baseColorBytes.data()), static_cast<std::streamsize>(baseColorBytes.size()));
        if (!out.good()) {
            return false;
        }
    }

    return true;
}

inline std::size_t ChunkGrid::chunkCount() const {
    return m_chunks.size();
}

inline void ChunkGrid::setChunks(std::vector<Chunk> chunks) {
    m_chunks = std::move(chunks);
}

inline std::vector<Chunk>& ChunkGrid::chunks() {
    return m_chunks;
}

inline const std::vector<Chunk>& ChunkGrid::chunks() const {
    return m_chunks;
}

} // namespace voxelsprout::world
