#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>

#include "world/voxel.h"

// World Chunk subsystem
// Responsible for: owning a small local collection of voxels.
// Should NOT do: global world streaming, simulation rules, or drawing.
namespace world {

class Chunk {
public:
    static constexpr int kSizeX = 32;
    static constexpr int kSizeY = 32;
    static constexpr int kSizeZ = 32;

    // Resolution-aware chunk layout:
    // - Chunk8: one macro cell represents an 8x8x8 block in chunk space.
    // - Chunk4: optional 2x2x2 subcells (each subcell is 4x4x4 voxels).
    // - Chunk1: optional full 8x8x8 micro voxels.
    static constexpr int kMacroVoxelSize = 8;
    static constexpr int kRefined4VoxelSize = 4;
    static constexpr int kRefined4CellsPerAxis = kMacroVoxelSize / kRefined4VoxelSize;
    static constexpr std::uint16_t kInvalidRefinementIndex = 0xFFFFu;
    static constexpr int kMacroSizeX = kSizeX / kMacroVoxelSize;
    static constexpr int kMacroSizeY = kSizeY / kMacroVoxelSize;
    static constexpr int kMacroSizeZ = kSizeZ / kMacroVoxelSize;

    enum class CellResolution : std::uint8_t {
        Uniform = 0,
        Refined4 = 1,
        Refined1 = 2
    };

    struct MacroCell {
        Voxel voxel{};
        CellResolution resolution = CellResolution::Uniform;
        std::uint16_t refined4Index = kInvalidRefinementIndex;
        std::uint16_t refined1Index = kInvalidRefinementIndex;
    };

    struct Chunk4Cell {
        std::array<Voxel, static_cast<std::size_t>(kRefined4CellsPerAxis * kRefined4CellsPerAxis * kRefined4CellsPerAxis)> subcells{};
    };

    struct Chunk1Cell {
        std::array<Voxel, static_cast<std::size_t>(kMacroVoxelSize * kMacroVoxelSize * kMacroVoxelSize)> voxels{};
    };

    Chunk();
    Chunk(int chunkX, int chunkY, int chunkZ);
    void setVoxel(int x, int y, int z, Voxel voxel);
    void setVoxelRefined4(int x, int y, int z, Voxel voxel);
    void setVoxelRefined(int x, int y, int z, Voxel voxel);
    void setFromSolidBitfield(const std::uint8_t* packedBits, std::size_t packedByteCount);
    void setFromTypedVoxelBytes(const std::uint8_t* typeBytes, std::size_t byteCount);
    void fillLayer(int y, Voxel voxel);
    Voxel voxelAt(int x, int y, int z) const;
    bool isSolid(int x, int y, int z) const;
    const std::vector<Voxel>& voxels() const;
    MacroCell macroCellAt(int mx, int my, int mz) const;
    bool isMacroSolid(int mx, int my, int mz) const;
    int chunkX() const;
    int chunkY() const;
    int chunkZ() const;

private:
    static std::size_t linearIndex(int x, int y, int z);
    static std::size_t macroLinearIndex(int mx, int my, int mz);
    static bool isInBounds(int x, int y, int z);
    static bool isMacroInBounds(int mx, int my, int mz);
    static std::size_t refined4LinearIndex(int sx, int sy, int sz);
    static std::size_t refined1LinearIndex(int lx, int ly, int lz);
    static VoxelType voxelTypeFromSerializedByte(std::uint8_t raw);
    std::size_t ensureChunk4IndexForCell(MacroCell& cell);
    std::size_t ensureChunk1IndexForCell(MacroCell& cell, int mx, int my, int mz);
    static bool isUniformChunk4(const Chunk4Cell& chunk4Cell);
    void rebuildMacroHierarchyFromDense();
    void syncMacroCellFromDense(int mx, int my, int mz);
    std::vector<Voxel> m_voxels;
    std::vector<MacroCell> m_macroCells;
    std::vector<Chunk4Cell> m_chunk4Cells;
    std::vector<Chunk1Cell> m_chunk1Cells;
    int m_chunkX = 0;
    int m_chunkY = 0;
    int m_chunkZ = 0;
};

inline Chunk::Chunk()
    : m_voxels(static_cast<std::size_t>(kSizeX * kSizeY * kSizeZ)),
      m_macroCells(static_cast<std::size_t>(kMacroSizeX * kMacroSizeY * kMacroSizeZ)) {}

inline Chunk::Chunk(int chunkX, int chunkY, int chunkZ)
    : m_voxels(static_cast<std::size_t>(kSizeX * kSizeY * kSizeZ)),
      m_macroCells(static_cast<std::size_t>(kMacroSizeX * kMacroSizeY * kMacroSizeZ)),
      m_chunkX(chunkX),
      m_chunkY(chunkY),
      m_chunkZ(chunkZ) {}

inline void Chunk::setVoxel(int x, int y, int z, Voxel voxel) {
    if (!isInBounds(x, y, z)) {
        return;
    }

    m_voxels[linearIndex(x, y, z)] = voxel;

    const int macroX = x / kMacroVoxelSize;
    const int macroY = y / kMacroVoxelSize;
    const int macroZ = z / kMacroVoxelSize;
    syncMacroCellFromDense(macroX, macroY, macroZ);
}

inline void Chunk::setVoxelRefined4(int x, int y, int z, Voxel voxel) {
    setVoxel(x, y, z, voxel);
}

inline void Chunk::fillLayer(int y, Voxel voxel) {
    if (y < 0 || y >= kSizeY) {
        return;
    }
    for (int z = 0; z < kSizeZ; ++z) {
        for (int x = 0; x < kSizeX; ++x) {
            setVoxel(x, y, z, voxel);
        }
    }
}

inline void Chunk::setVoxelRefined(int x, int y, int z, Voxel voxel) {
    setVoxel(x, y, z, voxel);
}

inline void Chunk::setFromSolidBitfield(const std::uint8_t* packedBits, std::size_t packedByteCount) {
    constexpr std::size_t kVoxelCount = static_cast<std::size_t>(kSizeX * kSizeY * kSizeZ);
    constexpr std::size_t kExpectedBytes = (kVoxelCount + 7u) / 8u;

    std::fill(m_voxels.begin(), m_voxels.end(), Voxel{VoxelType::Empty});
    std::fill(m_macroCells.begin(), m_macroCells.end(), MacroCell{});
    m_chunk4Cells.clear();
    m_chunk1Cells.clear();

    if (packedBits == nullptr || packedByteCount < kExpectedBytes) {
        return;
    }

    std::size_t voxelIndex = 0;
    for (int y = 0; y < kSizeY; ++y) {
        for (int z = 0; z < kSizeZ; ++z) {
            for (int x = 0; x < kSizeX; ++x, ++voxelIndex) {
                const std::size_t byteIndex = voxelIndex >> 3u;
                const std::uint8_t bitMask = static_cast<std::uint8_t>(1u << (voxelIndex & 7u));
                if ((packedBits[byteIndex] & bitMask) != 0u) {
                    m_voxels[voxelIndex] = Voxel{VoxelType::Solid};
                }
            }
        }
    }

    rebuildMacroHierarchyFromDense();
}

inline void Chunk::setFromTypedVoxelBytes(const std::uint8_t* typeBytes, std::size_t byteCount) {
    constexpr std::size_t kVoxelCount = static_cast<std::size_t>(kSizeX * kSizeY * kSizeZ);

    std::fill(m_voxels.begin(), m_voxels.end(), Voxel{VoxelType::Empty});
    std::fill(m_macroCells.begin(), m_macroCells.end(), MacroCell{});
    m_chunk4Cells.clear();
    m_chunk1Cells.clear();

    if (typeBytes == nullptr || byteCount < kVoxelCount) {
        return;
    }

    std::size_t voxelIndex = 0;
    for (int y = 0; y < kSizeY; ++y) {
        for (int z = 0; z < kSizeZ; ++z) {
            for (int x = 0; x < kSizeX; ++x, ++voxelIndex) {
                m_voxels[voxelIndex] = Voxel{voxelTypeFromSerializedByte(typeBytes[voxelIndex])};
            }
        }
    }

    rebuildMacroHierarchyFromDense();
}

inline Voxel Chunk::voxelAt(int x, int y, int z) const {
    if (!isInBounds(x, y, z)) {
        return Voxel{VoxelType::Empty};
    }
    return m_voxels[linearIndex(x, y, z)];
}

inline bool Chunk::isSolid(int x, int y, int z) const {
    return voxelAt(x, y, z).type != VoxelType::Empty;
}

inline const std::vector<Voxel>& Chunk::voxels() const {
    return m_voxels;
}

inline Chunk::MacroCell Chunk::macroCellAt(int mx, int my, int mz) const {
    if (!isMacroInBounds(mx, my, mz)) {
        return MacroCell{};
    }
    return m_macroCells[macroLinearIndex(mx, my, mz)];
}

inline bool Chunk::isMacroSolid(int mx, int my, int mz) const {
    return macroCellAt(mx, my, mz).voxel.type != VoxelType::Empty;
}

inline int Chunk::chunkX() const {
    return m_chunkX;
}

inline int Chunk::chunkY() const {
    return m_chunkY;
}

inline int Chunk::chunkZ() const {
    return m_chunkZ;
}

inline std::size_t Chunk::linearIndex(int x, int y, int z) {
    return static_cast<std::size_t>(x + (kSizeX * (z + (kSizeZ * y))));
}

inline std::size_t Chunk::macroLinearIndex(int mx, int my, int mz) {
    return static_cast<std::size_t>(mx + (kMacroSizeX * (mz + (kMacroSizeZ * my))));
}

inline std::size_t Chunk::refined4LinearIndex(int sx, int sy, int sz) {
    return static_cast<std::size_t>(sx + (kRefined4CellsPerAxis * (sz + (kRefined4CellsPerAxis * sy))));
}

inline std::size_t Chunk::refined1LinearIndex(int lx, int ly, int lz) {
    return static_cast<std::size_t>(lx + (kMacroVoxelSize * (lz + (kMacroVoxelSize * ly))));
}

inline VoxelType Chunk::voxelTypeFromSerializedByte(std::uint8_t raw) {
    switch (raw) {
    case static_cast<std::uint8_t>(VoxelType::Empty):
        return VoxelType::Empty;
    case static_cast<std::uint8_t>(VoxelType::Solid):
        return VoxelType::Stone;
    case static_cast<std::uint8_t>(VoxelType::SolidRed):
        return VoxelType::SolidRed;
    case static_cast<std::uint8_t>(VoxelType::Dirt):
        return VoxelType::Dirt;
    case static_cast<std::uint8_t>(VoxelType::Grass):
        return VoxelType::Grass;
    case static_cast<std::uint8_t>(VoxelType::Wood):
        return VoxelType::Wood;
    default:
        return VoxelType::Stone;
    }
}

inline bool Chunk::isInBounds(int x, int y, int z) {
    return x >= 0 && x < kSizeX && y >= 0 && y < kSizeY && z >= 0 && z < kSizeZ;
}

inline bool Chunk::isMacroInBounds(int mx, int my, int mz) {
    return mx >= 0 && mx < kMacroSizeX && my >= 0 && my < kMacroSizeY && mz >= 0 && mz < kMacroSizeZ;
}

inline std::size_t Chunk::ensureChunk4IndexForCell(MacroCell& cell) {
    if (cell.refined4Index != kInvalidRefinementIndex && cell.refined4Index < m_chunk4Cells.size()) {
        return static_cast<std::size_t>(cell.refined4Index);
    }

    Chunk4Cell newChunk4Cell{};
    for (Voxel& subcell : newChunk4Cell.subcells) {
        subcell = cell.voxel;
    }
    m_chunk4Cells.push_back(newChunk4Cell);
    const std::size_t index = m_chunk4Cells.size() - 1;
    cell.refined4Index = static_cast<std::uint16_t>(index);
    return index;
}

inline std::size_t Chunk::ensureChunk1IndexForCell(MacroCell& cell, int mx, int my, int mz) {
    if (cell.refined1Index != kInvalidRefinementIndex && cell.refined1Index < m_chunk1Cells.size()) {
        return static_cast<std::size_t>(cell.refined1Index);
    }

    Chunk1Cell newChunk1Cell{};
    const int baseX = mx * kMacroVoxelSize;
    const int baseY = my * kMacroVoxelSize;
    const int baseZ = mz * kMacroVoxelSize;
    for (int ly = 0; ly < kMacroVoxelSize; ++ly) {
        for (int lz = 0; lz < kMacroVoxelSize; ++lz) {
            for (int lx = 0; lx < kMacroVoxelSize; ++lx) {
                newChunk1Cell.voxels[refined1LinearIndex(lx, ly, lz)] =
                    m_voxels[linearIndex(baseX + lx, baseY + ly, baseZ + lz)];
            }
        }
    }

    m_chunk1Cells.push_back(newChunk1Cell);
    const std::size_t index = m_chunk1Cells.size() - 1;
    cell.refined1Index = static_cast<std::uint16_t>(index);
    return index;
}

inline bool Chunk::isUniformChunk4(const Chunk4Cell& chunk4Cell) {
    const VoxelType firstType = chunk4Cell.subcells[0].type;
    for (const Voxel& subcell : chunk4Cell.subcells) {
        if (subcell.type != firstType) {
            return false;
        }
    }
    return true;
}

inline void Chunk::rebuildMacroHierarchyFromDense() {
    for (int my = 0; my < kMacroSizeY; ++my) {
        for (int mz = 0; mz < kMacroSizeZ; ++mz) {
            for (int mx = 0; mx < kMacroSizeX; ++mx) {
                syncMacroCellFromDense(mx, my, mz);
            }
        }
    }
}

inline void Chunk::syncMacroCellFromDense(int mx, int my, int mz) {
    if (!isMacroInBounds(mx, my, mz)) {
        return;
    }

    const int beginX = mx * kMacroVoxelSize;
    const int beginY = my * kMacroVoxelSize;
    const int beginZ = mz * kMacroVoxelSize;

    const Voxel first = m_voxels[linearIndex(beginX, beginY, beginZ)];
    bool uniform = true;
    bool anySolid = first.type != VoxelType::Empty;

    for (int localY = 0; localY < kMacroVoxelSize; ++localY) {
        for (int localZ = 0; localZ < kMacroVoxelSize; ++localZ) {
            for (int localX = 0; localX < kMacroVoxelSize; ++localX) {
                const Voxel sample =
                    m_voxels[linearIndex(beginX + localX, beginY + localY, beginZ + localZ)];
                anySolid = anySolid || (sample.type != VoxelType::Empty);
                if (sample.type != first.type) {
                    uniform = false;
                }
            }
        }
    }

    MacroCell& cell = m_macroCells[macroLinearIndex(mx, my, mz)];
    if (uniform) {
        cell.voxel = first;
        cell.resolution = CellResolution::Uniform;
        cell.refined4Index = kInvalidRefinementIndex;
        cell.refined1Index = kInvalidRefinementIndex;
        return;
    }
    bool canRepresentAsChunk4 = true;
    Chunk4Cell chunk4Cell{};
    for (int subY = 0; subY < kRefined4CellsPerAxis; ++subY) {
        for (int subZ = 0; subZ < kRefined4CellsPerAxis; ++subZ) {
            for (int subX = 0; subX < kRefined4CellsPerAxis; ++subX) {
                const int subBeginX = beginX + (subX * kRefined4VoxelSize);
                const int subBeginY = beginY + (subY * kRefined4VoxelSize);
                const int subBeginZ = beginZ + (subZ * kRefined4VoxelSize);
                const Voxel subFirst = m_voxels[linearIndex(subBeginX, subBeginY, subBeginZ)];
                for (int ly = 0; ly < kRefined4VoxelSize; ++ly) {
                    for (int lz = 0; lz < kRefined4VoxelSize; ++lz) {
                        for (int lx = 0; lx < kRefined4VoxelSize; ++lx) {
                            if (m_voxels[linearIndex(subBeginX + lx, subBeginY + ly, subBeginZ + lz)].type != subFirst.type) {
                                canRepresentAsChunk4 = false;
                            }
                        }
                    }
                }
                chunk4Cell.subcells[refined4LinearIndex(subX, subY, subZ)] = subFirst;
            }
        }
    }

    cell.voxel = Voxel{anySolid ? VoxelType::Solid : VoxelType::Empty};
    if (canRepresentAsChunk4) {
        m_chunk4Cells.push_back(chunk4Cell);
        cell.refined4Index = static_cast<std::uint16_t>(m_chunk4Cells.size() - 1);
        cell.refined1Index = kInvalidRefinementIndex;
        cell.resolution = CellResolution::Refined4;
        return;
    }

    Chunk1Cell chunk1Cell{};
    for (int ly = 0; ly < kMacroVoxelSize; ++ly) {
        for (int lz = 0; lz < kMacroVoxelSize; ++lz) {
            for (int lx = 0; lx < kMacroVoxelSize; ++lx) {
                chunk1Cell.voxels[refined1LinearIndex(lx, ly, lz)] =
                    m_voxels[linearIndex(beginX + lx, beginY + ly, beginZ + lz)];
            }
        }
    }
    m_chunk1Cells.push_back(chunk1Cell);
    cell.refined1Index = static_cast<std::uint16_t>(m_chunk1Cells.size() - 1);
    cell.refined4Index = kInvalidRefinementIndex;
    cell.resolution = CellResolution::Refined1;
}

} // namespace world
