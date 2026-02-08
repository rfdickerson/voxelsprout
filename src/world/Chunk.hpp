#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "world/Voxel.hpp"

// World Chunk subsystem
// Responsible for: owning a small local collection of voxels.
// Should NOT do: global world streaming, simulation rules, or drawing.
namespace world {

class Chunk {
public:
    static constexpr int kSizeX = 32;
    static constexpr int kSizeY = 32;
    static constexpr int kSizeZ = 32;

    // Phase 1 multi-resolution layout: one macro cell represents a 4x4x4 block in chunk space.
    static constexpr int kMacroVoxelSize = 4;
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
    };

    Chunk();
    Chunk(int chunkX, int chunkY, int chunkZ);
    void setVoxel(int x, int y, int z, Voxel voxel);
    void setVoxelRefined(int x, int y, int z, Voxel voxel);
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
    void syncMacroCellFromDense(int mx, int my, int mz);
    std::vector<Voxel> m_voxels;
    std::vector<MacroCell> m_macroCells;
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

    const int macroX = x / kMacroVoxelSize;
    const int macroY = y / kMacroVoxelSize;
    const int macroZ = z / kMacroVoxelSize;
    if (!isMacroInBounds(macroX, macroY, macroZ)) {
        return;
    }

    MacroCell& cell = m_macroCells[macroLinearIndex(macroX, macroY, macroZ)];
    cell.voxel = voxel;
    cell.resolution = CellResolution::Uniform;

    // Keep dense per-voxel cache in sync so existing systems can still query voxelAt/isSolid.
    const int beginX = macroX * kMacroVoxelSize;
    const int beginY = macroY * kMacroVoxelSize;
    const int beginZ = macroZ * kMacroVoxelSize;
    for (int localY = 0; localY < kMacroVoxelSize; ++localY) {
        for (int localZ = 0; localZ < kMacroVoxelSize; ++localZ) {
            for (int localX = 0; localX < kMacroVoxelSize; ++localX) {
                m_voxels[linearIndex(beginX + localX, beginY + localY, beginZ + localZ)] = voxel;
            }
        }
    }
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
    if (!isInBounds(x, y, z)) {
        return;
    }

    m_voxels[linearIndex(x, y, z)] = voxel;

    const int macroX = x / kMacroVoxelSize;
    const int macroY = y / kMacroVoxelSize;
    const int macroZ = z / kMacroVoxelSize;
    syncMacroCellFromDense(macroX, macroY, macroZ);
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

inline bool Chunk::isInBounds(int x, int y, int z) {
    return x >= 0 && x < kSizeX && y >= 0 && y < kSizeY && z >= 0 && z < kSizeZ;
}

inline bool Chunk::isMacroInBounds(int mx, int my, int mz) {
    return mx >= 0 && mx < kMacroSizeX && my >= 0 && my < kMacroSizeY && mz >= 0 && mz < kMacroSizeZ;
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
        return;
    }

    cell.voxel = Voxel{anySolid ? VoxelType::Solid : VoxelType::Empty};
    cell.resolution = CellResolution::Refined1;
}

} // namespace world
