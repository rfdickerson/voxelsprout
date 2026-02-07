#pragma once

#include <cstddef>
#include <vector>

#include "world/Voxel.hpp"

// World Chunk subsystem
// Responsible for: owning a small local collection of voxels.
// Should NOT do: global world streaming, simulation rules, or drawing.
namespace world {

class Chunk {
public:
    static constexpr int kSizeX = 16;
    static constexpr int kSizeY = 16;
    static constexpr int kSizeZ = 16;

    Chunk();
    void setVoxel(int x, int y, int z, Voxel voxel);
    void fillLayer(int y, Voxel voxel);
    const std::vector<Voxel>& voxels() const;

private:
    static std::size_t linearIndex(int x, int y, int z);
    static bool isInBounds(int x, int y, int z);
    std::vector<Voxel> m_voxels;
};

inline Chunk::Chunk()
    : m_voxels(static_cast<std::size_t>(kSizeX * kSizeY * kSizeZ)) {}

inline void Chunk::setVoxel(int x, int y, int z, Voxel voxel) {
    if (!isInBounds(x, y, z)) {
        return;
    }
    m_voxels[linearIndex(x, y, z)] = voxel;
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

inline const std::vector<Voxel>& Chunk::voxels() const {
    return m_voxels;
}

inline std::size_t Chunk::linearIndex(int x, int y, int z) {
    return static_cast<std::size_t>(x + (kSizeX * (z + (kSizeZ * y))));
}

inline bool Chunk::isInBounds(int x, int y, int z) {
    return x >= 0 && x < kSizeX && y >= 0 && y < kSizeY && z >= 0 && z < kSizeZ;
}

} // namespace world
