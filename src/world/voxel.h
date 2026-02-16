#pragma once

#include <cstdint>

// World Voxel subsystem
// Responsible for: defining the smallest data unit in the voxel world.
// Should NOT do: perform simulation logic, chunk storage management, or rendering.
namespace voxelsprout::world {

enum class VoxelType : std::uint8_t {
    Empty = 0,
    Solid = 1,
    Stone = Solid,
    SolidRed = 2,
    Dirt = 3,
    Grass = 4,
    Wood = 5
};

struct Voxel {
    VoxelType type = VoxelType::Empty;
    // Optional 4-bit base-color index (0..15). 0xFF means "use material defaults".
    std::uint8_t baseColorIndex = 0xFFu;
};

} // namespace voxelsprout::world
