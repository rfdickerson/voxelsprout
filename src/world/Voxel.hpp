#pragma once

#include <cstdint>

// World Voxel subsystem
// Responsible for: defining the smallest data unit in the voxel world.
// Should NOT do: perform simulation logic, chunk storage management, or rendering.
namespace world {

enum class VoxelType : std::uint8_t {
    Empty = 0,
    Solid = 1
};

struct Voxel {
    VoxelType type = VoxelType::Empty;
};

} // namespace world
