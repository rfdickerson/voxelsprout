#include "render/packed_vertex.h"

namespace odai::world {
std::uint32_t PackedVoxelVertex::pack(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t z,
    std::uint32_t face,
    std::uint32_t corner,
    std::uint32_t ao,
    std::uint32_t material,
    std::uint32_t baseColorIndex,
    std::uint32_t lodLevel
) {
    return ((x & kMask5) << kShiftX) |
           ((y & kMask5) << kShiftY) |
           ((z & kMask5) << kShiftZ) |
           ((face & kMask3) << kShiftFace) |
           ((corner & kMask2) << kShiftCorner) |
           ((ao & kMask4) << kShiftAo) |
           ((material & kMask3) << kShiftMaterial) |
           ((baseColorIndex & kMask3) << kShiftBaseColor) |
           ((lodLevel & kMask2) << kShiftLodLevel);
}
}
