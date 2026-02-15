#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/Grid3.hpp"
#include "world/Chunk.hpp"
#include "world/Voxel.hpp"

// World CSG subsystem
// Responsible for: deterministic voxel-space CSG commands for building and carving toy-world structures.
// Should NOT do: chunk streaming, rendering, or simulation tick scheduling.
namespace world {

enum class BrushKind : std::uint8_t {
    Box = 0,
    PrismPipe = 1,
    Ramp = 2
};

enum class CsgOp : std::uint8_t {
    AddSolid = 0,
    SubtractSolid = 1,
    PaintMaterial = 2
};

inline constexpr std::uint16_t kCsgAffectEmpty = 1u << 0;
inline constexpr std::uint16_t kCsgAffectSolid = 1u << 1;
inline constexpr std::uint16_t kCsgAffectAll = 0xFFFFu;

struct Brush {
    BrushKind kind = BrushKind::Box;
    core::Cell3i minCell{};
    core::Cell3i maxCell{};
    core::Dir6 axis = core::Dir6::PosY;
    std::uint16_t radiusQ8 = 128;
};

struct CsgCommand {
    CsgOp op = CsgOp::AddSolid;
    Brush brush{};
    std::uint16_t materialId = 0;
    std::uint16_t affectMask = kCsgAffectAll;
};

struct CsgCell {
    Voxel voxel{};
    std::uint16_t materialId = 0;

    constexpr bool operator==(const CsgCell& rhs) const {
        return voxel.type == rhs.voxel.type && materialId == rhs.materialId;
    }
};

class CsgVolume {
public:
    CsgVolume() = default;
    CsgVolume(std::int32_t sizeX, std::int32_t sizeY, std::int32_t sizeZ, core::Cell3i originCell = {});

    bool isValid() const;
    core::Cell3i origin() const;
    std::int32_t sizeX() const;
    std::int32_t sizeY() const;
    std::int32_t sizeZ() const;
    core::CellAabb worldBounds() const;

    bool containsWorldCell(const core::Cell3i& worldCell) const;
    CsgCell cellAtWorld(const core::Cell3i& worldCell) const;
    void setCellAtWorld(const core::Cell3i& worldCell, const CsgCell& cell);

    const std::vector<CsgCell>& cells() const;
    std::vector<CsgCell>& cells();

private:
    std::size_t linearIndexFromWorld(const core::Cell3i& worldCell) const;

    core::Cell3i m_origin{};
    std::int32_t m_sizeX = 0;
    std::int32_t m_sizeY = 0;
    std::int32_t m_sizeZ = 0;
    std::vector<CsgCell> m_cells;
};

inline CsgVolume::CsgVolume(std::int32_t sizeX, std::int32_t sizeY, std::int32_t sizeZ, core::Cell3i originCell)
    : m_origin(originCell),
      m_sizeX(std::max<std::int32_t>(sizeX, 0)),
      m_sizeY(std::max<std::int32_t>(sizeY, 0)),
      m_sizeZ(std::max<std::int32_t>(sizeZ, 0)),
      m_cells(static_cast<std::size_t>(m_sizeX) * static_cast<std::size_t>(m_sizeY) * static_cast<std::size_t>(m_sizeZ)) {}

inline bool CsgVolume::isValid() const {
    return m_sizeX > 0 && m_sizeY > 0 && m_sizeZ > 0;
}

inline core::Cell3i CsgVolume::origin() const {
    return m_origin;
}

inline std::int32_t CsgVolume::sizeX() const {
    return m_sizeX;
}

inline std::int32_t CsgVolume::sizeY() const {
    return m_sizeY;
}

inline std::int32_t CsgVolume::sizeZ() const {
    return m_sizeZ;
}

inline core::CellAabb CsgVolume::worldBounds() const {
    if (!isValid()) {
        return core::CellAabb{};
    }
    core::CellAabb bounds{};
    bounds.valid = true;
    bounds.minInclusive = m_origin;
    bounds.maxExclusive = m_origin + core::Cell3i{m_sizeX, m_sizeY, m_sizeZ};
    return bounds;
}

inline bool CsgVolume::containsWorldCell(const core::Cell3i& worldCell) const {
    return worldBounds().contains(worldCell);
}

inline CsgCell CsgVolume::cellAtWorld(const core::Cell3i& worldCell) const {
    if (!containsWorldCell(worldCell)) {
        return CsgCell{};
    }
    return m_cells[linearIndexFromWorld(worldCell)];
}

inline void CsgVolume::setCellAtWorld(const core::Cell3i& worldCell, const CsgCell& cell) {
    if (!containsWorldCell(worldCell)) {
        return;
    }
    m_cells[linearIndexFromWorld(worldCell)] = cell;
}

inline const std::vector<CsgCell>& CsgVolume::cells() const {
    return m_cells;
}

inline std::vector<CsgCell>& CsgVolume::cells() {
    return m_cells;
}

inline std::size_t CsgVolume::linearIndexFromWorld(const core::Cell3i& worldCell) const {
    const core::Cell3i local = worldCell - m_origin;
    return static_cast<std::size_t>(
        local.x + (m_sizeX * (local.z + (m_sizeZ * local.y)))
    );
}

inline core::CellAabb brushBounds(const Brush& brush) {
    const core::Cell3i minCell{
        std::min(brush.minCell.x, brush.maxCell.x),
        std::min(brush.minCell.y, brush.maxCell.y),
        std::min(brush.minCell.z, brush.maxCell.z)
    };
    const core::Cell3i maxCell{
        std::max(brush.minCell.x, brush.maxCell.x),
        std::max(brush.minCell.y, brush.maxCell.y),
        std::max(brush.minCell.z, brush.maxCell.z)
    };

    if (maxCell.x <= minCell.x || maxCell.y <= minCell.y || maxCell.z <= minCell.z) {
        return core::CellAabb{};
    }

    core::CellAabb bounds{};
    bounds.valid = true;
    bounds.minInclusive = minCell;
    bounds.maxExclusive = maxCell;
    return bounds;
}

namespace detail {

inline bool brushContainsPrismPipeCell(const Brush& brush, const core::CellAabb& bounds, const core::Cell3i& cell) {
    if (!bounds.contains(cell)) {
        return false;
    }

    const std::int32_t radiusQ8 = std::max<std::int32_t>(1, static_cast<std::int32_t>(brush.radiusQ8));
    const std::int32_t cellXQ8 = (cell.x * 256) + 128;
    const std::int32_t cellYQ8 = (cell.y * 256) + 128;
    const std::int32_t cellZQ8 = (cell.z * 256) + 128;
    const std::int32_t centerXQ8 = (bounds.minInclusive.x + bounds.maxExclusive.x) * 128;
    const std::int32_t centerYQ8 = (bounds.minInclusive.y + bounds.maxExclusive.y) * 128;
    const std::int32_t centerZQ8 = (bounds.minInclusive.z + bounds.maxExclusive.z) * 128;

    switch (brush.axis) {
    case core::Dir6::PosX:
    case core::Dir6::NegX:
        return std::max(std::abs(cellYQ8 - centerYQ8), std::abs(cellZQ8 - centerZQ8)) <= radiusQ8;
    case core::Dir6::PosY:
    case core::Dir6::NegY:
        return std::max(std::abs(cellXQ8 - centerXQ8), std::abs(cellZQ8 - centerZQ8)) <= radiusQ8;
    case core::Dir6::PosZ:
    case core::Dir6::NegZ:
        return std::max(std::abs(cellXQ8 - centerXQ8), std::abs(cellYQ8 - centerYQ8)) <= radiusQ8;
    }
    return false;
}

inline bool brushContainsRampCell(const Brush& brush, const core::CellAabb& bounds, const core::Cell3i& cell) {
    if (!bounds.contains(cell)) {
        return false;
    }

    const std::int32_t height = bounds.maxExclusive.y - bounds.minInclusive.y;
    if (height <= 0) {
        return false;
    }

    auto riseForStep = [height](std::int32_t step, std::int32_t runLength) -> std::int32_t {
        if (runLength <= 0) {
            return 0;
        }
        const std::int32_t numerator = (step + 1) * height;
        return (numerator + runLength - 1) / runLength;
    };

    switch (brush.axis) {
    case core::Dir6::PosX:
    case core::Dir6::NegX: {
        const std::int32_t runLength = bounds.maxExclusive.x - bounds.minInclusive.x;
        if (runLength <= 0) {
            return false;
        }
        const std::int32_t step = brush.axis == core::Dir6::PosX
            ? cell.x - bounds.minInclusive.x
            : (bounds.maxExclusive.x - 1) - cell.x;
        const std::int32_t rise = std::clamp(riseForStep(step, runLength), 0, height);
        return cell.y < (bounds.minInclusive.y + rise);
    }
    case core::Dir6::PosZ:
    case core::Dir6::NegZ: {
        const std::int32_t runLength = bounds.maxExclusive.z - bounds.minInclusive.z;
        if (runLength <= 0) {
            return false;
        }
        const std::int32_t step = brush.axis == core::Dir6::PosZ
            ? cell.z - bounds.minInclusive.z
            : (bounds.maxExclusive.z - 1) - cell.z;
        const std::int32_t rise = std::clamp(riseForStep(step, runLength), 0, height);
        return cell.y < (bounds.minInclusive.y + rise);
    }
    case core::Dir6::PosY:
    case core::Dir6::NegY:
        return true;
    }
    return false;
}

inline bool brushContainsCell(const Brush& brush, const core::CellAabb& bounds, const core::Cell3i& cell) {
    switch (brush.kind) {
    case BrushKind::Box:
        return bounds.contains(cell);
    case BrushKind::PrismPipe:
        return brushContainsPrismPipeCell(brush, bounds, cell);
    case BrushKind::Ramp:
        return brushContainsRampCell(brush, bounds, cell);
    }
    return false;
}

inline bool affectMaskAllowsCell(const CsgCell& current, std::uint16_t affectMask) {
    if (affectMask == kCsgAffectAll) {
        return true;
    }
    const bool isSolid = current.voxel.type != VoxelType::Empty;
    if (isSolid) {
        return (affectMask & kCsgAffectSolid) != 0u;
    }
    return (affectMask & kCsgAffectEmpty) != 0u;
}

} // namespace detail

inline core::CellAabb applyCsgCommand(CsgVolume& volume, const CsgCommand& command) {
    core::CellAabb touched{};
    if (!volume.isValid()) {
        return touched;
    }

    const core::CellAabb bounds = core::intersectAabb(volume.worldBounds(), brushBounds(command.brush));
    if (!bounds.valid || bounds.empty()) {
        return touched;
    }

    for (std::int32_t y = bounds.minInclusive.y; y < bounds.maxExclusive.y; ++y) {
        for (std::int32_t z = bounds.minInclusive.z; z < bounds.maxExclusive.z; ++z) {
            for (std::int32_t x = bounds.minInclusive.x; x < bounds.maxExclusive.x; ++x) {
                const core::Cell3i worldCell{x, y, z};
                if (!detail::brushContainsCell(command.brush, bounds, worldCell)) {
                    continue;
                }

                CsgCell current = volume.cellAtWorld(worldCell);
                if (!detail::affectMaskAllowsCell(current, command.affectMask)) {
                    continue;
                }

                bool changed = false;
                switch (command.op) {
                case CsgOp::AddSolid:
                    if (current.voxel.type != VoxelType::Solid) {
                        current.voxel.type = VoxelType::Solid;
                        changed = true;
                    }
                    if (current.materialId != command.materialId) {
                        current.materialId = command.materialId;
                        changed = true;
                    }
                    break;
                case CsgOp::SubtractSolid:
                    if (current.voxel.type != VoxelType::Empty) {
                        current.voxel.type = VoxelType::Empty;
                        changed = true;
                    }
                    if (current.materialId != 0) {
                        current.materialId = 0;
                        changed = true;
                    }
                    break;
                case CsgOp::PaintMaterial:
                    // Paint only affects already-solid cells to avoid material noise in empty space.
                    if (current.voxel.type != VoxelType::Empty && current.materialId != command.materialId) {
                        current.materialId = command.materialId;
                        changed = true;
                    }
                    break;
                }

                if (!changed) {
                    continue;
                }
                volume.setCellAtWorld(worldCell, current);
                touched.includeCell(worldCell);
            }
        }
    }

    return touched;
}

inline core::CellAabb applyCsgCommands(CsgVolume& volume, const std::vector<CsgCommand>& commands) {
    core::CellAabb touched{};
    for (const CsgCommand& command : commands) {
        touched.includeAabb(applyCsgCommand(volume, command));
    }
    return touched;
}

inline core::CellAabb copyVolumeSolidsToChunk(const CsgVolume& volume, Chunk& chunk) {
    core::CellAabb touched{};
    if (!volume.isValid()) {
        return touched;
    }

    const core::Cell3i chunkOrigin{
        chunk.chunkX() * Chunk::kSizeX,
        chunk.chunkY() * Chunk::kSizeY,
        chunk.chunkZ() * Chunk::kSizeZ
    };
    core::CellAabb chunkBounds{};
    chunkBounds.valid = true;
    chunkBounds.minInclusive = chunkOrigin;
    chunkBounds.maxExclusive = chunkOrigin + core::Cell3i{Chunk::kSizeX, Chunk::kSizeY, Chunk::kSizeZ};

    const core::CellAabb overlap = core::intersectAabb(volume.worldBounds(), chunkBounds);
    if (!overlap.valid || overlap.empty()) {
        return touched;
    }

    for (std::int32_t y = overlap.minInclusive.y; y < overlap.maxExclusive.y; ++y) {
        for (std::int32_t z = overlap.minInclusive.z; z < overlap.maxExclusive.z; ++z) {
            for (std::int32_t x = overlap.minInclusive.x; x < overlap.maxExclusive.x; ++x) {
                const core::Cell3i worldCell{x, y, z};
                const CsgCell source = volume.cellAtWorld(worldCell);
                const std::int32_t localX = worldCell.x - chunkOrigin.x;
                const std::int32_t localY = worldCell.y - chunkOrigin.y;
                const std::int32_t localZ = worldCell.z - chunkOrigin.z;

                const Voxel existing = chunk.voxelAt(localX, localY, localZ);
                if (existing.type == source.voxel.type) {
                    continue;
                }

                chunk.setVoxel(localX, localY, localZ, source.voxel);
                touched.includeCell(worldCell);
            }
        }
    }

    return touched;
}

} // namespace world
