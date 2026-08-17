#include "import/cell_residency_planner.h"

#include <algorithm>
#include <cmath>

namespace odai::importer {

namespace {

// Fallout's world is Z-up: the horizontal plane is X/Y, and the cell grid's
// second axis is Y. The field is named `z` throughout this planner because the
// ESM calls it the Z grid coordinate (XCLC), but the world component it comes
// from is index 1, not 2. Getting this wrong ranks cells along the vertical
// axis and is invisible until nothing streams in.
constexpr int kWorldGridAxis = 1;

float cellCenterOffset(std::int32_t cellIndex, float cellSize) {
    return (static_cast<float>(cellIndex) + 0.5f) * cellSize;
}

}  // namespace

CellCoord CellResidencyPlanner::cellAt(float worldX, float worldZ) const {
    const float size = (m_config.cellSize > 0.0f) ? m_config.cellSize : 1.0f;
    return CellCoord{
        static_cast<std::int32_t>(std::floor(worldX / size)),
        static_cast<std::int32_t>(std::floor(worldZ / size))};
}

void CellResidencyPlanner::setPinnedCells(const std::vector<CellCoord>& cells) {
    m_pinnedCells.clear();
    m_pinnedCells.reserve(cells.size());
    for (const CellCoord& cell : cells) {
        m_pinnedCells.insert(cell);
    }
}

bool CellResidencyPlanner::isPinned(const CellCoord& cell) const {
    return m_pinnedCells.count(cell) != 0u;
}

std::int32_t CellResidencyPlanner::chebyshevDistance(const CellCoord& a, const CellCoord& b) const {
    return std::max(std::abs(a.x - b.x), std::abs(a.z - b.z));
}

void CellResidencyPlanner::update(const float position[3], const float velocity[3]) {
    m_toLoad.clear();
    m_toEvict.clear();

    const float size = (m_config.cellSize > 0.0f) ? m_config.cellSize : 1.0f;
    m_centerCell = cellAt(position[0], position[kWorldGridAxis]);

    // Aim ranking at where the player will be, not where they are, clamped so a
    // bad velocity sample cannot throw the budget at cells never reached.
    const float maxLead = m_config.maxLeadCells * size;
    float leadX = velocity[0] * m_config.leadTimeSeconds;
    float leadZ = velocity[kWorldGridAxis] * m_config.leadTimeSeconds;
    const float leadLength = std::sqrt((leadX * leadX) + (leadZ * leadZ));
    if (leadLength > maxLead && leadLength > 0.0f) {
        const float scale = maxLead / leadLength;
        leadX *= scale;
        leadZ *= scale;
    }
    const float aimX = position[0] + leadX;
    const float aimZ = position[kWorldGridAxis] + leadZ;

    // Evict first, so cells leaving the far radius free their budget in the same
    // update that new ones are proposed.
    for (const auto& [cell, state] : m_cells) {
        if (state != CellState::Resident) {
            continue;  // in-flight loads cannot be cancelled; Unavailable is permanent
        }
        if (isPinned(cell)) {
            continue;
        }
        if (chebyshevDistance(cell, m_centerCell) > m_config.unloadRadius) {
            m_toEvict.push_back(cell);
        }
    }
    std::sort(
        m_toEvict.begin(), m_toEvict.end(),
        [&](const CellCoord& a, const CellCoord& b) {
            // Farthest first: if the caller honours only part of the list, the
            // most useless memory is the memory it reclaims.
            const std::int32_t da = chebyshevDistance(a, m_centerCell);
            const std::int32_t db = chebyshevDistance(b, m_centerCell);
            if (da != db) {
                return da > db;
            }
            return (a.x != b.x) ? (a.x < b.x) : (a.z < b.z);  // stable tie-break
        });

    std::size_t loadsInFlight = 0;
    for (const auto& [cell, state] : m_cells) {
        if (state == CellState::Loading) {
            ++loadsInFlight;
        }
    }
    if (loadsInFlight >= m_config.maxLoadsInFlight) {
        return;  // budget already spent; nothing new this update
    }

    // Candidates: everything inside the load radius the planner is not already
    // tracking. Ranked by squared distance from the AIM point, so cells ahead of
    // the player win, with distance from the player as a tie-break so the ring
    // still fills in sensibly when standing still.
    struct Candidate {
        CellCoord cell;
        float aimDistanceSq;
        bool pinned = false;
    };
    std::vector<Candidate> candidates;
    const std::int32_t radius = std::max<std::int32_t>(0, m_config.loadRadius);
    candidates.reserve(
        m_pinnedCells.size() + static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));
    for (const CellCoord& cell : m_pinnedCells) {
        if (m_cells.count(cell) == 0) {
            candidates.push_back(Candidate{cell, 0.0f, true});
        }
    }
    for (std::int32_t dz = -radius; dz <= radius; ++dz) {
        for (std::int32_t dx = -radius; dx <= radius; ++dx) {
            const CellCoord cell{m_centerCell.x + dx, m_centerCell.z + dz};
            if (m_cells.count(cell) != 0 || isPinned(cell)) {
                continue;  // already loading, resident, or known missing
            }
            const float centerX = cellCenterOffset(cell.x, size);
            const float centerZ = cellCenterOffset(cell.z, size);
            const float deltaX = centerX - aimX;
            const float deltaZ = centerZ - aimZ;
            candidates.push_back(Candidate{cell, (deltaX * deltaX) + (deltaZ * deltaZ), false});
        }
    }

    std::sort(
        candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            if (a.pinned != b.pinned) {
                return a.pinned;
            }
            if (a.aimDistanceSq != b.aimDistanceSq) {
                return a.aimDistanceSq < b.aimDistanceSq;
            }
            // Deterministic ordering for exact ties, so behaviour is reproducible.
            return (a.cell.x != b.cell.x) ? (a.cell.x < b.cell.x) : (a.cell.z < b.cell.z);
        });

    const std::size_t budget = m_config.maxLoadsInFlight - loadsInFlight;
    const std::size_t take = std::min(budget, candidates.size());
    m_toLoad.reserve(take);
    for (std::size_t i = 0; i < take; ++i) {
        m_toLoad.push_back(candidates[i].cell);
    }
}

void CellResidencyPlanner::markLoadStarted(const CellCoord& cell) {
    const auto existing = m_cells.find(cell);
    if (existing != m_cells.end()) {
        return;  // already tracked; do not double-count the budget
    }
    m_cells.emplace(cell, CellState::Loading);
    ++m_stats.loadsStarted;
}

bool CellResidencyPlanner::markLoadFinished(const CellCoord& cell) {
    const auto existing = m_cells.find(cell);
    if (existing == m_cells.end() || existing->second != CellState::Loading) {
        return false;
    }
    // The player may have walked out of range while this was in flight. Uploading
    // it would burn GPU memory on geometry that is about to be evicted anyway.
    if (!isPinned(cell) && chebyshevDistance(cell, m_centerCell) > m_config.unloadRadius) {
        m_cells.erase(existing);
        ++m_stats.wastedLoads;
        return false;
    }
    existing->second = CellState::Resident;
    ++m_stats.loadsCompleted;
    return true;
}

void CellResidencyPlanner::markUnavailable(const CellCoord& cell) {
    const auto existing = m_cells.find(cell);
    if (existing != m_cells.end() && existing->second == CellState::Unavailable) {
        return;
    }
    m_cells[cell] = CellState::Unavailable;
    ++m_stats.unavailableCells;
}

void CellResidencyPlanner::markEvicted(const CellCoord& cell) {
    const auto existing = m_cells.find(cell);
    if (existing == m_cells.end() || existing->second != CellState::Resident) {
        return;
    }
    m_cells.erase(existing);
    ++m_stats.evictions;
}

bool CellResidencyPlanner::isResident(const CellCoord& cell) const {
    const auto existing = m_cells.find(cell);
    return existing != m_cells.end() && existing->second == CellState::Resident;
}

CellResidencyStats CellResidencyPlanner::stats() const {
    CellResidencyStats snapshot = m_stats;
    snapshot.residentCount = 0;
    snapshot.loadingCount = 0;
    for (const auto& [cell, state] : m_cells) {
        (void)cell;
        if (state == CellState::Resident) {
            ++snapshot.residentCount;
        } else if (state == CellState::Loading) {
            ++snapshot.loadingCount;
        }
    }
    return snapshot;
}

void CellResidencyPlanner::reset() {
    m_cells.clear();
    m_pinnedCells.clear();
    m_toLoad.clear();
    m_toEvict.clear();
    m_stats = CellResidencyStats{};
}

}  // namespace odai::importer
