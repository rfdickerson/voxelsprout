#pragma once

// Decides which exterior cells should be resident, which to start loading next,
// and which to evict -- from the player's position and velocity alone.
//
// Deliberately Vulkan-free, IO-free and thread-free: it makes decisions and
// nothing else. That is what makes the interesting behaviour testable, and the
// interesting behaviour is exactly the kind that fails quietly:
//   * a boundary that thrashes, reloading a cell every time the player steps
//     back and forth across one line;
//   * prediction that loads behind the player instead of ahead;
//   * an unbounded in-flight set that turns a sprint into a stall.
//
// The caller owns actually loading and evicting; it reports outcomes back so the
// planner can track residency. Modelled on world::ChunkMeshScheduler's
// main-thread-only contract.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace odai::importer {

// Exterior cell grid coordinates, as the ESM stores them (XCLC).
struct CellCoord {
    std::int32_t x = 0;
    std::int32_t z = 0;

    bool operator==(const CellCoord& other) const { return x == other.x && z == other.z; }
};

struct CellCoordHash {
    std::size_t operator()(const CellCoord& coord) const {
        // Cheap the two 32-bit halves into one 64-bit key; cell coordinates are
        // small and dense, so this never collides in practice.
        const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(coord.x)) << 32) |
                                  static_cast<std::uint32_t>(coord.z);
        return std::hash<std::uint64_t>{}(key);
    }
};

struct CellResidencyConfig {
    // Chebyshev radius in cells, matching how the game's own grid works.
    // unloadRadius MUST exceed loadRadius: the gap between them is the
    // hysteresis band, and without it a player standing on a boundary loads and
    // evicts the same cell forever.
    std::int32_t loadRadius = 4;
    std::int32_t unloadRadius = 6;
    // World units per cell. Fallout exteriors are 4096.
    float cellSize = 4096.0f;
    // How far ahead of the player to aim the ranking, in seconds of travel.
    // Ranking by raw position loads the ring symmetrically and the cells the
    // player is about to enter arrive last.
    float leadTimeSeconds = 2.0f;
    // Ceiling on the predicted lead, in cells. Without it a fast vehicle or a
    // bad velocity sample aims the whole budget at cells the player never
    // reaches.
    float maxLeadCells = 3.0f;
    // Concurrent loads. The cap is what keeps a sprint from queueing the entire
    // ring at once and starving the cells actually about to be entered.
    std::size_t maxLoadsInFlight = 4;
    // Chunks handed to the renderer in a single frame. Uploading geometry is
    // still synchronous (a queue submit and wait per buffer), so several cells
    // landing together costs their sum in one frame -- measured at 48 ms when
    // four arrived at once, which is a visible hitch. Spreading them turns one
    // long frame into several short ones. Raise only once uploads are async.
    std::size_t maxChunkAppliesPerFrame = 1;
};

enum class CellState : std::uint8_t {
    Loading,
    Resident,
    // Asked for and found not to exist in the plugin. Remembered so it is never
    // proposed again -- the worldspace is not a full rectangle and its edges
    // would otherwise be re-requested every frame forever.
    Unavailable,
};

struct CellResidencyStats {
    std::uint64_t loadsStarted = 0;
    std::uint64_t loadsCompleted = 0;
    std::uint64_t evictions = 0;
    // Loads that finished after their cell had already left the unload radius.
    // Real wasted work: counted, never hidden. A straight-line walk should keep
    // this at or near zero; a number that climbs means leadTime or the in-flight
    // budget is mistuned.
    std::uint64_t wastedLoads = 0;
    std::uint64_t unavailableCells = 0;
    std::size_t residentCount = 0;
    std::size_t loadingCount = 0;
};

class CellResidencyPlanner {
public:
    void setConfig(const CellResidencyConfig& config) { m_config = config; }
    [[nodiscard]] const CellResidencyConfig& config() const { return m_config; }

    // Keeps these cells resident and schedules them ahead of the ordinary
    // camera-centred ring. Used by deterministic captures, whose camera can
    // advance faster in simulation time than the asynchronous streamer can
    // build a new cell. Pins are deliberately explicit rather than a capture
    // special case in update(), so the normal residency policy stays testable.
    void setPinnedCells(const std::vector<CellCoord>& cells);

    // Recomputes what should happen next. `position` and `velocity` are in world
    // units, in the same space the cell grid is defined in. Results are read
    // through cellsToLoad() and cellsToEvict().
    void update(const float position[3], const float velocity[3]);

    // Cells that should begin loading now, nearest-to-predicted-position first
    // and already capped by maxLoadsInFlight. Valid until the next update().
    [[nodiscard]] const std::vector<CellCoord>& cellsToLoad() const { return m_toLoad; }

    // Cells that should be evicted now, farthest first so the most useless
    // memory is reclaimed even if the caller only honours part of the list.
    [[nodiscard]] const std::vector<CellCoord>& cellsToEvict() const { return m_toEvict; }

    // Caller reports outcomes. markLoadStarted must be called for each cell the
    // caller actually begins loading, or the in-flight budget cannot be enforced.
    void markLoadStarted(const CellCoord& cell);
    // Returns false if the cell has since left the unload radius, meaning the
    // result is wasted and the caller should discard rather than upload it.
    bool markLoadFinished(const CellCoord& cell);
    void markUnavailable(const CellCoord& cell);
    void markEvicted(const CellCoord& cell);

    [[nodiscard]] bool isResident(const CellCoord& cell) const;
    [[nodiscard]] bool isTracked(const CellCoord& cell) const { return m_cells.count(cell) != 0; }
    [[nodiscard]] CellResidencyStats stats() const;

    // Forgets everything, including the Unavailable set. For a worldspace change
    // or a teleport into a different region.
    void reset();

    // The cell containing a world position.
    [[nodiscard]] CellCoord cellAt(float worldX, float worldZ) const;

private:
    [[nodiscard]] std::int32_t chebyshevDistance(const CellCoord& a, const CellCoord& b) const;
    [[nodiscard]] bool isPinned(const CellCoord& cell) const;

    CellResidencyConfig m_config{};
    std::unordered_map<CellCoord, CellState, CellCoordHash> m_cells;
    std::unordered_set<CellCoord, CellCoordHash> m_pinnedCells;
    std::vector<CellCoord> m_toLoad;
    std::vector<CellCoord> m_toEvict;
    CellCoord m_centerCell{};
    CellResidencyStats m_stats{};
};

}  // namespace odai::importer
