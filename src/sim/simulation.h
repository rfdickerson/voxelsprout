#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "core/grid3.h"
#include "sim/belt.h"
#include "sim/belt_cargo.h"
#include "sim/pipe.h"
#include "sim/track.h"

// Simulation subsystem
// Responsible for: providing a single high-level update entry point for factory simulation.
// Should NOT do: world storage ownership, rendering, or OS-level app concerns.
namespace odai::sim {

class Simulation {
public:
    Simulation() = default;
    void initializeSingleBelt();

    void update(float dt);
    std::size_t beltCount() const;
    std::vector<Belt>& belts();
    const std::vector<Belt>& belts() const;
    // Preferred over `belts().emplace_back(...)` / `belts().erase(...)`: these two
    // methods bump `m_beltTopologyVersion` so `update()` can detect layout changes in
    // O(1) instead of re-hashing every belt every tick. The raw `belts()` reference is
    // still exposed for read-only iteration (rendering, hit-testing) and legacy callers;
    // any *other* code path that adds/removes/moves belts through it will silently skip
    // topology invalidation.
    void addBelt(int x, int y, int z, BeltDirection direction);
    bool removeBeltAt(std::size_t index);
    std::vector<Pipe>& pipes();
    std::size_t pipeCount() const;
    const std::vector<Pipe>& pipes() const;
    std::vector<Track>& tracks();
    std::size_t trackCount() const;
    const std::vector<Track>& tracks() const;
    const std::vector<BeltCargo>& beltCargoes() const;

private:
    static constexpr std::uint32_t kSpanQ16 = 1u << 16u;
    static constexpr float kCargoSpeedVoxelsPerSecond = 1.45f;
    static constexpr float kCargoLiftAboveBelt = 0.68f;
    static constexpr std::uint32_t kSpawnIntervalTicks = 18u;
    static constexpr std::uint32_t kSpawnMinSpacingQ16 = (kSpanQ16 * 5u) / 16u;
    static constexpr std::size_t kMaxCargoPerBelt = 3u;

    struct BeltTopologyNode {
        odai::core::Cell3i cell{};
        BeltDirection direction = BeltDirection::North;
        std::int32_t nextBeltIndex = -1;
        std::uint32_t incomingCount = 0;
    };

    static odai::core::Cell3i beltDirectionOffset(BeltDirection direction);
    static std::uint64_t beltCellKey(const odai::core::Cell3i& cell);
    void rebuildBeltTopology();
    void seedBeltCargo();
    void trySpawnBeltCargo();
    void updateCargoWorldPosition(BeltCargo& cargo);
    void updateBeltCargo(float dt);
    bool hasCargoBlockingEntry(std::int32_t entryIndex) const;

    std::vector<Belt> m_belts;
    std::vector<Pipe> m_pipes;
    std::vector<Track> m_tracks;
    std::vector<BeltTopologyNode> m_beltTopology;
    std::unordered_map<std::uint64_t, std::size_t> m_beltCellToIndex;
    std::vector<std::int32_t> m_beltEntryIndices;
    // Maps a belt-topology index to its slot in m_beltEntryIndices (-1 if the belt is
    // not an entry point). Rebuilt only alongside m_beltTopology (rare); lets the
    // per-tick spawn-spacing check below run in O(1) instead of O(cargo count).
    std::vector<std::int32_t> m_beltIndexToEntrySlot;
    // Minimum alongQ16 seen this tick among cargo currently on each entry belt.
    // Repopulated once per tick as a side effect of the cargo-advance pass in
    // updateBeltCargo() (which already visits every cargo item), so trySpawnBeltCargo()
    // never needs its own O(cargo) scan.
    std::vector<std::uint32_t> m_entrySlotMinAlongQ16;
    std::vector<BeltCargo> m_beltCargoes;
    std::uint64_t m_beltTopologyVersion = 0;
    std::uint64_t m_beltTopologyBuiltVersion = 0;
    std::uint32_t m_nextCargoId = 1;
    std::uint32_t m_tickCounter = 0;
};

inline void Simulation::initializeSingleBelt() {
    m_belts.clear();
    m_pipes.clear();
    m_tracks.clear();

    // Minimal simulation seed: one belt segment above flat ground.
    m_belts.emplace_back(0, 1, 0, BeltDirection::East);

    // Pipe toy seed used by the dedicated pipe render pass.
    m_pipes.emplace_back(2, 1, 2, odai::math::Vector3{1.0f, 0.0f, 0.0f}, 1.0f, 0.45f, odai::math::Vector3{0.95f, 0.95f, 0.95f});
    m_pipes.emplace_back(3, 1, 2, odai::math::Vector3{1.0f, 0.0f, 0.0f}, 1.0f, 0.45f, odai::math::Vector3{0.95f, 0.95f, 0.95f});

    // Track toy seed for primitive rail rendering.
    m_tracks.emplace_back(0, 1, 2, TrackDirection::East);
    m_tracks.emplace_back(1, 1, 2, TrackDirection::East);

    m_nextCargoId = 1;
    m_tickCounter = 0;
    ++m_beltTopologyVersion;
    rebuildBeltTopology();
    m_beltTopologyBuiltVersion = m_beltTopologyVersion;
    seedBeltCargo();
}

inline void Simulation::update(float dt) {
    // O(1) dirty check: m_beltTopologyVersion is only bumped by addBelt()/removeBeltAt(),
    // so this replaces what used to be an unconditional O(belts.size()) FNV hash of every
    // belt, every tick, purely to detect layout changes that in practice only happen on
    // discrete editor placement actions. See addBelt()/removeBeltAt() doc comment for the
    // caveat: mutating m_belts through the raw belts() reference bypasses this.
    if (m_beltTopologyBuiltVersion != m_beltTopologyVersion) {
        m_beltTopologyBuiltVersion = m_beltTopologyVersion;
        rebuildBeltTopology();
        seedBeltCargo();
    }
    updateBeltCargo(std::max(dt, 0.0f));
}

inline void Simulation::addBelt(int x, int y, int z, BeltDirection direction) {
    m_belts.emplace_back(x, y, z, direction);
    ++m_beltTopologyVersion;
}

inline bool Simulation::removeBeltAt(std::size_t index) {
    if (index >= m_belts.size()) {
        return false;
    }
    m_belts.erase(m_belts.begin() + static_cast<std::ptrdiff_t>(index));
    ++m_beltTopologyVersion;
    return true;
}

inline std::size_t Simulation::beltCount() const {
    return m_belts.size();
}

inline std::vector<Belt>& Simulation::belts() {
    return m_belts;
}

inline const std::vector<Belt>& Simulation::belts() const {
    return m_belts;
}

inline std::size_t Simulation::pipeCount() const {
    return m_pipes.size();
}

inline std::vector<Pipe>& Simulation::pipes() {
    return m_pipes;
}

inline const std::vector<Pipe>& Simulation::pipes() const {
    return m_pipes;
}

inline std::vector<Track>& Simulation::tracks() {
    return m_tracks;
}

inline std::size_t Simulation::trackCount() const {
    return m_tracks.size();
}

inline const std::vector<Track>& Simulation::tracks() const {
    return m_tracks;
}

inline const std::vector<BeltCargo>& Simulation::beltCargoes() const {
    return m_beltCargoes;
}

inline odai::core::Cell3i Simulation::beltDirectionOffset(BeltDirection direction) {
    switch (direction) {
    case BeltDirection::East:
        return odai::core::Cell3i{1, 0, 0};
    case BeltDirection::West:
        return odai::core::Cell3i{-1, 0, 0};
    case BeltDirection::South:
        return odai::core::Cell3i{0, 0, 1};
    case BeltDirection::North:
    default:
        return odai::core::Cell3i{0, 0, -1};
    }
}

inline std::uint64_t Simulation::beltCellKey(const odai::core::Cell3i& cell) {
    constexpr std::uint64_t kMask = (1ull << 21u) - 1ull;
    const std::uint64_t x = static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell.x) & kMask);
    const std::uint64_t y = static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell.y) & kMask);
    const std::uint64_t z = static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell.z) & kMask);
    return x | (y << 21u) | (z << 42u);
}

inline void Simulation::rebuildBeltTopology() {
    m_beltTopology.assign(m_belts.size(), {});
    m_beltCellToIndex.clear();
    m_beltEntryIndices.clear();
    m_beltIndexToEntrySlot.assign(m_belts.size(), -1);

    if (m_belts.empty()) {
        m_entrySlotMinAlongQ16.clear();
        return;
    }

    m_beltCellToIndex.reserve(m_belts.size() * 2u);
    for (std::size_t beltIndex = 0; beltIndex < m_belts.size(); ++beltIndex) {
        const Belt& belt = m_belts[beltIndex];
        BeltTopologyNode& node = m_beltTopology[beltIndex];
        node.cell = odai::core::Cell3i{belt.x, belt.y, belt.z};
        node.direction = belt.direction;
        node.nextBeltIndex = -1;
        node.incomingCount = 0;
        m_beltCellToIndex[beltCellKey(node.cell)] = beltIndex;
    }

    for (std::size_t beltIndex = 0; beltIndex < m_belts.size(); ++beltIndex) {
        BeltTopologyNode& node = m_beltTopology[beltIndex];
        const odai::core::Cell3i nextCell = node.cell + beltDirectionOffset(node.direction);
        const auto found = m_beltCellToIndex.find(beltCellKey(nextCell));
        if (found != m_beltCellToIndex.end()) {
            node.nextBeltIndex = static_cast<std::int32_t>(found->second);
            m_beltTopology[found->second].incomingCount += 1u;
        }
    }

    for (std::size_t beltIndex = 0; beltIndex < m_beltTopology.size(); ++beltIndex) {
        if (m_beltTopology[beltIndex].incomingCount == 0u) {
            m_beltEntryIndices.push_back(static_cast<std::int32_t>(beltIndex));
        }
    }
    if (m_beltEntryIndices.empty()) {
        m_beltEntryIndices.push_back(0);
    }

    for (std::size_t slot = 0; slot < m_beltEntryIndices.size(); ++slot) {
        const std::int32_t entryIndex = m_beltEntryIndices[slot];
        if (entryIndex >= 0 && static_cast<std::size_t>(entryIndex) < m_beltIndexToEntrySlot.size()) {
            m_beltIndexToEntrySlot[static_cast<std::size_t>(entryIndex)] = static_cast<std::int32_t>(slot);
        }
    }
    m_entrySlotMinAlongQ16.assign(m_beltEntryIndices.size(), kSpanQ16);
}

inline void Simulation::updateCargoWorldPosition(BeltCargo& cargo) {
    if (cargo.beltIndex < 0 || static_cast<std::size_t>(cargo.beltIndex) >= m_beltTopology.size()) {
        return;
    }
    const BeltTopologyNode& node = m_beltTopology[static_cast<std::size_t>(cargo.beltIndex)];
    const odai::core::Cell3i axis = beltDirectionOffset(node.direction);
    const float along01 = static_cast<float>(cargo.alongQ16) / static_cast<float>(kSpanQ16);
    const float alongCentered = along01 - 0.5f;
    cargo.currWorldPos[0] = static_cast<float>(node.cell.x) + 0.5f + (static_cast<float>(axis.x) * alongCentered);
    cargo.currWorldPos[1] = static_cast<float>(node.cell.y) + kCargoLiftAboveBelt;
    cargo.currWorldPos[2] = static_cast<float>(node.cell.z) + 0.5f + (static_cast<float>(axis.z) * alongCentered);
}

inline void Simulation::seedBeltCargo() {
    m_beltCargoes.clear();
    if (m_beltEntryIndices.empty()) {
        return;
    }

    const std::size_t maxSeedCount = std::min(m_beltEntryIndices.size(), static_cast<std::size_t>(24));
    m_beltCargoes.reserve(maxSeedCount);
    for (std::size_t i = 0; i < maxSeedCount; ++i) {
        const std::int32_t entryIndex = m_beltEntryIndices[i];
        if (entryIndex < 0 || static_cast<std::size_t>(entryIndex) >= m_beltTopology.size()) {
            continue;
        }
        BeltCargo cargo{};
        cargo.itemId = m_nextCargoId++;
        cargo.typeId = static_cast<std::uint16_t>(cargo.itemId % 5u);
        cargo.beltIndex = entryIndex;
        cargo.alongQ16 = 0u;
        updateCargoWorldPosition(cargo);
        cargo.prevWorldPos[0] = cargo.currWorldPos[0];
        cargo.prevWorldPos[1] = cargo.currWorldPos[1];
        cargo.prevWorldPos[2] = cargo.currWorldPos[2];
        m_beltCargoes.push_back(cargo);
    }
}

inline void Simulation::trySpawnBeltCargo() {
    if (m_beltEntryIndices.empty() || m_beltTopology.empty()) {
        return;
    }

    if ((m_tickCounter % kSpawnIntervalTicks) != 0u) {
        return;
    }

    const std::size_t maxCargoCount = std::max<std::size_t>(1u, m_belts.size() * kMaxCargoPerBelt);
    if (m_beltCargoes.size() >= maxCargoCount) {
        return;
    }

    const std::size_t entryCursor = static_cast<std::size_t>((m_tickCounter / kSpawnIntervalTicks) % m_beltEntryIndices.size());
    const std::int32_t entryIndex = m_beltEntryIndices[entryCursor];
    if (entryIndex < 0 || static_cast<std::size_t>(entryIndex) >= m_beltTopology.size()) {
        return;
    }

    if (hasCargoBlockingEntry(entryIndex)) {
        return;
    }

    BeltCargo cargo{};
    cargo.itemId = m_nextCargoId++;
    cargo.typeId = static_cast<std::uint16_t>(cargo.itemId % 5u);
    cargo.beltIndex = entryIndex;
    cargo.alongQ16 = 0u;
    updateCargoWorldPosition(cargo);
    cargo.prevWorldPos[0] = cargo.currWorldPos[0];
    cargo.prevWorldPos[1] = cargo.currWorldPos[1];
    cargo.prevWorldPos[2] = cargo.currWorldPos[2];
    m_beltCargoes.push_back(cargo);
}

inline bool Simulation::hasCargoBlockingEntry(std::int32_t entryIndex) const {
    // O(1) lookup instead of scanning every live cargo item: m_entrySlotMinAlongQ16 is
    // kept up to date as a side effect of the per-cargo pass in updateBeltCargo(), which
    // already visits every cargo item once per tick regardless. At thousands of cargo
    // and hundreds of belt entry points, the old linear scan over m_beltCargoes cost
    // O(total cargo) on every spawn-interval tick; this is O(1) here plus O(numEntries)
    // amortized in the pass that was happening anyway.
    if (entryIndex < 0 || static_cast<std::size_t>(entryIndex) >= m_beltIndexToEntrySlot.size()) {
        return false;
    }
    const std::int32_t slot = m_beltIndexToEntrySlot[static_cast<std::size_t>(entryIndex)];
    if (slot < 0 || static_cast<std::size_t>(slot) >= m_entrySlotMinAlongQ16.size()) {
        return false;
    }
    return m_entrySlotMinAlongQ16[static_cast<std::size_t>(slot)] < kSpawnMinSpacingQ16;
}

inline void Simulation::updateBeltCargo(float dt) {
    if (m_beltTopology.empty()) {
        m_beltCargoes.clear();
        return;
    }

    const float clampedDt = std::max(dt, 0.0f);
    const std::uint32_t stepQ16 = std::max(
        1u,
        static_cast<std::uint32_t>(std::lround(clampedDt * kCargoSpeedVoxelsPerSecond * static_cast<float>(kSpanQ16)))
    );

    for (BeltCargo& cargo : m_beltCargoes) {
        cargo.prevWorldPos[0] = cargo.currWorldPos[0];
        cargo.prevWorldPos[1] = cargo.currWorldPos[1];
        cargo.prevWorldPos[2] = cargo.currWorldPos[2];
    }

    std::fill(m_entrySlotMinAlongQ16.begin(), m_entrySlotMinAlongQ16.end(), kSpanQ16);

    std::erase_if(m_beltCargoes, [&](BeltCargo& cargo) {
        if (cargo.beltIndex < 0 || static_cast<std::size_t>(cargo.beltIndex) >= m_beltTopology.size()) {
            return true;
        }

        std::uint32_t alongQ16 = cargo.alongQ16 + stepQ16;
        std::int32_t beltIndex = cargo.beltIndex;
        std::size_t hopCount = 0;
        while (alongQ16 >= kSpanQ16) {
            alongQ16 -= kSpanQ16;
            const std::int32_t nextIndex = m_beltTopology[static_cast<std::size_t>(beltIndex)].nextBeltIndex;
            if (nextIndex < 0) {
                return true;
            }
            beltIndex = nextIndex;
            ++hopCount;
            if (hopCount > m_beltTopology.size()) {
                return true;
            }
        }

        cargo.beltIndex = beltIndex;
        cargo.alongQ16 = alongQ16;
        updateCargoWorldPosition(cargo);

        if (beltIndex >= 0 && static_cast<std::size_t>(beltIndex) < m_beltIndexToEntrySlot.size()) {
            const std::int32_t slot = m_beltIndexToEntrySlot[static_cast<std::size_t>(beltIndex)];
            if (slot >= 0 && static_cast<std::size_t>(slot) < m_entrySlotMinAlongQ16.size()) {
                std::uint32_t& slotMin = m_entrySlotMinAlongQ16[static_cast<std::size_t>(slot)];
                slotMin = std::min(slotMin, alongQ16);
            }
        }
        return false;
    });

    ++m_tickCounter;
    trySpawnBeltCargo();
}

} // namespace odai::sim
