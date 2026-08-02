#pragma once

#include <cstddef>
#include <cstdint>

#include "core/grid3.h"

// Core Hash subsystem
// Responsible for: integer mixing and spatial-key hashing shared by world,
// render, and sim code, so no subsystem re-invents an XOR-of-shifts that
// collides on axis-aligned coordinates.
// Should NOT do: cryptographic hashing, string hashing, or container definitions.
namespace odai::core {

// 64-bit avalanche finalizer (splitmix64's mix step). Every input bit reaches
// every output bit, which is what the XOR-of-shifts hashes this replaces did
// not do.
inline constexpr std::uint64_t mix64(std::uint64_t value) noexcept {
    value ^= value >> 30u;
    value *= 0xBF58476D1CE4E5B9ull;
    value ^= value >> 27u;
    value *= 0x94D049BB133111EBull;
    value ^= value >> 31u;
    return value;
}

// 32-bit avalanche finalizer (Murmur3 shape).
inline constexpr std::uint32_t mix32(std::uint32_t value) noexcept {
    value ^= value >> 16u;
    value *= 0x85EBCA6Bu;
    value ^= value >> 13u;
    value *= 0xC2B2AE35u;
    value ^= value >> 16u;
    return value;
}

// Hash for a 3D integer cell. Scatters each axis by a distinct large odd prime
// before mixing, so neighbouring chunk coordinates land in unrelated buckets.
// Replaces `hx ^ (hy << 1) ^ (hz << 2)`, which mapped whole axis-aligned rows
// of chunks onto a handful of buckets.
inline constexpr std::size_t hashCell3(
    std::int32_t x,
    std::int32_t y,
    std::int32_t z
) noexcept {
    const std::uint64_t hx = static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) * 73856093ull;
    const std::uint64_t hy = static_cast<std::uint64_t>(static_cast<std::uint32_t>(y)) * 19349663ull;
    const std::uint64_t hz = static_cast<std::uint64_t>(static_cast<std::uint32_t>(z)) * 83492791ull;
    return static_cast<std::size_t>(mix64(hx ^ (hy << 21u) ^ (hz << 42u)));
}

inline constexpr std::size_t hashCell3(const Cell3i& cell) noexcept {
    return hashCell3(cell.x, cell.y, cell.z);
}

// Functor form, for the Hash template argument of unordered_map/unordered_set.
struct Cell3Hash {
    [[nodiscard]] constexpr std::size_t operator()(const Cell3i& cell) const noexcept {
        return hashCell3(cell);
    }
};

// Position-keyed generator seed, 2D. Byte-identical to the implementation that
// was duplicated in procgen/rng.h and world/chunk_grid_worldgen.cc -- saved
// worlds reproduce through it, so the constants and their order are frozen.
// Note hashCoords2d(0, 0, 0) == 0; that is existing behavior, not an oversight
// to fix here.
inline constexpr std::uint32_t hashCoords2d(int x, int z, std::uint32_t salt = 0u) noexcept {
    std::uint32_t hash = static_cast<std::uint32_t>(x) * 0x9E3779B9u;
    hash ^= static_cast<std::uint32_t>(z) * 0x85EBCA6Bu;
    hash ^= salt * 0xC2B2AE35u;
    hash ^= hash >> 16u;
    hash *= 0x7FEB352Du;
    hash ^= hash >> 15u;
    hash *= 0x846CA68Bu;
    hash ^= hash >> 16u;
    return hash;
}

// The strategy-map generator's coordinate hash. A different algorithm from
// hashCoords2d, kept separate rather than merged because existing .smap files
// must keep reproducing. Byte-identical to the copies that were in
// game/game_sim.cc and tools/strategy_map_gen_main.cc.
inline constexpr std::uint32_t hashCoordsSeeded(
    std::int32_t x,
    std::int32_t y,
    std::uint32_t seed
) noexcept {
    std::uint32_t h = seed + 0x9E3779B9u;
    h ^= static_cast<std::uint32_t>(x) * 0x85EBCA77u;
    h = (h ^ (h >> 15)) * 0xC2B2AE3Du;
    h ^= static_cast<std::uint32_t>(y) * 0x27D4EB2Fu;
    h = (h ^ (h >> 13)) * 0x165667B1u;
    return h ^ (h >> 16);
}

}  // namespace odai::core
