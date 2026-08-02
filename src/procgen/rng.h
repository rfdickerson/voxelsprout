#pragma once

#include <array>
#include <cstdint>

#include "core/hash.h"
#include "core/lcg.h"

// Shared deterministic RNG for procedural generation. Same LCG constants the
// citybuilder app uses for its ambient effects, so extracting it here keeps
// every existing generator bit-exact per seed.
namespace odai::procgen {

struct Rng {
    std::uint32_t state;

    explicit Rng(std::uint32_t seed) : state(seed ? seed : 1u) {}

    std::uint32_t next() { return odai::core::lcgNext24(state); }

    // [0, 1) with 24 bits of resolution. Distinct from uniform(0, 1), which
    // quantises to 16 bits; both forms are in use in the tree.
    float unitFloat() { return static_cast<float>(next()) / static_cast<float>(1u << 24); }

    float uniform(float lo, float hi) {
        return lo + (hi - lo) * (static_cast<float>(next() & 0xffffu) / 65535.0f);
    }

    int range(int lo, int hiInclusive) {
        return lo + static_cast<int>(next() % static_cast<std::uint32_t>(hiInclusive - lo + 1));
    }

    bool chance(float p) { return uniform(0.0f, 1.0f) < p; }

    template <typename T, std::size_t N>
    const T& pick(const std::array<T, N>& pool) {
        return pool[next() % N];
    }
};

// Avalanche integer hash; use for position-keyed seeds where neighbouring
// inputs must decorrelate. The implementation now lives in core/hash.h, shared
// with the voxel worldgen that had its own identical copy.
inline std::uint32_t hash2d(int x, int z, std::uint32_t salt = 0u) {
    return odai::core::hashCoords2d(x, z, salt);
}

}  // namespace odai::procgen
