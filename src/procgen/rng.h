#pragma once

#include <array>
#include <cstdint>

// Shared deterministic RNG for procedural generation. Same LCG constants the
// citybuilder app uses for its ambient effects, so extracting it here keeps
// every existing generator bit-exact per seed.
namespace odai::procgen {

struct Rng {
    std::uint32_t state;

    explicit Rng(std::uint32_t seed) : state(seed ? seed : 1u) {}

    std::uint32_t next() {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    }

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

// Avalanche integer hash (same mix as the voxel worldgen's hashCoords); use
// for position-keyed seeds where neighbouring inputs must decorrelate.
inline std::uint32_t hash2d(int x, int z, std::uint32_t salt = 0u) {
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

}  // namespace odai::procgen
