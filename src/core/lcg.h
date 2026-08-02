#pragma once

#include <cstdint>

// Core Lcg subsystem
// Responsible for: the one deterministic 32-bit LCG the whole project uses.
// Should NOT do: seeding policy, distributions, or anything non-reproducible.
//
// Deliberately not std::mt19937 or std::minstd_rand: every generated world,
// .smap file, citybuilder layout, and inventory roll in this tree depends on
// these exact constants (Numerical Recipes) and on consuming the raw state.
// Changing either would silently reroll all of it.
namespace odai::core {

// Advances state in place and returns the new raw state. This is the form the
// existing generators consume, so it is the primitive rather than a wrapper.
inline constexpr std::uint32_t lcgNext(std::uint32_t& state) noexcept {
    state = state * 1664525u + 1013904223u;
    return state;
}

// The top 24 bits of the new state. An LCG's low bits have short periods, so
// prefer this over lcgNext for anything that then takes a modulus.
inline constexpr std::uint32_t lcgNext24(std::uint32_t& state) noexcept {
    return lcgNext(state) >> 8u;
}

// Object form for new code. Holds exactly the same state as the free functions
// above and produces the same sequence, so the two can be mixed.
class Lcg32 {
public:
    constexpr explicit Lcg32(std::uint32_t seed) noexcept : m_state(seed) {}

    constexpr std::uint32_t nextState() noexcept { return lcgNext(m_state); }
    constexpr std::uint32_t next24() noexcept { return lcgNext24(m_state); }

    [[nodiscard]] constexpr std::uint32_t state() const noexcept { return m_state; }
    constexpr void setState(std::uint32_t seed) noexcept { m_state = seed; }

private:
    std::uint32_t m_state;
};

}  // namespace odai::core
