#pragma once

#include <cstdint>

namespace voxelsprout::core {

class Pcg32 {
public:
    explicit Pcg32(std::uint64_t seed = 0x853c49e6748fea9bULL) : m_state(seed) {}

    std::uint32_t nextU32() {
        const std::uint64_t old = m_state;
        m_state = old * 6364136223846793005ULL + 1442695040888963407ULL;
        const std::uint32_t xorshifted = static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
        const std::uint32_t rot = static_cast<std::uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    float nextFloat01() {
        return static_cast<float>(nextU32() >> 8u) * (1.0f / 16777216.0f);
    }

private:
    std::uint64_t m_state;
};

} // namespace voxelsprout::core
