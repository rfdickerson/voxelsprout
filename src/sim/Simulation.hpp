#pragma once

#include <cstddef>
#include <vector>

#include "sim/Belt.hpp"

// Simulation subsystem
// Responsible for: providing a single high-level update entry point for factory simulation.
// Should NOT do: world storage ownership, rendering, or OS-level app concerns.
namespace sim {

class Simulation {
public:
    Simulation() = default;
    void initializeSingleBelt();

    void update(float dt);
    std::size_t beltCount() const;
    const std::vector<Belt>& belts() const;

private:
    std::vector<Belt> m_belts;
};

inline void Simulation::initializeSingleBelt() {
    m_belts.clear();

    // Minimal simulation seed: one belt segment above flat ground.
    m_belts.emplace_back(0, 1, 0, BeltDirection::East);
}

inline void Simulation::update(float dt) {
    (void)dt;
}

inline std::size_t Simulation::beltCount() const {
    return m_belts.size();
}

inline const std::vector<Belt>& Simulation::belts() const {
    return m_belts;
}

} // namespace sim
