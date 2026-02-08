#pragma once

#include <cstddef>
#include <vector>

#include "sim/Belt.hpp"
#include "sim/Pipe.hpp"

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
    std::vector<Pipe>& pipes();
    std::size_t pipeCount() const;
    const std::vector<Pipe>& pipes() const;

private:
    std::vector<Belt> m_belts;
    std::vector<Pipe> m_pipes;
};

inline void Simulation::initializeSingleBelt() {
    m_belts.clear();
    m_pipes.clear();

    // Minimal simulation seed: one belt segment above flat ground.
    m_belts.emplace_back(0, 1, 0, BeltDirection::East);

    // Pipe toy seed used by the dedicated pipe render pass.
    m_pipes.emplace_back(2, 1, 2, math::Vector3{1.0f, 0.0f, 0.0f}, 2.0f, 0.45f, math::Vector3{0.74f, 0.70f, 0.62f});
    m_pipes.emplace_back(2, 1, 3, math::Vector3{0.0f, 1.0f, 0.0f}, 1.0f, 0.45f, math::Vector3{0.70f, 0.76f, 0.68f});
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

inline std::size_t Simulation::pipeCount() const {
    return m_pipes.size();
}

inline std::vector<Pipe>& Simulation::pipes() {
    return m_pipes;
}

inline const std::vector<Pipe>& Simulation::pipes() const {
    return m_pipes;
}

} // namespace sim
