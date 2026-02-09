#pragma once

#include <cstddef>
#include <vector>

#include "sim/Belt.hpp"
#include "sim/Pipe.hpp"
#include "sim/Track.hpp"

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
    std::vector<Belt>& belts();
    const std::vector<Belt>& belts() const;
    std::vector<Pipe>& pipes();
    std::size_t pipeCount() const;
    const std::vector<Pipe>& pipes() const;
    std::vector<Track>& tracks();
    std::size_t trackCount() const;
    const std::vector<Track>& tracks() const;

private:
    std::vector<Belt> m_belts;
    std::vector<Pipe> m_pipes;
    std::vector<Track> m_tracks;
};

inline void Simulation::initializeSingleBelt() {
    m_belts.clear();
    m_pipes.clear();
    m_tracks.clear();

    // Minimal simulation seed: one belt segment above flat ground.
    m_belts.emplace_back(0, 1, 0, BeltDirection::East);

    // Pipe toy seed used by the dedicated pipe render pass.
    m_pipes.emplace_back(2, 1, 2, math::Vector3{1.0f, 0.0f, 0.0f}, 1.0f, 0.45f, math::Vector3{0.95f, 0.95f, 0.95f});
    m_pipes.emplace_back(3, 1, 2, math::Vector3{1.0f, 0.0f, 0.0f}, 1.0f, 0.45f, math::Vector3{0.95f, 0.95f, 0.95f});

    // Track toy seed for primitive rail rendering.
    m_tracks.emplace_back(0, 1, 2, TrackDirection::East);
    m_tracks.emplace_back(1, 1, 2, TrackDirection::East);
}

inline void Simulation::update(float dt) {
    (void)dt;
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

} // namespace sim
