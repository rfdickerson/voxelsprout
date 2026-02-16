#pragma once

#include <cstdint>

// Simulation Belt Cargo subsystem
// Responsible for: deterministic per-item belt movement state used by simulation and rendering.
// Should NOT do: rendering, random effects, or gameplay authoring logic.
namespace sim {

struct BeltCargo {
    std::uint32_t itemId = 0;
    std::uint16_t typeId = 0;
    std::int32_t beltIndex = -1;
    std::uint32_t alongQ16 = 0;
    float prevWorldPos[3] = {0.0f, 0.0f, 0.0f};
    float currWorldPos[3] = {0.0f, 0.0f, 0.0f};
};

} // namespace sim
