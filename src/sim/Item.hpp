#pragma once

// Simulation Item subsystem
// Responsible for: defining a minimal item data placeholder for factory entities.
// Should NOT do: inventory rules, transport logic, or serialization yet.
namespace sim {

struct Item {
    int id = 0;
};

} // namespace sim
