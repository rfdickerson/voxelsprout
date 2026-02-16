#pragma once

// Simulation Belt subsystem
// Responsible for: representing a transport machine placeholder in the simulation.
// Should NOT do: move items, resolve collisions, or interact with rendering yet.
namespace sim {

enum class BeltDirection {
    North,
    East,
    South,
    West
};

class Belt {
public:
    Belt() = default;
    Belt(int xIn, int yIn, int zIn, BeltDirection directionIn);

    int x = 0;
    int y = 0;
    int z = 0;
    BeltDirection direction = BeltDirection::North;
};

inline Belt::Belt(int xIn, int yIn, int zIn, BeltDirection directionIn)
    : x(xIn), y(yIn), z(zIn), direction(directionIn) {}

} // namespace sim
