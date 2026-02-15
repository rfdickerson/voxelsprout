#pragma once

// Simulation Track subsystem
// Responsible for: representing simple rail track segments for rendering and placement.
// Should NOT do: train routing, signaling, or rendering API details.
namespace sim {

enum class TrackDirection {
    North,
    East,
    South,
    West
};

class Track {
public:
    Track() = default;
    Track(int xIn, int yIn, int zIn, TrackDirection directionIn);

    int x = 0;
    int y = 0;
    int z = 0;
    TrackDirection direction = TrackDirection::North;
};

inline Track::Track(int xIn, int yIn, int zIn, TrackDirection directionIn)
    : x(xIn), y(yIn), z(zIn), direction(directionIn) {}

} // namespace sim
