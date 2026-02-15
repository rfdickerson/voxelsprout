#pragma once

#include "math/Math.hpp"

// Simulation Pipe subsystem
// Responsible for: representing simple straight pipe segments for rendering and routing.
// Should NOT do: fluid simulation, pressure calculations, or rendering API details.
namespace sim {

class Pipe {
public:
    Pipe() = default;
    Pipe(
        int xIn,
        int yIn,
        int zIn,
        math::Vector3 axisIn,
        float lengthIn,
        float radiusIn,
        math::Vector3 tintIn
    );

    int x = 0;
    int y = 0;
    int z = 0;
    math::Vector3 axis{0.0f, 1.0f, 0.0f};
    float length = 1.0f;
    float radius = 0.45f;
    math::Vector3 tint{0.95f, 0.95f, 0.95f};
};

inline Pipe::Pipe(
    int xIn,
    int yIn,
    int zIn,
    math::Vector3 axisIn,
    float lengthIn,
    float radiusIn,
    math::Vector3 tintIn
)
    : x(xIn),
      y(yIn),
      z(zIn),
      axis(math::normalize(axisIn)),
      length(lengthIn),
      radius(radiusIn),
      tint(tintIn) {}

} // namespace sim
