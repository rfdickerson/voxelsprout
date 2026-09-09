#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace odai::render {
// WTHR headings are clockwise degrees from Bethesda north (+Y, engine -Z).
// Direction range bounds a smooth deterministic gust; it is not extra speed.
inline std::array<float, 3> sampleAuthoredWeatherWind(
    float headingDegrees, float rangeDegrees, float normalizedSpeed, double seconds) {
    if (!std::isfinite(headingDegrees) || !std::isfinite(rangeDegrees) ||
        !std::isfinite(normalizedSpeed) || !std::isfinite(seconds)) return {0, -1, 0};
    const double gust = std::sin(seconds * 0.11) * 0.65 + std::sin(seconds * 0.037) * 0.35;
    const double angle = (headingDegrees + std::clamp(rangeDegrees, 0.0f, 180.0f) * gust) *
        (3.141592653589793 / 180.0);
    return {float(std::sin(angle)), float(-std::cos(angle)),
            std::clamp(normalizedSpeed, 0.0f, 1.0f)};
}
}
