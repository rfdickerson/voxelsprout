#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "math/Math.hpp"
#include "sim/NetworkGraph.hpp"

// Simulation NetworkProcedural subsystem
// Responsible for: deterministic helper utilities to build and classify transport topology.
// Should NOT do: global world edits, chunk meshing, or renderer-specific math.
namespace sim {

template <typename OccupancyFn>
std::uint8_t neighborMask6(const core::Cell3i& cell, OccupancyFn&& isOccupied) {
    std::uint8_t mask = 0;
    for (const core::Dir6 dir : core::kAllDir6) {
        if (isOccupied(core::neighborCell(cell, dir))) {
            mask = static_cast<std::uint8_t>(mask | core::dirBit(dir));
        }
    }
    return mask;
}

inline std::vector<core::Cell3i> rasterizeSpan(const EdgeSpan& span) {
    std::vector<core::Cell3i> cells;
    if (!isValidEdgeSpan(span)) {
        return cells;
    }

    cells.reserve(static_cast<std::size_t>(span.lengthVoxels));
    core::Cell3i cursor = span.start;
    const core::Cell3i step = core::dirToOffset(span.dir);
    for (std::uint16_t i = 0; i < span.lengthVoxels; ++i) {
        cells.push_back(cursor);
        cursor += step;
    }
    return cells;
}

inline std::uint32_t connectionCount(std::uint8_t neighborMask) {
    return static_cast<std::uint32_t>(std::popcount(static_cast<unsigned int>(neighborMask & 0x3Fu)));
}

enum class JoinPiece : std::uint8_t {
    Isolated = 0,
    EndCap = 1,
    Straight = 2,
    Elbow = 3,
    Tee = 4,
    Cross = 5
};

inline JoinPiece classifyJoinPiece(std::uint8_t neighborMask) {
    const std::uint8_t mask = static_cast<std::uint8_t>(neighborMask & 0x3Fu);
    const std::uint32_t degree = connectionCount(mask);
    if (degree == 0u) {
        return JoinPiece::Isolated;
    }
    if (degree == 1u) {
        return JoinPiece::EndCap;
    }
    if (degree == 2u) {
        core::Dir6 first = core::Dir6::PosX;
        core::Dir6 second = core::Dir6::PosX;
        std::uint32_t found = 0;
        for (const core::Dir6 dir : core::kAllDir6) {
            if ((mask & core::dirBit(dir)) == 0u) {
                continue;
            }
            if (found == 0u) {
                first = dir;
            } else {
                second = dir;
            }
            ++found;
        }
        return core::areOpposite(first, second) ? JoinPiece::Straight : JoinPiece::Elbow;
    }
    if (degree == 3u) {
        return JoinPiece::Tee;
    }
    return JoinPiece::Cross;
}

inline std::int32_t quantizeFixed(float value, int fractionalBits) {
    const double scale = std::ldexp(1.0, fractionalBits);
    const double scaled = static_cast<double>(value) * scale;
    const double clamped = std::clamp(
        scaled,
        static_cast<double>(std::numeric_limits<std::int32_t>::min()),
        static_cast<double>(std::numeric_limits<std::int32_t>::max())
    );
    return static_cast<std::int32_t>(std::llround(clamped));
}

inline float dequantizeFixed(std::int32_t value, int fractionalBits) {
    const double scale = std::ldexp(1.0, fractionalBits);
    return static_cast<float>(static_cast<double>(value) / scale);
}

inline std::int16_t quantizeAngleDegQ10(float degrees) {
    const double wrapped = std::remainder(static_cast<double>(degrees), 360.0);
    const double scaled = wrapped * (1024.0 / 180.0);
    const double clamped = std::clamp(
        scaled,
        static_cast<double>(std::numeric_limits<std::int16_t>::min()),
        static_cast<double>(std::numeric_limits<std::int16_t>::max())
    );
    return static_cast<std::int16_t>(std::llround(clamped));
}

inline float dequantizeAngleDegQ10(std::int16_t quantized) {
    return static_cast<float>(static_cast<double>(quantized) * (180.0 / 1024.0));
}

struct QuantizedTransform {
    std::int32_t txQ12 = 0;
    std::int32_t tyQ12 = 0;
    std::int32_t tzQ12 = 0;
    std::int16_t yawDegQ10 = 0;
    std::int16_t pitchDegQ10 = 0;
    std::int16_t rollDegQ10 = 0;
};

inline QuantizedTransform quantizeTransform(const math::Vector3& positionMeters, const math::Vector3& eulerDegrees) {
    QuantizedTransform output{};
    output.txQ12 = quantizeFixed(positionMeters.x, 12);
    output.tyQ12 = quantizeFixed(positionMeters.y, 12);
    output.tzQ12 = quantizeFixed(positionMeters.z, 12);
    output.yawDegQ10 = quantizeAngleDegQ10(eulerDegrees.y);
    output.pitchDegQ10 = quantizeAngleDegQ10(eulerDegrees.x);
    output.rollDegQ10 = quantizeAngleDegQ10(eulerDegrees.z);
    return output;
}

inline math::Vector3 dequantizePosition(const QuantizedTransform& transform) {
    return math::Vector3{
        dequantizeFixed(transform.txQ12, 12),
        dequantizeFixed(transform.tyQ12, 12),
        dequantizeFixed(transform.tzQ12, 12)
    };
}

inline math::Vector3 dequantizeEulerDegrees(const QuantizedTransform& transform) {
    return math::Vector3{
        dequantizeAngleDegQ10(transform.pitchDegQ10),
        dequantizeAngleDegQ10(transform.yawDegQ10),
        dequantizeAngleDegQ10(transform.rollDegQ10)
    };
}

} // namespace sim
