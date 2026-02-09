#pragma once

#include <array>
#include <cmath>
#include <cstdint>

#include "math/Math.hpp"

// Core Grid subsystem
// Responsible for: defining deterministic integer-grid primitives shared by world and simulation code.
// Should NOT do: simulation state ownership, rendering behavior, or file serialization.
namespace core {

struct Cell3i {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;

    constexpr Cell3i() = default;
    constexpr Cell3i(std::int32_t xIn, std::int32_t yIn, std::int32_t zIn) : x(xIn), y(yIn), z(zIn) {}

    constexpr bool operator==(const Cell3i&) const = default;

    constexpr Cell3i operator+(const Cell3i& rhs) const {
        return Cell3i{x + rhs.x, y + rhs.y, z + rhs.z};
    }

    constexpr Cell3i operator-(const Cell3i& rhs) const {
        return Cell3i{x - rhs.x, y - rhs.y, z - rhs.z};
    }

    constexpr Cell3i operator*(std::int32_t scalar) const {
        return Cell3i{x * scalar, y * scalar, z * scalar};
    }

    constexpr Cell3i& operator+=(const Cell3i& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    constexpr Cell3i& operator-=(const Cell3i& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }
};

inline constexpr Cell3i operator*(std::int32_t scalar, const Cell3i& cell) {
    return cell * scalar;
}

struct CellAabb {
    Cell3i minInclusive{};
    Cell3i maxExclusive{};
    bool valid = false;

    constexpr bool empty() const {
        if (!valid) {
            return true;
        }
        return maxExclusive.x <= minInclusive.x ||
               maxExclusive.y <= minInclusive.y ||
               maxExclusive.z <= minInclusive.z;
    }

    constexpr bool contains(const Cell3i& cell) const {
        if (!valid || empty()) {
            return false;
        }
        return cell.x >= minInclusive.x && cell.x < maxExclusive.x &&
               cell.y >= minInclusive.y && cell.y < maxExclusive.y &&
               cell.z >= minInclusive.z && cell.z < maxExclusive.z;
    }

    constexpr void includeCell(const Cell3i& cell) {
        if (!valid) {
            minInclusive = cell;
            maxExclusive = cell + Cell3i{1, 1, 1};
            valid = true;
            return;
        }

        if (cell.x < minInclusive.x) minInclusive.x = cell.x;
        if (cell.y < minInclusive.y) minInclusive.y = cell.y;
        if (cell.z < minInclusive.z) minInclusive.z = cell.z;

        const Cell3i cellMax = cell + Cell3i{1, 1, 1};
        if (cellMax.x > maxExclusive.x) maxExclusive.x = cellMax.x;
        if (cellMax.y > maxExclusive.y) maxExclusive.y = cellMax.y;
        if (cellMax.z > maxExclusive.z) maxExclusive.z = cellMax.z;
    }

    constexpr void includeAabb(const CellAabb& other) {
        if (!other.valid || other.empty()) {
            return;
        }
        if (!valid || empty()) {
            *this = other;
            return;
        }

        if (other.minInclusive.x < minInclusive.x) minInclusive.x = other.minInclusive.x;
        if (other.minInclusive.y < minInclusive.y) minInclusive.y = other.minInclusive.y;
        if (other.minInclusive.z < minInclusive.z) minInclusive.z = other.minInclusive.z;
        if (other.maxExclusive.x > maxExclusive.x) maxExclusive.x = other.maxExclusive.x;
        if (other.maxExclusive.y > maxExclusive.y) maxExclusive.y = other.maxExclusive.y;
        if (other.maxExclusive.z > maxExclusive.z) maxExclusive.z = other.maxExclusive.z;
    }
};

inline constexpr CellAabb intersectAabb(const CellAabb& lhs, const CellAabb& rhs) {
    if (!lhs.valid || lhs.empty() || !rhs.valid || rhs.empty()) {
        return CellAabb{};
    }

    CellAabb result{};
    result.valid = true;
    result.minInclusive.x = lhs.minInclusive.x > rhs.minInclusive.x ? lhs.minInclusive.x : rhs.minInclusive.x;
    result.minInclusive.y = lhs.minInclusive.y > rhs.minInclusive.y ? lhs.minInclusive.y : rhs.minInclusive.y;
    result.minInclusive.z = lhs.minInclusive.z > rhs.minInclusive.z ? lhs.minInclusive.z : rhs.minInclusive.z;
    result.maxExclusive.x = lhs.maxExclusive.x < rhs.maxExclusive.x ? lhs.maxExclusive.x : rhs.maxExclusive.x;
    result.maxExclusive.y = lhs.maxExclusive.y < rhs.maxExclusive.y ? lhs.maxExclusive.y : rhs.maxExclusive.y;
    result.maxExclusive.z = lhs.maxExclusive.z < rhs.maxExclusive.z ? lhs.maxExclusive.z : rhs.maxExclusive.z;

    if (result.empty()) {
        return CellAabb{};
    }
    return result;
}

enum class Dir6 : std::uint8_t {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5
};

inline constexpr std::array<Dir6, 6> kAllDir6 = {
    Dir6::PosX,
    Dir6::NegX,
    Dir6::PosY,
    Dir6::NegY,
    Dir6::PosZ,
    Dir6::NegZ
};

inline constexpr std::uint8_t dirIndex(Dir6 dir) {
    return static_cast<std::uint8_t>(dir);
}

inline constexpr std::uint8_t dirBit(Dir6 dir) {
    return static_cast<std::uint8_t>(1u << dirIndex(dir));
}

inline constexpr Cell3i dirToOffset(Dir6 dir) {
    switch (dir) {
    case Dir6::PosX: return Cell3i{1, 0, 0};
    case Dir6::NegX: return Cell3i{-1, 0, 0};
    case Dir6::PosY: return Cell3i{0, 1, 0};
    case Dir6::NegY: return Cell3i{0, -1, 0};
    case Dir6::PosZ: return Cell3i{0, 0, 1};
    case Dir6::NegZ: return Cell3i{0, 0, -1};
    }
    return Cell3i{0, 0, 0};
}

inline constexpr Dir6 oppositeDir(Dir6 dir) {
    switch (dir) {
    case Dir6::PosX: return Dir6::NegX;
    case Dir6::NegX: return Dir6::PosX;
    case Dir6::PosY: return Dir6::NegY;
    case Dir6::NegY: return Dir6::PosY;
    case Dir6::PosZ: return Dir6::NegZ;
    case Dir6::NegZ: return Dir6::PosZ;
    }
    return Dir6::PosY;
}

inline constexpr bool areOpposite(Dir6 a, Dir6 b) {
    return oppositeDir(a) == b;
}

inline constexpr Cell3i neighborCell(const Cell3i& cell, Dir6 dir) {
    return cell + dirToOffset(dir);
}

inline constexpr math::Vector3 dirToUnitVector(Dir6 dir) {
    switch (dir) {
    case Dir6::PosX: return math::Vector3{1.0f, 0.0f, 0.0f};
    case Dir6::NegX: return math::Vector3{-1.0f, 0.0f, 0.0f};
    case Dir6::PosY: return math::Vector3{0.0f, 1.0f, 0.0f};
    case Dir6::NegY: return math::Vector3{0.0f, -1.0f, 0.0f};
    case Dir6::PosZ: return math::Vector3{0.0f, 0.0f, 1.0f};
    case Dir6::NegZ: return math::Vector3{0.0f, 0.0f, -1.0f};
    }
    return math::Vector3{0.0f, 1.0f, 0.0f};
}

struct AxisFrame {
    math::Vector3 forward{0.0f, 1.0f, 0.0f};
    math::Vector3 right{1.0f, 0.0f, 0.0f};
    math::Vector3 up{0.0f, 0.0f, 1.0f};
};

inline AxisFrame buildAxisFrame(Dir6 forwardDir) {
    AxisFrame frame{};
    frame.forward = dirToUnitVector(forwardDir);

    const math::Vector3 fallbackUp = std::abs(frame.forward.y) > 0.99f
        ? math::Vector3{0.0f, 0.0f, 1.0f}
        : math::Vector3{0.0f, 1.0f, 0.0f};
    frame.right = math::normalize(math::cross(frame.forward, fallbackUp));
    if (math::lengthSquared(frame.right) <= 0.000001f) {
        frame.right = math::Vector3{1.0f, 0.0f, 0.0f};
    }
    frame.up = math::normalize(math::cross(frame.right, frame.forward));
    return frame;
}

} // namespace core
