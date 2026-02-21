#pragma once

#include <cmath>

namespace voxelsprout::core {

constexpr float kPi = 3.14159265358979323846f;

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    Vec3(float px, float py, float pz) : x(px), y(py), z(pz) {}

    Vec3 operator+(const Vec3& rhs) const { return Vec3{x + rhs.x, y + rhs.y, z + rhs.z}; }
    Vec3 operator-(const Vec3& rhs) const { return Vec3{x - rhs.x, y - rhs.y, z - rhs.z}; }
    Vec3 operator*(float s) const { return Vec3{x * s, y * s, z * s}; }
    Vec3& operator+=(const Vec3& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }
};

inline float dot(const Vec3& a, const Vec3& b) {
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

inline float length(const Vec3& v) {
    return std::sqrt(dot(v, v));
}

inline Vec3 normalize(const Vec3& v) {
    const float len = length(v);
    if (len <= 0.0f) {
        return Vec3{};
    }
    return Vec3{v.x / len, v.y / len, v.z / len};
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3{
        (a.y * b.z) - (a.z * b.y),
        (a.z * b.x) - (a.x * b.z),
        (a.x * b.y) - (a.y * b.x)};
}

inline float radians(float degrees) {
    return degrees * (kPi / 180.0f);
}

} // namespace voxelsprout::core
