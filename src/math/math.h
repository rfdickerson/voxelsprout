#pragma once

#include <cmath>

namespace math {

constexpr float kPi = 3.14159265358979323846f;

inline float radians(float degreesValue) {
    return degreesValue * (kPi / 180.0f);
}

inline float degrees(float radiansValue) {
    return radiansValue * (180.0f / kPi);
}

struct Vector3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    constexpr Vector3() = default;
    constexpr Vector3(float xIn, float yIn, float zIn) : x(xIn), y(yIn), z(zIn) {}

    constexpr Vector3 operator+(const Vector3& rhs) const { return Vector3{x + rhs.x, y + rhs.y, z + rhs.z}; }
    constexpr Vector3 operator-(const Vector3& rhs) const { return Vector3{x - rhs.x, y - rhs.y, z - rhs.z}; }
    constexpr Vector3 operator-() const { return Vector3{-x, -y, -z}; }
    constexpr Vector3 operator*(float scalar) const { return Vector3{x * scalar, y * scalar, z * scalar}; }
    constexpr Vector3 operator/(float scalar) const { return Vector3{x / scalar, y / scalar, z / scalar}; }

    Vector3& operator+=(const Vector3& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    Vector3& operator-=(const Vector3& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    Vector3& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    Vector3& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }
};

inline constexpr Vector3 operator*(float scalar, const Vector3& v) {
    return v * scalar;
}

inline float dot(const Vector3& a, const Vector3& b) {
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

inline Vector3 cross(const Vector3& a, const Vector3& b) {
    return Vector3{
        (a.y * b.z) - (a.z * b.y),
        (a.z * b.x) - (a.x * b.z),
        (a.x * b.y) - (a.y * b.x)
    };
}

inline float lengthSquared(const Vector3& v) {
    return dot(v, v);
}

inline float length(const Vector3& v) {
    return std::sqrt(lengthSquared(v));
}

inline Vector3 normalize(const Vector3& v) {
    const float len = length(v);
    if (len <= 0.0f) {
        return Vector3{};
    }
    return v / len;
}

struct Vector4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;

    constexpr Vector4() = default;
    constexpr Vector4(float xIn, float yIn, float zIn, float wIn) : x(xIn), y(yIn), z(zIn), w(wIn) {}
    constexpr explicit Vector4(const Vector3& xyz, float wIn) : x(xyz.x), y(xyz.y), z(xyz.z), w(wIn) {}
};

struct Matrix4 {
    // Row-major storage.
    float m[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    static Matrix4 identity() {
        return Matrix4{};
    }

    static Matrix4 translation(const Vector3& t) {
        Matrix4 result = identity();
        result(0, 3) = t.x;
        result(1, 3) = t.y;
        result(2, 3) = t.z;
        return result;
    }

    static Matrix4 scale(const Vector3& s) {
        Matrix4 result = identity();
        result(0, 0) = s.x;
        result(1, 1) = s.y;
        result(2, 2) = s.z;
        return result;
    }

    static Matrix4 rotationX(float radians) {
        Matrix4 result = identity();
        const float c = std::cos(radians);
        const float s = std::sin(radians);
        result(1, 1) = c;
        result(1, 2) = -s;
        result(2, 1) = s;
        result(2, 2) = c;
        return result;
    }

    static Matrix4 rotationY(float radians) {
        Matrix4 result = identity();
        const float c = std::cos(radians);
        const float s = std::sin(radians);
        result(0, 0) = c;
        result(0, 2) = s;
        result(2, 0) = -s;
        result(2, 2) = c;
        return result;
    }

    static Matrix4 rotationZ(float radians) {
        Matrix4 result = identity();
        const float c = std::cos(radians);
        const float s = std::sin(radians);
        result(0, 0) = c;
        result(0, 1) = -s;
        result(1, 0) = s;
        result(1, 1) = c;
        return result;
    }

    // Right-handed perspective matrix with OpenGL-style clip space z in [-1, 1].
    static Matrix4 perspective(float fovYRadians, float aspectRatio, float nearPlane, float farPlane) {
        Matrix4 result{};
        for (int i = 0; i < 16; ++i) {
            result.m[i] = 0.0f;
        }

        const float f = 1.0f / std::tan(fovYRadians * 0.5f);
        result(0, 0) = f / aspectRatio;
        result(1, 1) = f;
        result(2, 2) = (farPlane + nearPlane) / (nearPlane - farPlane);
        result(2, 3) = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
        result(3, 2) = -1.0f;
        return result;
    }

    float& operator()(int row, int col) {
        return m[(row * 4) + col];
    }

    const float& operator()(int row, int col) const {
        return m[(row * 4) + col];
    }
};

// Vulkan perspective with depth range [0, 1], near -> 0, far -> 1.
inline Matrix4 perspectiveVulkan(float fovYRadians, float aspectRatio, float nearPlane, float farPlane) {
    Matrix4 result{};
    for (int i = 0; i < 16; ++i) {
        result.m[i] = 0.0f;
    }

    const float f = 1.0f / std::tan(fovYRadians * 0.5f);
    result(0, 0) = f / aspectRatio;
    result(1, 1) = -f;
    result(2, 2) = farPlane / (nearPlane - farPlane);
    result(2, 3) = (farPlane * nearPlane) / (nearPlane - farPlane);
    result(3, 2) = -1.0f;
    return result;
}

// Vulkan reverse-Z perspective with depth range [0, 1], near -> 1, far -> 0.
inline Matrix4 perspectiveVulkanReverseZ(float fovYRadians, float aspectRatio, float nearPlane, float farPlane) {
    Matrix4 result{};
    for (int i = 0; i < 16; ++i) {
        result.m[i] = 0.0f;
    }

    const float f = 1.0f / std::tan(fovYRadians * 0.5f);
    result(0, 0) = f / aspectRatio;
    result(1, 1) = -f;
    result(2, 2) = nearPlane / (farPlane - nearPlane);
    result(2, 3) = (nearPlane * farPlane) / (farPlane - nearPlane);
    result(3, 2) = -1.0f;
    return result;
}

// Vulkan orthographic with depth range [0, 1], near -> 0, far -> 1.
inline Matrix4 orthographicVulkan(
    float left,
    float right,
    float bottom,
    float top,
    float nearPlane,
    float farPlane
) {
    Matrix4 result{};
    for (int i = 0; i < 16; ++i) {
        result.m[i] = 0.0f;
    }

    result(0, 0) = 2.0f / (right - left);
    result(1, 1) = -2.0f / (top - bottom);
    result(2, 2) = 1.0f / (nearPlane - farPlane);
    result(3, 3) = 1.0f;
    result(0, 3) = -(right + left) / (right - left);
    result(1, 3) = -(top + bottom) / (top - bottom);
    result(2, 3) = nearPlane / (nearPlane - farPlane);
    return result;
}

// Vulkan reverse-Z orthographic with depth range [0, 1], near -> 1, far -> 0.
inline Matrix4 orthographicVulkanReverseZ(
    float left,
    float right,
    float bottom,
    float top,
    float nearPlane,
    float farPlane
) {
    Matrix4 result{};
    for (int i = 0; i < 16; ++i) {
        result.m[i] = 0.0f;
    }

    result(0, 0) = 2.0f / (right - left);
    result(1, 1) = -2.0f / (top - bottom);
    result(2, 2) = 1.0f / (farPlane - nearPlane);
    result(3, 3) = 1.0f;
    result(0, 3) = -(right + left) / (right - left);
    result(1, 3) = -(top + bottom) / (top - bottom);
    result(2, 3) = farPlane / (farPlane - nearPlane);
    return result;
}

inline Matrix4 multiply(const Matrix4& a, const Matrix4& b) {
    Matrix4 result{};
    for (int i = 0; i < 16; ++i) {
        result.m[i] = 0.0f;
    }

    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a(row, k) * b(k, col);
            }
            result(row, col) = sum;
        }
    }

    return result;
}

inline Vector4 multiply(const Matrix4& m, const Vector4& v) {
    return Vector4{
        (m(0, 0) * v.x) + (m(0, 1) * v.y) + (m(0, 2) * v.z) + (m(0, 3) * v.w),
        (m(1, 0) * v.x) + (m(1, 1) * v.y) + (m(1, 2) * v.z) + (m(1, 3) * v.w),
        (m(2, 0) * v.x) + (m(2, 1) * v.y) + (m(2, 2) * v.z) + (m(2, 3) * v.w),
        (m(3, 0) * v.x) + (m(3, 1) * v.y) + (m(3, 2) * v.z) + (m(3, 3) * v.w)
    };
}

inline Matrix4 operator*(const Matrix4& a, const Matrix4& b) {
    return multiply(a, b);
}

inline Vector4 operator*(const Matrix4& m, const Vector4& v) {
    return multiply(m, v);
}

inline Vector3 transformPoint(const Matrix4& m, const Vector3& p) {
    const Vector4 result = m * Vector4{p, 1.0f};
    if (result.w == 0.0f) {
        return Vector3{result.x, result.y, result.z};
    }
    return Vector3{result.x / result.w, result.y / result.w, result.z / result.w};
}

inline Vector3 transformDirection(const Matrix4& m, const Vector3& d) {
    const Vector4 result = m * Vector4{d, 0.0f};
    return Vector3{result.x, result.y, result.z};
}

} // namespace math
