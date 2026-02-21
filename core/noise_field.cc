#include "core/noise_field.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace voxelsprout::core {
namespace {

float clamp01(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

std::uint32_t hash3i(std::int32_t x, std::int32_t y, std::int32_t z, std::uint32_t seed) {
    std::uint32_t h = seed;
    h ^= static_cast<std::uint32_t>(x) * 0x8da6b343u;
    h ^= static_cast<std::uint32_t>(y) * 0xd8163841u;
    h ^= static_cast<std::uint32_t>(z) * 0xcb1ab31fu;
    h ^= (h >> 13u);
    h *= 0x85ebca6bu;
    h ^= (h >> 16u);
    return h;
}

float random01(std::uint32_t h) {
    return static_cast<float>(h & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

float perlinGrad(std::uint32_t h, float x, float y, float z) {
    switch (h & 0x0fu) {
        case 0x0u:
            return x + y;
        case 0x1u:
            return -x + y;
        case 0x2u:
            return x - y;
        case 0x3u:
            return -x - y;
        case 0x4u:
            return x + z;
        case 0x5u:
            return -x + z;
        case 0x6u:
            return x - z;
        case 0x7u:
            return -x - z;
        case 0x8u:
            return y + z;
        case 0x9u:
            return -y + z;
        case 0xau:
            return y - z;
        case 0xbu:
            return -y - z;
        case 0xcu:
            return x + y;
        case 0xdu:
            return -x + y;
        case 0xeu:
            return x - y;
        default:
            return -x - y;
    }
}

float perlinNoise3D(double x, double y, double z, std::uint32_t seed) {
    const std::int32_t x0 = static_cast<std::int32_t>(std::floor(x));
    const std::int32_t y0 = static_cast<std::int32_t>(std::floor(y));
    const std::int32_t z0 = static_cast<std::int32_t>(std::floor(z));
    const std::int32_t x1 = x0 + 1;
    const std::int32_t y1 = y0 + 1;
    const std::int32_t z1 = z0 + 1;

    const float fx = static_cast<float>(x - static_cast<double>(x0));
    const float fy = static_cast<float>(y - static_cast<double>(y0));
    const float fz = static_cast<float>(z - static_cast<double>(z0));
    const float u = fade(fx);
    const float v = fade(fy);
    const float w = fade(fz);

    const float n000 = perlinGrad(hash3i(x0, y0, z0, seed), fx, fy, fz);
    const float n100 = perlinGrad(hash3i(x1, y0, z0, seed), fx - 1.0f, fy, fz);
    const float n010 = perlinGrad(hash3i(x0, y1, z0, seed), fx, fy - 1.0f, fz);
    const float n110 = perlinGrad(hash3i(x1, y1, z0, seed), fx - 1.0f, fy - 1.0f, fz);
    const float n001 = perlinGrad(hash3i(x0, y0, z1, seed), fx, fy, fz - 1.0f);
    const float n101 = perlinGrad(hash3i(x1, y0, z1, seed), fx - 1.0f, fy, fz - 1.0f);
    const float n011 = perlinGrad(hash3i(x0, y1, z1, seed), fx, fy - 1.0f, fz - 1.0f);
    const float n111 = perlinGrad(hash3i(x1, y1, z1, seed), fx - 1.0f, fy - 1.0f, fz - 1.0f);

    const float nx00 = lerp(n000, n100, u);
    const float nx10 = lerp(n010, n110, u);
    const float nx01 = lerp(n001, n101, u);
    const float nx11 = lerp(n011, n111, u);
    const float nxy0 = lerp(nx00, nx10, v);
    const float nxy1 = lerp(nx01, nx11, v);
    return lerp(nxy0, nxy1, w);
}

float fbmPerlin3D(
    double x,
    double y,
    double z,
    double baseFrequency,
    std::int32_t octaves,
    double lacunarity,
    double persistence,
    std::uint32_t seed) {
    double frequency = baseFrequency;
    float amplitude = 1.0f;
    float amplitudeSum = 0.0f;
    float valueSum = 0.0f;
    for (std::int32_t octave = 0; octave < octaves; ++octave) {
        const std::uint32_t octaveSeed = seed + static_cast<std::uint32_t>(octave) * 0x9e3779b9u;
        valueSum += amplitude * perlinNoise3D(x * frequency, y * frequency, z * frequency, octaveSeed);
        amplitudeSum += amplitude;
        frequency *= lacunarity;
        amplitude *= static_cast<float>(persistence);
    }
    if (amplitudeSum <= 0.0f) {
        return 0.0f;
    }
    return valueSum / amplitudeSum;
}

float worleyF1Distance(double x, double y, double z, double frequency, std::uint32_t seed) {
    const double px = x * frequency;
    const double py = y * frequency;
    const double pz = z * frequency;
    const std::int32_t cellX = static_cast<std::int32_t>(std::floor(px));
    const std::int32_t cellY = static_cast<std::int32_t>(std::floor(py));
    const std::int32_t cellZ = static_cast<std::int32_t>(std::floor(pz));

    double minDistance2 = std::numeric_limits<double>::max();
    for (std::int32_t dz = -1; dz <= 1; ++dz) {
        for (std::int32_t dy = -1; dy <= 1; ++dy) {
            for (std::int32_t dx = -1; dx <= 1; ++dx) {
                const std::int32_t nx = cellX + dx;
                const std::int32_t ny = cellY + dy;
                const std::int32_t nz = cellZ + dz;
                const std::uint32_t baseHash = hash3i(nx, ny, nz, seed);
                const double fx = static_cast<double>(nx) + static_cast<double>(random01(baseHash ^ 0x68bc21ebu));
                const double fy = static_cast<double>(ny) + static_cast<double>(random01(baseHash ^ 0x02e5be93u));
                const double fz = static_cast<double>(nz) + static_cast<double>(random01(baseHash ^ 0x967a889bu));
                const double ox = fx - px;
                const double oy = fy - py;
                const double oz = fz - pz;
                const double distance2 = (ox * ox) + (oy * oy) + (oz * oz);
                minDistance2 = std::min(minDistance2, distance2);
            }
        }
    }
    return static_cast<float>(std::sqrt(minDistance2));
}

} // namespace

NoiseSamples sampleLibNoise(double x, double y, double z) {
    const float perlinValue = fbmPerlin3D(x, y, z, 0.08, 5, 2.0, 0.5, 1337u);
    const float worleyDistance = worleyF1Distance(x, y, z, 0.08, 4242u);
    const float worleyNormalized = worleyDistance / std::sqrt(3.0f);

    NoiseSamples result{};
    result.perlin = clamp01(0.5f + (0.5f * perlinValue));
    result.worley = clamp01(worleyNormalized);
    return result;
}

} // namespace voxelsprout::core
