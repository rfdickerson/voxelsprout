#pragma once

#include <cmath>

// CPU twin of src/render/shaders/noise.slang.
//
// These functions mirror the GPU value-noise / fbm used for terrain
// displacement so off-GPU code (e.g. prop placement) can query an approximate
// ground height that agrees with what the tessellation shaders draw.
//
// Parity caveat: the lattice hash uses sin(), whose last-bit results differ
// between CPU and GPU. The lattice is evaluated at exact integer coordinates of
// modest magnitude, so divergence stays small — fine for placing trees/rocks
// (a few centimetres of error is invisible) but NOT a bit-exact match. Keep this
// file in lockstep with noise.slang; any edit to one must be mirrored in the other.
namespace odai::math {

inline float noiseFract(float v) {
    return v - std::floor(v);
}

inline float noiseLerp(float a, float b, float t) {
    return a + ((b - a) * t);
}

// Hash a lattice point to [0, 1). Matches hashNoise() in noise.slang.
inline float hashNoise(float px, float py) {
    return noiseFract(std::sin((px * 127.1f) + (py * 311.7f)) * 43758.5453f);
}

// Smooth value noise in [-1, 1]. Matches valueNoise() in noise.slang.
inline float valueNoise(float px, float py) {
    const float ix = std::floor(px);
    const float iy = std::floor(py);
    const float fx = px - ix;
    const float fy = py - iy;
    const float ux = fx * fx * (3.0f - (2.0f * fx));
    const float uy = fy * fy * (3.0f - (2.0f * fy));
    const float a = hashNoise(ix, iy);
    const float b = hashNoise(ix + 1.0f, iy);
    const float c = hashNoise(ix, iy + 1.0f);
    const float d = hashNoise(ix + 1.0f, iy + 1.0f);
    return (noiseLerp(noiseLerp(a, b, ux), noiseLerp(c, d, ux), uy) * 2.0f) - 1.0f;
}

// Ridged variant. Matches ridgedNoise() in noise.slang.
inline float ridgedNoise(float px, float py) {
    return 1.0f - std::fabs(valueNoise(px, py));
}

// Two-octave displacement fbm. Matches fbm2() in noise.slang — this is the one
// prop placement should use to follow the tessellated hex terrain surface.
inline float fbm2(float px, float py) {
    return (valueNoise(px, py) * 0.70f) +
           (valueNoise((px * 2.07f) + 19.3f, (py * 2.07f) - 8.7f) * 0.30f);
}

// Three-octave macro fbm. Matches fbm3() in noise.slang.
inline float fbm3(float px, float py) {
    return (valueNoise(px, py) * 0.52f) +
           (valueNoise((px * 2.03f) + 19.3f, (py * 2.03f) - 8.7f) * 0.30f) +
           (valueNoise((px * 4.11f) - 31.1f, (py * 4.11f) + 24.6f) * 0.18f);
}

}  // namespace odai::math
