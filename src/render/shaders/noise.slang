// Shared procedural noise primitives for terrain displacement and detail.
//
// These are intentionally simple value-noise / fbm functions (not a novel
// algorithm): a sin-hash lattice with smoothstep interpolation. The same math
// has a CPU twin in src/math/noise.h so prop placement can query terrain height
// off-GPU and agree (closely) with what the tessellation shaders displace.
//
// Keep this file and src/math/noise.h in lockstep: any change here must be
// mirrored there, or CPU-placed props will float above / sink into the ground.
module noise;

// Hash a lattice point to [0, 1). Classic sin-hash; cheap but precision-sensitive
// for large coordinates (the source of CPU/GPU divergence — tolerable for props).
public float hashNoise(float2 p)
{
    return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

// Smooth value noise in [-1, 1], bilinear-interpolated over the unit lattice with
// a smoothstep fade.
public float valueNoise(float2 p)
{
    const float2 i = floor(p);
    const float2 f = frac(p);
    const float2 u = f * f * (3.0 - (2.0 * f));
    const float a = hashNoise(i);
    const float b = hashNoise(i + float2(1.0, 0.0));
    const float c = hashNoise(i + float2(0.0, 1.0));
    const float d = hashNoise(i + float2(1.0, 1.0));
    return (lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y) * 2.0) - 1.0;
}

// Ridged variant (peaks where value noise crosses zero); used for sharper relief.
public float ridgedNoise(float2 p)
{
    return 1.0 - abs(valueNoise(p));
}

// Two-octave fbm weighted toward the low frequency so the relief reads as smooth
// rolling forms rather than high-frequency spikes. This is the canonical
// displacement fbm — its CPU twin (fbm2 in src/math/noise.h) drives prop placement.
public float fbm2(float2 p)
{
    return (valueNoise(p) * 0.70) + (valueNoise((p * 2.07) + float2(19.3, -8.7)) * 0.30);
}

// Three-octave fbm for richer macro terrain shaping (heightmap path).
public float fbm3(float2 p)
{
    return
        (valueNoise(p) * 0.52) +
        (valueNoise((p * 2.03) + float2(19.3, -8.7)) * 0.30) +
        (valueNoise((p * 4.11) + float2(-31.1, 24.6)) * 0.18);
}
