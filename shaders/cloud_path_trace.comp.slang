static const float kPi = 3.1415926535f;
static const float3 kCloudAabbMin = float3(-10.0f, 0.0f, -10.0f);
static const float3 kCloudAabbMax = float3(10.0f, 10.0f, 10.0f);
static const uint kMaxBouncesHardLimit = 12u;
static const uint kSunTrMaxSteps = 32u;
static const uint kApproxSunTrSteps = 12u;
static const uint kCloudApproxSteps = 32u;
static const float kSunAngularRadiusRadians = 0.004f; // ~0.23 degrees

struct CameraPush {
    float4 cameraPositionFov;
    float4 cameraForward;
    float4 cameraRight;
    float4 cameraUp;
    float4 sunDirectionIntensity;
    float4 mediumParams; // x = density scale, y = g, z = albedo
    float4 cloudShapeParams; // x = macro scale, y = detail scale
    float4 cloudProfileParams; // x = coverage, y = weather scale, z = cloud base, w = cloud top
    float4 cloudWarpParams; // x = reserved, y = erosion strength, z = reserved
    float4 cloudLightParams; // x = brightness boost, y = ambient lift, z = max bounces, w = debug sun transmittance
    float4 frameParams;  // x = width, y = height, z = frame index, w = reset flag
};

[[vk::push_constant]]
ConstantBuffer<CameraPush> gPush;

[[vk::binding(0, 0)]]
RWTexture2D<float4> gAccumulation;
[[vk::binding(1, 0)]]
RWTexture2D<uint> gRngState;
[[vk::binding(2, 0)]]
Texture2D<float4> gBlueNoise;
[[vk::binding(3, 0)]]
Texture3D<float4> gCloudNoiseVolume;
[[vk::binding(4, 0)]]
SamplerState gVolumeSampler;

uint hash3(uint3 p) {
    uint h = p.x * 747796405u;
    h ^= p.y * 2891336453u;
    h ^= p.z * 1181783497u;
    h ^= (h >> 16u);
    h *= 2246822519u;
    h ^= (h >> 13u);
    h *= 3266489917u;
    h ^= (h >> 16u);
    return h;
}

float rand01(inout uint state) {
    state = state * 1664525u + 1013904223u;
    return float(state & 0x00ffffffu) * (1.0f / 16777216.0f);
}

float hashToUnitFloat(uint h) {
    uint v = h ^ 0x9e3779b9u;
    return float(v & 0x00ffffffu) * (1.0f / 16777216.0f);
}

float3 fade3(float3 t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float perlinGrad(uint h, float3 p) {
    uint g = h & 15u;
    float u = (g < 8u) ? p.x : p.y;
    float v = (g < 4u) ? p.y : ((g == 12u || g == 14u) ? p.x : p.z);
    float a = ((g & 1u) != 0u) ? -u : u;
    float b = ((g & 2u) != 0u) ? -v : v;
    return a + b;
}

float perlinNoise3(float3 p) {
    int3 i0 = int3(floor(p));
    int3 i1 = i0 + int3(1, 1, 1);
    float3 f = frac(p);
    float3 u = fade3(f);

    float n000 = perlinGrad(hash3(uint3(i0.x, i0.y, i0.z)), f - float3(0.0f, 0.0f, 0.0f));
    float n100 = perlinGrad(hash3(uint3(i1.x, i0.y, i0.z)), f - float3(1.0f, 0.0f, 0.0f));
    float n010 = perlinGrad(hash3(uint3(i0.x, i1.y, i0.z)), f - float3(0.0f, 1.0f, 0.0f));
    float n110 = perlinGrad(hash3(uint3(i1.x, i1.y, i0.z)), f - float3(1.0f, 1.0f, 0.0f));
    float n001 = perlinGrad(hash3(uint3(i0.x, i0.y, i1.z)), f - float3(0.0f, 0.0f, 1.0f));
    float n101 = perlinGrad(hash3(uint3(i1.x, i0.y, i1.z)), f - float3(1.0f, 0.0f, 1.0f));
    float n011 = perlinGrad(hash3(uint3(i0.x, i1.y, i1.z)), f - float3(0.0f, 1.0f, 1.0f));
    float n111 = perlinGrad(hash3(uint3(i1.x, i1.y, i1.z)), f - float3(1.0f, 1.0f, 1.0f));

    float nx00 = lerp(n000, n100, u.x);
    float nx10 = lerp(n010, n110, u.x);
    float nx01 = lerp(n001, n101, u.x);
    float nx11 = lerp(n011, n111, u.x);
    float nxy0 = lerp(nx00, nx10, u.y);
    float nxy1 = lerp(nx01, nx11, u.y);
    return lerp(nxy0, nxy1, u.z);
}

float perlinFbm(float3 p) {
    float sum = 0.0f;
    float amp = 0.5f;
    float freq = 1.0f;
    float norm = 0.0f;

    [unroll]
    for (int i = 0; i < 5; ++i) {
        float n = perlinNoise3(p * freq);
        sum += amp * (0.5f + 0.5f * n);
        norm += amp;
        freq *= 2.03f;
        amp *= 0.5f;
    }

    return (norm > 0.0f) ? (sum / norm) : 0.0f;
}

float2 worleyF1F2(float3 p) {
    int3 cell = int3(floor(p));
    float f1 = 1e9f;
    float f2 = 1e9f;

    [loop]
    for (int z = -1; z <= 1; ++z) {
        [loop]
        for (int y = -1; y <= 1; ++y) {
            [loop]
            for (int x = -1; x <= 1; ++x) {
                int3 n = cell + int3(x, y, z);
                uint h = hash3(uint3(n));
                float3 j = float3(
                    hashToUnitFloat(h),
                    hashToUnitFloat(h * 1664525u + 1013904223u),
                    hashToUnitFloat(h * 22695477u + 1u));
                float3 feature = float3(n) + j;
                float d = length(p - feature);
                if (d < f1) {
                    f2 = f1;
                    f1 = d;
                } else if (d < f2) {
                    f2 = d;
                }
            }
        }
    }

    return float2(f1, f2);
}

float3 curlNoise(float3 p) {
    const float e = 0.18f;
    float3 offA = float3(19.1f, -7.3f, 4.7f);
    float3 offB = float3(-13.8f, 11.5f, -2.4f);
    float3 offC = float3(5.2f, 3.9f, 14.1f);

    float n1_y1 = perlinNoise3(p + float3(0.0f, e, 0.0f) + offA);
    float n1_y0 = perlinNoise3(p - float3(0.0f, e, 0.0f) + offA);
    float n1_z1 = perlinNoise3(p + float3(0.0f, 0.0f, e) + offA);
    float n1_z0 = perlinNoise3(p - float3(0.0f, 0.0f, e) + offA);

    float n2_x1 = perlinNoise3(p + float3(e, 0.0f, 0.0f) + offB);
    float n2_x0 = perlinNoise3(p - float3(e, 0.0f, 0.0f) + offB);
    float n2_z1 = perlinNoise3(p + float3(0.0f, 0.0f, e) + offB);
    float n2_z0 = perlinNoise3(p - float3(0.0f, 0.0f, e) + offB);

    float n3_x1 = perlinNoise3(p + float3(e, 0.0f, 0.0f) + offC);
    float n3_x0 = perlinNoise3(p - float3(e, 0.0f, 0.0f) + offC);
    float n3_y1 = perlinNoise3(p + float3(0.0f, e, 0.0f) + offC);
    float n3_y0 = perlinNoise3(p - float3(0.0f, e, 0.0f) + offC);

    float dA_dy = (n1_y1 - n1_y0) / (2.0f * e);
    float dA_dz = (n1_z1 - n1_z0) / (2.0f * e);
    float dB_dx = (n2_x1 - n2_x0) / (2.0f * e);
    float dB_dz = (n2_z1 - n2_z0) / (2.0f * e);
    float dC_dx = (n3_x1 - n3_x0) / (2.0f * e);
    float dC_dy = (n3_y1 - n3_y0) / (2.0f * e);

    float3 curl = float3(
        dC_dy - dB_dz,
        dA_dz - dC_dx,
        dB_dx - dA_dy);
    float len = length(curl);
    return (len > 1e-4f) ? (curl / len) : float3(0.0f, 0.0f, 0.0f);
}

float4 sampleCloudNoise(float3 p) {
    return gCloudNoiseVolume.SampleLevel(gVolumeSampler, p, 0.0f);
}

float remap(float value, float originalMin, float originalMax, float newMin, float newMax) {
    float denom = originalMax - originalMin;
    if (abs(denom) < 1e-5f) {
        return newMin;
    }
    float t = (value - originalMin) / denom;
    return lerp(newMin, newMax, t);
}

float densityField(float3 p) {
    const float cloudBase = gPush.cloudProfileParams.z;
    const float cloudTop = max(gPush.cloudProfileParams.w, cloudBase + 0.25f);
    const float macroScale = max(gPush.cloudShapeParams.x, 0.05f);
    const float detailScale = max(gPush.cloudShapeParams.y, 0.08f);
    const float erosionStrength = max(gPush.cloudWarpParams.y, 0.0f);

    if (p.y < cloudBase || p.y > cloudTop) {
        return 0.0f;
    }

    const float densityScale = max(gPush.mediumParams.x, 0.01f);

    // 1. Dome container (squashed and flat-bottomed).
    const float3 center = float3(0.0f, cloudBase + 1.5f, 0.0f);
    const float3 localP = (p - center) * float3(1.0f, 1.8f, 1.0f);
    const float distFromCenter = length(localP);
    const float domeRadius = 7.0f;
    const float domeMask = 1.0f - smoothstep(domeRadius * 0.1f, domeRadius, distFromCenter);

    const float h = saturate((p.y - cloudBase) / max((cloudTop - cloudBase), 0.25f));
    const float bottomRamp = smoothstep(0.0f, 0.15f, h);
    const float structuralMask = domeMask * bottomRamp;
    if (structuralMask <= 0.0f) {
        return 0.0f;
    }

    // 2. Base shape from low-frequency noise.
    const float3 baseUvw = p * macroScale * 0.15f;
    const float baseNoise = sampleCloudNoise(baseUvw).r;
    float baseDensity = remap(baseNoise, 1.0f - structuralMask, 1.0f, 0.0f, 1.0f);
    baseDensity = saturate(baseDensity);
    if (baseDensity <= 0.0f) {
        return 0.0f;
    }

    // 3. Cauliflower edge carving from Worley channels.
    const float3 detailUvw = p * detailScale * 0.4f;
    const float4 detailNoise = sampleCloudNoise(detailUvw);
    const float highFreqFbm = detailNoise.g * 0.625f + detailNoise.b * 0.25f + detailNoise.w * 0.125f;
    const float heightBias = lerp(0.2f, 1.0f, h);
    const float erosionModifier = lerp(highFreqFbm, 0.0f, baseDensity) * heightBias;
    const float finalDensity = saturate(baseDensity - (erosionModifier * erosionStrength));

    return saturate(finalDensity) * densityScale;
}

bool intersectAabb(float3 rayOrigin, float3 rayDir, float3 bmin, float3 bmax, out float tMin, out float tMax) {
    float3 invD = 1.0f / rayDir;
    float3 t0 = (bmin - rayOrigin) * invD;
    float3 t1 = (bmax - rayOrigin) * invD;

    float3 tSmall = min(t0, t1);
    float3 tLarge = max(t0, t1);

    tMin = max(max(tSmall.x, tSmall.y), max(tSmall.z, 0.0f));
    tMax = min(min(tLarge.x, tLarge.y), tLarge.z);
    return tMax > tMin;
}

float henyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    float d = 1.0f + g2 - 2.0f * g * cosTheta;
    return (1.0f - g2) / (4.0f * kPi * pow(max(d, 1e-4f), 1.5f));
}

float dualHenyeyGreenstein(float cosTheta, float g) {
    // Cloud droplets: dominant forward lobe + weaker backward lobe.
    float forwardLobe = henyeyGreenstein(cosTheta, g);
    float backwardLobe = henyeyGreenstein(cosTheta, -0.3f);
    return lerp(backwardLobe, forwardLobe, 0.7f);
}

float3 makeOrthoX(float3 n) {
    return (abs(n.z) < 0.999f) ? normalize(cross(float3(0.0f, 0.0f, 1.0f), n)) : normalize(cross(float3(0.0f, 1.0f, 0.0f), n));
}

float3 sampleUniformCone(float3 axis, float cosThetaMax, inout uint rng) {
    float xi1 = rand01(rng);
    float xi2 = rand01(rng);

    float cosTheta = lerp(1.0f, cosThetaMax, xi1);
    float sinTheta = sqrt(max(1.0f - cosTheta * cosTheta, 0.0f));
    float phi = 2.0f * kPi * xi2;

    float3 t = makeOrthoX(axis);
    float3 b = cross(axis, t);
    return normalize((sinTheta * cos(phi)) * t + (sinTheta * sin(phi)) * b + cosTheta * axis);
}

float3 sampleHenyeyGreenstein(float3 wo, float g, inout uint rng) {
    float xi1 = rand01(rng);
    float xi2 = rand01(rng);

    float cosTheta = 0.0f;
    if (abs(g) < 1e-3f) {
        cosTheta = 1.0f - 2.0f * xi1;
    } else {
        float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi1);
        cosTheta = (1.0f + g * g - s * s) / (2.0f * g);
    }
    float sinTheta = sqrt(max(1.0f - cosTheta * cosTheta, 0.0f));
    float phi = 2.0f * kPi * xi2;

    float3 t = makeOrthoX(wo);
    float3 b = cross(wo, t);
    return normalize((sinTheta * cos(phi)) * t + (sinTheta * sin(phi)) * b + cosTheta * wo);
}

bool sampleDistance(
    float3 rayOrigin,
    float3 rayDir,
    float tMin,
    float tMax,
    float sigmaMajorant,
    float densityScale,
    inout uint rng,
    out float3 outPosition
) {
    outPosition = 0.0f;

    float t = tMin;
    [loop]
    for (uint i = 0u; i < 180u; ++i) {
        float u = max(rand01(rng), 1e-6f);
        t += -log(u) / sigmaMajorant;
        if (t >= tMax) {
            return false;
        }

        float3 x = rayOrigin + rayDir * t;
        float sigmaT = densityField(x) * densityScale * 1.5f;
        float accept = saturate(sigmaT / sigmaMajorant);

        if (rand01(rng) < accept) {
            outPosition = x;
            return true;
        }
    }

    return false;
}

float estimateSunTransmittanceSteps(float3 p, float3 sunDir, float densityScale, uint maxSteps) {
    float tMin = 0.0f;
    float tMax = 0.0f;
    if (!intersectAabb(p + sunDir * 0.05f, sunDir, kCloudAabbMin, kCloudAabbMax, tMin, tMax)) {
        return 1.0f;
    }

    float marchDistance = tMax;
    float stepSize = marchDistance / float(max(maxSteps, 1u));
    if (stepSize <= 0.0f) {
        return 1.0f;
    }

    float t = stepSize * 0.5f;
    float opticalDepth = 0.0f;
    [loop]
    for (uint i = 0; i < maxSteps; ++i) {
        float3 x = p + sunDir * t;
        float density = densityField(x) * densityScale * 1.5f;
        opticalDepth += density * stepSize;
        t += stepSize;
    }

    return exp(-opticalDepth);
}

float estimateSunTransmittance(float3 p, float3 sunDir, float densityScale) {
    return estimateSunTransmittanceSteps(p, sunDir, densityScale, kSunTrMaxSteps);
}

float3 integrateCloudApproximation(
    float3 rayOrigin,
    float3 rayDir,
    float3 sunDir,
    float3 sunRadiance,
    float sunIntensity,
    float densityScale,
    float g,
    float3 cloudAlbedo,
    float brightnessBoost,
    float ambientLift,
    bool debugSunTransmittance
) {
    float tMin = 0.0f;
    float tMax = 0.0f;
    if (!intersectAabb(rayOrigin, rayDir, kCloudAabbMin, kCloudAabbMax, tMin, tMax)) {
        return skyColor(rayDir, sunDir, sunIntensity);
    }

    float marchDistance = tMax - tMin;
    if (marchDistance <= 0.0f) {
        return skyColor(rayDir, sunDir, sunIntensity);
    }

    const float stepSize = marchDistance / float(kCloudApproxSteps);
    float t = tMin + (0.5f * stepSize);
    float3 L = 0.0f;
    float3 throughput = 1.0f;
    float3 ambientSky = skyAmbientLighting(sunDir, sunIntensity);
    float3 ambientTerm = ambientSky * (0.05f / (4.0f * kPi)) * ambientLift;

    [loop]
    for (uint i = 0u; i < kCloudApproxSteps; ++i) {
        float maxThroughput = max(throughput.x, max(throughput.y, throughput.z));
        if (maxThroughput < 1.0e-4f) {
            break;
        }

        float3 x = rayOrigin + (rayDir * t);
        float sigmaT = densityField(x) * densityScale * 1.5f;
        if (sigmaT > 0.0f) {
            float segmentTransmit = exp(-sigmaT * stepSize);
            float segmentScatter = 1.0f - segmentTransmit;

            float cosTheta = clamp(dot(-rayDir, -sunDir), -1.0f, 1.0f);
            float phaseSun = dualHenyeyGreenstein(cosTheta, g);
            float trSun = estimateSunTransmittanceSteps(x, sunDir, densityScale, kApproxSunTrSteps);
            if (debugSunTransmittance) {
                return float3(trSun, trSun, trSun);
            }

            float3 direct = sunRadiance * phaseSun * trSun * cloudAlbedo * brightnessBoost;
            float powder = 1.0f - exp(-sigmaT * 1.2f);
            float multipleScatteringApprox = 0.25f + (0.65f * powder);

            L += throughput * segmentScatter * (direct + (direct * multipleScatteringApprox * powder));
            L += throughput * segmentScatter * ambientTerm * cloudAlbedo;

            throughput *= segmentTransmit;
        }

        t += stepSize;
    }

    return L + (throughput * skyColor(rayDir, sunDir, sunIntensity));
}

float phaseRayleigh(float cosTheta) {
    return 3.0f * (1.0f + cosTheta * cosTheta) / (16.0f * kPi);
}

float phaseMie(float cosTheta, float g) {
    float g2 = g * g;
    float d = max(1.0f + g2 - 2.0f * g * cosTheta, 1e-4f);
    return (1.0f - g2) / (4.0f * kPi * d * sqrt(d));
}

float airMass(float upDot) {
    float clamped = clamp(upDot, 0.01f, 1.0f);
    float zenith = acos(clamped);
    float zenithDeg = zenith * (180.0f / kPi);
    float denom = clamped + 0.15f * pow(max(93.885f - zenithDeg, 1.0f), -1.253f);
    return 1.0f / max(denom, 0.02f);
}

float3 skyColor(float3 dir, float3 sunDir, float sunIntensity) {
    float3 v = normalize(dir);
    float mu = clamp(dot(v, sunDir), -1.0f, 1.0f);

    float mView = airMass(max(v.y, 0.01f));
    float mSun = airMass(max(sunDir.y, 0.01f));

    // Tuned to produce a plausible daylight atmosphere response in HDR.
    float3 betaR = float3(5.5e-6f, 13.0e-6f, 22.4e-6f) * 14000.0f;
    float3 betaM = float3(21.0e-6f, 21.0e-6f, 21.0e-6f) * 7000.0f;

    float3 tau = betaR * mView + betaM * mView;
    float3 extinction = exp(-tau);

    float rayleigh = phaseRayleigh(mu);
    float mie = phaseMie(mu, 0.82f);
    float3 inscatter = (betaR * rayleigh + betaM * mie) * exp(-(betaR + betaM) * mSun);
    inscatter *= (1.0f - extinction) / max(betaR + betaM, float3(1e-4f, 1e-4f, 1e-4f));

    float sunDisk = smoothstep(0.9996f, 0.99995f, mu);
    float sunHalo = pow(saturate(mu), 64.0f);
    float horizonBoost = smoothstep(-0.08f, 0.25f, v.y);
    float twilight = smoothstep(-0.18f, 0.08f, sunDir.y);

    float3 skyBase = inscatter * (0.9f + 0.25f * horizonBoost) * twilight;
    float3 sunColor = float3(1.0f, 0.96f, 0.90f);
    float3 sunTerm = sunColor * (sunDisk * 24.0f + sunHalo * 1.8f);
    return (skyBase + sunTerm) * max(sunIntensity, 0.01f);
}

float3 skyAmbientLighting(float3 sunDir, float sunIntensity) {
    float3 up = skyColor(float3(0.0f, 1.0f, 0.0f), sunDir, sunIntensity);
    float3 horizonSun = skyColor(normalize(float3(sunDir.x, 0.15f, sunDir.z)), sunDir, sunIntensity);
    float3 horizonOpp = skyColor(normalize(float3(-sunDir.x, 0.15f, -sunDir.z)), sunDir, sunIntensity);
    return (0.45f * up) + (0.35f * horizonSun) + (0.20f * horizonOpp);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint width = (uint)gPush.frameParams.x;
    uint height = (uint)gPush.frameParams.y;
    if (dispatchThreadID.x >= width || dispatchThreadID.y >= height) {
        return;
    }

    uint2 pixel = dispatchThreadID.xy;
    uint frameIndexUint = (uint)gPush.frameParams.z;

    uint2 blueNoiseSize;
    gBlueNoise.GetDimensions(blueNoiseSize.x, blueNoiseSize.y);
    blueNoiseSize = max(blueNoiseSize, uint2(1u, 1u));

    float2 goldenOffset = frac(float2(0.7548776f, 0.5698403f) * (float)frameIndexUint);
    uint2 pixelOffset = uint2(goldenOffset * float2(blueNoiseSize));
    uint2 blueNoiseCoord = uint2(
        (pixel.x + pixelOffset.x) % blueNoiseSize.x,
        (pixel.y + pixelOffset.y) % blueNoiseSize.y);
    float4 blueNoise = gBlueNoise.Load(int3(blueNoiseCoord, 0));

    uint rngState = gRngState[pixel];
    if (rngState == 0u || gPush.frameParams.w > 0.5f) {
        uint bnInt = (uint)(blueNoise.z * 16777216.0f);
        rngState = hash3(uint3(pixel, bnInt ^ frameIndexUint));
    }

    float2 jitter = blueNoise.xy;
    float2 uv = ((float2(pixel) + jitter) / float2(width, height)) * 2.0f - 1.0f;
    uv.y = -uv.y;

    float aspect = float(width) / float(height);
    float tanHalfFov = tan(radians(gPush.cameraPositionFov.w) * 0.5f);

    float3 rayOrigin = gPush.cameraPositionFov.xyz;
    float3 rayDir = normalize(
        gPush.cameraForward.xyz +
        (uv.x * aspect * tanHalfFov) * gPush.cameraRight.xyz +
        (uv.y * tanHalfFov) * gPush.cameraUp.xyz);

    float3 sunDir = normalize(gPush.sunDirectionIntensity.xyz);
    float sunIntensity = max(gPush.sunDirectionIntensity.w, 0.01f);
    float sunCosThetaMax = cos(kSunAngularRadiusRadians);
    float3 sunRadiance = float3(1.0f, 0.97f, 0.9f) * sunIntensity;

    float densityScale = max(gPush.mediumParams.x, 0.01f);
    float g = clamp(gPush.mediumParams.y, 0.0f, 0.90f);
    float3 cloudAlbedo = float3(gPush.mediumParams.z, gPush.mediumParams.z, gPush.mediumParams.z);
    float brightnessBoost = max(gPush.cloudLightParams.x, 0.0f);
    float ambientLift = max(gPush.cloudLightParams.y, 0.0f);
    bool debugSunTransmittance = gPush.cloudLightParams.w > 0.5f;
    uint maxBounces = (uint)clamp(gPush.cloudLightParams.z, 1.0f, (float)kMaxBouncesHardLimit);
    float sigmaMajorant = max(densityScale * (1.85f + 0.65f * saturate(gPush.cloudShapeParams.w)), 0.05f);

    float3 L = 0.0f;
    float3 throughput = 1.0f;

    if (maxBounces <= 1u) {
        L = integrateCloudApproximation(
            rayOrigin,
            rayDir,
            sunDir,
            sunRadiance,
            sunIntensity,
            densityScale,
            g,
            cloudAlbedo,
            brightnessBoost,
            ambientLift,
            debugSunTransmittance);
    } else {
        [loop]
        for (uint bounce = 0u; bounce < maxBounces; ++bounce) {
            float tMin = 0.0f;
            float tMax = 0.0f;
            if (!intersectAabb(rayOrigin, rayDir, kCloudAabbMin, kCloudAabbMax, tMin, tMax)) {
                if (!debugSunTransmittance) {
                    L += throughput * skyColor(rayDir, sunDir, sunIntensity);
                }
                break;
            }

            float3 scatterPos = 0.0f;
            bool hit = sampleDistance(
                rayOrigin,
                rayDir,
                tMin,
                tMax,
                sigmaMajorant,
                densityScale,
                rngState,
                scatterPos);

            if (!hit) {
                if (!debugSunTransmittance) {
                    L += throughput * skyColor(rayDir, sunDir, sunIntensity);
                }
                break;
            }

            if (debugSunTransmittance && bounce == 0u) {
                float trSunDebug = estimateSunTransmittance(scatterPos, sunDir, densityScale);
                L = float3(trSunDebug, trSunDebug, trSunDebug);
                break;
            }

            float3 wo = -rayDir;
            float3 sunDirSample = sampleUniformCone(sunDir, sunCosThetaMax, rngState);
            float trSun = estimateSunTransmittance(scatterPos, sunDirSample, densityScale);
            const float localDensity = densityField(scatterPos) * densityScale;
            const float powderEffect = 1.0f - exp(-localDensity * 2.0f);
            trSun *= powderEffect;
            // Conventions:
            // - viewDir points from previous vertex/camera toward the scattering point.
            // - lightDir points from sun toward the scattering point.
            // HG uses cos(theta) = dot(lightDir, -viewDir).
            float3 viewDir = rayDir;
            float3 lightDir = -sunDirSample;
            float cosTheta = clamp(dot(lightDir, -viewDir), -1.0f, 1.0f);
            float phaseSun = dualHenyeyGreenstein(cosTheta, g);
            // sunRadiance here represents integrated sun-disk radiance under our UI intensity convention.
            L += throughput * cloudAlbedo * brightnessBoost * trSun * phaseSun * sunRadiance;

            float3 ambientSky = skyAmbientLighting(sunDir, sunIntensity);
            L += throughput * cloudAlbedo * (1.0f / (4.0f * kPi)) * ambientSky * 0.05f;

            if (bounce + 1u >= maxBounces) {
                break;
            }

            float3 wi = sampleHenyeyGreenstein(wo, g, rngState);
            rayOrigin = scatterPos + wi * 0.02f;
            rayDir = wi;
            throughput *= cloudAlbedo;

            // Start RR at path depth 5 (bounce index 4) to preserve deeper backlit glow.
            if (bounce >= 4u) {
                const float rrSurvive = 0.9f;
                if (rand01(rngState) > rrSurvive) {
                    break;
                }
                throughput /= rrSurvive;
            }
        }
    }

    float4 prev = gAccumulation[pixel];
    float frameIndex = gPush.frameParams.z;
    if (gPush.frameParams.w > 0.5f) {
        gAccumulation[pixel] = float4(L, 1.0f);
    } else {
        float3 accum = (prev.xyz * frameIndex + L) / (frameIndex + 1.0f);
        gAccumulation[pixel] = float4(accum, 1.0f);
    }

    gRngState[pixel] = rngState;
}
