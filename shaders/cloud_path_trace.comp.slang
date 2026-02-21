static const float kPi = 3.1415926535f;
static const float3 kCloudAabbMin = float3(-10.0f, 0.0f, -10.0f);
static const float3 kCloudAabbMax = float3(10.0f, 10.0f, 10.0f);
static const uint kMaxBouncesHardLimit = 6u;

struct CameraPush {
    float4 cameraPositionFov;
    float4 cameraForward;
    float4 cameraRight;
    float4 cameraUp;
    float4 sunDirectionIntensity;
    float4 mediumParams; // x = density scale, y = g, z = albedo
    float4 cloudShapeParams; // x = macro scale, y = detail scale, z = density cutoff, w = chunkiness
    float4 cloudProfileParams; // x = coverage, y = weather scale, z = cloud base, w = cloud top
    float4 cloudWarpParams; // x = warp strength, y = erosion strength
    float4 cloudLightParams; // x = brightness boost, y = ambient lift, z = max bounces
    float4 frameParams;  // x = width, y = height, z = frame index, w = reset flag
};

[[vk::push_constant]]
ConstantBuffer<CameraPush> gPush;

[[vk::binding(0, 0)]]
RWTexture2D<float4> gAccumulation;
[[vk::binding(1, 0)]]
RWTexture2D<uint> gRngState;

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

float densityField(float3 p) {
    const float3 cloudCenter = float3(0.0f, 4.2f, -1.8f);

    float macroScale = max(gPush.cloudShapeParams.x, 0.05f);
    float detailScale = max(gPush.cloudShapeParams.y, 0.08f);
    float cutoff = saturate(gPush.cloudShapeParams.z);
    float chunkiness = max(gPush.cloudShapeParams.w, 0.05f);

    float coverage = saturate(gPush.cloudProfileParams.x);
    float weatherScale = max(gPush.cloudProfileParams.y, 0.01f);
    float cloudBase = gPush.cloudProfileParams.z;
    float cloudTop = max(gPush.cloudProfileParams.w, cloudBase + 0.25f);

    float warpStrength = max(gPush.cloudWarpParams.x, 0.0f);
    float erosionStrength = max(gPush.cloudWarpParams.y, 0.0f);

    // Vertical profile: build-up at base, billowy core, soft top dissipation.
    float h = (p.y - cloudBase) / max(cloudTop - cloudBase, 0.25f);
    float baseRamp = smoothstep(0.0f, 0.10f, h);
    float topFade = 1.0f - smoothstep(0.72f, 1.0f, h);
    float midBulge = saturate(1.0f - abs(h - 0.45f) * 2.2f);
    float vertical = baseRamp * topFade * (0.62f + 0.38f * midBulge);
    if (vertical <= 0.0f) {
        return 0.0f;
    }

    // Low-frequency weather/coverage field for broad regional breakup.
    float3 weatherP = float3(p.x, (p.y - cloudBase) * 0.35f, p.z) * weatherScale;
    float weather = perlinFbm(weatherP + float3(31.7f, -12.3f, 5.9f));
    float coverageThreshold = 1.0f - coverage;
    float coverageMask = smoothstep(coverageThreshold - 0.12f, coverageThreshold + 0.12f, weather);
    if (coverageMask <= 0.0f) {
        return 0.0f;
    }

    float3 pLocal = p - cloudCenter;

    // Curl warp for turbulent, non-spherical silhouette distortion.
    float3 warp = curlNoise(pLocal * 0.22f + float3(7.1f, -4.6f, 2.8f));
    pLocal += warpStrength * 1.45f * warp;

    // Multi-lobe macro scaffold to avoid oval symmetry.
    float lobeA = 1.0f - dot(pLocal / float3(4.9f, 2.5f, 3.7f), pLocal / float3(4.9f, 2.5f, 3.7f));
    float3 pb = pLocal - float3(2.3f, -0.1f, 0.8f);
    float lobeB = 1.0f - dot(pb / float3(2.8f, 1.7f, 2.3f), pb / float3(2.8f, 1.7f, 2.3f));
    float3 pc = pLocal - float3(-2.6f, 0.6f, -1.2f);
    float lobeC = 1.0f - dot(pc / float3(3.0f, 1.9f, 2.5f), pc / float3(3.0f, 1.9f, 2.5f));

    float macro = max(max(lobeA, lobeB), lobeC);
    if (macro <= 0.0f) {
        return 0.0f;
    }
    macro = smoothstep(0.0f, 1.0f, macro);
    macro *= vertical * coverageMask;

    // Perlin + Worley(F1/F2) stack for believable cloud cell structure.
    float perlin = perlinFbm((pLocal + float3(12.3f, 7.1f, -4.7f)) * macroScale);
    float2 wf = worleyF1F2((pLocal + float3(-5.3f, 3.8f, 9.2f)) * detailScale);
    float f1 = saturate(wf.x / 1.35f);
    float cellInterior = 1.0f - f1;
    float cellRidge = saturate((wf.y - wf.x) * 2.8f);

    float clump = smoothstep(0.28f, lerp(0.90f, 0.66f, saturate(chunkiness * 0.5f)), perlin);
    float detailBlend = saturate(0.35f + 0.42f * chunkiness);
    float detail = lerp(cellInterior, clump, detailBlend);

    float skirt = smoothstep(0.0f, 0.22f, macro) * (1.0f - smoothstep(0.74f, 1.0f, macro));
    float puffy = 0.48f + 0.24f * saturate(chunkiness);
    float density = macro * (puffy + (1.0f - puffy) * detail) + (0.10f + 0.18f * saturate(chunkiness)) * skirt * clump;

    // High-frequency erosion driven by Worley ridge/cellular contrast.
    float erosion = erosionStrength * lerp((1.0f - perlin) * 0.25f, cellRidge, 0.75f);
    density -= erosion * (0.35f + 0.65f * (1.0f - macro));

    return max(density - cutoff, 0.0f);
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

float3 makeOrthoX(float3 n) {
    return (abs(n.z) < 0.999f) ? normalize(cross(float3(0.0f, 0.0f, 1.0f), n)) : normalize(cross(float3(0.0f, 1.0f, 0.0f), n));
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

bool sampleDeltaTrackingEvent(
    float3 rayOrigin,
    float3 rayDir,
    float tMin,
    float tMax,
    float sigmaMajorant,
    float densityScale,
    inout uint rng,
    out float3 outPosition,
    out float outTr,
    out float outSigmaS
) {
    outPosition = 0.0f;
    outTr = 1.0f;
    outSigmaS = 0.0f;

    float t = tMin;
    [loop]
    for (uint i = 0u; i < 180u; ++i) {
        float u = max(rand01(rng), 1e-6f);
        t += -log(u) / sigmaMajorant;
        if (t >= tMax) {
            return false;
        }

        float3 x = rayOrigin + rayDir * t;
        float sigmaS = densityField(x) * densityScale * 1.5f;
        float accept = saturate(sigmaS / sigmaMajorant);

        if (rand01(rng) < accept) {
            outPosition = x;
            outSigmaS = sigmaS;
            return true;
        }

        outTr *= max(1.0f - accept, 0.0f);
        if (outTr < 1e-4f) {
            return false;
        }
    }

    return false;
}

float estimateSunTransmittance(float3 p, float3 sunDir, float sigmaMajorant, float densityScale, inout uint rng) {
    float tMin = 0.0f;
    float tMax = 0.0f;
    if (!intersectAabb(p + sunDir * 0.02f, sunDir, kCloudAabbMin, kCloudAabbMax, tMin, tMax)) {
        return 1.0f;
    }

    float tr = 1.0f;
    float t = tMin;
    [loop]
    for (uint i = 0; i < 128u; ++i) {
        float u = max(rand01(rng), 1e-6f);
        t += -log(u) / sigmaMajorant;
        if (t >= tMax) {
            break;
        }

        float3 x = p + sunDir * t;
        float sigmaT = densityField(x) * densityScale * 1.5f;
        tr *= max(1.0f - (sigmaT / sigmaMajorant), 0.0f);
        if (tr < 1e-3f) {
            return 0.0f;
        }
    }

    return tr;
}

float3 skyColor(float3 dir) {
    float t = saturate(0.5f + 0.5f * dir.y);
    return lerp(float3(0.62f, 0.72f, 0.85f), float3(0.28f, 0.46f, 0.72f), t);
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
    uint rngState = gRngState[pixel];
    if (rngState == 0u || gPush.frameParams.w > 0.5f) {
        rngState = hash3(uint3(pixel, 1u));
    }

    float2 jitter = float2(rand01(rngState), rand01(rngState));
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
    float3 sunRadiance = float3(
        gPush.sunDirectionIntensity.w,
        gPush.sunDirectionIntensity.w,
        gPush.sunDirectionIntensity.w);

    float densityScale = max(gPush.mediumParams.x, 0.01f);
    float g = clamp(gPush.mediumParams.y, 0.0f, 0.95f);
    float3 cloudAlbedo = float3(gPush.mediumParams.z, gPush.mediumParams.z, gPush.mediumParams.z);
    float brightnessBoost = max(gPush.cloudLightParams.x, 0.1f);
    float ambientLift = max(gPush.cloudLightParams.y, 0.0f);
    uint maxBounces = (uint)clamp(gPush.cloudLightParams.z, 1.0f, (float)kMaxBouncesHardLimit);
    float sigmaMajorant = max(densityScale * (1.85f + 0.65f * saturate(gPush.cloudShapeParams.w)), 0.05f);

    float3 L = 0.0f;
    float3 throughput = 1.0f;

    [loop]
    for (uint bounce = 0u; bounce < maxBounces; ++bounce) {
        float tMin = 0.0f;
        float tMax = 0.0f;
        if (!intersectAabb(rayOrigin, rayDir, kCloudAabbMin, kCloudAabbMax, tMin, tMax)) {
            L += throughput * skyColor(rayDir);
            break;
        }

        float3 scatterPos = 0.0f;
        float trSegment = 1.0f;
        float sigmaS = 0.0f;
        bool hit = sampleDeltaTrackingEvent(
            rayOrigin,
            rayDir,
            tMin,
            tMax,
            sigmaMajorant,
            densityScale,
            rngState,
            scatterPos,
            trSegment,
            sigmaS);

        if (!hit) {
            L += throughput * trSegment * skyColor(rayDir);
            break;
        }

        throughput *= trSegment;

        float3 wo = -rayDir;
        float trSun = estimateSunTransmittance(scatterPos, sunDir, sigmaMajorant, densityScale, rngState);
        float phaseSun = henyeyGreenstein(dot(wo, sunDir), g);
        L += throughput * cloudAlbedo * brightnessBoost * trSun * sigmaS * phaseSun * sunRadiance;

        float ambientBase = 0.10f + 0.20f * ambientLift;
        float3 ambientSky = ambientLift * skyColor(float3(0.0f, 1.0f, 0.0f)) + float3(ambientBase, ambientBase, ambientBase);
        L += throughput * cloudAlbedo * brightnessBoost * sigmaS * (1.0f / (4.0f * kPi)) * ambientSky;

        if (bounce + 1u >= maxBounces) {
            break;
        }

        float3 wi = sampleHenyeyGreenstein(wo, g, rngState);
        rayOrigin = scatterPos + wi * 0.02f;
        rayDir = wi;
        throughput *= lerp(float3(1.0f, 1.0f, 1.0f), cloudAlbedo, 0.35f);

        if (bounce >= 1u) {
            const float rrSurvive = 0.8f;
            if (rand01(rngState) > rrSurvive) {
                break;
            }
            throughput /= rrSurvive;
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
