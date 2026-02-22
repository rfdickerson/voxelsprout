static const float kPi = 3.1415926535f;
static const float3 kCloudAabbMin = float3(-10.0f, 0.0f, -10.0f);
static const float3 kCloudAabbMax = float3(10.0f, 10.0f, 10.0f);
static const uint kViewMarchSteps = 32u;

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
    float4 cloudLightParams; // x = brightness boost, y = ambient lift, z = reserved, w = debug sun transmittance
    float4 frameParams;  // x = width, y = height, z = frame index, w = reset flag
    float4 prevCameraPositionFov;
    float4 prevCameraForward;
    float4 prevCameraRight;
    float4 prevCameraUp;
    float4 historyParams; // x = use history, y = blend weight, z = clamp strength
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
[[vk::binding(5, 0)]]
Texture3D<float> gSunTransmittance;
[[vk::binding(6, 0)]]
Texture3D<float4> gMultiScatterField;
[[vk::binding(7, 0)]]
Texture3D<float> gDensityVolume;
[[vk::binding(8, 0)]]
Texture2D<float4> gPrevAccumulation;
[[vk::binding(9, 0)]]
Texture3D<float> gDensityEdgeVolume;

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

float sampleSunTransmittance(float3 p) {
    float3 uvw = saturate((p - kCloudAabbMin) / (kCloudAabbMax - kCloudAabbMin));
    return gSunTransmittance.SampleLevel(gVolumeSampler, uvw, 0.0f).r;
}

float3 sampleMultiScatter(float3 p) {
    float3 uvw = saturate((p - kCloudAabbMin) / (kCloudAabbMax - kCloudAabbMin));
    return gMultiScatterField.SampleLevel(gVolumeSampler, uvw, 0.0f).xyz;
}

float sampleDensity(float3 p) {
    float3 uvw = saturate((p - kCloudAabbMin) / (kCloudAabbMax - kCloudAabbMin));
    float densityScale = max(gPush.mediumParams.x, 0.01f);
    return gDensityVolume.SampleLevel(gVolumeSampler, uvw, 0.0f) * densityScale;
}

float sampleDensityEdge(float3 p) {
    float3 uvw = saturate((p - kCloudAabbMin) / (kCloudAabbMax - kCloudAabbMin));
    return gDensityEdgeVolume.SampleLevel(gVolumeSampler, uvw, 0.0f);
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

void projectSkySH1(
    float3 sunDir,
    float sunIntensity,
    out float3 c0,
    out float3 cx,
    out float3 cy,
    out float3 cz
) {
    const float3 up = skyColor(float3(0.0f, 1.0f, 0.0f), sunDir, sunIntensity);
    const float3 down = skyColor(float3(0.0f, -1.0f, 0.0f), sunDir, sunIntensity);
    const float3 right = skyColor(float3(1.0f, 0.0f, 0.0f), sunDir, sunIntensity);
    const float3 left = skyColor(float3(-1.0f, 0.0f, 0.0f), sunDir, sunIntensity);
    const float3 front = skyColor(float3(0.0f, 0.0f, 1.0f), sunDir, sunIntensity);
    const float3 back = skyColor(float3(0.0f, 0.0f, -1.0f), sunDir, sunIntensity);

    c0 = (up + down + right + left + front + back) * (1.0f / 6.0f);
    cx = 0.5f * (right - left);
    cy = 0.5f * (up - down);
    cz = 0.5f * (front - back);
}

float3 evalSkySH1(float3 dir, float3 c0, float3 cx, float3 cy, float3 cz) {
    return c0 + cx * dir.x + cy * dir.y + cz * dir.z;
}

float3 integrateViewMarchFinal(
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
    float3 ambientSkyBottom,
    float3 ambientSkyTop,
    bool debugSunTransmittance,
    inout uint rngState,
    out float3 outHistoryAnchor,
    out bool outHasHistoryAnchor
) {
    outHistoryAnchor = 0.0f;
    outHasHistoryAnchor = false;

    float tMin = 0.0f;
    float tMax = 0.0f;
    if (!intersectAabb(rayOrigin, rayDir, kCloudAabbMin, kCloudAabbMax, tMin, tMax)) {
        return skyColor(rayDir, sunDir, sunIntensity);
    }

    float marchDistance = tMax - tMin;
    if (marchDistance <= 0.0f) {
        return skyColor(rayDir, sunDir, sunIntensity);
    }

    const float ds = marchDistance / float(kViewMarchSteps);
    if (ds <= 0.0f) {
        return skyColor(rayDir, sunDir, sunIntensity);
    }

    float3 Lss = 0.0f;
    float3 Lms = 0.0f;
    float Tview = 1.0f;
    const float stepJitter = rand01(rngState);
    const float cosTheta = clamp(dot(-sunDir, -rayDir), -1.0f, 1.0f);
    const float phase = dualHenyeyGreenstein(cosTheta, g);
    const float ambientStrength = lerp(0.05f, 0.6f, saturate(ambientLift));
    const float cloudBase = gPush.cloudProfileParams.z;
    const float cloudTop = max(gPush.cloudProfileParams.w, cloudBase + 0.25f);
    const float cloudHeightSpan = max(cloudTop - cloudBase, 0.25f);
    const float3 groundAlbedoColor = float3(0.18f, 0.22f, 0.18f);
    const float3 earthBounce = sunRadiance * max(sunDir.y, 0.0f) * groundAlbedoColor;

    [loop]
    for (uint i = 0u; i < kViewMarchSteps; ++i) {
        float t = tMin + (float(i) + stepJitter) * ds;
        float3 x = rayOrigin + rayDir * t;
        float rho = sampleDensity(x);
        if (rho <= 0.0f) {
            continue;
        }
        if (!outHasHistoryAnchor) {
            outHistoryAnchor = x;
            outHasHistoryAnchor = true;
        }

        float Tsun = sampleSunTransmittance(x);
        if (debugSunTransmittance) {
            return float3(Tsun, Tsun, Tsun);
        }

        float edge = sampleDensityEdge(x);
        float edgeBoost = lerp(1.0f, 1.35f, saturate(edge));
        float3 E = sampleMultiScatter(x);
        float powderFromDensity = 1.0f - exp(-rho * 1.2f);
        float powder = lerp(powderFromDensity, saturate(edge), 0.65f);
        float3 direct = rho * Tsun * phase * sunRadiance * cloudAlbedo * (0.25f + 0.75f * powder) * edgeBoost;
        float rhoMask = 0.2f + 0.8f * saturate(rho);
        float3 indirect = E * rhoMask * (0.65f + 0.35f * powder);
        float hFrac = saturate((x.y - cloudBase) / cloudHeightSpan);
        float3 ambientSky = max(lerp(ambientSkyBottom, ambientSkyTop, hFrac), 0.0f);
        float3 localAmbient = max(lerp(earthBounce, ambientSky, hFrac), 0.0f);
        float3 ambientFloor = sunRadiance * 0.22f;
        float3 ambientColor = max(localAmbient, ambientFloor);
        float3 ambient = rho * ambientColor * cloudAlbedo * ambientStrength * (1.0f / (4.0f * kPi));
        Lss += Tview * direct * ds * max(brightnessBoost, 0.0f);
        Lms += Tview * (indirect + ambient) * ds;

        // Use explicit view transmittance update for stable deterministic marching.
        Tview *= exp(-rho * 1.5f * ds);
        if (Tview < 1.0e-4f) {
            break;
        }
    }

    return Lss + Lms + (Tview * skyColor(rayDir, sunDir, sunIntensity));
}

bool reprojectToPrevPixel(float3 worldPoint, uint width, uint height, out int2 prevPixel) {
    float3 toPoint = worldPoint - gPush.prevCameraPositionFov.xyz;
    float3 prevForward = gPush.prevCameraForward.xyz;
    float3 prevRight = gPush.prevCameraRight.xyz;
    float3 prevUp = gPush.prevCameraUp.xyz;

    float z = dot(toPoint, prevForward);
    if (z <= 0.01f) {
        prevPixel = int2(0, 0);
        return false;
    }

    float x = dot(toPoint, prevRight);
    float y = dot(toPoint, prevUp);
    float aspect = float(width) / float(height);
    float tanHalfFov = tan(radians(gPush.prevCameraPositionFov.w) * 0.5f);
    if (tanHalfFov <= 1e-5f) {
        prevPixel = int2(0, 0);
        return false;
    }

    float ndcX = x / (z * tanHalfFov * aspect);
    float ndcY = y / (z * tanHalfFov);
    float2 uv;
    uv.x = ndcX * 0.5f + 0.5f;
    uv.y = 0.5f - ndcY * 0.5f;
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        prevPixel = int2(0, 0);
        return false;
    }

    prevPixel = int2(uv * float2(width, height));
    prevPixel = clamp(prevPixel, int2(0, 0), int2(int(width) - 1, int(height) - 1));
    return true;
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
    
    // 1. Differentiate Sky vs Ground
    bool isGround = v.y < 0.0f;
    // If looking at the ground, mirror the view vector upwards to calculate the sky haze above it
    float3 viewDir = isGround ? float3(v.x, -v.y, v.z) : v; 
    
    float mu = clamp(dot(viewDir, sunDir), -1.0f, 1.0f);
    float mView = airMass(max(viewDir.y, 0.001f));
    float mSun = airMass(max(sunDir.y, 0.001f));

    float3 betaR = float3(5.8e-6f, 13.5e-6f, 33.1e-6f) * 11000.0f;
    float3 betaM = float3(21.0e-6f, 21.0e-6f, 21.0e-6f) * 2000.0f;

    float3 tau = betaR * mView + betaM * mView;
    float3 extinction = exp(-tau);

    float rayleigh = phaseRayleigh(mu);
    float mie = phaseMie(mu, 0.82f);
    float3 inscatter = (betaR * rayleigh + betaM * mie) * exp(-(betaR + betaM) * mSun);
    inscatter *= (1.0f - extinction) / max(betaR + betaM, float3(1e-4f, 1e-4f, 1e-4f));

    // 2. Add Horizon Haze (Multiple scattering approximation for the sky)
    // This adds a bright, realistic atmospheric glow at the horizon line
    float horizonBlend = exp(-viewDir.y * 4.0f); 
    float3 horizonGlow = float3(0.5f, 0.65f, 0.8f) * 0.025f; // Bright sky-blue/cyan haze
    inscatter += horizonGlow * horizonBlend;

    float sunDisk = smoothstep(0.9996f, 0.99995f, mu);
    float sunHalo = pow(saturate(mu), 64.0f);
    float twilight = smoothstep(-0.18f, 0.08f, sunDir.y);

    float3 sunColor = float3(1.0f, 0.99f, 0.98f);
    float3 sunTerm = sunColor * (sunDisk * 24.0f + sunHalo * 1.8f);
    
    float3 finalSky = (inscatter * twilight + sunTerm) * max(sunIntensity, 0.01f);

    // 3. Paint the Ground
    if (isGround) {
        float3 groundAlbedo = float3(0.18f, 0.22f, 0.18f); // Earthy green/brown
        float3 groundLit = groundAlbedo * max(sunDir.y, 0.0f) * sunIntensity * 0.15f;
        
        // Fog the ground heavily so it blends seamlessly into the horizon haze
        float groundFog = exp(v.y * 12.0f); // v.y is negative here, so this approaches 0 looking down
        
        // Find what the sky color is exactly AT the horizon to blend with
        float3 baseHorizonColor = (inscatter * twilight) * max(sunIntensity, 0.01f);
        
        finalSky = lerp(groundLit, baseHorizonColor, groundFog);
    }

    return finalSky;
}

float3 skyAmbientLighting(float3 dir, float3 sunDir, float sunIntensity) {
    float3 shC0;
    float3 shCx;
    float3 shCy;
    float3 shCz;
    projectSkySH1(sunDir, sunIntensity, shC0, shCx, shCy, shCz);
    return max(evalSkySH1(normalize(dir), shC0, shCx, shCy, shCz), 0.0f);
}

float3 cloudTwilightTint(float3 viewDir, float3 sunDir, float sunIntensity) {
    const float sunWarm = pow(1.0f - clamp(sunDir.y, 0.0f, 1.0f), 1.2f);

    const float3 viewSky = max(skyColor(-viewDir, sunDir, sunIntensity), 0.0f);
    const float3 rimDir = normalize(-viewDir + (-sunDir * 0.55f));
    const float3 rimSky = max(skyColor(rimDir, sunDir, sunIntensity), 0.0f);
    const float3 glowDir = normalize(0.75f * (-viewDir) - 0.25f * sunDir);
    const float3 glowSky = max(skyColor(glowDir, sunDir, sunIntensity), 0.0f);

    const float viewBlend = 0.45f + 0.55f * sunWarm;
    float3 tinted = lerp(viewSky, rimSky, viewBlend);
    return lerp(tinted, glowSky, 0.35f * sunWarm);
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
    float3 sunRadiance = float3(1.0f, 1.0f, 0.98f) * sunIntensity;

    float densityScale = max(gPush.mediumParams.x, 0.01f);
    float g = clamp(gPush.mediumParams.y, 0.0f, 0.90f);
    float3 cloudAlbedo = float3(gPush.mediumParams.z, gPush.mediumParams.z, gPush.mediumParams.z);
    float brightnessBoost = max(gPush.cloudLightParams.x, 0.0f);
    float ambientLift = max(gPush.cloudLightParams.y, 0.0f);
    bool debugSunTransmittance = gPush.cloudLightParams.w > 0.5f;

    float3 shC0;
    float3 shCx;
    float3 shCy;
    float3 shCz;
    projectSkySH1(sunDir, sunIntensity, shC0, shCx, shCy, shCz);
    float3 ambientSkyBottom = max(evalSkySH1(float3(0.0f, -1.0f, 0.0f), shC0, shCx, shCy, shCz), 0.0f);
    float3 ambientSkyTop = max(evalSkySH1(float3(0.0f, 1.0f, 0.0f), shC0, shCx, shCy, shCz), 0.0f);

    // Deep shadows should retain a deep blue/cyan tint from sky scattering.
    const float3 skyBlueTint = float3(0.55f, 0.75f, 1.25f);
    ambientSkyTop *= skyBlueTint;
    ambientSkyBottom *= lerp(skyBlueTint, float3(1.0f, 1.0f, 1.0f), 0.5f);

    float3 historyAnchor = 0.0f;
    bool hasHistoryAnchor = false;
    float3 L = integrateViewMarchFinal(
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
        ambientSkyBottom,
        ambientSkyTop,
        debugSunTransmittance,
        rngState,
        historyAnchor,
        hasHistoryAnchor);

    if (gPush.historyParams.x > 0.5f && hasHistoryAnchor) {
        int2 prevPixel;
        if (reprojectToPrevPixel(historyAnchor, width, height, prevPixel)) {
            float3 history = gPrevAccumulation.Load(int3(prevPixel, 0)).xyz;
            float clampStrength = max(gPush.historyParams.z, 1.0f);
            const float3 lumaWeights = float3(0.2126f, 0.7152f, 0.0722f);
            float l = dot(L, lumaWeights);
            float lh = dot(history, lumaWeights);
            float ratio = lh / max(l, 1.0e-4f);
            if (ratio > clampStrength) {
                history *= (clampStrength / ratio);
            }
            float blend = saturate(gPush.historyParams.y);
            L = lerp(L, history, blend);
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
