static const float kPi = 3.1415926535f;
static const float3 kCloudAabbMin = float3(-10.0f, 0.0f, -10.0f);
static const float3 kCloudAabbMax = float3(10.0f, 10.0f, 10.0f);
static const uint kViewMarchStepsMax = 256u;

struct CameraPush {
    float4 cameraPositionFov;
    float4 cameraForward;
    float4 cameraRight;
    float4 cameraUp;
    float4 sunDirectionIntensity;
    float4 mediumParams; // x = density scale, y = g, z = albedo, w = extinction coeff
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
    return gDensityVolume.SampleLevel(gVolumeSampler, uvw, 0.0f);
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
    float g,
    float3 cloudAlbedo,
    float brightnessBoost,
    float ambientLift,
    float3 ambientSkyBottom,
    float3 ambientSkyTop,
    bool debugSunTransmittance,
    float rayJitterOffset
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

    const float targetDs = (kCloudAabbMax.x - kCloudAabbMin.x) / 256.0f;
    if (targetDs <= 0.0f) {
        return skyColor(rayDir, sunDir, sunIntensity);
    }
    uint steps = (uint)ceil(marchDistance / targetDs);
    steps = min(steps, kViewMarchStepsMax);
    const float ds = marchDistance / float(max(steps, 1u));

    float3 Lss = 0.0f;
    float3 Lms = 0.0f;
    float Tview = 1.0f;
    const float densityScale = max(gPush.mediumParams.x, 0.01f);
    const float erosionStrength = max(gPush.cloudWarpParams.y, 0.0f);
    const float cosTheta = clamp(dot(-sunDir, -rayDir), -1.0f, 1.0f);
    const float phase = dualHenyeyGreenstein(cosTheta, g);
    const float ambientStrength = lerp(0.05f, 0.6f, saturate(ambientLift));
    const float extinctionCoeff = max(gPush.mediumParams.w, 0.01f);
    const float cloudBase = gPush.cloudProfileParams.z;
    const float cloudTop = max(gPush.cloudProfileParams.w, cloudBase + 0.25f);
    const float cloudHeightSpan = max(cloudTop - cloudBase, 0.25f);
    const float3 groundAlbedoColor = float3(0.18f, 0.22f, 0.18f);
    const float3 earthBounce = sunRadiance * max(sunDir.y, 0.0f) * groundAlbedoColor;

    float startJitter = saturate(rayJitterOffset) * ds;

    [loop]
    for (uint i = 0u; i < steps; ++i) {
        float t = tMin + startJitter + float(i) * ds;
        if (t > tMax) {
            break;
        }
        float3 x = rayOrigin + rayDir * t;
        float bakedDensity = sampleDensity(x);
        if (bakedDensity <= 0.01f) {
            continue;
        }
        float detailScale = max(gPush.cloudShapeParams.y, 0.01f);
        float3 detailUvw = x * detailScale * 0.5f;
        float4 detailTex = gCloudNoiseVolume.SampleLevel(gVolumeSampler, detailUvw, 0.0f);
        float worleyFbm = detailTex.g * 0.625f + detailTex.b * 0.25f + detailTex.a * 0.125f;
        worleyFbm += (detailTex.r - 0.5f) * 0.05f;

        float erosionModifier = (1.0f - worleyFbm) * erosionStrength;
        float erodedDensity = bakedDensity - erosionModifier;
        float sharpDensity = saturate(erodedDensity * 10.0f) * bakedDensity;
        float rho = sharpDensity * densityScale;
        if (rho <= 0.0f) {
            continue;
        }
        float sigmaT = rho * extinctionCoeff;

        float Tsun = sampleSunTransmittance(x);
        if (debugSunTransmittance) {
            return float3(Tsun, Tsun, Tsun);
        }

        float powder = 1.0f - exp(-sigmaT * 2.0f);
        float3 sigmaS = sigmaT * cloudAlbedo;
        float3 directLight = Tsun * phase * sunRadiance * (0.35f + 0.65f * powder);
        float hFrac = saturate((x.y - cloudBase) / cloudHeightSpan);
        float3 skyFill = lerp(ambientSkyBottom, ambientSkyTop, hFrac);
        float3 localAmbient = max(lerp(earthBounce, skyFill, hFrac), 0.0f);
        float3 E = sampleMultiScatter(x) * 2.5f;
        float thickness = saturate(1.0f - Tview);
        thickness = smoothstep(0.0f, 0.65f, thickness);
        float backlit = saturate(1.0f - Tsun);
        float3 shadowBlue = ambientSkyTop * 0.35f;
        float3 indirectLight = (E * thickness) + (shadowBlue * backlit * 0.3f) + (localAmbient * ambientStrength * 0.5f);
        Lss += Tview * sigmaS * directLight * ds * max(brightnessBoost, 1.0f);
        Lms += Tview * sigmaS * indirectLight * ds;

        // Use explicit view transmittance update for stable deterministic marching.
        Tview *= exp(-sigmaT * ds);
        if (Tview < 1.0e-4f) {
            break;
        }
    }

    const float fogDensity = 0.012f;
    float fogExtinction = exp(-tMin * fogDensity);
    float3 fogColor = lerp(ambientSkyBottom, ambientSkyTop, clamp(rayDir.y * 0.5f + 0.5f, 0.0f, 1.0f));
    float3 finalCloudLight = Lss + Lms;
    finalCloudLight = lerp(fogColor * (1.0f - Tview), finalCloudLight, fogExtinction);

    return finalCloudLight + (Tview * skyColor(rayDir, sunDir, sunIntensity));
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

float3 dynamicSunColor(float sunElevation) {
    float3 sunNoon = float3(1.0f, 0.99f, 0.98f);
    float3 sunSet = float3(1.0f, 0.22f, 0.03f);
    // Keep warm tones active through low/mid sun elevations.
    float t = smoothstep(-0.10f, 0.45f, sunElevation);
    float3 color = lerp(sunSet, sunNoon, t);
    // Extra warm bias close to the horizon so clouds do not read as white.
    float lowSun = 1.0f - smoothstep(-0.02f, 0.30f, sunElevation);
    float3 warmTint = lerp(float3(1.0f, 1.0f, 1.0f), float3(1.15f, 0.82f, 0.55f), lowSun);
    return color * warmTint;
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

    // Rayleigh (blue sky), Mie (haze), and ozone (green/yellow absorption).
    float3 betaR = float3(5.8e-6f, 13.5e-6f, 33.1e-6f) * 11000.0f;
    float3 betaM = float3(21.0e-6f, 21.0e-6f, 21.0e-6f) * 2000.0f;
    float3 betaO = float3(3.426e-7f, 8.298e-7f, 0.356e-7f) * 3000.0f;

    float3 tau = betaR * mView + betaM * mView + betaO * mView;
    float3 extinction = exp(-tau);

    float rayleigh = phaseRayleigh(mu);
    float mie = phaseMie(mu, 0.82f);
    float3 inscatter = (betaR * rayleigh + betaM * mie) * exp(-(betaR + betaM + betaO) * mSun);
    inscatter *= (1.0f - extinction) / max(betaR + betaM, float3(1e-4f, 1e-4f, 1e-4f));

    // Shift haze from magenta at low sun to blue during daytime.
    float horizonBlend = exp(-viewDir.y * 5.0f);
    float3 dayHaze = float3(0.5f, 0.65f, 0.8f);
    float3 sunriseHaze = float3(1.2f, 0.35f, 0.65f);
    float3 horizonColor = lerp(sunriseHaze, dayHaze, smoothstep(-0.1f, 0.15f, sunDir.y));
    inscatter += horizonColor * horizonBlend * 0.03f;

    float sunDisk = smoothstep(0.9997f, 0.99995f, mu);
    float sunHalo = pow(saturate(mu), 64.0f);
    float twilight = smoothstep(-0.18f, 0.08f, sunDir.y);

    float3 sunColor = dynamicSunColor(sunDir.y);
    float3 sunTerm = sunColor * (sunDisk * 24.0f + sunHalo * 1.8f);
    
    float3 finalSky = (inscatter * twilight + sunTerm) * max(sunIntensity, 0.01f);

    // 3. Paint the Ground
    if (isGround) {
        float3 groundAlbedo = float3(0.18f, 0.22f, 0.18f); // Earthy green/brown
        float3 groundLit = groundAlbedo * max(sunDir.y, 0.0f) * sunIntensity * sunColor * 0.15f;
        
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

    float2 jitter = blueNoise.xy;
    float rayJitterOffset = blueNoise.r;
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
    float3 sunRadiance = dynamicSunColor(sunDir.y) * sunIntensity;

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

    // Deep shadows inherit atmospheric color: purple at twilight, blue at daytime.
    float3 twilightPurple = float3(0.9f, 0.35f, 0.85f);
    float3 dayBlue = float3(0.55f, 0.75f, 1.25f);
    float3 currentAmbientTint = lerp(twilightPurple, dayBlue, smoothstep(-0.1f, 0.2f, sunDir.y));
    ambientSkyTop *= currentAmbientTint;
    ambientSkyBottom *= lerp(currentAmbientTint, float3(1.0f, 1.0f, 1.0f), 0.5f);

    float3 L = integrateViewMarchFinal(
        rayOrigin,
        rayDir,
        sunDir,
        sunRadiance,
        sunIntensity,
        g,
        cloudAlbedo,
        brightnessBoost,
        ambientLift,
        ambientSkyBottom,
        ambientSkyTop,
        debugSunTransmittance,
        rayJitterOffset);

    gAccumulation[pixel] = float4(L, 1.0f);

}
