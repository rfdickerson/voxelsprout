import camera_uniform;
import chunk_push_constants;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

[[vk::binding(6, 0)]]
Sampler2D<float4> normalDepthTexture;
[[vk::binding(5, 0)]]
Sampler2D<float4> opaqueSceneColorTexture;

#ifdef ODAI_RT_REFLECTIONS
[[vk::binding(12, 0)]]
RaytracingAccelerationStructure reflectionScene;
#endif

static const float kPi = 3.14159265359;
static const float kAirIor = 1.0;
static const float kWaterIor = 1.333;
static const float kAirToWaterEta = kAirIor / kWaterIor;
static const uint kBindlessTextureIndexWaterNormal = 8u;
static const float3 kWaterShallowTint = float3(0.025, 0.58, 0.52);
static const float3 kWaterDeepTint = float3(0.005, 0.16, 0.19);
static const float3 kWaterAbsorption = float3(0.070, 0.028, 0.010);

struct FragmentInput
{
    float4 inPosition : SV_Position;
    float3 inWorldPosition : TEXCOORD0;
    float2 inUv : TEXCOORD1;
};

struct SceneWaterSample
{
    float valid;
    float thickness;
    float contact;
    float3 bottomTint;
};

float3 proceduralSkyNoSunDisk(float3 direction)
{
    const float3 dir = normalize(direction);
    const float3 toSun = -normalize(camera.sunDirectionIntensity.xyz);
    const float rayleighStrength = max(camera.skyConfig0.x, 0.01);
    const float mieStrength = max(camera.skyConfig0.y, 0.01);
    const float skyExposure = max(camera.skyConfig0.w, 0.01);
    const float sunAltitude = clamp(toSun.y, -1.0, 1.0);
    const float dayFactor = smoothstep(0.05, 0.65, sunAltitude);
    const float twilightFactor = 1.0 - dayFactor;
    const float horizonBand = clamp(1.0 - (abs(sunAltitude) / 0.35), 0.0, 1.0);
    const float warmAmount = twilightFactor * pow(horizonBand, 1.2);
    const float pinkAmount = warmAmount * clamp((0.10 - sunAltitude) / 0.30, 0.0, 1.0);

    const float horizonT = clamp((dir.y * 0.5) + 0.5, 0.0, 1.0);
    const float skyT = pow(horizonT, 0.35);

    float3 horizonRayleigh = lerp(float3(0.54, 0.70, 1.00), float3(0.74, 0.44, 0.52), warmAmount);
    float3 horizonMie = lerp(float3(1.00, 0.74, 0.42), float3(1.18, 0.54, 0.30), warmAmount);
    horizonRayleigh = lerp(horizonRayleigh, float3(0.70, 0.36, 0.68), pinkAmount * 0.70);
    horizonMie = lerp(horizonMie, float3(1.08, 0.46, 0.72), pinkAmount * 0.85);

    const float zenithWarm = twilightFactor * 0.58;
    const float3 zenithRayleigh = lerp(float3(0.06, 0.24, 0.54), float3(0.16, 0.12, 0.30), zenithWarm);
    const float3 zenithMie = lerp(float3(0.22, 0.20, 0.15), float3(0.30, 0.18, 0.24), zenithWarm);

    const float3 horizonColor =
        (horizonRayleigh * rayleighStrength) +
        (horizonMie * (mieStrength * 0.58));
    const float3 zenithColor =
        (zenithRayleigh * rayleighStrength) +
        (zenithMie * (mieStrength * 0.25));
    const float3 sky = lerp(horizonColor, zenithColor, skyT);

    const float3 groundColor = float3(0.05, 0.06, 0.07);
    const float belowHorizon = clamp(-dir.y, 0.0, 1.0);
    const float3 horizonGroundColor = horizonColor * 0.32;
    const float3 ground = lerp(horizonGroundColor, groundColor, pow(belowHorizon, 0.55));
    const float skyWeight = smoothstep(-0.18, 0.02, dir.y);
    const float3 color = lerp(ground, sky, skyWeight);
    return max(color * skyExposure, float3(0.0, 0.0, 0.0));
}

float3 proceduralWaterReflectionSky(float3 direction)
{
    const float3 dir = normalize(direction);
    const float3 toSun = -normalize(camera.sunDirectionIntensity.xyz);
    const float sunDot = saturate(dot(dir, toSun));
    const float sunDisk = pow(sunDot, 1200.0);
    const float sunHalo = pow(sunDot, 42.0);
    const float3 baseSky = proceduralSkyNoSunDisk(dir);
    return baseSky + (camera.sunColorShadow.xyz * ((sunHalo * 0.032) + (sunDisk * 0.24)));
}

float3 applyAtmosphericFog(float3 litColor, float3 worldPosition, float viewDepth, float3 sunDirection)
{
    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float3 viewDir = normalize(worldPosition - cameraWorld);
    const float3 skyColor = proceduralSkyNoSunDisk(viewDir);

    const float3 sigmaRayleigh = float3(0.0012, 0.0023, 0.0048) * max(camera.skyConfig0.x, 0.01);
    const float sigmaMie = 0.0038 * max(camera.skyConfig0.y, 0.01);
    const float3 extinction = sigmaRayleigh + float3(sigmaMie);

    const float sunView = max(dot(viewDir, -sunDirection), 0.0);
    const float g = clamp(camera.skyConfig0.z, 0.0, 0.98);
    const float phaseMie = (1.0 - (g * g)) / (4.0 * kPi * pow(1.0 + (g * g) - (2.0 * g * sunView), 1.5));
    const float phaseBoost = 1.0 + (phaseMie * 1.6 * sunView);
    const float3 inscatterColor = skyColor * phaseBoost;

    const float3 transmittance = exp(-extinction * max(viewDepth, 0.0));
    return (litColor * transmittance) + (inscatterColor * (float3(1.0, 1.0, 1.0) - transmittance));
}

float stableWaterNoise(float2 p)
{
    const float2 i = floor(p);
    const float2 f = frac(p);
    const float2 u = f * f * (3.0 - (2.0 * f));
    const float a = frac(sin(dot(i + float2(0.0, 0.0), float2(127.1, 311.7))) * 43758.5453123);
    const float b = frac(sin(dot(i + float2(1.0, 0.0), float2(127.1, 311.7))) * 43758.5453123);
    const float c = frac(sin(dot(i + float2(0.0, 1.0), float2(127.1, 311.7))) * 43758.5453123);
    const float d = frac(sin(dot(i + float2(1.0, 1.0), float2(127.1, 311.7))) * 43758.5453123);
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

float3 decodeViewNormal(float3 encodedNormal)
{
    return normalize((encodedNormal * 2.0) - 1.0);
}

float2 waterScreenUv(float4 screenPosition)
{
    uint texW = 0u;
    uint texH = 0u;
    normalDepthTexture.GetDimensions(texW, texH);
    const float2 fullResolutionExtent = float2(float(max(texW, 1u)), float(max(texH, 1u))) * 2.0;
    return clamp(screenPosition.xy / fullResolutionExtent, float2(0.0, 0.0), float2(1.0, 1.0));
}

float3 viewRayFromUv(float2 uv)
{
    float projX = camera.proj[0][0];
    float projY = camera.proj[1][1];
    if (abs(projX) < 1e-6) projX = (projX < 0.0) ? -1e-6 : 1e-6;
    if (abs(projY) < 1e-6) projY = (projY < 0.0) ? -1e-6 : 1e-6;

    const float2 ndc = (uv * 2.0) - 1.0;
    return normalize(float3(ndc.x / projX, ndc.y / projY, -1.0));
}

SceneWaterSample sampleSceneWater(float4 screenPosition, float3 waterWorldPosition, float3 normal)
{
    SceneWaterSample result;
    result.valid = 0.0;
    result.thickness = 18.0;
    result.contact = 0.0;
    result.bottomTint = float3(0.42, 0.36, 0.24);

    uint texW = 0u;
    uint texH = 0u;
    normalDepthTexture.GetDimensions(texW, texH);
    const float2 texelSize = 1.0 / float2(float(max(texW, 1u)), float(max(texH, 1u)));
    const float2 baseUv = waterScreenUv(screenPosition);
    const float2 refractedUv = clamp(
        baseUv + (normal.xz * texelSize * 7.0),
        float2(0.0, 0.0),
        float2(1.0, 1.0));
    const float4 sceneNormalDepth = normalDepthTexture.SampleLevel(refractedUv, 0.0);
    const float sceneViewDepth = sceneNormalDepth.w;
    if (sceneViewDepth <= 0.0001)
    {
        return result;
    }

    const float waterViewDepth = max(-mul(camera.view, float4(waterWorldPosition, 1.0)).z, 0.0);
    const float depthBehindWater = sceneViewDepth - waterViewDepth;
    if (depthBehindWater <= 0.02)
    {
        return result;
    }

    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float3 viewRay = viewRayFromUv(refractedUv);
    const float3x3 invViewRotation = transpose((float3x3)camera.view);
    const float3 worldViewDir = normalize(mul(invViewRotation, viewRay));
    const float travelDistance = sceneViewDepth / max(-viewRay.z, 1e-4);
    const float3 sceneWorldPosition = cameraWorld + (worldViewDir * travelDistance);
    const float waterColumnHeight = max(waterWorldPosition.y - sceneWorldPosition.y, 0.0);
    const float bottomBelowSurface = step(sceneWorldPosition.y, waterWorldPosition.y + 0.35);
    if (bottomBelowSurface <= 0.0)
    {
        return result;
    }

    const float3 sceneViewNormal = decodeViewNormal(sceneNormalDepth.xyz);
    const float3 sceneWorldNormal = normalize(mul(invViewRotation, sceneViewNormal));
    const float floorFactor = saturate(sceneWorldNormal.y * 1.35);
    const float sunFacing = saturate(dot(
        normalize(float3(-worldViewDir.x, 0.45, -worldViewDir.z)),
        -normalize(camera.sunDirectionIntensity.xyz)));
    const float3 sandTint = lerp(
        float3(0.44, 0.37, 0.24),
        float3(0.68, 0.58, 0.36),
        sunFacing);
    const float3 rockTint = lerp(
        float3(0.20, 0.22, 0.18),
        float3(0.36, 0.38, 0.28),
        sunFacing);
    const float3 wallTint = lerp(
        float3(0.14, 0.15, 0.13),
        float3(0.30, 0.28, 0.20),
        sunFacing);
    const float depthThickness = max(waterColumnHeight, depthBehindWater * 0.36);
    const float sedimentNoise =
        (stableWaterNoise(sceneWorldPosition.xz * 0.035) * 0.65) +
        (stableWaterNoise(sceneWorldPosition.zx * 0.083) * 0.35);

    result.valid = 1.0;
    result.thickness = clamp(depthThickness, 0.25, 120.0);
    result.contact = saturate(1.0 - smoothstep(1.5, 11.0, result.thickness));
    result.bottomTint = lerp(wallTint, lerp(sandTint, rockTint, 0.22 + (0.24 * sedimentNoise)), floorFactor);
    result.bottomTint *= 0.96 + (0.16 * sedimentNoise);
    return result;
}

struct ScreenRefractionSample
{
    float3 color;
    float weight;
    float distortionPixels;
};

ScreenRefractionSample sampleAnimatedSceneRefraction(
    float2 baseUv,
    float3 normal,
    float sceneThickness,
    float shallowFactor,
    float overheadView,
    float contactFactor,
    float3 shallowWater,
    float3 fallbackColor)
{
    ScreenRefractionSample result;
    result.color = fallbackColor;
    result.weight = 0.0;
    result.distortionPixels = 0.0;

    uint texW = 0u;
    uint texH = 0u;
    opaqueSceneColorTexture.GetDimensions(texW, texH);
    const float2 texelSize = 1.0 / float2(float(max(texW, 1u)), float(max(texH, 1u)));
    const float refractionStrength = max(camera.dofConfig2.y, 0.0);
    const float refractionDistortionPixels = max(camera.dofConfig2.z, 0.0);
    if (refractionStrength <= 0.0001)
    {
        return result;
    }

    const float depthFade = lerp(0.24, 1.0, 1.0 - smoothstep(44.0, 180.0, sceneThickness));
    const float shorelineDamping = lerp(1.0, 0.42, contactFactor);
    const float viewFade = 0.48 + (0.52 * overheadView);
    const float2 refractiveSlope = normal.xz / max(normal.y, 0.35);
    const float distortionPixels =
        refractionDistortionPixels *
        lerp(0.52, 1.0, shallowFactor) *
        depthFade *
        shorelineDamping *
        viewFade;
    result.distortionPixels = distortionPixels;
    const float2 refractedUv = clamp(
        baseUv + (refractiveSlope * texelSize * distortionPixels),
        float2(0.0, 0.0),
        float2(1.0, 1.0));
    const float3 opaqueSceneColor = opaqueSceneColorTexture.SampleLevel(refractedUv, 0.0).xyz;
    const float3 attenuation = exp(-kWaterAbsorption * max(sceneThickness, 0.0) * 0.34);
    const float3 refractedSceneColor =
        (opaqueSceneColor * attenuation * lerp(0.78, 1.04, shallowFactor)) +
        (shallowWater * (0.035 + (0.10 * shallowFactor)));
    const float sceneWeight =
        refractionStrength *
        0.56 *
        viewFade *
        depthFade *
        lerp(0.42, 0.95, shallowFactor) *
        lerp(1.0, 0.72, contactFactor);
    result.weight = saturate(sceneWeight);
    result.color = lerp(fallbackColor, refractedSceneColor, result.weight);
    return result;
}

float waterCausticPattern(float3 worldPosition, float depthDistance)
{
    const float time = camera.skyConfig1.z * max(camera.waterConfig.x, 0.01);
    const float2 p0 = (worldPosition.xz * 0.033) + float2(time * 0.030, -time * 0.021);
    const float2 p1 = (worldPosition.zx * 0.047) + float2(-time * 0.026, time * 0.018);
    const float2 p2 = (worldPosition.xz * 0.021) + float2(-time * 0.013, time * 0.016);
    const float line0 = pow(1.0 - abs(frac(p0.x + sin(p0.y) * 0.18) - 0.5) * 2.0, 8.0);
    const float line1 = pow(1.0 - abs(frac(p1.x + cos(p1.y) * 0.16) - 0.5) * 2.0, 10.0);
    const float breakup = 0.72 + (0.28 * stableWaterNoise(p2 * 2.7));
    const float depthFade = 1.0 - smoothstep(5.0, 34.0, depthDistance);
    return (line0 * 0.58 + line1 * 0.42) * breakup * depthFade;
}

float3 waterNormal(float3 worldPosition, float2 uv)
{
    const float time = camera.skyConfig1.z;
    const float animationSpeed = max(camera.waterConfig.x, 0.01);
    const float normalStrength = max(camera.waterConfig.y, 0.01);
    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float viewDistance = length(cameraWorld - worldPosition);
    const float detailFade = 1.0 - smoothstep(220.0, 1800.0, viewDistance);
    const float swell0 = sin((worldPosition.x * 0.0056) + (worldPosition.z * 0.0041) + (time * 0.092 * animationSpeed));
    const float swell1 = cos((worldPosition.x * 0.0044) - (worldPosition.z * 0.0063) - (time * 0.071 * animationSpeed));
    const float windRipple = sin((worldPosition.x * 0.018) - (worldPosition.z * 0.014) + (time * 0.32 * animationSpeed));
    const float microRipple = sin((uv.x * 6.28318 * 0.72) + (uv.y * 6.28318 * 0.58) + (time * 0.58 * animationSpeed));
    const float dx = (swell0 * 0.0140) + (swell1 * 0.0090) + (windRipple * 0.0056) + (microRipple * 0.0028 * detailFade);
    const float dz = (swell1 * 0.0120) - (windRipple * 0.0056) - (microRipple * 0.0028 * detailFade);
    const float2 worldUv = float2(worldPosition.x, worldPosition.z) * 0.00195;
    const float2 warp = float2(
        stableWaterNoise((worldUv * 19.0) + float2(time * 0.030, 0.0)),
        stableWaterNoise((worldUv.yx * 17.0) + float2(0.0, -time * 0.026))) - 0.5;
    const float2 scroll0 = float2(time * 0.030, -time * 0.021) * animationSpeed;
    const float2 scroll1 = float2(-time * 0.020, time * 0.027) * animationSpeed;
    const float2 scroll2 = float2(time * 0.014, time * 0.018) * animationSpeed;
    const float2 sampleUv0 = (worldUv * 1.07) + scroll0 + (warp * 0.022);
    const float2 sampleUv1 = (worldUv * 1.69) + scroll1 - (warp * 0.015);
    const float2 sampleUv2 = (float2(worldUv.y, -worldUv.x) * 1.41) + scroll2 + (warp.yx * 0.012);
    const float2 waterNormalRg0 = bindlessTextures[kBindlessTextureIndexWaterNormal].SampleLevel(sampleUv0, 0.0).xy * 2.0 - 1.0;
    const float2 waterNormalRg1 = bindlessTextures[kBindlessTextureIndexWaterNormal].SampleLevel(sampleUv1, 0.0).xy * 2.0 - 1.0;
    const float2 waterNormalRg2 = bindlessTextures[kBindlessTextureIndexWaterNormal].SampleLevel(sampleUv2, 0.0).xy * 2.0 - 1.0;
    const float2 blendedNormalRg =
        (waterNormalRg0 * 0.42) +
        (waterNormalRg1 * 0.36) +
        (waterNormalRg2 * 0.22);
    const float2 textureSlope = blendedNormalRg * (0.46 + (0.28 * detailFade)) * normalStrength;
    const float2 proceduralSlope = float2(-dx, -dz) * normalStrength;
    const float2 waveSlope = (textureSlope * 0.95) + (proceduralSlope * 1.65);
    return normalize(float3(waveSlope.x, 1.0, waveSlope.y));
}

#ifdef ODAI_RT_REFLECTIONS
float3 traceWaterReflection(float3 origin, float3 reflectionDir, float3 skyReflection)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = reflectionDir;
    ray.TMin = 0.05;
    ray.TMax = max(camera.shadowCascadeSplits.w * 1.8, 768.0);

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> rayQuery;
    rayQuery.TraceRayInline(reflectionScene, 0, 0xFF, ray);
    while (rayQuery.Proceed()) {}

    if (rayQuery.CommittedStatus() == COMMITTED_NOTHING)
    {
        return skyReflection;
    }

    const float hitDistance = rayQuery.CommittedRayT();
    const float3 hitPosition = origin + (reflectionDir * hitDistance);
    const float heightAboveWater = max(hitPosition.y - origin.y, 0.0);
    const float buildingFactor = smoothstep(120.0, 900.0, heightAboveWater);
    const float distanceFade = exp(-hitDistance * 0.0035);
    const float skyOcclusion = saturate(1.0 - (abs(reflectionDir.y) * 1.35));
    const float sunFacing = saturate(dot(
        normalize(float3(-reflectionDir.x, 0.35, -reflectionDir.z)),
        -normalize(camera.sunDirectionIntensity.xyz)));
    const float3 terrainTint = lerp(
        float3(0.14, 0.13, 0.11),
        float3(0.34, 0.27, 0.18),
        sunFacing);
    const float3 buildingTint = lerp(
        float3(0.26, 0.22, 0.18),
        float3(0.66, 0.50, 0.32),
        sunFacing);
    const float3 reflectedSurface = lerp(terrainTint, buildingTint, buildingFactor);
    const float sceneWeight = saturate((0.30 + (skyOcclusion * 0.55)) * (0.65 + (0.35 * distanceFade)));
    return lerp(skyReflection, reflectedSurface + (skyReflection * 0.18), sceneWeight);
}

float3 traceWaterRefraction(
    float3 origin,
    float3 refractionDir,
    float3 deepWater,
    float3 shallowWater,
    out float outHitDistance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = refractionDir;
    ray.TMin = 0.02;
    ray.TMax = 224.0;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> rayQuery;
    rayQuery.TraceRayInline(reflectionScene, 0, 0xFF, ray);
    while (rayQuery.Proceed()) {}

    if (rayQuery.CommittedStatus() == COMMITTED_NOTHING)
    {
        outHitDistance = ray.TMax;
        return deepWater * 0.88;
    }

    const float hitDistance = rayQuery.CommittedRayT();
    const float3 hitPosition = origin + (refractionDir * hitDistance);
    outHitDistance = hitDistance;
    const float depthFactor = saturate(hitDistance / 150.0);
    const float shallowFactor = 1.0 - depthFactor;
    const float refractionDecay = max(camera.waterConfig.w, 0.01);
    const float murkFactor = saturate(hitDistance / max(34.0 / refractionDecay, 1.0));
    const float riverbedHeightFactor = saturate((hitPosition.y + 256.0) / 768.0);
    const float riverbedSlopeFactor = saturate(abs(refractionDir.y) * 1.35);
    const float sunFacing = saturate(dot(
        normalize(float3(-refractionDir.x, 0.55, -refractionDir.z)),
        -normalize(camera.sunDirectionIntensity.xyz)));
    const float3 siltTint = lerp(
        float3(0.42, 0.35, 0.22),
        float3(0.62, 0.53, 0.33),
        riverbedHeightFactor);
    const float3 rockTint = lerp(
        float3(0.20, 0.22, 0.18),
        float3(0.35, 0.38, 0.28),
        sunFacing);
    const float3 hitSurfaceTint = lerp(siltTint, rockTint, riverbedSlopeFactor * 0.48);
    const float caustics = waterCausticPattern(hitPosition, hitDistance) * (0.20 + (0.42 * sunFacing));
    const float3 bottomTint = lerp((hitSurfaceTint * (1.06 + caustics)), deepWater * 0.22, depthFactor);
    const float3 attenuation = exp(-kWaterAbsorption * (hitDistance * refractionDecay));
    const float3 transmittedColor = (bottomTint * attenuation) + (shallowWater * 0.18 * shallowFactor);
    const float3 murkyVolumeColor = lerp(shallowWater * 0.78, deepWater * 1.08, murkFactor);
    return lerp(transmittedColor, murkyVolumeColor, saturate(murkFactor * 0.52));
}
#endif

float4 main(FragmentInput input) : SV_Target0
{
    if (chunkPc.cascadeData.z > 0.5)
    {
        return float4(0.02, 0.48, 1.0, 1.0);
    }

    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float3 viewDir = normalize(cameraWorld - input.inWorldPosition);
    const float3 normal = waterNormal(input.inWorldPosition, input.inUv);
    const float waterDebugMode = camera.dofConfig2.w;
    if (waterDebugMode > 0.5 && waterDebugMode < 1.5)
    {
        return float4((normal * 0.5) + 0.5, 1.0);
    }

    const float3 reflectionDir = normalize(reflect(-viewDir, normal));
    const float nDotV = saturate(dot(normal, viewDir));
    const float rippleSlope = saturate(length(normal.xz) * 2.25);
    const float overheadView = pow(nDotV, 2.2);
    const float fresnelBase = 0.020;
    const float fresnel = fresnelBase + ((1.0 - fresnelBase) * pow(1.0 - nDotV, 5.0));
    const float fresnelReflect = saturate((fresnel * 1.28) + (rippleSlope * 0.075));
    if (waterDebugMode > 5.5 && waterDebugMode < 6.5)
    {
        return float4(fresnelReflect.xxx, 1.0);
    }

    const float reflectionStrength = max(camera.waterConfig.z, 0.01);
    const float3 refractionDir = refract(-viewDir, normal, kAirToWaterEta);

    const SceneWaterSample sceneWater = sampleSceneWater(input.inPosition, input.inWorldPosition, normal);
    const float fallbackDepthHint = stableWaterNoise((input.inWorldPosition.xz * 0.010) + float2(17.0, -9.0));
    const float fallbackThickness = lerp(10.0, 42.0, fallbackDepthHint);
    const float sceneThickness = lerp(fallbackThickness, sceneWater.thickness, sceneWater.valid);
    const float shoreFactor = lerp(
        1.0 - smoothstep(6.0, 32.0, fallbackThickness),
        1.0 - smoothstep(3.0, 34.0, sceneWater.thickness),
        sceneWater.valid);
    const float3 shallowWater = kWaterShallowTint;
    const float3 deepWater = kWaterDeepTint;
    float refractedHitDistance = 18.0;
    float rtDepthFactor = 0.0;
    float rtShallowFactor = shoreFactor;
    const float3 waterBody = lerp(deepWater, shallowWater, shoreFactor);

    const float3 skyReflection = proceduralWaterReflectionSky(reflectionDir);
    float3 reflectionColor = skyReflection;
#ifdef ODAI_RT_REFLECTIONS
    reflectionColor = traceWaterReflection(input.inWorldPosition + (normal * 0.08), reflectionDir, skyReflection);
#endif
    if (waterDebugMode > 4.5 && waterDebugMode < 5.5)
    {
        return float4(reflectionColor, 1.0);
    }

    float3 refractionColor = waterBody;
#ifdef ODAI_RT_REFLECTIONS
    if (dot(refractionDir, refractionDir) > 0.0001)
    {
        refractionColor = traceWaterRefraction(
            input.inWorldPosition - float3(0.0, 0.03, 0.0),
            refractionDir,
            deepWater,
            shallowWater,
            refractedHitDistance);
        rtDepthFactor = saturate(refractedHitDistance / 14.0);
        rtShallowFactor = 1.0 - rtDepthFactor;
    }
#else
    refractedHitDistance = sceneThickness;
    rtDepthFactor = saturate(refractedHitDistance / 34.0);
    rtShallowFactor = 1.0 - smoothstep(4.0, 30.0, refractedHitDistance);
    const float refractionDecay = max(camera.waterConfig.w, 0.01);
    const float3 attenuation = exp(-kWaterAbsorption * (refractedHitDistance * refractionDecay));
    const float3 fallbackBottomTint = lerp(float3(0.48, 0.42, 0.27), deepWater * 0.25, rtDepthFactor);
    const float3 bottomTint = lerp(fallbackBottomTint, sceneWater.bottomTint, sceneWater.valid);
    const float murkFactor = smoothstep(9.0, 62.0, refractedHitDistance * refractionDecay);
    const float caustics = waterCausticPattern(
        input.inWorldPosition - float3(0.0, min(refractedHitDistance, 32.0), 0.0),
        refractedHitDistance);
    const float3 transmittedColor =
        (bottomTint * (1.0 + (caustics * 0.16 * rtShallowFactor)) * attenuation) +
        (shallowWater * (0.08 + (0.12 * rtShallowFactor)));
    const float3 volumeColor = lerp(shallowWater * 0.86, deepWater * 1.18, smoothstep(0.12, 1.0, rtDepthFactor));
    refractionColor = lerp(transmittedColor, volumeColor, saturate(murkFactor * 0.55));
#endif
    if (waterDebugMode > 1.5 && waterDebugMode < 2.5)
    {
        const float depthViz = saturate(refractedHitDistance / 60.0);
        const float3 shallowViz = float3(0.02, 0.82, 0.92);
        const float3 deepViz = float3(0.02, 0.04, 0.16);
        const float3 contactViz = float3(sceneWater.contact * 0.85, 0.0, 0.0);
        return float4(saturate(lerp(shallowViz, deepViz, depthViz) + contactViz), 1.0);
    }

    const float2 baseScreenUv = waterScreenUv(input.inPosition);
    const ScreenRefractionSample screenRefraction = sampleAnimatedSceneRefraction(
        baseScreenUv,
        normal,
        refractedHitDistance,
        rtShallowFactor,
        overheadView,
        sceneWater.contact,
        shallowWater,
        refractionColor);
    refractionColor = screenRefraction.color;
    if (waterDebugMode > 2.5 && waterDebugMode < 3.5)
    {
        return float4(screenRefraction.color, 1.0);
    }
    if (waterDebugMode > 3.5 && waterDebugMode < 4.5)
    {
        const float warpViz = saturate((screenRefraction.distortionPixels * length(normal.xz)) / 18.0);
        return float4(warpViz, screenRefraction.weight, 1.0 - warpViz, 1.0);
    }

    const float bottomMottle0 = stableWaterNoise((input.inWorldPosition.xz * 0.028) + float2(13.7, 4.1));
    const float bottomMottle1 = stableWaterNoise((input.inWorldPosition.zx * 0.061) + float2(-7.3, 18.9));
    const float bottomMottle = (bottomMottle0 * 0.62) + (bottomMottle1 * 0.38);
    const float3 overheadBottomTint = lerp(float3(0.48, 0.44, 0.30), float3(0.04, 0.18, 0.19), rtDepthFactor);
    const float screenRefractionPresence = screenRefraction.weight * (0.55 + (0.45 * rtShallowFactor));
    const float overheadClarity =
        overheadView *
        (0.34 + (0.26 * rtShallowFactor)) *
        lerp(1.0, 0.28, screenRefractionPresence);
    refractionColor = lerp(
        refractionColor,
        (refractionColor * (0.92 + (0.12 * bottomMottle))) + (overheadBottomTint * (0.060 + (0.045 * bottomMottle))),
        overheadClarity);

    const float3 sunDir = -normalize(camera.sunDirectionIntensity.xyz);
    const float3 halfVector = normalize(viewDir + sunDir);
    const float ndotl = saturate(dot(normal, sunDir));
    const float ndoth = saturate(dot(normal, halfVector));
    const float vdoth = saturate(dot(viewDir, halfVector));
    const float sunGlint = pow(saturate(dot(reflectionDir, sunDir)), 64.0);
    const float windNoise = stableWaterNoise((input.inWorldPosition.xz * 0.014) + (camera.skyConfig1.z * 0.010));
    const float broadSpecular = pow(ndoth, 10.0) * (0.105 + (0.22 * fresnelReflect));
    const float tightSpecular = pow(ndoth, lerp(58.0, 96.0, windNoise)) * (0.070 + (0.22 * fresnelReflect));
    const float directionalSpecular = (broadSpecular + tightSpecular) * (0.38 + (0.62 * ndotl)) * (0.55 + (0.45 * vdoth));
    const float3 sunScatterTint = shallowWater * lerp(0.025, 0.085, ndotl);
    const float grazingFactor = saturate(1.0 - nDotV);
    const float angleReflectionFactor = saturate(pow(grazingFactor, 0.58));
    const float bottomDepthFactor = saturate(refractedHitDistance / 14.0);
    const float shallowTransmissionFactor = saturate(rtShallowFactor);
    const float reflectionFloor =
        (0.045 + (0.080 * bottomDepthFactor) + (0.030 * rippleSlope)) *
        (0.65 + (0.35 * angleReflectionFactor));
    const float reflectionDriver =
        (angleReflectionFactor * 0.66) +
        (bottomDepthFactor * 0.25) +
        ((1.0 - shallowTransmissionFactor) * 0.10) +
        (rippleSlope * 0.035);
    const float reflectionWeight = saturate(
        max(fresnelReflect, max(reflectionFloor, reflectionDriver * 0.62)) *
        reflectionStrength *
        lerp(0.48, 0.90, bottomDepthFactor) *
        lerp(1.0, 0.62, overheadView));
    const float3 reflectionBoost = lerp(float3(0.96, 0.98, 0.96), float3(1.08, 1.11, 1.10), angleReflectionFactor);
    const float finalReflectionMix = saturate(
        (reflectionWeight * 0.72) +
        (fresnelReflect * reflectionStrength * 0.18) +
        (rippleSlope * 0.030));
    float3 litColor = lerp(screenRefraction.color, reflectionColor * reflectionBoost, finalReflectionMix);
    litColor = lerp(litColor, refractionColor, saturate((1.0 - screenRefraction.weight) * 0.65));
    const float contactFilmMask = sceneWater.contact *
        (0.35 + (0.65 * stableWaterNoise((input.inWorldPosition.xz * 0.075) + float2(camera.skyConfig1.z * 0.012, 3.1))));
    const float foamMask = max(
        saturate((rtShallowFactor - 0.52) * 1.65) *
            saturate((rippleSlope - 0.16) * 3.2) *
            (0.45 + (0.55 * stableWaterNoise((input.inWorldPosition.xz * 0.055) + (camera.skyConfig1.z * 0.018)))),
        contactFilmMask * 0.42);
    const float causticLift = waterCausticPattern(input.inWorldPosition - float3(0.0, refractedHitDistance, 0.0), refractedHitDistance);
    const float3 slopeLightDirection = normalize(float3(0.42, 0.58, 0.70));
    const float slopeLight = saturate((dot(normal, slopeLightDirection) * 0.5) + 0.5);
    const float slopeContrast = (slopeLight - 0.50) * rippleSlope;
    const float warpVisibility =
        screenRefraction.weight *
        saturate(screenRefraction.distortionPixels / 80.0) *
        saturate(length(normal.xz) * 3.0);
    litColor *= 0.98 + (0.16 * ndotl) + (0.07 * rippleSlope);
    litColor *= 1.0 + (slopeContrast * 0.38);
    litColor += lerp(float3(0.08, 0.34, 0.34), float3(0.36, 0.74, 0.66), slopeLight) *
        (rippleSlope * (0.030 + (0.070 * overheadView)));
    litColor += lerp(float3(0.02, 0.22, 0.20), float3(0.35, 0.78, 0.68), slopeLight) *
        (warpVisibility * (0.030 + (0.060 * overheadView)));
    litColor += shallowWater * (causticLift * (0.030 + (0.040 * ndotl)) * saturate(rtShallowFactor + 0.20));
    litColor += lerp(float3(0.05, 0.34, 0.30), float3(0.02, 0.12, 0.14), bottomDepthFactor) *
        (overheadView * (0.045 + (0.040 * causticLift)) * (0.45 + (0.55 * rtShallowFactor)));
    litColor += lerp(float3(0.54, 0.82, 0.74), float3(0.84, 0.94, 0.86), ndotl) * (foamMask * 0.045);
    litColor += float3(0.22, 0.44, 0.36) * (contactFilmMask * 0.025);
    const float rawRefractionPull = saturate(
        screenRefraction.weight *
        (0.16 + (0.18 * overheadView) + (0.14 * shallowTransmissionFactor)) *
        (1.0 - (angleReflectionFactor * 0.62)));
    litColor = lerp(litColor, screenRefraction.color, rawRefractionPull);
    litColor += camera.sunColorShadow.xyz * (sunGlint * (0.070 + (fresnelReflect * 0.18)));
    litColor += camera.sunColorShadow.xyz * directionalSpecular;
    litColor += sunScatterTint;
    litColor += lerp(shallowWater * 0.080, deepWater * 0.035, bottomDepthFactor);
    return float4(litColor, 1.0);
}
