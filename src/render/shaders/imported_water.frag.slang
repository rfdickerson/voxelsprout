import camera_uniform;
import chunk_push_constants;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

#ifdef ODAI_RT_REFLECTIONS
[[vk::binding(12, 0)]]
RaytracingAccelerationStructure reflectionScene;
#endif

static const float kPi = 3.14159265359;
static const float kAirIor = 1.0;
static const float kWaterIor = 1.333;
static const float kAirToWaterEta = kAirIor / kWaterIor;
static const uint kBindlessTextureIndexWaterNormal = 8u;
static const float3 kWaterShallowTint = float3(0.18, 0.44, 0.39);
static const float3 kWaterDeepTint = float3(0.018, 0.075, 0.105);
static const float3 kWaterAbsorption = float3(0.034, 0.021, 0.012);

struct FragmentInput
{
    float3 inWorldPosition : TEXCOORD0;
    float2 inUv : TEXCOORD1;
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
    const float swell0 = sin((worldPosition.x * 0.0074) + (worldPosition.z * 0.0057) + (time * 0.0042 * animationSpeed));
    const float swell1 = cos((worldPosition.x * 0.0061) - (worldPosition.z * 0.0082) - (time * 0.0031 * animationSpeed));
    const float windRipple = sin((worldPosition.x * 0.026) - (worldPosition.z * 0.020) + (time * 0.014 * animationSpeed));
    const float microRipple = sin((uv.x * 6.28318 * 1.08) + (uv.y * 6.28318 * 0.81) + (time * 0.0060 * animationSpeed));
    const float dx = (swell0 * 0.0080) + (swell1 * 0.0052) + (windRipple * 0.0038) + (microRipple * 0.0024 * detailFade);
    const float dz = (swell1 * 0.0074) - (windRipple * 0.0038) - (microRipple * 0.0024 * detailFade);
    const float2 worldUv = float2(worldPosition.x, worldPosition.z) * 0.00195;
    const float2 warp = float2(
        stableWaterNoise((worldUv * 19.0) + float2(time * 0.010, 0.0)),
        stableWaterNoise((worldUv.yx * 17.0) + float2(0.0, -time * 0.009))) - 0.5;
    const float2 scroll0 = float2(time * 0.0054, -time * 0.0038) * animationSpeed;
    const float2 scroll1 = float2(-time * 0.0033, time * 0.0046) * animationSpeed;
    const float2 scroll2 = float2(time * 0.0024, time * 0.0020) * animationSpeed;
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
    const float2 textureSlope = blendedNormalRg * (0.38 + (0.20 * detailFade)) * normalStrength;
    const float2 proceduralSlope = float2(-dx, -dz) * normalStrength;
    const float2 waveSlope = (textureSlope * 0.82) + (proceduralSlope * 1.05);
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
        float3(0.24, 0.21, 0.15),
        float3(0.36, 0.32, 0.23),
        riverbedHeightFactor);
    const float3 rockTint = lerp(
        float3(0.15, 0.18, 0.14),
        float3(0.29, 0.34, 0.25),
        sunFacing);
    const float3 hitSurfaceTint = lerp(siltTint, rockTint, riverbedSlopeFactor * 0.48);
    const float caustics = waterCausticPattern(hitPosition, hitDistance) * (0.20 + (0.42 * sunFacing));
    const float3 bottomTint = lerp((hitSurfaceTint * (1.04 + caustics)), deepWater * 0.30, depthFactor);
    const float3 attenuation = exp(-kWaterAbsorption * (hitDistance * refractionDecay));
    const float3 transmittedColor = (bottomTint * attenuation) + (shallowWater * 0.18 * shallowFactor);
    const float3 murkyVolumeColor = lerp(shallowWater * 0.52, deepWater * 0.96, murkFactor);
    return lerp(transmittedColor, murkyVolumeColor, saturate(murkFactor * 0.74));
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
    const float3 reflectionDir = normalize(reflect(-viewDir, normal));
    const float nDotV = saturate(dot(normal, viewDir));
    const float rippleSlope = saturate(length(normal.xz) * 2.6);
    const float overheadView = pow(nDotV, 2.2);
    const float fresnelBase = 0.020;
    const float fresnel = fresnelBase + ((1.0 - fresnelBase) * pow(1.0 - nDotV, 5.0));
    const float fresnelReflect = saturate((fresnel * 1.28) + (rippleSlope * 0.035));
    const float reflectionStrength = max(camera.waterConfig.z, 0.01);
    const float3 refractionDir = refract(-viewDir, normal, kAirToWaterEta);

    const float depthHint =
        saturate((input.inWorldPosition.y - camera.shadowCascadeSplits.x) / max(camera.shadowCascadeSplits.w, 1.0));
    const float shoreFactor = saturate(1.0 - (depthHint * 1.35));
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
    refractedHitDistance = lerp(4.0, 46.0, saturate(depthHint * 1.35));
    rtDepthFactor = saturate(refractedHitDistance / 18.0);
    rtShallowFactor = 1.0 - rtDepthFactor;
    const float3 attenuation = exp(-kWaterAbsorption * (refractedHitDistance * max(camera.waterConfig.w, 0.01)));
    refractionColor = lerp(shallowWater * 0.98, deepWater * 0.96, rtDepthFactor) * attenuation +
        waterBody * (0.22 + (0.18 * shoreFactor));
#endif

    const float bottomMottle0 = stableWaterNoise((input.inWorldPosition.xz * 0.028) + float2(13.7, 4.1));
    const float bottomMottle1 = stableWaterNoise((input.inWorldPosition.zx * 0.061) + float2(-7.3, 18.9));
    const float bottomMottle = (bottomMottle0 * 0.62) + (bottomMottle1 * 0.38);
    const float3 overheadBottomTint = lerp(float3(0.20, 0.22, 0.17), float3(0.055, 0.105, 0.105), rtDepthFactor);
    const float overheadClarity = overheadView * (0.38 + (0.34 * rtShallowFactor));
    refractionColor = lerp(
        refractionColor,
        (refractionColor * (0.88 + (0.18 * bottomMottle))) + (overheadBottomTint * (0.10 + (0.08 * bottomMottle))),
        overheadClarity);

    const float3 sunDir = -normalize(camera.sunDirectionIntensity.xyz);
    const float3 halfVector = normalize(viewDir + sunDir);
    const float ndotl = saturate(dot(normal, sunDir));
    const float ndoth = saturate(dot(normal, halfVector));
    const float vdoth = saturate(dot(viewDir, halfVector));
    const float sunGlint = pow(saturate(dot(reflectionDir, sunDir)), 64.0);
    const float windNoise = stableWaterNoise((input.inWorldPosition.xz * 0.014) + (camera.skyConfig1.z * 0.010));
    const float broadSpecular = pow(ndoth, 18.0) * (0.045 + (0.12 * fresnelReflect));
    const float tightSpecular = pow(ndoth, lerp(86.0, 148.0, windNoise)) * (0.08 + (0.26 * fresnelReflect));
    const float directionalSpecular = (broadSpecular + tightSpecular) * (0.24 + (0.76 * ndotl)) * (0.55 + (0.45 * vdoth));
    const float3 sunScatterTint = waterBody * lerp(0.06, 0.16, ndotl);
    const float grazingFactor = saturate(1.0 - nDotV);
    const float angleReflectionFactor = saturate(pow(grazingFactor, 0.58));
    const float bottomDepthFactor = saturate(refractedHitDistance / 14.0);
    const float shallowTransmissionFactor = saturate(rtShallowFactor);
    const float reflectionFloor =
        (0.050 + (0.070 * bottomDepthFactor) + (0.032 * rippleSlope)) *
        (0.65 + (0.35 * angleReflectionFactor));
    const float reflectionDriver =
        (angleReflectionFactor * 0.66) +
        (bottomDepthFactor * 0.25) +
        ((1.0 - shallowTransmissionFactor) * 0.14) +
        (rippleSlope * 0.045);
    const float reflectionWeight = saturate(
        max(fresnelReflect, max(reflectionFloor, reflectionDriver * 0.62)) *
        reflectionStrength *
        lerp(0.62, 0.92, bottomDepthFactor) *
        lerp(1.0, 0.58, overheadView));
    const float transmissionWeight = 1.0 - reflectionWeight;
    const float3 reflectionBoost = lerp(float3(0.96, 0.98, 0.96), float3(1.08, 1.11, 1.10), angleReflectionFactor);
    float3 litColor = (refractionColor * transmissionWeight) + ((reflectionColor * reflectionBoost) * reflectionWeight);
    const float foamMask = saturate((rtShallowFactor - 0.52) * 1.65) *
        saturate((rippleSlope - 0.16) * 3.2) *
        (0.45 + (0.55 * stableWaterNoise((input.inWorldPosition.xz * 0.055) + (camera.skyConfig1.z * 0.018))));
    const float causticLift = waterCausticPattern(input.inWorldPosition - float3(0.0, refractedHitDistance, 0.0), refractedHitDistance);
    litColor *= 0.96 + (0.20 * ndotl) + (0.10 * rippleSlope);
    litColor += shallowWater * (causticLift * (0.050 + (0.065 * ndotl)) * saturate(rtShallowFactor + 0.25));
    litColor += lerp(float3(0.16, 0.26, 0.22), float3(0.05, 0.13, 0.14), bottomDepthFactor) *
        (overheadView * (0.030 + (0.060 * causticLift)) * (0.45 + (0.55 * rtShallowFactor)));
    litColor += lerp(float3(0.70, 0.82, 0.78), float3(0.88, 0.94, 0.88), ndotl) * (foamMask * 0.09);
    litColor += camera.sunColorShadow.xyz * (sunGlint * (0.070 + (fresnelReflect * 0.18)));
    litColor += camera.sunColorShadow.xyz * directionalSpecular;
    litColor += sunScatterTint;
    litColor += waterBody * 0.025;
    const float alphaBase =
        0.12 +
        (bottomDepthFactor * 0.22) +
        (angleReflectionFactor * 0.18) +
        (reflectionWeight * 0.11) +
        ((1.0 - shallowTransmissionFactor) * 0.10) +
        (foamMask * 0.03);
    const float alpha = saturate(alphaBase * lerp(1.0, 0.74, overheadView));
    return float4(litColor, alpha);
}
