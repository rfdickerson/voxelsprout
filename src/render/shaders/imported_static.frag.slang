import camera_uniform;
import chunk_push_constants;
import sh_lighting;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

[[vk::binding(4, 0)]]
Sampler2DShadow shadowMap;

[[vk::binding(9, 0)]]
Sampler3D<float4> voxelGiVolume;

[[vk::binding(11, 0)]]
Sampler3D<float4> voxelGiOccupancyVolume;

#ifdef ODAI_RT_SHADOWS
[[vk::binding(12, 0)]]
RaytracingAccelerationStructure shadowScene;
#endif

static const uint kImportedMaterialFlagAlphaTest = 1u;
static const uint kInvalidImportedTextureIndex = 0xffffffffu;

struct FragmentInput
{
    float3 inWorldPosition : TEXCOORD0;
    float3 inWorldNormal : TEXCOORD1;
    float3 inColor : TEXCOORD2;
    float2 inUv : TEXCOORD3;
    nointerpolation uint inTextureIndex : TEXCOORD4;
    nointerpolation uint inFlags : TEXCOORD5;
};

float4 sampleImportedAlbedo(FragmentInput input)
{
    const bool disableTextures = chunkPc.cascadeData.z > 0.5;
    if (input.inTextureIndex == kInvalidImportedTextureIndex)
    {
        return float4(input.inColor, 1.0);
    }
    float4 sampled = bindlessTextures[input.inTextureIndex].Sample(input.inUv);
    if ((input.inFlags & kImportedMaterialFlagAlphaTest) != 0u && sampled.a < 0.5)
    {
        discard;
    }
    if (disableTextures)
    {
        return float4(input.inColor, 1.0);
    }
    return float4(sampled.rgb, sampled.a);
}

float3 sampleImportedVoxelGi(float3 worldPosition, float3 normal)
{
    const float cellSize = max(camera.voxelGiGridOriginCellSize.w, 0.001);
    const float3 gridOrigin = camera.voxelGiGridOriginCellSize.xyz;
    const float3 gridExtent = max(camera.voxelGiGridExtentStrength.xyz, float3(cellSize, cellSize, cellSize));
    const float3 uvwCenter = (worldPosition - gridOrigin) / gridExtent;
    if (any(uvwCenter < float3(0.0, 0.0, 0.0)) || any(uvwCenter > float3(1.0, 1.0, 1.0)))
    {
        return float3(0.0, 0.0, 0.0);
    }

    const float3 uvwForward = (worldPosition + (normal * (cellSize * 0.35)) - gridOrigin) / gridExtent;
    const float3 uvwBack = (worldPosition - (normal * (cellSize * 0.35)) - gridOrigin) / gridExtent;
    const float3 giCenter = voxelGiVolume.SampleLevel(uvwCenter, 0.0).xyz;
    const float3 giForward =
        (any(uvwForward < float3(0.0, 0.0, 0.0)) || any(uvwForward > float3(1.0, 1.0, 1.0)))
            ? giCenter
            : voxelGiVolume.SampleLevel(uvwForward, 0.0).xyz;
    const float3 giBack =
        (any(uvwBack < float3(0.0, 0.0, 0.0)) || any(uvwBack > float3(1.0, 1.0, 1.0)))
            ? giCenter
            : voxelGiVolume.SampleLevel(uvwBack, 0.0).xyz;

    return ((giCenter * 0.60) + (giForward * 0.25) + (giBack * 0.15)) *
        max(camera.voxelGiGridExtentStrength.w, 0.0);
}

float giLumaFromColor(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float3 sampleImportedVoxelGiDebug(float3 worldPosition, float3 normal)
{
    const float cellSize = max(camera.voxelGiGridOriginCellSize.w, 0.001);
    const float3 gridOrigin = camera.voxelGiGridOriginCellSize.xyz;
    const float3 gridExtent = max(camera.voxelGiGridExtentStrength.xyz, float3(cellSize, cellSize, cellSize));
    const float3 uvwCenter = (worldPosition - gridOrigin) / gridExtent;
    if (any(uvwCenter < float3(0.0, 0.0, 0.0)) || any(uvwCenter > float3(1.0, 1.0, 1.0)))
    {
        return float3(0.0, 0.0, 0.0);
    }

    return sampleImportedVoxelGi(worldPosition, normal);
}

float3 sampleImportedVoxelGiOccupancyAlbedo(float3 worldPosition, float3 normal)
{
    const float cellSize = max(camera.voxelGiGridOriginCellSize.w, 0.001);
    const float3 gridOrigin = camera.voxelGiGridOriginCellSize.xyz;
    const float3 gridExtent = max(camera.voxelGiGridExtentStrength.xyz, float3(cellSize, cellSize, cellSize));
    const float3 uvwCenter = (worldPosition - gridOrigin) / gridExtent;
    if (any(uvwCenter < float3(0.0, 0.0, 0.0)) || any(uvwCenter > float3(1.0, 1.0, 1.0)))
    {
        return float3(0.0, 0.0, 0.0);
    }

    const float probeDistance = cellSize * 0.90;
    const float3 uvwForward = (worldPosition + (normal * probeDistance) - gridOrigin) / gridExtent;
    const float3 uvwBack = (worldPosition - (normal * probeDistance) - gridOrigin) / gridExtent;
    const float4 center = voxelGiOccupancyVolume.SampleLevel(uvwCenter, 0.0);
    const float4 forward =
        (any(uvwForward < float3(0.0, 0.0, 0.0)) || any(uvwForward > float3(1.0, 1.0, 1.0)))
            ? center
            : voxelGiOccupancyVolume.SampleLevel(uvwForward, 0.0);
    const float4 back =
        (any(uvwBack < float3(0.0, 0.0, 0.0)) || any(uvwBack > float3(1.0, 1.0, 1.0)))
            ? center
            : voxelGiOccupancyVolume.SampleLevel(uvwBack, 0.0);

    float4 best = center;
    if (forward.x > best.x)
    {
        best = forward;
    }
    if (back.x > best.x)
    {
        best = back;
    }
    return (best.x > 0.5) ? saturate(best.yzw) : float3(0.0, 0.0, 0.0);
}

float3 voxelGiFalseColor(float t)
{
    const float x = clamp(t, 0.0, 1.0);
    if (x < 0.25) return lerp(float3(0.02, 0.02, 0.10), float3(0.00, 0.42, 1.00), x / 0.25);
    if (x < 0.50) return lerp(float3(0.00, 0.42, 1.00), float3(0.00, 0.95, 0.45), (x - 0.25) / 0.25);
    if (x < 0.75) return lerp(float3(0.00, 0.95, 0.45), float3(1.00, 0.92, 0.12), (x - 0.50) / 0.25);
    return lerp(float3(1.00, 0.92, 0.12), float3(1.00, 0.16, 0.12), (x - 0.75) / 0.25);
}

#ifdef ODAI_RT_SHADOWS
static const float kPi = 3.14159265;
static const float kGoldenAngle = 2.39996323;

float2 rtDiskSampleOffset(int sampleIndex, int sampleCount)
{
    const float count = max(float(sampleCount), 1.0);
    const float radius = sqrt((float(sampleIndex) + 0.5) / count);
    const float angle = (float(sampleIndex) + 1.0) * kGoldenAngle;
    return float2(cos(angle), sin(angle)) * radius;
}

float hashRtShadowRotation(float3 worldPosition)
{
    return frac(sin(dot(worldPosition, float3(12.9898, 78.233, 37.719))) * 43758.5453);
}

float2 rotateRtDiskOffset(float2 offset, float angle)
{
    const float s = sin(angle);
    const float c = cos(angle);
    return float2((offset.x * c) - (offset.y * s), (offset.x * s) + (offset.y * c));
}

float sampleRayTracedShadowRay(float3 origin, float3 direction, float tMin, float tMax)
{
    if (tMax <= tMin)
    {
        return 1.0;
    }

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> rayQuery;
    rayQuery.TraceRayInline(shadowScene, 0, 0xFF, ray);
    while (rayQuery.Proceed()) {}
    return (rayQuery.CommittedStatus() == COMMITTED_NOTHING) ? 1.0 : 0.0;
}

float sampleRayTracedPointLightShadow(
    float3 worldPosition,
    float3 normal,
    float3 lightDirection,
    float lightDistance,
    float ndotl)
{
    const float originBias = lerp(0.45, 0.18, saturate(ndotl));
    const float3 origin = worldPosition + (normal * originBias) + (lightDirection * 0.12);
    return sampleRayTracedShadowRay(origin, lightDirection, 0.03, max(lightDistance - 0.30, 0.03));
}

float sampleRayTracedDirectionalShadow(float3 worldPosition, float3 normal, float ndotl)
{
    const float shadowMapReceiverOffset =
        lerp(camera.shadowConfig0.x, camera.shadowConfig0.y, 0.35) * (1.0 + (1.0 - ndotl));
    const float normalOffset = clamp(shadowMapReceiverOffset * 4.0, 0.18, 0.85);
    const float3 toSun = -normalize(camera.sunDirectionIntensity.xyz);
    const float3 shadowSamplePosition = worldPosition + (normal * normalOffset) + (toSun * 0.12);
    const float tMin = 0.03;
    const float tMax = max(camera.shadowCascadeSplits.w * 8.0, 4096.0);
    const int sampleCount = clamp(int(camera.shadowConfig4.x + 0.5), 1, 8);
    const float sunAngularRadiusDegrees = max(camera.shadowConfig4.y, 0.0);
    const float coneRadius = tan(sunAngularRadiusDegrees * (kPi / 180.0));

    if (sampleCount <= 1 || coneRadius <= 1e-5)
    {
        return sampleRayTracedShadowRay(shadowSamplePosition, toSun, tMin, tMax);
    }

    const float3 basisUp = (abs(toSun.y) < 0.999) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 tangent = normalize(cross(basisUp, toSun));
    const float3 bitangent = cross(toSun, tangent);
    const float rotationAngle = hashRtShadowRotation(worldPosition) * (2.0 * kPi);

    float visibility = 0.0;
    [unroll]
    for (int sampleIndex = 0; sampleIndex < 8; ++sampleIndex)
    {
        if (sampleIndex >= sampleCount) break;
        const float2 diskOffset = rotateRtDiskOffset(rtDiskSampleOffset(sampleIndex, sampleCount), rotationAngle);
        const float3 rayDirection = normalize(
            toSun +
            ((tangent * diskOffset.x) + (bitangent * diskOffset.y)) * coneRadius
        );
        visibility += sampleRayTracedShadowRay(shadowSamplePosition, rayDirection, tMin, tMax);
    }
    return visibility / float(sampleCount);
}
#endif

float3 evaluateImportedLocalLights(float3 worldPosition, float3 normal)
{
    const int lightCount = clamp(int(camera.importedLightConfig.x + 0.5), 0, 64);
    const float globalIntensity = max(camera.importedLightConfig.y, 0.0);
    if (lightCount <= 0 || globalIntensity <= 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }

    float3 lighting = float3(0.0, 0.0, 0.0);
    [loop]
    for (int lightIndex = 0; lightIndex < 64; ++lightIndex)
    {
        if (lightIndex >= lightCount) break;
        const float4 positionRadius = camera.importedLightPositionRadius[lightIndex];
        const float4 colorIntensity = camera.importedLightColorIntensity[lightIndex];
        const float3 toLight = positionRadius.xyz - worldPosition;
        const float distanceSquared = dot(toLight, toLight);
        const float radius = max(positionRadius.w, 1.0);
        const float radiusSquared = radius * radius;
        if (distanceSquared >= radiusSquared)
        {
            continue;
        }

        const float distanceToLight = sqrt(max(distanceSquared, 1e-4));
        const float3 lightDirection = toLight / distanceToLight;
        const float ndotl = saturate(dot(normal, lightDirection));
        if (ndotl <= 0.0001)
        {
            continue;
        }
        const float normalizedDistance = saturate(distanceToLight / radius);
        const float falloff = saturate(1.0 - (normalizedDistance * normalizedDistance));
        const float attenuation = falloff * falloff;
        float visibility = 1.0;
#ifdef ODAI_RT_SHADOWS
        visibility = sampleRayTracedPointLightShadow(
            worldPosition,
            normal,
            lightDirection,
            distanceToLight,
            ndotl);
#endif
        lighting += colorIntensity.rgb * (colorIntensity.w * globalIntensity * ndotl * attenuation * visibility);
    }
    return lighting;
}

float3 evaluateImportedLocalLightDebug(float3 worldPosition)
{
    const int lightCount = clamp(int(camera.importedLightConfig.x + 0.5), 0, 64);
    const float globalIntensity = max(camera.importedLightConfig.y, 0.0);
    if (lightCount <= 0 || globalIntensity <= 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }

    float3 lighting = float3(0.0, 0.0, 0.0);
    [loop]
    for (int lightIndex = 0; lightIndex < 64; ++lightIndex)
    {
        if (lightIndex >= lightCount) break;
        const float4 positionRadius = camera.importedLightPositionRadius[lightIndex];
        const float4 colorIntensity = camera.importedLightColorIntensity[lightIndex];
        const float3 toLight = positionRadius.xyz - worldPosition;
        const float distanceSquared = dot(toLight, toLight);
        const float radius = max(positionRadius.w, 1.0);
        if (distanceSquared >= radius * radius)
        {
            continue;
        }

        const float distanceToLight = sqrt(max(distanceSquared, 1e-4));
        const float normalizedDistance = saturate(distanceToLight / radius);
        const float falloff = saturate(1.0 - (normalizedDistance * normalizedDistance));
        lighting += colorIntensity.rgb * (colorIntensity.w * globalIntensity * falloff * falloff);
    }
    return lighting;
}

float3 evaluateMorrowindInteriorGi(float3 worldPosition, float3 normal, float3 albedo)
{
    const int lightCount = clamp(int(camera.importedLightConfig.x + 0.5), 0, 64);
    const float globalIntensity = max(camera.importedLightConfig.y, 0.0);
    if (lightCount <= 0 || globalIntensity <= 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }

    float3 irradiance = float3(0.0, 0.0, 0.0);
    [loop]
    for (int lightIndex = 0; lightIndex < 64; ++lightIndex)
    {
        if (lightIndex >= lightCount)
        {
            break;
        }

        const float4 positionRadius = camera.importedLightPositionRadius[lightIndex];
        const float4 colorIntensity = camera.importedLightColorIntensity[lightIndex];
        const float radius = max(positionRadius.w, 1.0);
        const float3 toLight = positionRadius.xyz - worldPosition;
        const float distanceSquared = dot(toLight, toLight);
        if (distanceSquared >= radius * radius)
        {
            continue;
        }

        const float distanceToLight = sqrt(max(distanceSquared, 1e-4));
        const float3 lightDirection = toLight / distanceToLight;
        const float normalWrap = saturate((dot(normal, lightDirection) * 0.35) + 0.65);
        const float normalizedDistance = saturate(distanceToLight / radius);
        const float falloff = saturate(1.0 - (normalizedDistance * normalizedDistance));
        const float attenuation = falloff * falloff;
        irradiance += colorIntensity.rgb * (colorIntensity.w * globalIntensity * attenuation * normalWrap);
    }

    const float albedoLuma = max(dot(albedo, float3(0.2126, 0.7152, 0.0722)), 0.08);
    const float3 colorBleed = saturate(lerp(float3(albedoLuma, albedoLuma, albedoLuma), albedo, 0.55));
    return irradiance * colorBleed * 0.18;
}

int chooseShadowCascade(float viewDepth)
{
    if (viewDepth <= camera.shadowCascadeSplits.x) return 0;
    if (viewDepth <= camera.shadowCascadeSplits.y) return 1;
    if (viewDepth <= camera.shadowCascadeSplits.z) return 2;
    return 3;
}

float sampleImportedShadowPcf(float3 worldPosition, float3 normal, int cascadeIndex, float ndotl)
{
    const float cascadeT = float(cascadeIndex) / 3.0;
    const float normalOffset =
        lerp(camera.shadowConfig0.x, camera.shadowConfig0.y, cascadeT) * (1.0 + (1.0 - ndotl));
    const float3 shadowSamplePosition = worldPosition + (normal * normalOffset);
    const float4 shadowClip = mul(camera.lightViewProj[cascadeIndex], float4(shadowSamplePosition, 1.0));
    if (shadowClip.w <= 0.0) return 1.0;

    const float3 shadowNdc = shadowClip.xyz / shadowClip.w;
    const float2 shadowUvLocal = (shadowNdc.xy * 0.5) + 0.5;
    const float4 atlasUvRect = camera.shadowAtlasUvRects[cascadeIndex];
    const float2 shadowUv = atlasUvRect.xy + (shadowUvLocal * atlasUvRect.zw);
    const float shadowDepthRef = shadowNdc.z;

    if (
        shadowUvLocal.x <= 0.0 || shadowUvLocal.x >= 1.0 ||
        shadowUvLocal.y <= 0.0 || shadowUvLocal.y >= 1.0 ||
        shadowDepthRef <= 0.0 || shadowDepthRef >= 1.0)
    {
        return 1.0;
    }

    uint texW = 0;
    uint texH = 0;
    shadowMap.GetDimensions(texW, texH);
    const float2 texelSize = 1.0 / float2(float(texW), float(texH));
    const float texelScale = max(texelSize.x, texelSize.y);
    const float bias =
        lerp(camera.shadowConfig0.z, camera.shadowConfig0.w, cascadeT) * texelScale +
        lerp(camera.shadowConfig1.x, camera.shadowConfig1.y, cascadeT) * texelScale * (1.0 - ndotl);
    const int kernelRadius = clamp(int(camera.shadowConfig3.w + 0.5), 1, 3);

    float visibility = 0.0;
    float weightSum = 0.0;
    for (int y = -3; y <= 3; ++y)
    {
        for (int x = -3; x <= 3; ++x)
        {
            if (abs(x) > kernelRadius || abs(y) > kernelRadius) continue;
            const float2 uv = shadowUv + float2(float(x), float(y)) * texelSize;
            const float wx = float(kernelRadius + 1 - abs(x));
            const float wy = float(kernelRadius + 1 - abs(y));
            const float weight = wx * wy;
            visibility += shadowMap.SampleCmpLevelZero(uv, shadowDepthRef + bias) * weight;
            weightSum += weight;
        }
    }
    return visibility / max(weightSum, 0.0001);
}

float sampleImportedCascadedShadow(float3 worldPosition, float3 normal, float viewDepth, float ndotl)
{
    const int primaryCascade = chooseShadowCascade(viewDepth);
    float shadow = sampleImportedShadowPcf(worldPosition, normal, primaryCascade, ndotl);
    if (primaryCascade >= 3) return shadow;

    const float prevSplit = (primaryCascade == 0) ? 0.0 : camera.shadowCascadeSplits[primaryCascade - 1];
    const float split = camera.shadowCascadeSplits[primaryCascade];
    const float blendRange = max(camera.shadowConfig1.z, (split - prevSplit) * camera.shadowConfig1.w);
    const float blendT = smoothstep(split - blendRange, split, viewDepth);
    if (blendT <= 0.0) return shadow;

    const float nextShadow = sampleImportedShadowPcf(worldPosition, normal, primaryCascade + 1, ndotl);
    return lerp(shadow, nextShadow, blendT);
}

float4 main(FragmentInput input) : SV_Target0
{
    const float4 albedo = sampleImportedAlbedo(input);
    if (chunkPc.cascadeData.w > 0.5) {
        return float4(saturate(albedo.rgb * 1.25), 1.0);
    }
    const float3 normal = normalize(input.inWorldNormal);
    const float3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const float3 shNormalIrradiance = evaluateShIrradiance(normal);
    const float3 shHemisphereIrradiance = evaluateShHemisphereIrradiance(normal);
    const float3 ambientIrradiance = lerp(shNormalIrradiance, shHemisphereIrradiance, 0.6);
    const float ndotl = max(dot(normal, -sunDirection), 0.0);
    const float wrappedNdotL = saturate((ndotl * 0.65) + 0.35);
    const float viewDepth = max(-mul(camera.view, float4(input.inWorldPosition, 1.0)).z, 0.0);
    float shadowVisibility = 1.0;
    if (ndotl > 0.0001)
    {
#ifdef ODAI_RT_SHADOWS
        shadowVisibility = sampleRayTracedDirectionalShadow(input.inWorldPosition, normal, ndotl);
#else
        shadowVisibility = sampleImportedCascadedShadow(input.inWorldPosition, normal, viewDepth, ndotl);
#endif
    }
    const bool interiorMode = chunkPc.cascadeData.y > 0.5;
    const float3 skyProbeNormal = float3(0.0, 1.0, 0.0);
    const float skyProbeNdotL = max(dot(skyProbeNormal, -sunDirection), 0.25);
    float skyVisibility = 1.0;
    if (interiorMode)
    {
#ifdef ODAI_RT_SHADOWS
        skyVisibility = sampleRayTracedDirectionalShadow(input.inWorldPosition, skyProbeNormal, skyProbeNdotL);
#else
        skyVisibility = sampleImportedCascadedShadow(input.inWorldPosition, skyProbeNormal, viewDepth, skyProbeNdotL);
#endif
    }
    const float ambientOcclusion = interiorMode ? lerp(0.28, 1.0, skyVisibility) : 1.0;
    const float3 ambient =
        (interiorMode ? float3(0.045, 0.047, 0.052) : float3(0.18, 0.19, 0.21)) +
        (ambientIrradiance * float3(0.92, 0.95, 1.00) * (interiorMode ? 0.22 : 0.58));
    const float3 sampledVoxelGi = sampleImportedVoxelGi(input.inWorldPosition, normal);
    const float3 morrowindGi = interiorMode
        ? evaluateMorrowindInteriorGi(input.inWorldPosition, normal, albedo.rgb)
        : float3(0.0, 0.0, 0.0);
    const float3 voxelGi = interiorMode
        ? max(sampledVoxelGi * 1.35, morrowindGi)
        : float3(0.0, 0.0, 0.0);
    const float3 voxelGiDebug = interiorMode
        ? max(sampleImportedVoxelGiDebug(input.inWorldPosition, normal), morrowindGi)
        : sampleImportedVoxelGiDebug(input.inWorldPosition, normal);
    const int voxelGiDebugMode = int(camera.shadowVoxelGridOrigin.z + 0.5);
    if (voxelGiDebugMode == 1)
    {
        const float3 giClamped = max(voxelGiDebug, float3(0.0, 0.0, 0.0));
        const float peak = max(max(giClamped.x, giClamped.y), giClamped.z);
        const float3 hue = (peak > 0.0001) ? (giClamped / peak) : float3(0.0, 0.0, 0.0);
        const float intensity = clamp(log2(1.0 + (peak * 8.0)) / 4.0, 0.0, 1.0);
        return float4(hue * intensity, 1.0);
    }
    if (voxelGiDebugMode == 2)
    {
        const float giLuma = giLumaFromColor(voxelGiDebug);
        const float giMapped = clamp(log2(1.0 + (giLuma * 6.0)) / 3.0, 0.0, 1.0);
        return float4(voxelGiFalseColor(giMapped), 1.0);
    }
    if (voxelGiDebugMode == 3)
    {
        const float giLuma = giLumaFromColor(voxelGiDebug);
        const float giMapped = clamp(log2(1.0 + (giLuma * 4.0)) / 3.0, 0.0, 1.0);
        return float4(giMapped, giMapped, giMapped, 1.0);
    }
    if (voxelGiDebugMode == 4)
    {
        const float3 occupancyAlbedo = sampleImportedVoxelGiOccupancyAlbedo(input.inWorldPosition, normal);
        return float4(max(occupancyAlbedo, saturate(albedo.rgb)), 1.0);
    }
    if (voxelGiDebugMode == 5)
    {
        const float3 lightDebug = max(evaluateImportedLocalLightDebug(input.inWorldPosition), float3(0.0, 0.0, 0.0));
        const float peak = max(max(lightDebug.x, lightDebug.y), lightDebug.z);
        const float3 hue = (peak > 0.0001) ? (lightDebug / peak) : float3(0.0, 0.0, 0.0);
        const float intensity = clamp(log2(1.0 + (peak * 2.0)) / 4.0, 0.0, 1.0);
        return float4(hue * intensity, 1.0);
    }
    const float3 importedLights =
        evaluateImportedLocalLights(input.inWorldPosition, normal) * (interiorMode ? 1.25 : 1.0);
    const float directShadowFactor = lerp(1.0, shadowVisibility, clamp(camera.sunColorShadow.w, 0.0, 1.0));
    const float3 direct =
        camera.sunColorShadow.xyz *
        (sunIntensity * wrappedNdotL * directShadowFactor * (interiorMode ? 0.24 : 0.62));
    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float3 viewDir = normalize(cameraWorld - input.inWorldPosition);
    const float rim = pow(1.0 - saturate(dot(normal, viewDir)), 2.0);
    const float3 shaded =
        ((((ambient * ambientOcclusion) + voxelGi + direct + importedLights) * albedo.rgb) +
         (float3(1.0, 0.98, 0.94) * (rim * (interiorMode ? 0.010 : 0.022))));
    return float4(shaded, 1.0);
}
