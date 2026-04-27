import camera_uniform;
import chunk_push_constants;
import sh_lighting;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

[[vk::binding(9, 0)]]
Sampler3D<float4> voxelGiVolume;

[[vk::binding(11, 0)]]
Sampler3D<float4> voxelGiOccupancyVolume;

static const uint kImportedMaterialFlagAlphaTest = 1u;
static const uint kInvalidImportedTextureIndex = 0xffffffffu;

#ifdef ODAI_RT_SHADOWS
[[vk::binding(12, 0)]]
RaytracingAccelerationStructure shadowScene;
static const float kPi = 3.14159265359;

float2 rtDiskSampleOffset(int sampleIndex, int sampleCount)
{
    static const float kGoldenAngle = 2.39996322973;
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

float sampleRayTracedShadow(float3 worldPosition, float3 normal, float ndotl)
{
    const float shadowMapReceiverOffset =
        lerp(camera.shadowConfig0.x, camera.shadowConfig0.y, 0.35) * (1.0 + (1.0 - ndotl));
    const float normalOffset = clamp(shadowMapReceiverOffset * 0.12, 0.0025, 0.015);
    const float3 shadowSamplePosition = worldPosition + (normal * normalOffset);
    const float3 toSun = -normalize(camera.sunDirectionIntensity.xyz);
    const float tMin = max(0.001, normalOffset * 0.25);
    const float tMax = max(camera.shadowCascadeSplits.w * 1.5, 512.0);
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

struct FragmentInput
{
    float3 inWorldPosition : TEXCOORD0;
    float3 inWorldNormal : TEXCOORD1;
    float3 inColor : TEXCOORD2;
    float2 inUv : TEXCOORD3;
    nointerpolation uint inTextureIndex : TEXCOORD4;
    nointerpolation uint inFlags : TEXCOORD5;
};

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

    const float3 uvwForward = (worldPosition + (normal * (cellSize * 0.55)) - gridOrigin) / gridExtent;
    const float3 uvwInterior = (worldPosition - (normal * (cellSize * 0.35)) - gridOrigin) / gridExtent;
    const float3 giCenter = voxelGiVolume.SampleLevel(uvwCenter, 0.0).xyz;
    const float3 giForward =
        (any(uvwForward < float3(0.0, 0.0, 0.0)) || any(uvwForward > float3(1.0, 1.0, 1.0)))
            ? giCenter
            : voxelGiVolume.SampleLevel(uvwForward, 0.0).xyz;

    const float interiorOccupancy =
        (any(uvwInterior < float3(0.0, 0.0, 0.0)) || any(uvwInterior > float3(1.0, 1.0, 1.0)))
            ? 0.0
            : voxelGiOccupancyVolume.SampleLevel(uvwInterior, 0.0).x;
    const float oneSidedBlend = (interiorOccupancy > 0.5) ? 1.0 : 0.0;
    const float3 baseBlend = (giCenter * 0.70) + (giForward * 0.30);
    const float3 oneSidedBlendTarget = (giCenter * 0.20) + (giForward * 0.80);
    const float3 giRadiance = lerp(baseBlend, oneSidedBlendTarget, oneSidedBlend);
    return giRadiance * max(camera.voxelGiGridExtentStrength.w, 0.0);
}

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
        const float normalizedDistance = saturate(distanceToLight / radius);
        const float falloff = saturate(1.0 - (normalizedDistance * normalizedDistance));
        const float attenuation = falloff * falloff;
        lighting += colorIntensity.rgb * (colorIntensity.w * globalIntensity * ndotl * attenuation);
    }
    return lighting;
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
    float shadowVisibility = 1.0;
#ifdef ODAI_RT_SHADOWS
    shadowVisibility = sampleRayTracedShadow(input.inWorldPosition, normal, ndotl);
#endif
    const float3 ambient =
        float3(0.18, 0.19, 0.21) +
        (ambientIrradiance * float3(0.92, 0.95, 1.00) * 0.58);
    const float3 voxelGi = sampleImportedVoxelGi(input.inWorldPosition, normal) * 1.35;
    const float3 importedLights = evaluateImportedLocalLights(input.inWorldPosition, normal);
    const float3 direct =
        camera.sunColorShadow.xyz * (sunIntensity * wrappedNdotL * shadowVisibility * 0.62);
    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float3 viewDir = normalize(cameraWorld - input.inWorldPosition);
    const float rim = pow(1.0 - saturate(dot(normal, viewDir)), 2.0);
    const float3 shaded =
        (((ambient + voxelGi + direct + importedLights) * albedo.rgb) + (float3(1.0, 0.98, 0.94) * (rim * 0.022)));
    return float4(shaded, 1.0);
}
