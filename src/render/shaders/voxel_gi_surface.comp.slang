// Compile for Vulkan SPIR-V with column-major matrices to match renderer.cc uniform layout:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/voxel_gi_surface.comp.slang -o src/render/shaders/voxel_gi_surface.comp.slang.spv

import camera_uniform;
import sh_lighting;

[[vk::binding(1, 0)]]
Sampler2DShadow shadowMap;

[[vk::binding(5, 0)]]
Texture3D<float4> voxelGiOccupancy;

[[vk::binding(6, 0)]]
RWTexture3D<float4> voxelGiSurfacePosXOut;

[[vk::binding(7, 0)]]
RWTexture3D<float4> voxelGiSurfaceNegXOut;

[[vk::binding(8, 0)]]
RWTexture3D<float4> voxelGiSurfacePosYOut;

[[vk::binding(9, 0)]]
RWTexture3D<float4> voxelGiSurfaceNegYOut;

[[vk::binding(10, 0)]]
RWTexture3D<float4> voxelGiSurfacePosZOut;

[[vk::binding(11, 0)]]
RWTexture3D<float4> voxelGiSurfaceNegZOut;

[[vk::binding(12, 0)]]
RWTexture3D<float4> voxelGiSkyExposure;

#ifdef VOXEL_GI_RT_SURFACE
[[vk::binding(16, 0)]]
RaytracingAccelerationStructure giSurfaceScene;
#endif

int chooseShadowCascade(float viewDepth)
{
    if (viewDepth <= camera.shadowCascadeSplits.x) return 0;
    if (viewDepth <= camera.shadowCascadeSplits.y) return 1;
    if (viewDepth <= camera.shadowCascadeSplits.z) return 2;
    return 3;
}

float sampleShadowVisibility(float3 worldPosition)
{
    const float viewDepth = max(-mul(camera.view, float4(worldPosition, 1.0)).z, 0.0);
    const int cascadeIndex = chooseShadowCascade(viewDepth);
    const float4 shadowClip = mul(camera.lightViewProj[cascadeIndex], float4(worldPosition, 1.0));
    if (shadowClip.w <= 0.0) return 1.0;

    const float3 shadowNdc = shadowClip.xyz / shadowClip.w;
    const float2 shadowUvLocal = (shadowNdc.xy * 0.5) + 0.5;
    if (any(shadowUvLocal <= float2(0.0, 0.0)) || any(shadowUvLocal >= float2(1.0, 1.0)))
    {
        return 1.0;
    }

    const float4 atlasRect = camera.shadowAtlasUvRects[cascadeIndex];
    const float2 shadowUv = atlasRect.xy + (shadowUvLocal * atlasRect.zw);
    const float shadowDepthRef = shadowNdc.z;
    if (shadowDepthRef <= 0.0 || shadowDepthRef >= 1.0)
    {
        return 1.0;
    }

    uint texW = 0u;
    uint texH = 0u;
    shadowMap.GetDimensions(texW, texH);
    const float2 atlasTexelSize = float2(
        1.0 / max(float(texW), 1.0),
        1.0 / max(float(texH), 1.0)
    );
    const float2 localTexelSize = atlasTexelSize / max(atlasRect.zw, float2(1e-6, 1e-6));
    const float cascadeT = float(cascadeIndex) / 3.0;
    const float baseBias = lerp(camera.shadowConfig0.z, camera.shadowConfig0.w, cascadeT);
    const float bias = baseBias * max(localTexelSize.x, localTexelSize.y);
    return shadowMap.SampleCmpLevelZero(shadowUv, shadowDepthRef + bias);
}

bool inBounds(int3 cell, int3 dims)
{
    return
        cell.x >= 0 && cell.y >= 0 && cell.z >= 0 &&
        cell.x < dims.x && cell.y < dims.y && cell.z < dims.z;
}

float sampleSunVoxelVisibility(int3 cell, int3 faceOffset, int3 dims, float3 sunDirection)
{
    const float3 toSun = normalize(-sunDirection);
    float3 rayPos = float3(cell) + 0.5 + (float3(faceOffset) * 0.51);
    const float rayStepCells = 1.0;
    const int maxRaySteps = 8;
    const float blockerStepTransmission = 0.78;
    float visibility = 1.0;
    for (int step = 0; step < maxRaySteps; ++step)
    {
        rayPos += toSun * rayStepCells;
        const int3 sampleCell = int3(floor(rayPos));
        if (!inBounds(sampleCell, dims))
        {
            return visibility;
        }

        const float sampleOccupancy = voxelGiOccupancy.Load(int4(sampleCell, 0)).x;
        if (sampleOccupancy > 0.5)
        {
            visibility *= blockerStepTransmission;
            if (visibility <= 0.20)
            {
                return 0.20;
            }
        }
    }

    return visibility;
}

#ifdef VOXEL_GI_RT_SURFACE
static const float kPi = 3.14159265;

float2 rtSurfaceDiskSampleOffset(int sampleIndex)
{
    if (sampleIndex == 0) return float2(0.0, 0.0);
    if (sampleIndex == 1) return float2(0.72, 0.0);
    if (sampleIndex == 2) return float2(-0.36, 0.6235);
    return float2(-0.36, -0.6235);
}

float hashRtSurfaceRotation(int3 cell, int faceIndex)
{
    const float3 p = float3(cell) + float3(float(faceIndex) * 0.173, float(faceIndex) * 0.071, float(faceIndex) * 0.119);
    return frac(sin(dot(p, float3(12.9898, 78.233, 37.719))) * 43758.5453);
}

float2 rotateRtSurfaceOffset(float2 offset, float angle)
{
    const float s = sin(angle);
    const float c = cos(angle);
    return float2((offset.x * c) - (offset.y * s), (offset.x * s) + (offset.y * c));
}

float sampleRtSurfaceRay(float3 origin, float3 direction, float tMin, float tMax)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> rayQuery;
    rayQuery.TraceRayInline(giSurfaceScene, 0, 0xFF, ray);
    while (rayQuery.Proceed()) {}
    return (rayQuery.CommittedStatus() == COMMITTED_NOTHING) ? 1.0 : 0.0;
}

float sampleRtSunVisibility(float3 faceWorldPosition, float3 normal, float ndotl, int3 cell, int faceIndex)
{
    const float cellSize = max(camera.voxelGiGridOriginCellSize.w, 0.001);
    const float receiverBiasScale = clamp(camera.shadowConfig4.w, 0.25, 4.0);
    const float normalOffset = max(cellSize * 0.01 * receiverBiasScale, cellSize * 0.0025);
    const float3 origin = faceWorldPosition + (normal * normalOffset);
    const float3 toSun = normalize(-camera.sunDirectionIntensity.xyz);
    const float tMin = max(cellSize * 0.002, normalOffset * 0.25);
    const float tMax = max(camera.shadowCascadeSplits.w * 1.5, 512.0);
    const int sampleCount = clamp(int(camera.shadowConfig4.z + 0.5), 1, 2);
    const float sunAngularRadiusDegrees = max(camera.shadowConfig4.y, 0.0);
    const float coneRadius = tan(sunAngularRadiusDegrees * (kPi / 180.0));

    if (sampleCount <= 1 || coneRadius <= 1e-5)
    {
        return sampleRtSurfaceRay(origin, toSun, tMin, tMax);
    }

    const float3 basisUp = (abs(toSun.y) < 0.999) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 tangent = normalize(cross(basisUp, toSun));
    const float3 bitangent = cross(toSun, tangent);
    const float rotationAngle = hashRtSurfaceRotation(cell, faceIndex) * (2.0 * kPi);

    float visibility = 0.0;
    [unroll]
    for (int sampleIndex = 0; sampleIndex < 2; ++sampleIndex)
    {
        if (sampleIndex >= sampleCount) break;
        const float2 diskOffset = rotateRtSurfaceOffset(rtSurfaceDiskSampleOffset(sampleIndex), rotationAngle);
        const float3 direction = normalize(
            toSun +
            ((tangent * diskOffset.x) + (bitangent * diskOffset.y)) * coneRadius
        );
        visibility += sampleRtSurfaceRay(origin, direction, tMin, tMax);
    }
    return visibility / float(sampleCount);
}
#endif

[numthreads(4, 4, 4)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint gridW = 0u;
    uint gridH = 0u;
    uint gridD = 0u;
    voxelGiSurfacePosXOut.GetDimensions(gridW, gridH, gridD);
    if (dispatchThreadId.x >= gridW || dispatchThreadId.y >= gridH || dispatchThreadId.z >= gridD)
    {
        return;
    }

    const float4 occupancySample = voxelGiOccupancy.Load(int4(int3(dispatchThreadId), 0));
    if (occupancySample.x <= 0.5)
    {
        voxelGiSurfacePosXOut[dispatchThreadId] = float4(0.0, 0.0, 0.0, 1.0);
        voxelGiSurfaceNegXOut[dispatchThreadId] = float4(0.0, 0.0, 0.0, 1.0);
        voxelGiSurfacePosYOut[dispatchThreadId] = float4(0.0, 0.0, 0.0, 1.0);
        voxelGiSurfaceNegYOut[dispatchThreadId] = float4(0.0, 0.0, 0.0, 1.0);
        voxelGiSurfacePosZOut[dispatchThreadId] = float4(0.0, 0.0, 0.0, 1.0);
        voxelGiSurfaceNegZOut[dispatchThreadId] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float cellSize = max(camera.voxelGiGridOriginCellSize.w, 0.001);
    const float3 gridOrigin = camera.voxelGiGridOriginCellSize.xyz;
    const float3 worldPosition = gridOrigin + ((float3(dispatchThreadId) + 0.5) * cellSize);
    const float3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const float injectSunScale = clamp(camera.shadowConfig2.w, 0.0, 4.0);
    const float injectShScale = clamp(camera.shadowConfig3.x, 0.0, 4.0);

    const float3 normals[6] = {
        float3(1.0, 0.0, 0.0),
        float3(-1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, -1.0, 0.0),
        float3(0.0, 0.0, 1.0),
        float3(0.0, 0.0, -1.0)
    };
    const int3 faceOffsets[6] = {
        int3(1, 0, 0),
        int3(-1, 0, 0),
        int3(0, 1, 0),
        int3(0, -1, 0),
        int3(0, 0, 1),
        int3(0, 0, -1)
    };
    const int3 cell = int3(dispatchThreadId);
    const int3 dims = int3(int(gridW), int(gridH), int(gridD));
    const float skyColumnExposure = clamp(voxelGiSkyExposure[dispatchThreadId].x, 0.0, 1.0);

    float3 cachedSurfaceIrradiancePosX = float3(0.0, 0.0, 0.0);
    float3 cachedSurfaceIrradianceNegX = float3(0.0, 0.0, 0.0);
    float3 cachedSurfaceIrradiancePosY = float3(0.0, 0.0, 0.0);
    float3 cachedSurfaceIrradianceNegY = float3(0.0, 0.0, 0.0);
    float3 cachedSurfaceIrradiancePosZ = float3(0.0, 0.0, 0.0);
    float3 cachedSurfaceIrradianceNegZ = float3(0.0, 0.0, 0.0);
    const float skyBlockedStepVisibility = 0.78;
    for (int i = 0; i < 6; ++i)
    {
        const float3 normal = normals[i];
        const int3 adjacentCell = cell + faceOffsets[i];
        bool faceExposed = true;
        if (inBounds(adjacentCell, dims))
        {
            const float adjacentOccupancy = voxelGiOccupancy.Load(int4(adjacentCell, 0)).x;
            faceExposed = adjacentOccupancy <= 0.5;
        }
        if (!faceExposed)
        {
            continue;
        }

        // Cheap local occlusion heuristic: one additional probe beyond the face-open check.
        float skyLocalOcclusion = 1.0;
        const int3 forwardCell = adjacentCell + faceOffsets[i];
        if (inBounds(forwardCell, dims))
        {
            const float sampleOccupancy = voxelGiOccupancy.Load(int4(forwardCell, 0)).x;
            if (sampleOccupancy > 0.5)
            {
                skyLocalOcclusion = skyBlockedStepVisibility;
            }
        }
        const float ndotl = max(dot(normal, -sunDirection), 0.0);
        const float3 faceCenterOffset = normal * (cellSize * 0.51);
#ifndef VOXEL_GI_RT_SURFACE
        const float3 receiverBiasOffset = normal * (cellSize * 0.02);
        const float shadowVisibility =
            (ndotl > 0.0001) ? sampleShadowVisibility(worldPosition + faceCenterOffset + receiverBiasOffset) : 0.0;
        const float voxelSunVisibility =
            (ndotl > 0.0001) ? sampleSunVoxelVisibility(cell, faceOffsets[i], dims, sunDirection) : 0.0;
        const float directVisibility = shadowVisibility * voxelSunVisibility;
#else
        const float directVisibility =
            (ndotl > 0.0001) ? sampleRtSunVisibility(worldPosition + faceCenterOffset, normal, ndotl, cell, i) : 0.0;
#endif
        const float skyFactor = lerp(0.25, 1.0, skyColumnExposure) * skyLocalOcclusion;
        const float upFactor = saturate((normal.y * 0.5) + 0.5);
        const float ambientBoost = lerp(0.6, 1.2, upFactor);
        const float3 skyIrradiance = evaluateShIrradiance(normal) * (injectShScale * skyFactor * ambientBoost);
        const float3 directIrradiance =
            camera.sunColorShadow.xyz * (sunIntensity * ndotl * directVisibility * injectSunScale);
        const float3 faceIrradianceColor = max(directIrradiance + skyIrradiance, float3(0.0, 0.0, 0.0));
        if (i == 0) cachedSurfaceIrradiancePosX = faceIrradianceColor;
        if (i == 1) cachedSurfaceIrradianceNegX = faceIrradianceColor;
        if (i == 2) cachedSurfaceIrradiancePosY = faceIrradianceColor;
        if (i == 3) cachedSurfaceIrradianceNegY = faceIrradianceColor;
        if (i == 4) cachedSurfaceIrradiancePosZ = faceIrradianceColor;
        if (i == 5) cachedSurfaceIrradianceNegZ = faceIrradianceColor;
    }
    const float maxCachedSurfaceIrradiance = 8.0;
    const float3 maxIrradiance = float3(
        maxCachedSurfaceIrradiance,
        maxCachedSurfaceIrradiance,
        maxCachedSurfaceIrradiance
    );
    voxelGiSurfacePosXOut[dispatchThreadId] = float4(min(cachedSurfaceIrradiancePosX, maxIrradiance), 1.0);
    voxelGiSurfaceNegXOut[dispatchThreadId] = float4(min(cachedSurfaceIrradianceNegX, maxIrradiance), 1.0);
    voxelGiSurfacePosYOut[dispatchThreadId] = float4(min(cachedSurfaceIrradiancePosY, maxIrradiance), 1.0);
    voxelGiSurfaceNegYOut[dispatchThreadId] = float4(min(cachedSurfaceIrradianceNegY, maxIrradiance), 1.0);
    voxelGiSurfacePosZOut[dispatchThreadId] = float4(min(cachedSurfaceIrradiancePosZ, maxIrradiance), 1.0);
    voxelGiSurfaceNegZOut[dispatchThreadId] = float4(min(cachedSurfaceIrradianceNegZ, maxIrradiance), 1.0);
}
