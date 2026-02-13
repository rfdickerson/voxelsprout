// Compile for Vulkan SPIR-V with column-major matrices to match Renderer.cpp uniform layout:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/voxel_gi_inject.comp.slang -o src/render/shaders/voxel_gi_inject.comp.slang.spv

import camera_uniform;
import sh_lighting;

[[vk::binding(1, 0)]]
Sampler2DShadow shadowMap;

[[vk::binding(2, 0)]]
RWTexture3D<float4> voxelGiInjectOut;

[[vk::binding(5, 0)]]
Sampler3D<float4> voxelGiOccupancy;

bool inBounds(int3 cell, int3 dims)
{
    return
        cell.x >= 0 && cell.y >= 0 && cell.z >= 0 &&
        cell.x < dims.x && cell.y < dims.y && cell.z < dims.z;
}

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
    const float texelSize = 1.0 / max(float(max(texW, texH)), 1.0);
    const float cascadeT = float(cascadeIndex) / 3.0;
    const float baseBias = lerp(camera.shadowConfig0.z, camera.shadowConfig0.w, cascadeT);
    const float bias = baseBias * texelSize;
    return shadowMap.SampleCmpLevelZero(shadowUv, shadowDepthRef + bias);
}

[numthreads(4, 4, 4)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint gridW = 0u;
    uint gridH = 0u;
    uint gridD = 0u;
    voxelGiInjectOut.GetDimensions(gridW, gridH, gridD);
    if (dispatchThreadId.x >= gridW || dispatchThreadId.y >= gridH || dispatchThreadId.z >= gridD)
    {
        return;
    }

    const int3 dims = int3(int(gridW), int(gridH), int(gridD));
    const int3 cell = int3(dispatchThreadId);
    const float3 occupancyUv = (float3(dispatchThreadId) + 0.5) / float3(float(gridW), float(gridH), float(gridD));
    const float4 occupancySample = voxelGiOccupancy.SampleLevel(occupancyUv, 0.0);
    const float occupancy = occupancySample.x;
    if (occupancy > 0.5)
    {
        voxelGiInjectOut[dispatchThreadId] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float cellSize = max(camera.voxelGiGridOriginCellSize.w, 0.001);
    const float3 gridOrigin = camera.voxelGiGridOriginCellSize.xyz;
    const float3 worldPosition = gridOrigin + ((float3(dispatchThreadId) + 0.5) * cellSize);
    const float3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const float injectSunScale = clamp(camera.shadowConfig2.w, 0.0, 4.0);
    const float injectShScale = clamp(camera.shadowConfig3.x, 0.0, 4.0);
    const float injectBounceScale = clamp(camera.shadowConfig3.y, 0.0, 4.0);
    const int3 offsets[6] = {
        int3(1, 0, 0),
        int3(-1, 0, 0),
        int3(0, 1, 0),
        int3(0, -1, 0),
        int3(0, 0, 1),
        int3(0, 0, -1)
    };

    float3 bouncedLightSum = float3(0.0, 0.0, 0.0);
    int bouncedLightCount = 0;
    for (int i = 0; i < 6; ++i)
    {
        const int3 neighborCell = cell + offsets[i];
        if (!inBounds(neighborCell, dims))
        {
            continue;
        }

        const float3 neighborUv =
            (float3(neighborCell) + 0.5) / float3(float(gridW), float(gridH), float(gridD));
        const float4 neighborSample = voxelGiOccupancy.SampleLevel(neighborUv, 0.0);
        if (neighborSample.x <= 0.5)
        {
            continue;
        }

        const float3 surfaceAlbedo = saturate(neighborSample.yzw);
        const float3 faceNormal = normalize(-float3(offsets[i]));
        const float3 skyIrradiance = evaluateShIrradiance(faceNormal) * injectShScale;
        const float ndotl = max(dot(faceNormal, -sunDirection), 0.0);
        const float3 faceWorldPosition = worldPosition + (faceNormal * (cellSize * 0.5));
        const float shadowVisibility =
            (ndotl > 0.0001) ? sampleShadowVisibility(faceWorldPosition + (faceNormal * (cellSize * 0.05))) : 0.0;
        const float3 directOnSurface =
            camera.sunColorShadow.xyz * (sunIntensity * ndotl * shadowVisibility * injectSunScale);
        bouncedLightSum += (directOnSurface + skyIrradiance) * surfaceAlbedo;
        bouncedLightCount += 1;
    }

    const float3 injected =
        (bouncedLightCount > 0)
            ? max((bouncedLightSum / float(bouncedLightCount)) * injectBounceScale, float3(0.0, 0.0, 0.0))
            : float3(0.0, 0.0, 0.0);
    voxelGiInjectOut[dispatchThreadId] = float4(injected, 1.0);
}
