// Compile for Vulkan SPIR-V with column-major matrices to match Renderer.cpp uniform layout:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/voxel_gi_surface.comp.slang -o src/render/shaders/voxel_gi_surface.comp.slang.spv

import camera_uniform;
import sh_lighting;

[[vk::binding(1, 0)]]
Sampler2DShadow shadowMap;

[[vk::binding(5, 0)]]
Texture3D<float4> voxelGiOccupancy;

[[vk::binding(6, 0)]]
RWTexture3D<float4> voxelGiSurfacePosOut;

[[vk::binding(7, 0)]]
RWTexture3D<float4> voxelGiSurfaceNegOut;

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
    const float blockerStepTransmission = 0.55;
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
            if (visibility <= 0.08)
            {
                return 0.08;
            }
        }
    }

    return visibility;
}

[numthreads(4, 4, 4)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint gridW = 0u;
    uint gridH = 0u;
    uint gridD = 0u;
    voxelGiSurfacePosOut.GetDimensions(gridW, gridH, gridD);
    if (dispatchThreadId.x >= gridW || dispatchThreadId.y >= gridH || dispatchThreadId.z >= gridD)
    {
        return;
    }

    const float4 occupancySample = voxelGiOccupancy.Load(int4(int3(dispatchThreadId), 0));
    if (occupancySample.x <= 0.5)
    {
        voxelGiSurfacePosOut[dispatchThreadId] = float4(0.0, 0.0, 0.0, 1.0);
        voxelGiSurfaceNegOut[dispatchThreadId] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float3 surfaceAlbedo = saturate(occupancySample.yzw);
    const float albedoPeak = max(max(surfaceAlbedo.x, surfaceAlbedo.y), surfaceAlbedo.z);
    const float albedoMin = min(min(surfaceAlbedo.x, surfaceAlbedo.y), surfaceAlbedo.z);
    const float albedoSaturation = (albedoPeak > 0.0001) ? ((albedoPeak - albedoMin) / albedoPeak) : 0.0;
    // Small art-directed boost so saturated materials (like red blocks) bleed color more visibly.
    const float colorBleedBoost = lerp(1.0, 1.45, clamp(albedoSaturation, 0.0, 1.0));
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

    const float3 lumaWeights = float3(0.2126, 0.7152, 0.0722);
    float3 cachedSurfaceIrradiancePos = float3(0.0, 0.0, 0.0);
    float3 cachedSurfaceIrradianceNeg = float3(0.0, 0.0, 0.0);
    const int skyOcclusionSteps = 2;
    const float skyBlockedStepVisibility = 0.60;
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

        float skyVisibility = 1.0;
        for (int step = 1; step <= skyOcclusionSteps; ++step)
        {
            const int3 sampleCell = cell + (faceOffsets[i] * step);
            if (!inBounds(sampleCell, dims))
            {
                break;
            }

            const float sampleOccupancy = voxelGiOccupancy.Load(int4(sampleCell, 0)).x;
            if (sampleOccupancy > 0.5)
            {
                // Multiple blocked steps quickly darken enclosed spaces.
                skyVisibility *= skyBlockedStepVisibility;
            }
        }
        const float3 skyIrradiance = evaluateShIrradiance(normal) * (injectShScale * skyVisibility);
        const float ndotl = max(dot(normal, -sunDirection), 0.0);
        const float3 faceCenterOffset = normal * (cellSize * 0.51);
        const float3 receiverBiasOffset = normal * (cellSize * 0.02);
        const float shadowVisibility =
            (ndotl > 0.0001) ? sampleShadowVisibility(worldPosition + faceCenterOffset + receiverBiasOffset) : 0.0;
        const float voxelSunVisibility =
            (ndotl > 0.0001) ? sampleSunVoxelVisibility(cell, faceOffsets[i], dims, sunDirection) : 0.0;
        const float directVisibility = shadowVisibility * voxelSunVisibility;
        const float3 directIrradiance =
            camera.sunColorShadow.xyz * (sunIntensity * ndotl * directVisibility * injectSunScale);
        const float3 faceIrradianceColor = max(directIrradiance + skyIrradiance, float3(0.0, 0.0, 0.0));
        const float faceIrradiance = max(dot(faceIrradianceColor, lumaWeights) * colorBleedBoost, 0.0);
        if (i == 0) cachedSurfaceIrradiancePos.x = faceIrradiance;
        if (i == 1) cachedSurfaceIrradianceNeg.x = faceIrradiance;
        if (i == 2) cachedSurfaceIrradiancePos.y = faceIrradiance;
        if (i == 3) cachedSurfaceIrradianceNeg.y = faceIrradiance;
        if (i == 4) cachedSurfaceIrradiancePos.z = faceIrradiance;
        if (i == 5) cachedSurfaceIrradianceNeg.z = faceIrradiance;
    }
    const float maxCachedSurfaceIrradiance = 8.0;
    const float3 maxIrradiance = float3(
        maxCachedSurfaceIrradiance,
        maxCachedSurfaceIrradiance,
        maxCachedSurfaceIrradiance
    );
    voxelGiSurfacePosOut[dispatchThreadId] = float4(min(cachedSurfaceIrradiancePos, maxIrradiance), 1.0);
    voxelGiSurfaceNegOut[dispatchThreadId] = float4(min(cachedSurfaceIrradianceNeg, maxIrradiance), 1.0);
}
