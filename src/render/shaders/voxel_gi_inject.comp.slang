// Compile for Vulkan SPIR-V with column-major matrices to match renderer.cc uniform layout:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/voxel_gi_inject.comp.slang -o src/render/shaders/voxel_gi_inject.comp.slang.spv

import camera_uniform;
import sh_lighting;
[[vk::binding(2, 0)]]
RWTexture3D<float4> voxelGiInjectOut;

[[vk::binding(5, 0)]]
Texture3D<float4> voxelGiOccupancy;

[[vk::binding(6, 0)]]
RWTexture3D<float4> voxelGiSurfacePosXCache;

[[vk::binding(7, 0)]]
RWTexture3D<float4> voxelGiSurfaceNegXCache;

[[vk::binding(8, 0)]]
RWTexture3D<float4> voxelGiSurfacePosYCache;

[[vk::binding(9, 0)]]
RWTexture3D<float4> voxelGiSurfaceNegYCache;

[[vk::binding(10, 0)]]
RWTexture3D<float4> voxelGiSurfacePosZCache;

[[vk::binding(11, 0)]]
RWTexture3D<float4> voxelGiSurfaceNegZCache;

[[vk::binding(12, 0)]]
RWTexture3D<float4> voxelGiSkyExposure;

bool inBounds(int3 cell, int3 dims)
{
    return
        cell.x >= 0 && cell.y >= 0 && cell.z >= 0 &&
        cell.x < dims.x && cell.y < dims.y && cell.z < dims.z;
}

float3 sampleDirectionalSurfaceIrradiance(int3 cell, int3 directionToReceiver, int3 dims)
{
    const float3 surfacePosX = max(voxelGiSurfacePosXCache[uint3(cell)].xyz, float3(0.0, 0.0, 0.0));
    const float3 surfaceNegX = max(voxelGiSurfaceNegXCache[uint3(cell)].xyz, float3(0.0, 0.0, 0.0));
    const float3 surfacePosY = max(voxelGiSurfacePosYCache[uint3(cell)].xyz, float3(0.0, 0.0, 0.0));
    const float3 surfaceNegY = max(voxelGiSurfaceNegYCache[uint3(cell)].xyz, float3(0.0, 0.0, 0.0));
    const float3 surfacePosZ = max(voxelGiSurfacePosZCache[uint3(cell)].xyz, float3(0.0, 0.0, 0.0));
    const float3 surfaceNegZ = max(voxelGiSurfaceNegZCache[uint3(cell)].xyz, float3(0.0, 0.0, 0.0));
    const float3 isotropicRadiance =
        (surfacePosX + surfaceNegX + surfacePosY + surfaceNegY + surfacePosZ + surfaceNegZ) * (1.0 / 6.0);

    const float3 dir = float3(directionToReceiver);
    const float3 absDir = abs(dir);
    if ((absDir.x + absDir.y + absDir.z) <= 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }

    // Estimate local openness around the emitter cell to modulate isotropic fallback.
    int openNeighborFaces = 0;
    const int3 neighborOffsets[6] = {
        int3(1, 0, 0),
        int3(-1, 0, 0),
        int3(0, 1, 0),
        int3(0, -1, 0),
        int3(0, 0, 1),
        int3(0, 0, -1)
    };
    for (int n = 0; n < 6; ++n)
    {
        const int3 adjacent = cell + neighborOffsets[n];
        if (!inBounds(adjacent, dims))
        {
            openNeighborFaces += 1;
            continue;
        }
        const float adjacentOccupancy = voxelGiOccupancy.Load(int4(adjacent, 0)).x;
        if (adjacentOccupancy <= 0.5)
        {
            openNeighborFaces += 1;
        }
    }
    const float openRatio = float(openNeighborFaces) / 6.0;

    // Blend +/- axis faces by direction magnitudes (cubemap-style) to avoid dominant-axis aliasing.
    const float weightDenom = absDir.x + absDir.y + absDir.z;
    const float3 axisWeights = absDir / weightDenom;
    float3 directional = float3(0.0, 0.0, 0.0);
    bool receiverFacingSideOpen = false;
    if (axisWeights.x > 0.0001)
    {
        const int3 faceOffsetX = (dir.x >= 0.0) ? int3(1, 0, 0) : int3(-1, 0, 0);
        const float3 faceRadianceX = (dir.x >= 0.0) ? surfacePosX : surfaceNegX;
        float opennessX = 1.0;
        const int3 adjacentCellX = cell + faceOffsetX;
        if (inBounds(adjacentCellX, dims))
        {
            const float adjacentOccupancyX = voxelGiOccupancy.Load(int4(adjacentCellX, 0)).x;
            if (adjacentOccupancyX > 0.5)
            {
                opennessX = 0.0;
            }
        }
        if (opennessX > 0.0)
        {
            receiverFacingSideOpen = true;
            directional += faceRadianceX * axisWeights.x;
        }
    }
    if (axisWeights.y > 0.0001)
    {
        const int3 faceOffsetY = (dir.y >= 0.0) ? int3(0, 1, 0) : int3(0, -1, 0);
        const float3 faceRadianceY = (dir.y >= 0.0) ? surfacePosY : surfaceNegY;
        float opennessY = 1.0;
        const int3 adjacentCellY = cell + faceOffsetY;
        if (inBounds(adjacentCellY, dims))
        {
            const float adjacentOccupancyY = voxelGiOccupancy.Load(int4(adjacentCellY, 0)).x;
            if (adjacentOccupancyY > 0.5)
            {
                opennessY = 0.0;
            }
        }
        if (opennessY > 0.0)
        {
            receiverFacingSideOpen = true;
            directional += faceRadianceY * axisWeights.y;
        }
    }
    if (axisWeights.z > 0.0001)
    {
        const int3 faceOffsetZ = (dir.z >= 0.0) ? int3(0, 0, 1) : int3(0, 0, -1);
        const float3 faceRadianceZ = (dir.z >= 0.0) ? surfacePosZ : surfaceNegZ;
        float opennessZ = 1.0;
        const int3 adjacentCellZ = cell + faceOffsetZ;
        if (inBounds(adjacentCellZ, dims))
        {
            const float adjacentOccupancyZ = voxelGiOccupancy.Load(int4(adjacentCellZ, 0)).x;
            if (adjacentOccupancyZ > 0.5)
            {
                opennessZ = 0.0;
            }
        }
        if (opennessZ > 0.0)
        {
            receiverFacingSideOpen = true;
            directional += faceRadianceZ * axisWeights.z;
        }
    }

    const float isotropicFloorScaleOpen = lerp(0.03, 0.12, openRatio);
    const float isotropicFloorScaleBlocked = lerp(0.005, 0.04, openRatio);
    const float isotropicFloorScale =
        receiverFacingSideOpen ? isotropicFloorScaleOpen : isotropicFloorScaleBlocked;

    // Openness-driven isotropic floor reduces starvation while limiting wraparound in tight spaces.
    return directional + (isotropicRadiance * isotropicFloorScale);
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
    const float4 occupancySample = voxelGiOccupancy.Load(int4(int3(dispatchThreadId), 0));
    const float occupancy = occupancySample.x;
    if (occupancy > 0.5)
    {
        voxelGiInjectOut[dispatchThreadId] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float injectBounceScale = clamp(camera.shadowConfig3.y, 0.0, 4.0);
    const float injectShScale = clamp(camera.shadowConfig3.x, 0.0, 4.0);
    // Lambertian conversion from irradiance to radiance.
    const float lambertInvPi = 0.31830988618;
    const int3 offsets[6] = {
        int3(1, 0, 0),
        int3(-1, 0, 0),
        int3(0, 1, 0),
        int3(0, -1, 0),
        int3(0, 0, 1),
        int3(0, 0, -1)
    };

    float3 bouncedLightSum = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < 6; ++i)
    {
        const int3 neighborCell = cell + offsets[i];
        if (!inBounds(neighborCell, dims))
        {
            continue;
        }

        const float4 neighborSample = voxelGiOccupancy.Load(int4(neighborCell, 0));
        if (neighborSample.x <= 0.5)
        {
            continue;
        }

        const float3 surfaceAlbedo = saturate(neighborSample.yzw);
        const float albedoPeak = max(max(surfaceAlbedo.x, surfaceAlbedo.y), surfaceAlbedo.z);
        const float albedoMin = min(min(surfaceAlbedo.x, surfaceAlbedo.y), surfaceAlbedo.z);
        const float albedoSaturation = (albedoPeak > 0.0001) ? ((albedoPeak - albedoMin) / albedoPeak) : 0.0;
        const float colorBleedBoost = lerp(1.0, 1.12, clamp(albedoSaturation, 0.0, 1.0));
        // Keep near-field bounce color faithful to the source surface albedo.
        const float3 bounceAlbedo = surfaceAlbedo;
        const float3 neighborIrradiance = sampleDirectionalSurfaceIrradiance(neighborCell, -offsets[i], dims);
        bouncedLightSum += bounceAlbedo * (neighborIrradiance * colorBleedBoost);
    }

    int openNeighborFaces = 0;
    for (int i = 0; i < 6; ++i)
    {
        const int3 neighborCell = cell + offsets[i];
        if (!inBounds(neighborCell, dims))
        {
            openNeighborFaces += 1;
            continue;
        }
        const float neighborOccupancy = voxelGiOccupancy.Load(int4(neighborCell, 0)).x;
        if (neighborOccupancy <= 0.5)
        {
            openNeighborFaces += 1;
        }
    }
    const float localOpenRatio = float(openNeighborFaces) / 6.0;
    const float skyExposure = clamp(voxelGiSkyExposure[dispatchThreadId].x, 0.0, 1.0);
    const float ambientGate = skyExposure * lerp(0.20, 1.0, localOpenRatio);
    // Small SH inject term keeps sky-shadowed outdoor regions from going unnaturally black.
    const float ambientInjectScale = injectShScale * 0.20;
    const float3 ambientIrradiance =
        evaluateShHemisphereIrradiance(float3(0.0, 1.0, 0.0)) * (ambientInjectScale * ambientGate);

    const float3 injected =
        max((bouncedLightSum * injectBounceScale + ambientIrradiance) * lambertInvPi, float3(0.0, 0.0, 0.0));
    voxelGiInjectOut[dispatchThreadId] = float4(injected, 1.0);
}
