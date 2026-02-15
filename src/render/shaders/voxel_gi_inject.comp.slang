// Compile for Vulkan SPIR-V with column-major matrices to match Renderer.cpp uniform layout:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/voxel_gi_inject.comp.slang -o src/render/shaders/voxel_gi_inject.comp.slang.spv

import camera_uniform;
import sh_lighting;

[[vk::binding(2, 0)]]
RWTexture3D<float4> voxelGiInjectOut;

[[vk::binding(5, 0)]]
Texture3D<float4> voxelGiOccupancy;

[[vk::binding(6, 0)]]
RWTexture3D<float4> voxelGiSurfacePosCache;

[[vk::binding(7, 0)]]
RWTexture3D<float4> voxelGiSurfaceNegCache;

bool inBounds(int3 cell, int3 dims)
{
    return
        cell.x >= 0 && cell.y >= 0 && cell.z >= 0 &&
        cell.x < dims.x && cell.y < dims.y && cell.z < dims.z;
}

float sampleDirectionalSurfaceIrradiance(int3 cell, int3 directionToReceiver, int3 dims, out float3 selectedFaceNormalOut)
{
    const float3 surfacePos = max(voxelGiSurfacePosCache[uint3(cell)].xyz, float3(0.0, 0.0, 0.0));
    const float3 surfaceNeg = max(voxelGiSurfaceNegCache[uint3(cell)].xyz, float3(0.0, 0.0, 0.0));
    const float isotropicIrradiance =
        (surfacePos.x + surfacePos.y + surfacePos.z + surfaceNeg.x + surfaceNeg.y + surfaceNeg.z) / 6.0;

    const float3 dir = float3(directionToReceiver);
    const float3 absDir = abs(dir);
    if ((absDir.x + absDir.y + absDir.z) <= 0.0)
    {
        selectedFaceNormalOut = float3(0.0, 1.0, 0.0);
        return 0.0;
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
    float directional = 0.0;
    float3 blendedFaceNormal = float3(0.0, 0.0, 0.0);
    bool receiverFacingSideOpen = false;
    if (axisWeights.x > 0.0001)
    {
        const int3 faceOffsetX = (dir.x >= 0.0) ? int3(1, 0, 0) : int3(-1, 0, 0);
        const float faceIrradianceX = (dir.x >= 0.0) ? surfacePos.x : surfaceNeg.x;
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
            directional += faceIrradianceX * axisWeights.x;
            blendedFaceNormal += float3(faceOffsetX) * axisWeights.x;
        }
    }
    if (axisWeights.y > 0.0001)
    {
        const int3 faceOffsetY = (dir.y >= 0.0) ? int3(0, 1, 0) : int3(0, -1, 0);
        const float faceIrradianceY = (dir.y >= 0.0) ? surfacePos.y : surfaceNeg.y;
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
            directional += faceIrradianceY * axisWeights.y;
            blendedFaceNormal += float3(faceOffsetY) * axisWeights.y;
        }
    }
    if (axisWeights.z > 0.0001)
    {
        const int3 faceOffsetZ = (dir.z >= 0.0) ? int3(0, 0, 1) : int3(0, 0, -1);
        const float faceIrradianceZ = (dir.z >= 0.0) ? surfacePos.z : surfaceNeg.z;
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
            directional += faceIrradianceZ * axisWeights.z;
            blendedFaceNormal += float3(faceOffsetZ) * axisWeights.z;
        }
    }

    const float isotropicFloorScaleOpen = lerp(0.04, 0.18, openRatio);
    const float isotropicFloorScaleBlocked = lerp(0.01, 0.06, openRatio);
    const float isotropicFloorScale =
        receiverFacingSideOpen ? isotropicFloorScaleOpen : isotropicFloorScaleBlocked;

    if (dot(blendedFaceNormal, blendedFaceNormal) <= 0.0001)
    {
        selectedFaceNormalOut = normalize(dir);
        return isotropicIrradiance * isotropicFloorScale;
    }

    selectedFaceNormalOut = normalize(blendedFaceNormal);
    // Openness-driven isotropic floor reduces starvation while limiting wraparound in tight spaces.
    return max(directional, isotropicIrradiance * isotropicFloorScale);
}

float3 sampleIncidentLightTint(float3 faceNormal)
{
    const float3 lumaWeights = float3(0.2126, 0.7152, 0.0722);
    const float3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const float ndotl = max(dot(faceNormal, -sunDirection), 0.0);
    const float3 sunLight = camera.sunColorShadow.xyz * (sunIntensity * ndotl);
    const float3 skyLight = evaluateShIrradiance(faceNormal);
    const float3 incident = max(sunLight + skyLight, float3(0.0, 0.0, 0.0));
    const float incidentLuma = dot(incident, lumaWeights);
    if (incidentLuma <= 0.0001)
    {
        return float3(1.0, 1.0, 1.0);
    }
    // Normalize by luminance so tint affects hue/chroma while scalar irradiance controls energy.
    return incident / incidentLuma;
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

    const float3 lumaWeights = float3(0.2126, 0.7152, 0.0722);
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
        const float surfaceLuma = dot(surfaceAlbedo, lumaWeights);
        // Compress bounce chroma so highly saturated voxels do not over-tint distant surfaces.
        const float3 bounceAlbedo = lerp(float3(surfaceLuma, surfaceLuma, surfaceLuma), surfaceAlbedo, 0.40);
        float3 emitterFaceNormal = float3(0.0, 1.0, 0.0);
        const float neighborIrradiance =
            sampleDirectionalSurfaceIrradiance(neighborCell, -offsets[i], dims, emitterFaceNormal);
        const float3 incidentTint = sampleIncidentLightTint(emitterFaceNormal);
        bouncedLightSum += bounceAlbedo * (neighborIrradiance * incidentTint);
    }

    const float3 injected =
        max(bouncedLightSum * (injectBounceScale * lambertInvPi), float3(0.0, 0.0, 0.0));
    voxelGiInjectOut[dispatchThreadId] = float4(injected, 1.0);
}
