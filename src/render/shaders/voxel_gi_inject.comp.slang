// Compile for Vulkan SPIR-V with column-major matrices to match Renderer.cpp uniform layout:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/voxel_gi_inject.comp.slang -o src/render/shaders/voxel_gi_inject.comp.slang.spv

import camera_uniform;

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

float sampleDirectionalSurfaceIrradiance(int3 cell, int3 directionToReceiver)
{
    const float3 surfacePos = max(voxelGiSurfacePosCache[uint3(cell)].xyz, float3(0.0, 0.0, 0.0));
    const float3 surfaceNeg = max(voxelGiSurfaceNegCache[uint3(cell)].xyz, float3(0.0, 0.0, 0.0));

    const float3 dir = float3(directionToReceiver);
    const float3 absDir = abs(dir);
    if ((absDir.x + absDir.y + absDir.z) <= 0.0)
    {
        return 0.0;
    }

    // Pick the face from the dominant axis; ties resolve deterministically (X, then Y, then Z).
    if (absDir.x >= absDir.y && absDir.x >= absDir.z)
    {
        return (dir.x >= 0.0) ? surfacePos.x : surfaceNeg.x;
    }
    if (absDir.y >= absDir.z)
    {
        return (dir.y >= 0.0) ? surfacePos.y : surfaceNeg.y;
    }
    return (dir.z >= 0.0) ? surfacePos.z : surfaceNeg.z;
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
        const float neighborIrradiance = sampleDirectionalSurfaceIrradiance(neighborCell, -offsets[i]);
        bouncedLightSum += surfaceAlbedo * neighborIrradiance;
    }

    const float3 injected =
        max(bouncedLightSum * (injectBounceScale * lambertInvPi), float3(0.0, 0.0, 0.0));
    voxelGiInjectOut[dispatchThreadId] = float4(injected, 1.0);
}
