// Compile for Vulkan SPIR-V with column-major matrices to match Renderer.cpp uniform layout:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/voxel_gi_propagate.comp.slang -o src/render/shaders/voxel_gi_propagate.comp.slang.spv

import camera_uniform;

[[vk::binding(3, 0)]]
RWTexture3D<float4> voxelGiPropagateIn;

[[vk::binding(4, 0)]]
RWTexture3D<float4> voxelGiPropagateOut;

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
    voxelGiPropagateIn.GetDimensions(gridW, gridH, gridD);
    if (dispatchThreadId.x >= gridW || dispatchThreadId.y >= gridH || dispatchThreadId.z >= gridD)
    {
        return;
    }

    const int3 dims = int3(int(gridW), int(gridH), int(gridD));
    const int3 cell = int3(dispatchThreadId);
    const float propagateBlend = clamp(camera.shadowConfig3.z, 0.0, 1.0);
    const float propagateDecay = clamp(camera.skyConfig2.w, 0.0, 1.0);
    const float occupancy = voxelGiOccupancy.Load(int4(int3(dispatchThreadId), 0)).x;
    if (occupancy > 0.5)
    {
        voxelGiPropagateOut[dispatchThreadId] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float3 center = voxelGiPropagateIn[dispatchThreadId].xyz;

    const int3 offsets[6] = {
        int3(1, 0, 0),
        int3(-1, 0, 0),
        int3(0, 1, 0),
        int3(0, -1, 0),
        int3(0, 0, 1),
        int3(0, 0, -1)
    };

    // Occupied boundary cells are persistent colored sources; boost their influence slightly
    // so local bleed reads more clearly without adding extra directional storage yet.
    const float occupiedBoundaryWeight = 1.35;
    const float diffusionScale = (1.0 / 6.0);
    const float thinWallTransmission = 0.25;
    float3 neighborWeightedSum = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < 6; ++i)
    {
        const int3 neighborCell = cell + offsets[i];
        if (!inBounds(neighborCell, dims))
        {
            continue;
        }
        const float4 neighborOccupancySample = voxelGiOccupancy.Load(int4(neighborCell, 0));
        if (neighborOccupancySample.x > 0.5)
        {
            const float3 boundaryAlbedo = saturate(neighborOccupancySample.yzw);
            const float boundaryIrradiance = sampleDirectionalSurfaceIrradiance(neighborCell, -offsets[i]);
            const float3 boundaryRadiance = boundaryAlbedo * boundaryIrradiance;
            neighborWeightedSum += boundaryRadiance * occupiedBoundaryWeight;
            continue;
        }
        float neighborWeight = 1.0;
        const int3 blockerCell = cell + (offsets[i] * 2);
        if (inBounds(blockerCell, dims))
        {
            const float blockerOccupancy = voxelGiOccupancy.Load(int4(blockerCell, 0)).x;
            if (blockerOccupancy > 0.5)
            {
                neighborWeight = thinWallTransmission;
            }
        }
        neighborWeightedSum += voxelGiPropagateIn[uint3(neighborCell)].xyz * neighborWeight;
    }

    const float3 diffusion = neighborWeightedSum * diffusionScale;
    const float3 mixed = lerp(center, diffusion, propagateBlend);
    const float3 propagated = max(mixed * propagateDecay, float3(0.0, 0.0, 0.0));
    voxelGiPropagateOut[dispatchThreadId] = float4(propagated, 1.0);
}
