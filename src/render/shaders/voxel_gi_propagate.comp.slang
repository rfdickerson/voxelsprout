// Compile for Vulkan SPIR-V with column-major matrices to match Renderer.cpp uniform layout:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/voxel_gi_propagate.comp.slang -o src/render/shaders/voxel_gi_propagate.comp.slang.spv

import camera_uniform;

[[vk::binding(3, 0)]]
RWTexture3D<float4> voxelGiPropagateIn;

[[vk::binding(4, 0)]]
RWTexture3D<float4> voxelGiPropagateOut;

[[vk::binding(5, 0)]]
Texture3D<float4> voxelGiOccupancy;

bool inBounds(int3 cell, int3 dims)
{
    return
        cell.x >= 0 && cell.y >= 0 && cell.z >= 0 &&
        cell.x < dims.x && cell.y < dims.y && cell.z < dims.z;
}

float sampleOpenCell(int3 cell, int3 dims)
{
    if (!inBounds(cell, dims))
    {
        return 1.0;
    }
    const float occupancy = voxelGiOccupancy.Load(int4(cell, 0)).x;
    return (occupancy <= 0.5) ? 1.0 : 0.0;
}

float transportOpenRatio(int3 cell, int3 direction, int3 dims)
{
    int3 orthoA = int3(1, 0, 0);
    int3 orthoB = int3(0, 1, 0);
    if (abs(direction.x) > 0)
    {
        orthoA = int3(0, 1, 0);
        orthoB = int3(0, 0, 1);
    }
    else if (abs(direction.y) > 0)
    {
        orthoA = int3(1, 0, 0);
        orthoB = int3(0, 0, 1);
    }
    else
    {
        orthoA = int3(1, 0, 0);
        orthoB = int3(0, 1, 0);
    }

    const int3 neighbor = cell + direction;
    float openSamples = 0.0;
    openSamples += sampleOpenCell(cell + orthoA, dims);
    openSamples += sampleOpenCell(cell - orthoA, dims);
    openSamples += sampleOpenCell(cell + orthoB, dims);
    openSamples += sampleOpenCell(cell - orthoB, dims);
    openSamples += sampleOpenCell(neighbor + orthoA, dims);
    openSamples += sampleOpenCell(neighbor - orthoA, dims);
    openSamples += sampleOpenCell(neighbor + orthoB, dims);
    openSamples += sampleOpenCell(neighbor - orthoB, dims);
    return openSamples * (1.0 / 8.0);
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

    float3 neighborSum = float3(0.0, 0.0, 0.0);
    float neighborWeight = 0.0;
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
            continue;
        }
        const float openRatio = transportOpenRatio(cell, offsets[i], dims);
        const float transportWeight = lerp(0.10, 1.0, openRatio);
        neighborSum += voxelGiPropagateIn[uint3(neighborCell)].xyz * transportWeight;
        neighborWeight += transportWeight;
    }

    const float3 diffusion = (neighborWeight > 0.0) ? (neighborSum / neighborWeight) : center;
    const float3 mixed = lerp(center, diffusion, propagateBlend);
    const float3 propagated = max(mixed * propagateDecay, float3(0.0, 0.0, 0.0));
    voxelGiPropagateOut[dispatchThreadId] = float4(propagated, 1.0);
}
