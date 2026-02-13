// Compile for Vulkan SPIR-V with column-major matrices to match Renderer.cpp uniform layout:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/voxel_gi_propagate.comp.slang -o src/render/shaders/voxel_gi_propagate.comp.slang.spv

[[vk::binding(3, 0)]]
RWTexture3D<float4> voxelGiPropagateIn;

[[vk::binding(4, 0)]]
RWTexture3D<float4> voxelGiPropagateOut;

[[vk::binding(5, 0)]]
Sampler3D<float4> voxelGiOccupancy;

bool inBounds(int3 cell, int3 dims)
{
    return
        cell.x >= 0 && cell.y >= 0 && cell.z >= 0 &&
        cell.x < dims.x && cell.y < dims.y && cell.z < dims.z;
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
    const float3 occupancyUv =
        (float3(dispatchThreadId) + 0.5) / float3(float(gridW), float(gridH), float(gridD));
    const float occupancy = voxelGiOccupancy.SampleLevel(occupancyUv, 0.0).x;
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
    int neighborCount = 0;
    for (int i = 0; i < 6; ++i)
    {
        const int3 neighborCell = cell + offsets[i];
        if (!inBounds(neighborCell, dims))
        {
            continue;
        }
        const float3 neighborUv =
            (float3(neighborCell) + 0.5) / float3(float(gridW), float(gridH), float(gridD));
        const float neighborOccupancy = voxelGiOccupancy.SampleLevel(neighborUv, 0.0).x;
        if (neighborOccupancy > 0.5)
        {
            continue;
        }
        neighborSum += voxelGiPropagateIn[uint3(neighborCell)].xyz;
        neighborCount += 1;
    }

    const float3 neighborAverage = (neighborCount > 0) ? (neighborSum / float(neighborCount)) : center;
    const float3 mixed = lerp(center, neighborAverage, 0.62);
    const float3 propagated = max(mixed * 0.93, float3(0.0, 0.0, 0.0));
    voxelGiPropagateOut[dispatchThreadId] = float4(propagated, 1.0);
}
