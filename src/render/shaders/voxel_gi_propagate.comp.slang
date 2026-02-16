// Compile for Vulkan SPIR-V with column-major matrices to match renderer.cc uniform layout:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/voxel_gi_propagate.comp.slang -o src/render/shaders/voxel_gi_propagate.comp.slang.spv

import camera_uniform;

[[vk::binding(3, 0)]]
Texture3D<float4> voxelGiPropagateIn;

[[vk::binding(4, 0)]]
RWTexture3D<float4> voxelGiPropagateOut;

[[vk::binding(5, 0)]]
Texture3D<float4> voxelGiOccupancy;

static const int kGroupDim = 4;
static const int kHalo = 1;
static const int kSharedDim = kGroupDim + (2 * kHalo);
static const int kSharedVoxelCount = kSharedDim * kSharedDim * kSharedDim;
groupshared float4 sRadiance[kSharedDim][kSharedDim][kSharedDim];
groupshared float sOpen[kSharedDim][kSharedDim][kSharedDim];

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

bool inSharedBounds(int3 localCell)
{
    return
        localCell.x >= 0 && localCell.y >= 0 && localCell.z >= 0 &&
        localCell.x < kSharedDim && localCell.y < kSharedDim && localCell.z < kSharedDim;
}

float sampleOpenCellCached(int3 localCell, int3 globalCell, int3 dims)
{
    if (inSharedBounds(localCell))
    {
        return sOpen[localCell.x][localCell.y][localCell.z];
    }
    return sampleOpenCell(globalCell, dims);
}

float transportOpenRatio(int3 localCell, int3 cell, int3 direction, int3 dims)
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
    const int3 neighborLocal = localCell + direction;
    float openSamples = 0.0;
    openSamples += sampleOpenCellCached(localCell + orthoA, cell + orthoA, dims);
    openSamples += sampleOpenCellCached(localCell - orthoA, cell - orthoA, dims);
    openSamples += sampleOpenCellCached(localCell + orthoB, cell + orthoB, dims);
    openSamples += sampleOpenCellCached(localCell - orthoB, cell - orthoB, dims);
    openSamples += sampleOpenCellCached(neighborLocal + orthoA, neighbor + orthoA, dims);
    openSamples += sampleOpenCellCached(neighborLocal - orthoA, neighbor - orthoA, dims);
    openSamples += sampleOpenCellCached(neighborLocal + orthoB, neighbor + orthoB, dims);
    openSamples += sampleOpenCellCached(neighborLocal - orthoB, neighbor - orthoB, dims);
    return openSamples * (1.0 / 8.0);
}

[numthreads(4, 4, 4)]
void main(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID,
    uint groupIndex : SV_GroupIndex
)
{
    uint gridW = 0u;
    uint gridH = 0u;
    uint gridD = 0u;
    voxelGiPropagateIn.GetDimensions(gridW, gridH, gridD);
    const bool inDispatchBounds = dispatchThreadId.x < gridW && dispatchThreadId.y < gridH && dispatchThreadId.z < gridD;

    const int3 dims = int3(int(gridW), int(gridH), int(gridD));
    const int3 cell = int3(dispatchThreadId);
    const float propagateBlend = clamp(camera.shadowConfig3.z, 0.0, 1.0);
    const float propagateDecay = clamp(camera.skyConfig2.w, 0.0, 1.0);
    const int3 localCell = int3(groupThreadId) + int3(kHalo, kHalo, kHalo);

    const int3 sharedBaseCell = (int3(groupId) * kGroupDim) - int3(kHalo, kHalo, kHalo);
    for (uint sharedIndex = groupIndex; sharedIndex < kSharedVoxelCount; sharedIndex += (kGroupDim * kGroupDim * kGroupDim))
    {
        const int sx = int(sharedIndex % kSharedDim);
        const int sy = int((sharedIndex / kSharedDim) % kSharedDim);
        const int sz = int(sharedIndex / (kSharedDim * kSharedDim));
        const int3 localSharedCell = int3(sx, sy, sz);
        const int3 globalSharedCell = sharedBaseCell + localSharedCell;

        float occupancySample = 1.0;
        float4 radianceSample = float4(0.0, 0.0, 0.0, 1.0);
        if (inBounds(globalSharedCell, dims))
        {
            occupancySample = voxelGiOccupancy.Load(int4(globalSharedCell, 0)).x;
            radianceSample = voxelGiPropagateIn.Load(int4(globalSharedCell, 0));
        }
        sOpen[sx][sy][sz] = (occupancySample <= 0.5) ? 1.0 : 0.0;
        sRadiance[sx][sy][sz] = radianceSample;
    }
    GroupMemoryBarrierWithGroupSync();

    if (!inDispatchBounds)
    {
        return;
    }

    const float occupancy = 1.0 - sOpen[localCell.x][localCell.y][localCell.z];

    if (occupancy > 0.5)
    {
        voxelGiPropagateOut[dispatchThreadId] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float3 center = sRadiance[localCell.x][localCell.y][localCell.z].xyz;

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
        const int3 neighborLocalCell = localCell + offsets[i];
        const float neighborOpen = sampleOpenCellCached(neighborLocalCell, neighborCell, dims);
        if (neighborOpen < 0.5)
        {
            continue;
        }
        const float openRatio = transportOpenRatio(localCell, cell, offsets[i], dims);
        const float transportWeight = lerp(0.10, 1.0, openRatio);
        const float3 neighborRadiance = inSharedBounds(neighborLocalCell)
            ? sRadiance[neighborLocalCell.x][neighborLocalCell.y][neighborLocalCell.z].xyz
            : voxelGiPropagateIn.Load(int4(neighborCell, 0)).xyz;
        neighborSum += neighborRadiance * transportWeight;
        neighborWeight += transportWeight;
    }

    const float3 diffusion = (neighborWeight > 0.0) ? (neighborSum / neighborWeight) : center;
    const float3 mixed = lerp(center, diffusion, propagateBlend);
    const float3 propagated = max(mixed * propagateDecay, float3(0.0, 0.0, 0.0));
    voxelGiPropagateOut[dispatchThreadId] = float4(propagated, 1.0);
}
