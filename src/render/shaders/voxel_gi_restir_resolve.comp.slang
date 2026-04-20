import camera_uniform;
import sh_lighting;

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

struct RestirReservoir {
    float4 radianceWeight;
    float4 state;
};

[[vk::binding(17, 0)]]
RWStructuredBuffer<RestirReservoir> restirCurrent;

uint restirReservoirIndex(uint3 cell, int faceIndex, uint3 dims)
{
    const uint cellIndex = (cell.z * dims.y * dims.x) + (cell.y * dims.x) + cell.x;
    return (cellIndex * 6u) + uint(faceIndex);
}

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
    const int3 dims = int3(int(gridW), int(gridH), int(gridD));
    const int3 cell = int3(dispatchThreadId);
    const float injectShScale = clamp(camera.shadowConfig3.x, 0.0, 4.0);
    const float skyColumnExposure = clamp(voxelGiSkyExposure[dispatchThreadId].x, 0.0, 1.0);

    float3 faceIrradiance[6];
    [unroll]
    for (int i = 0; i < 6; ++i) {
        faceIrradiance[i] = float3(0.0, 0.0, 0.0);
    }

    [unroll]
    for (int faceIndex = 0; faceIndex < 6; ++faceIndex)
    {
        const int3 adjacentCell = cell + faceOffsets[faceIndex];
        bool faceExposed = true;
        if (inBounds(adjacentCell, dims))
        {
            faceExposed = voxelGiOccupancy.Load(int4(adjacentCell, 0)).x <= 0.5;
        }
        if (!faceExposed)
        {
            continue;
        }
        const float3 normal = normals[faceIndex];
        const float upFactor = saturate((normal.y * 0.5) + 0.5);
        const float ambientBoost = lerp(0.6, 1.2, upFactor);
        const float3 skyIrradiance = evaluateShIrradiance(normal) * (injectShScale * lerp(0.25, 1.0, skyColumnExposure) * ambientBoost);
        const RestirReservoir reservoir = restirCurrent[restirReservoirIndex(dispatchThreadId, faceIndex, uint3(gridW, gridH, gridD))];
        const float3 directIrradiance = (reservoir.state.w > 0.0) ? reservoir.radianceWeight.xyz : float3(0.0, 0.0, 0.0);
        faceIrradiance[faceIndex] = max(directIrradiance + skyIrradiance, float3(0.0, 0.0, 0.0));
    }

    const float3 maxIrradiance = float3(8.0, 8.0, 8.0);
    voxelGiSurfacePosXOut[dispatchThreadId] = float4(min(faceIrradiance[0], maxIrradiance), 1.0);
    voxelGiSurfaceNegXOut[dispatchThreadId] = float4(min(faceIrradiance[1], maxIrradiance), 1.0);
    voxelGiSurfacePosYOut[dispatchThreadId] = float4(min(faceIrradiance[2], maxIrradiance), 1.0);
    voxelGiSurfaceNegYOut[dispatchThreadId] = float4(min(faceIrradiance[3], maxIrradiance), 1.0);
    voxelGiSurfacePosZOut[dispatchThreadId] = float4(min(faceIrradiance[4], maxIrradiance), 1.0);
    voxelGiSurfaceNegZOut[dispatchThreadId] = float4(min(faceIrradiance[5], maxIrradiance), 1.0);
}
