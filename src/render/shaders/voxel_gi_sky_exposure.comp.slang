// Compile for Vulkan SPIR-V with column-major matrices to match Renderer.cpp uniform layout:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/voxel_gi_sky_exposure.comp.slang -o src/render/shaders/voxel_gi_sky_exposure.comp.slang.spv

import camera_uniform;

[[vk::binding(5, 0)]]
Texture3D<float4> voxelGiOccupancy;

[[vk::binding(12, 0)]]
RWTexture3D<float4> voxelGiSkyExposureOut;

bool inBounds(int3 cell, int3 dims)
{
    return
        cell.x >= 0 && cell.y >= 0 && cell.z >= 0 &&
        cell.x < dims.x && cell.y < dims.y && cell.z < dims.z;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint gridW = 0u;
    uint gridH = 0u;
    uint gridD = 0u;
    voxelGiSkyExposureOut.GetDimensions(gridW, gridH, gridD);
    if (dispatchThreadId.x >= gridW || dispatchThreadId.y >= gridD)
    {
        return;
    }

    const int x = int(dispatchThreadId.x);
    const int z = int(dispatchThreadId.y);
    const int topY = int(gridH) - 1;
    bool blockedAbove = false;
    for (int y = topY; y >= 0; --y)
    {
        const int3 cell = int3(x, y, z);
        const float occupancy = voxelGiOccupancy.Load(int4(cell, 0)).x;
        const float skyExposure = (!blockedAbove && occupancy <= 0.5) ? 1.0 : 0.0;
        voxelGiSkyExposureOut[uint3(cell)] = float4(skyExposure, skyExposure, skyExposure, 1.0);
        if (occupancy > 0.5)
        {
            blockedAbove = true;
        }
    }
}
