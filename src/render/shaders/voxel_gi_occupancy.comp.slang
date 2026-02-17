// Compile for Vulkan SPIR-V with column-major matrices to match renderer.cc uniform layout:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/voxel_gi_occupancy.comp.slang -o src/render/shaders/voxel_gi_occupancy.comp.slang.spv

import camera_uniform;

struct VoxelGiChunkMeta
{
    int worldMinX;
    int worldMinY;
    int worldMinZ;
    uint voxelOffset;
};

[[vk::binding(13, 0)]]
RWTexture3D<float4> voxelGiOccupancyOut;

[[vk::binding(14, 0)]]
StructuredBuffer<VoxelGiChunkMeta> voxelGiChunkMetaBuffer;

[[vk::binding(15, 0)]]
StructuredBuffer<uint> voxelGiChunkVoxelsBuffer;

float3 voxelTypeDefaultAlbedo(uint voxelType)
{
    if (voxelType == 1u) return float3(150.0 / 255.0, 154.0 / 255.0, 160.0 / 255.0);
    if (voxelType == 2u) return float3(255.0 / 255.0, 71.0 / 255.0, 56.0 / 255.0);
    if (voxelType == 3u) return float3(122.0 / 255.0, 93.0 / 255.0, 58.0 / 255.0);
    if (voxelType == 4u) return float3(80.0 / 255.0, 142.0 / 255.0, 63.0 / 255.0);
    if (voxelType == 5u) return float3(141.0 / 255.0, 106.0 / 255.0, 64.0 / 255.0);
    return float3(0.0, 0.0, 0.0);
}

float3 resolveVoxelAlbedo(uint voxelType, uint baseColorIndex)
{
    if (baseColorIndex <= 15u)
    {
        const float3 paletteColor = saturate(camera.voxelBaseColorPalette[baseColorIndex].xyz);
        const float peak = max(max(paletteColor.x, paletteColor.y), paletteColor.z);
        const float floorValue = min(min(paletteColor.x, paletteColor.y), paletteColor.z);
        const float saturation = (peak > 1e-4) ? ((peak - floorValue) / peak) : 0.0;
        // If palette entry is effectively grayscale, fall back to stable material albedo for GI chroma.
        if (saturation > 0.06)
        {
            return paletteColor;
        }
    }
    return voxelTypeDefaultAlbedo(voxelType);
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint localX = dispatchThreadId.x;
    const uint localY = dispatchThreadId.y;
    const uint chunkLayerZ = dispatchThreadId.z;
    if (localX >= 32u || localY >= 32u)
    {
        return;
    }

    uint chunkCount = 0u;
    uint chunkMetaStride = 0u;
    voxelGiChunkMetaBuffer.GetDimensions(chunkCount, chunkMetaStride);
    const uint chunkIndex = chunkLayerZ / 32u;
    const uint localZ = chunkLayerZ % 32u;
    if (chunkIndex >= chunkCount || localZ >= 32u)
    {
        return;
    }

    const VoxelGiChunkMeta chunkMeta = voxelGiChunkMetaBuffer[chunkIndex];
    const uint localLinearIndex = localX + (32u * (localZ + (32u * localY)));
    const uint packedVoxel = voxelGiChunkVoxelsBuffer[chunkMeta.voxelOffset + localLinearIndex];
    const uint voxelType = packedVoxel & 0xFFu;
    const uint baseColorIndex = (packedVoxel >> 8u) & 0xFFu;

    const float3 worldPosition = float3(
        float(chunkMeta.worldMinX) + float(localX) + 0.5,
        float(chunkMeta.worldMinY) + float(localY) + 0.5,
        float(chunkMeta.worldMinZ) + float(localZ) + 0.5
    );
    const float cellSize = max(camera.voxelGiGridOriginCellSize.w, 0.001);
    const int3 gridCell = int3(floor((worldPosition - camera.voxelGiGridOriginCellSize.xyz) / cellSize));
    uint gridW = 0u;
    uint gridH = 0u;
    uint gridD = 0u;
    voxelGiOccupancyOut.GetDimensions(gridW, gridH, gridD);
    if (gridCell.x < 0 || gridCell.y < 0 || gridCell.z < 0 ||
        gridCell.x >= int(gridW) || gridCell.y >= int(gridH) || gridCell.z >= int(gridD))
    {
        return;
    }

    if (voxelType == 0u)
    {
        voxelGiOccupancyOut[uint3(gridCell)] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    const float3 albedo = resolveVoxelAlbedo(voxelType, baseColorIndex);
    voxelGiOccupancyOut[uint3(gridCell)] = float4(1.0, albedo.x, albedo.y, albedo.z);
}
