import camera_uniform;

struct RestirReservoir {
    float4 radianceWeight;
    float4 state;
};

[[vk::binding(17, 0)]]
RWStructuredBuffer<RestirReservoir> restirCurrent;

[[vk::binding(18, 0)]]
StructuredBuffer<RestirReservoir> restirPrevious;

[[vk::binding(19, 0)]]
RWStructuredBuffer<RestirReservoir> restirScratch;

uint restirReservoirIndex(uint3 cell, int faceIndex, uint3 dims)
{
    const uint cellIndex = (cell.z * dims.y * dims.x) + (cell.y * dims.x) + cell.x;
    return (cellIndex * 6u) + uint(faceIndex);
}

float reservoirScore(RestirReservoir reservoir)
{
    return max(reservoir.radianceWeight.w, 0.0);
}

[numthreads(4, 4, 4)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint gridW = 64u;
    const uint gridH = 64u;
    const uint gridD = 64u;
    if (dispatchThreadId.x >= gridW || dispatchThreadId.y >= gridH || dispatchThreadId.z >= gridD)
    {
        return;
    }

    const bool historyValid = camera.voxelGiRestirConfig0.w > 0.5 && camera.voxelGiRestirConfig0.z > 0.5;
    const float ageCap = max(camera.voxelGiRestirConfig1.z, 1.0);
    const uint3 dims = uint3(gridW, gridH, gridD);
    [unroll]
    for (int faceIndex = 0; faceIndex < 6; ++faceIndex)
    {
        const uint reservoirIndex = restirReservoirIndex(dispatchThreadId, faceIndex, dims);
        RestirReservoir currentReservoir = restirCurrent[reservoirIndex];
        RestirReservoir outputReservoir = currentReservoir;
        if (historyValid)
        {
            RestirReservoir previousReservoir = restirPrevious[reservoirIndex];
            if (previousReservoir.state.w > 0.0 && previousReservoir.state.z < ageCap)
            {
                previousReservoir.state.z += 1.0;
                if (reservoirScore(previousReservoir) >= reservoirScore(currentReservoir))
                {
                    outputReservoir = previousReservoir;
                }
            }
        }
        restirScratch[reservoirIndex] = outputReservoir;
    }
}
