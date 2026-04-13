import camera_uniform;

struct RestirReservoir {
    float4 radianceWeight;
    float4 state;
};

[[vk::binding(17, 0)]]
RWStructuredBuffer<RestirReservoir> restirCurrent;

[[vk::binding(19, 0)]]
RWStructuredBuffer<RestirReservoir> restirScratch;

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

    const bool useSpatial = camera.voxelGiRestirConfig1.x > 0.5;
    const int radius = clamp(int(camera.voxelGiRestirConfig1.y + 0.5), 1, 2);
    const int3 dims = int3(gridW, gridH, gridD);
    const uint3 udims = uint3(gridW, gridH, gridD);
    const int3 cell = int3(dispatchThreadId);
    const int3 neighborOffsets[6] = {
        int3(1, 0, 0),
        int3(-1, 0, 0),
        int3(0, 1, 0),
        int3(0, -1, 0),
        int3(0, 0, 1),
        int3(0, 0, -1)
    };

    [unroll]
    for (int faceIndex = 0; faceIndex < 6; ++faceIndex)
    {
        const uint reservoirIndex = restirReservoirIndex(dispatchThreadId, faceIndex, udims);
        RestirReservoir bestReservoir = restirScratch[reservoirIndex];
        if (useSpatial)
        {
            [unroll]
            for (int neighborIndex = 0; neighborIndex < 6; ++neighborIndex)
            {
                const int3 neighborCell = cell + (neighborOffsets[neighborIndex] * radius);
                if (!inBounds(neighborCell, dims))
                {
                    continue;
                }
                const RestirReservoir neighborReservoir = restirScratch[restirReservoirIndex(uint3(neighborCell), faceIndex, udims)];
                if (neighborReservoir.state.w <= 0.0)
                {
                    continue;
                }
                if (reservoirScore(neighborReservoir) > reservoirScore(bestReservoir))
                {
                    bestReservoir = neighborReservoir;
                }
            }
        }
        restirCurrent[reservoirIndex] = bestReservoir;
    }
}
