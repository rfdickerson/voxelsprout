import camera_uniform;
import sh_lighting;

[[vk::binding(5, 0)]]
Texture3D<float4> voxelGiOccupancy;

[[vk::binding(12, 0)]]
RWTexture3D<float4> voxelGiSkyExposure;

[[vk::binding(16, 0)]]
RaytracingAccelerationStructure giSurfaceScene;

struct RestirReservoir {
    float4 radianceWeight;
    float4 state;
};

[[vk::binding(17, 0)]]
RWStructuredBuffer<RestirReservoir> restirCurrent;

static const float kPi = 3.14159265;

bool inBounds(int3 cell, int3 dims)
{
    return
        cell.x >= 0 && cell.y >= 0 && cell.z >= 0 &&
        cell.x < dims.x && cell.y < dims.y && cell.z < dims.z;
}

uint restirReservoirIndex(uint3 cell, int faceIndex, uint3 dims)
{
    const uint cellIndex = (cell.z * dims.y * dims.x) + (cell.y * dims.x) + cell.x;
    return (cellIndex * 6u) + uint(faceIndex);
}

float hash1(float4 p)
{
    return frac(sin(dot(p, float4(12.9898, 78.233, 37.719, 19.913))) * 43758.5453);
}

float2 vogelDiskOffset(int sampleIndex, int sampleCount, float rotation)
{
    const float goldenAngle = 2.39996323;
    const float radius = sqrt((float(sampleIndex) + 0.5) / max(float(sampleCount), 1.0));
    const float angle = (goldenAngle * float(sampleIndex)) + rotation;
    return float2(cos(angle), sin(angle)) * radius;
}

float sampleRtSurfaceRay(float3 origin, float3 direction, float tMin, float tMax)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> rayQuery;
    rayQuery.TraceRayInline(giSurfaceScene, 0, 0xFF, ray);
    while (rayQuery.Proceed()) {}
    return (rayQuery.CommittedStatus() == COMMITTED_NOTHING) ? 1.0 : 0.0;
}

float sampleCandidateVisibility(float3 faceWorldPosition, float3 normal, int3 cell, int faceIndex, int sampleIndex, int sampleCount)
{
    const float3 toSun = normalize(-camera.sunDirectionIntensity.xyz);
    const float cellSize = max(camera.voxelGiGridOriginCellSize.w, 0.001);
    const float receiverBiasScale = clamp(camera.shadowConfig4.w, 0.25, 4.0);
    const float normalOffset = max(cellSize * 0.01 * receiverBiasScale, cellSize * 0.0025);
    const float3 origin = faceWorldPosition + (normal * normalOffset);
    const float tMin = max(cellSize * 0.002, normalOffset * 0.25);
    const float tMax = max(camera.shadowCascadeSplits.w * 1.5, 512.0);
    const float sunAngularRadiusDegrees = max(camera.shadowConfig4.y, 0.0);
    const float coneRadius = tan(sunAngularRadiusDegrees * (kPi / 180.0));
    if (coneRadius <= 1e-5)
    {
        return sampleRtSurfaceRay(origin, toSun, tMin, tMax);
    }

    const float3 basisUp = (abs(toSun.y) < 0.999) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 tangent = normalize(cross(basisUp, toSun));
    const float3 bitangent = cross(toSun, tangent);
    const float rotation = hash1(float4(float3(cell), float(faceIndex) * 0.37)) * (2.0 * kPi);
    const float2 diskOffset = vogelDiskOffset(sampleIndex, sampleCount, rotation);
    const float3 direction = normalize(toSun + ((tangent * diskOffset.x) + (bitangent * diskOffset.y)) * coneRadius);
    return sampleRtSurfaceRay(origin, direction, tMin, tMax);
}

[numthreads(4, 4, 4)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint gridW = 0u;
    uint gridH = 0u;
    uint gridD = 0u;
    voxelGiOccupancy.GetDimensions(gridW, gridH, gridD);
    if (dispatchThreadId.x >= gridW || dispatchThreadId.y >= gridH || dispatchThreadId.z >= gridD)
    {
        return;
    }

    const float4 occupancySample = voxelGiOccupancy.Load(int4(int3(dispatchThreadId), 0));
    const uint3 dims = uint3(gridW, gridH, gridD);
    const int3 cell = int3(dispatchThreadId);
    const float cellSize = max(camera.voxelGiGridOriginCellSize.w, 0.001);
    const float3 gridOrigin = camera.voxelGiGridOriginCellSize.xyz;
    const float3 worldPosition = gridOrigin + ((float3(dispatchThreadId) + 0.5) * cellSize);
    const float3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const float injectSunScale = clamp(camera.shadowConfig2.w, 0.0, 4.0);
    const float skyColumnExposure = clamp(voxelGiSkyExposure[dispatchThreadId].x, 0.0, 1.0);
    const int sampleCount = clamp(int(camera.voxelGiRestirConfig0.y + 0.5), 1, 8);

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

    [unroll]
    for (int faceIndex = 0; faceIndex < 6; ++faceIndex)
    {
        RestirReservoir reservoir;
        reservoir.radianceWeight = float4(0.0, 0.0, 0.0, 0.0);
        reservoir.state = float4(0.0, 0.0, 0.0, 0.0);
        const uint reservoirIndex = restirReservoirIndex(dispatchThreadId, faceIndex, dims);
        if (occupancySample.x <= 0.5)
        {
            restirCurrent[reservoirIndex] = reservoir;
            continue;
        }

        const int3 adjacentCell = cell + faceOffsets[faceIndex];
        bool faceExposed = true;
        if (inBounds(adjacentCell, int3(dims)))
        {
            faceExposed = voxelGiOccupancy.Load(int4(adjacentCell, 0)).x <= 0.5;
        }
        if (!faceExposed)
        {
            restirCurrent[reservoirIndex] = reservoir;
            continue;
        }

        const float3 normal = normals[faceIndex];
        const float ndotl = max(dot(normal, -sunDirection), 0.0);
        if (ndotl <= 0.0001)
        {
            restirCurrent[reservoirIndex] = reservoir;
            continue;
        }

        const float3 faceCenter = worldPosition + (normal * (cellSize * 0.51));
        float bestWeight = 0.0;
        float3 bestRadiance = float3(0.0, 0.0, 0.0);
        [loop]
        for (int sampleIndex = 0; sampleIndex < 8; ++sampleIndex)
        {
            if (sampleIndex >= sampleCount) break;
            const float visibility = sampleCandidateVisibility(faceCenter, normal, cell, faceIndex, sampleIndex, sampleCount);
            const float3 radiance = camera.sunColorShadow.xyz * (sunIntensity * ndotl * visibility * injectSunScale);
            const float weight = max(dot(radiance, float3(0.2126, 0.7152, 0.0722)), 0.0);
            if (weight > bestWeight)
            {
                bestWeight = weight;
                bestRadiance = radiance;
            }
        }

        if (bestWeight > 0.0)
        {
            reservoir.radianceWeight = float4(bestRadiance, bestWeight);
            reservoir.state = float4(float(sampleCount), float(sampleCount), 0.0, skyColumnExposure);
        }
        restirCurrent[reservoirIndex] = reservoir;
    }
}
