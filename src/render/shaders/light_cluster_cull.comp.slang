import camera_uniform;
import light_clusters;

// One thread per cluster. Builds the cluster's view-space AABB, tests every
// uploaded local light against it, and writes a 64-bit mask of the ones that
// reach it. See light_clusters.slang for why this is a mask rather than the
// usual compacted index list.

// Binding 0 of set 0 is the camera UBO, declared by camera_uniform.slang.
[[vk::binding(1, 0)]]
RWStructuredBuffer<uint2> clusterLightMask;

struct LightClusterPushConstants
{
    uint3 gridDim;
    uint tileSize;
    float sliceScale;
    float sliceBias;
    uint2 renderExtent;
};

[[vk::push_constant]]
LightClusterPushConstants pc;

// Inverse of the projection's x/y scale, so a screen NDC and a view depth give
// a view-space position. Taken from the projection matrix rather than from a
// separately-uploaded fov, because the fov is animated (the dialogue camera
// dollies) and two sources would disagree for a frame.
float2 viewScaleFromProjection()
{
    const float projX = abs(camera.proj[0][0]) < 1e-6 ? 1e-6 : camera.proj[0][0];
    const float projY = abs(camera.proj[1][1]) < 1e-6 ? 1e-6 : camera.proj[1][1];
    return float2(1.0 / projX, 1.0 / projY);
}

// The depth at the near edge of `slice`, inverting lightClusterSlice.
float sliceNearDepth(uint slice)
{
    return exp2((float(slice) - pc.sliceBias) / max(pc.sliceScale, 1e-6));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint clusterCount = pc.gridDim.x * pc.gridDim.y * pc.gridDim.z;
    const uint clusterIndex = dispatchThreadId.x;
    if (clusterIndex >= clusterCount)
    {
        return;
    }

    const uint tilesPerSlice = pc.gridDim.x * pc.gridDim.y;
    const uint slice = clusterIndex / tilesPerSlice;
    const uint tileIndex = clusterIndex - (slice * tilesPerSlice);
    const uint2 tile = uint2(tileIndex % pc.gridDim.x, tileIndex / pc.gridDim.x);

    // Tile rect in pixels, clamped to the render extent: the last tile in each
    // row is usually partial, and letting it run past the edge inflates its
    // AABB and pulls in lights that cannot touch any of its pixels.
    const float2 tileMinPixels = float2(tile * pc.tileSize);
    const float2 tileMaxPixels = min(tileMinPixels + float2(pc.tileSize, pc.tileSize),
                                     float2(pc.renderExtent));

    // Pixels -> NDC. Vulkan's y runs down the screen; the sign is irrelevant to
    // an axis-aligned box as long as both corners use the same convention.
    const float2 extent = max(float2(pc.renderExtent), float2(1.0, 1.0));
    const float2 ndcMin = ((tileMinPixels / extent) * 2.0) - 1.0;
    const float2 ndcMax = ((tileMaxPixels / extent) * 2.0) - 1.0;

    const float nearDepth = sliceNearDepth(slice);
    const float farDepth = sliceNearDepth(slice + 1u);
    const float2 viewScale = viewScaleFromProjection();

    // A view-space x is ndc.x * viewScale.x * depth, i.e. linear in depth, so
    // the extremes of the frustum wedge are at the corner/depth extremes. Eight
    // points, min/max -- no need to walk the wedge.
    float3 aabbMin = float3(1e30, 1e30, 1e30);
    float3 aabbMax = float3(-1e30, -1e30, -1e30);
    for (uint corner = 0u; corner < 8u; ++corner)
    {
        const float ndcX = ((corner & 1u) != 0u) ? ndcMax.x : ndcMin.x;
        const float ndcY = ((corner & 2u) != 0u) ? ndcMax.y : ndcMin.y;
        const float depth = ((corner & 4u) != 0u) ? farDepth : nearDepth;
        // View space here is right-handed with -z forward, matching
        // camera.view: a point at positive distance `depth` sits at z = -depth.
        const float3 point = float3(ndcX * viewScale.x * depth, ndcY * viewScale.y * depth, -depth);
        aabbMin = min(aabbMin, point);
        aabbMax = max(aabbMax, point);
    }

    const int lightCount = clamp(int(camera.importedLightConfig.x + 0.5), 0, 64);
    const float globalIntensity = max(camera.importedLightConfig.y, 0.0);
    uint2 mask = uint2(0u, 0u);
    if (globalIntensity > 0.0)
    {
        for (int lightIndex = 0; lightIndex < lightCount; ++lightIndex)
        {
            const float4 positionRadius = camera.importedLightPositionRadius[lightIndex];
            // Lights are uploaded in WORLD space (the fragment shader shades in
            // world space), so they come into view space here rather than the
            // cluster going out to world space -- one transform per light per
            // cluster either way, but this keeps the AABB axis-aligned, which
            // is the whole point of building it.
            const float3 viewPosition = mul(camera.view, float4(positionRadius.xyz, 1.0)).xyz;
            const float radius = max(positionRadius.w, 1.0);

            // Squared distance from the sphere centre to the box: zero on every
            // axis the centre is already inside.
            const float3 outside = max(max(aabbMin - viewPosition, viewPosition - aabbMax), float3(0.0, 0.0, 0.0));
            if (dot(outside, outside) > (radius * radius))
            {
                continue;
            }
            if (lightIndex < 32)
            {
                mask.x |= (1u << uint(lightIndex));
            }
            else
            {
                mask.y |= (1u << uint(lightIndex - 32));
            }
        }
    }
    clusterLightMask[clusterIndex] = mask;
}
