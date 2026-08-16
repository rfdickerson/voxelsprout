// XeGTAO depth prefilter: builds the 5-level viewspace depth pyramid the GTAO
// pass marches against.
//
// This is the single change that matters most on an Intel Xe iGPU. The GTAO
// integral marches a screen-space circle whose radius in pixels grows as the
// surface gets closer; at contact range that circle is tens of pixels across,
// and sampling a full-resolution depth texture along it is a scattered read per
// step with essentially no reuse between neighbouring threads. Xe's texture
// cache is small and shared with the EUs, so that pattern spends most of its
// time waiting on memory rather than doing the (cheap) horizon arithmetic.
//
// The pyramid turns that into a near-local read: the main pass picks a MIP whose
// texel footprint matches the step length, so a march that covers 32 pixels
// reads ~8 texels from MIP2 instead of 32 from MIP0, and adjacent threads hit
// the same cache lines. Same integral, same horizons, a fraction of the traffic.
//
// The filter is NOT a box average. Averaging depth across a silhouette invents a
// surface halfway between the near object and the far background, and the GTAO
// march would then find a horizon that does not exist -- which reads as a dark
// halo around every object against the sky. XeGTAO weights each of the four
// children by how close it is to the NEAREST of them, on the same falloff curve
// the AO effect radius uses, so a mixed 2x2 collapses toward the near surface
// instead of splitting the difference.
//
// Layout follows XeGTAO's PrefilterDepths16x16: an 8x8 group, each thread
// loading a 2x2 quad, so one group produces a 16x16 tile of MIP0 and then
// reduces it through groupshared memory to MIP1 (8x8), MIP2 (4x4), MIP3 (2x2)
// and MIP4 (1x1). All five levels are written by ONE dispatch, which is why
// there are no per-MIP barriers -- the alternative, a dispatch per level, is
// five barriers over images that shrink to nothing and is dominated by sync.

import camera_uniform;

// Source: the normal-depth prepass target. .w is linear viewspace depth in world
// units (positive, 0 = nothing rendered); .xyz is the encoded view normal, which
// this pass ignores.
[[vk::binding(1, 0)]]
Sampler2D<float4> normalDepthTexture;

// One binding per level rather than an array: the reduction below indexes them
// only with compile-time-constant levels (the loop is unrolled), and a storage
// IMAGE array would need descriptor-array offset math that the descriptor-buffer
// helpers only provide for combined image samplers.
[[vk::binding(2, 0)]] RWTexture2D<float> depthPyramidMip0;
[[vk::binding(3, 0)]] RWTexture2D<float> depthPyramidMip1;
[[vk::binding(4, 0)]] RWTexture2D<float> depthPyramidMip2;
[[vk::binding(5, 0)]] RWTexture2D<float> depthPyramidMip3;
[[vk::binding(6, 0)]] RWTexture2D<float> depthPyramidMip4;

void writeMip(int level, int2 pixel, float value)
{
    if (level == 0) { depthPyramidMip0[pixel] = value; }
    else if (level == 1) { depthPyramidMip1[pixel] = value; }
    else if (level == 2) { depthPyramidMip2[pixel] = value; }
    else if (level == 3) { depthPyramidMip3[pixel] = value; }
    else { depthPyramidMip4[pixel] = value; }
}

struct PushConstants
{
    // Extent of MIP0, i.e. the resolution the GTAO pass runs at.
    uint width;
    uint height;
    // Effect radius in world units, mirrored from the GTAO pass so the MIP
    // filter's falloff matches the falloff the march will apply. A pyramid built
    // against a different radius biases every horizon it feeds.
    float effectRadius;
    float falloffRange;
};

[[vk::push_constant]]
PushConstants pushConstants;

// 16x16 tile of MIP0 for this group, reduced in place down the chain.
groupshared float g_tileDepths[16][16];

// XeGTAO_DepthMIPFilter. Weighted toward the NEAREST of the four children -- see
// the halo note in the header. The weights are the AO falloff curve evaluated on
// each child's distance behind the nearest one, so a 2x2 spanning a silhouette
// keeps the near surface and a 2x2 on a flat wall averages normally.
float depthMipFilter(float depth0, float depth1, float depth2, float depth3)
{
    // 0.75 is XeGTAO's depthRangeScaleFactor: the MIP filter deliberately uses a
    // slightly tighter radius than the effect itself, because a child that is
    // already at the edge of the effect radius contributes nothing to the
    // horizon and should not drag the averaged depth outward.
    const float effectRadius = 0.75 * max(pushConstants.effectRadius, 1e-4);
    const float falloffRange = max(pushConstants.falloffRange * effectRadius, 1e-6);
    const float falloffFrom = effectRadius * (1.0 - pushConstants.falloffRange);
    const float falloffMul = -1.0 / falloffRange;
    const float falloffAdd = (falloffFrom / falloffRange) + 1.0;

    // Nearest child wins: distances are measured from the smallest depth, so the
    // near surface always has weight 1 and anything further falls off.
    const float nearest = min(min(depth0, depth1), min(depth2, depth3));
    const float weight0 = saturate(((depth0 - nearest) * falloffMul) + falloffAdd);
    const float weight1 = saturate(((depth1 - nearest) * falloffMul) + falloffAdd);
    const float weight2 = saturate(((depth2 - nearest) * falloffMul) + falloffAdd);
    const float weight3 = saturate(((depth3 - nearest) * falloffMul) + falloffAdd);

    const float weightSum = weight0 + weight1 + weight2 + weight3;
    if (weightSum < 1e-6)
    {
        return nearest;
    }
    return ((weight0 * depth0) + (weight1 * depth1) + (weight2 * depth2) + (weight3 * depth3)) /
           weightSum;
}

// Sky/background texels carry depth 0 from the prepass clear. Left at 0 they
// would be the "nearest" surface everywhere and pull every MIP to zero, so they
// are pushed to the far end instead -- an unoccluded direction, which is what
// empty space is.
float loadSourceDepth(int2 pixel, int2 maxPixel)
{
    const int2 clamped = clamp(pixel, int2(0, 0), maxPixel);
    const float2 uv = (float2(clamped) + 0.5) / float2(maxPixel + int2(1, 1));
    const float depth = normalDepthTexture.SampleLevel(uv, 0.0).w;
    return (depth <= 1e-4) ? 3.0e38 : depth;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupThreadId : SV_GroupThreadID,
          uint3 groupId : SV_GroupID)
{
    const int2 maxPixel = int2(int(pushConstants.width) - 1, int(pushConstants.height) - 1);

    // Each thread owns a 2x2 quad of MIP0, so an 8x8 group covers 16x16.
    const int2 basePixel = int2(groupId.xy) * 16 + (int2(groupThreadId.xy) * 2);
    const int2 tileBase = int2(groupThreadId.xy) * 2;

    float quad[4];
    [unroll]
    for (int q = 0; q < 4; ++q)
    {
        const int2 offset = int2(q & 1, q >> 1);
        const float depth = loadSourceDepth(basePixel + offset, maxPixel);
        quad[q] = depth;
        g_tileDepths[tileBase.y + offset.y][tileBase.x + offset.x] = depth;

        const int2 writePixel = basePixel + offset;
        if (writePixel.x <= maxPixel.x && writePixel.y <= maxPixel.y)
        {
            writeMip(0, writePixel, depth);
        }
    }

    // MIP1: every thread reduces its own 2x2. No barrier needed yet -- each
    // thread reads only the quad it just wrote.
    const float mip1Depth = depthMipFilter(quad[0], quad[1], quad[2], quad[3]);
    g_tileDepths[tileBase.y][tileBase.x] = mip1Depth;
    {
        const int2 writePixel = (int2(groupId.xy) * 8) + int2(groupThreadId.xy);
        const int2 mipMax = max(maxPixel >> 1, int2(0, 0));
        if (writePixel.x <= mipMax.x && writePixel.y <= mipMax.y)
        {
            writeMip(1, writePixel, mip1Depth);
        }
    }

    // MIP2..MIP4: each level quarters the number of active threads. Results live
    // at stride-2^level positions in the tile so the next level can find them
    // without a second buffer.
    [unroll]
    for (int level = 2; level <= 4; ++level)
    {
        GroupMemoryBarrierWithGroupSync();

        const int stride = 1 << (level - 1);      // spacing of inputs in the tile
        const int activeThreads = 8 >> (level - 1);  // 4, then 2, then 1
        if (int(groupThreadId.x) < activeThreads && int(groupThreadId.y) < activeThreads)
        {
            const int2 src = int2(groupThreadId.xy) * (stride * 2);
            const float d0 = g_tileDepths[src.y][src.x];
            const float d1 = g_tileDepths[src.y][src.x + stride];
            const float d2 = g_tileDepths[src.y + stride][src.x];
            const float d3 = g_tileDepths[src.y + stride][src.x + stride];
            const float reduced = depthMipFilter(d0, d1, d2, d3);
            g_tileDepths[src.y][src.x] = reduced;

            const int2 writePixel =
                (int2(groupId.xy) * activeThreads) + int2(groupThreadId.xy);
            const int2 mipMax = max(maxPixel >> level, int2(0, 0));
            if (writePixel.x <= mipMax.x && writePixel.y <= mipMax.y)
            {
                writeMip(level, writePixel, reduced);
            }
        }
    }
}
