import camera_uniform;

[[vk::binding(1, 0)]]
Sampler2D<float4> normalDepthTexture;

[[vk::binding(3, 0)]]
RWStructuredBuffer<float2> depthHierarchy;

struct ContactShadowPushConstants
{
    uint4 extent;       // render width/height, half width/height
    uint4 dispatch;     // mip, source offset, destination offset, temporal phase
    uint4 cluster;      // grid x/y/z, tile size
    float4 params;      // slice scale/bias, max ray distance, unused
};

[[vk::push_constant]]
ContactShadowPushConstants pc;

uint2 mipExtent(uint mip)
{
    uint2 result = max(pc.extent.zw, uint2(1u, 1u));
    for (uint i = 0u; i < mip; ++i)
    {
        result = max((result + 1u) / 2u, uint2(1u, 1u));
    }
    return result;
}

float2 mergeDepth(float2 accumulated, float2 sampleRange)
{
    if (sampleRange.y <= 0.0)
    {
        return accumulated;
    }
    if (accumulated.y <= 0.0)
    {
        return sampleRange;
    }
    return float2(min(accumulated.x, sampleRange.x), max(accumulated.y, sampleRange.y));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint mip = pc.dispatch.x;
    const uint2 destinationExtent = mipExtent(mip);
    if (any(dispatchThreadId.xy >= destinationExtent))
    {
        return;
    }

    float2 range = float2(0.0, 0.0);
    if (mip == 0u)
    {
        const uint2 fullBase = dispatchThreadId.xy * 2u;
        [unroll]
        for (uint y = 0u; y < 2u; ++y)
        {
            [unroll]
            for (uint x = 0u; x < 2u; ++x)
            {
                const uint2 fullPixel = min(fullBase + uint2(x, y), pc.extent.xy - 1u);
                const float2 uv = (float2(fullPixel) + 0.5) / float2(pc.extent.xy);
                const float depth = normalDepthTexture.SampleLevel(uv, 0.0).w;
                if (depth > 1e-4)
                {
                    range = mergeDepth(range, float2(depth, depth));
                }
            }
        }
    }
    else
    {
        const uint2 sourceExtent = mipExtent(mip - 1u);
        const uint2 sourceBase = dispatchThreadId.xy * 2u;
        [unroll]
        for (uint y = 0u; y < 2u; ++y)
        {
            [unroll]
            for (uint x = 0u; x < 2u; ++x)
            {
                const uint2 sourcePixel = min(sourceBase + uint2(x, y), sourceExtent - 1u);
                const uint sourceIndex = pc.dispatch.y + sourcePixel.y * sourceExtent.x + sourcePixel.x;
                range = mergeDepth(range, depthHierarchy[sourceIndex]);
            }
        }
    }

    const uint destinationIndex =
        pc.dispatch.z + dispatchThreadId.y * destinationExtent.x + dispatchThreadId.x;
    depthHierarchy[destinationIndex] = range;
}
