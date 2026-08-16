[[vk::binding(0, 0)]]
Sampler2D<float4> normalDepthTexture;

[[vk::binding(1, 0)]]
Sampler2D<float4> ssaoRawTexture;

[[vk::binding(2, 0)]]
RWTexture2D<float> ssaoBlurTexture;

struct PushConstants
{
    uint width;
    uint height;
};

[[vk::push_constant]]
PushConstants pushConstants;

static const int kSsaoBlurRadius = 4;
static const float kSsaoBlurSigma = 2.2;

float3 decodeViewNormal(float3 encodedNormal)
{
    return normalize((encodedNormal * 2.0) - 1.0);
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= pushConstants.width || dispatchThreadId.y >= pushConstants.height)
    {
        return;
    }

    const float2 uv = (float2(dispatchThreadId.xy) + 0.5) / float2(float(pushConstants.width), float(pushConstants.height));

    // The estimator's resolution, which is independent of this pass's -- see the
    // upsample below. Queried rather than pushed so the two cannot disagree.
    uint rawW = 0;
    uint rawH = 0;
    ssaoRawTexture.GetDimensions(rawW, rawH);

    const float4 centerNd = normalDepthTexture.SampleLevel(uv, 0.0);
    const float3 centerNormal = decodeViewNormal(centerNd.xyz);
    const float centerDepth = centerNd.w;
    // Depth here is view distance in WORLD UNITS (imported_static_normaldepth
    // writes -view.z), and at Bethesda scale -- ~70 units to the metre, depths
    // out to 50000 -- an absolute falloff like exp(-|d| * 4.0) is zero for
    // every tap on any surface that is not parallel to the screen. What that
    // renders as is not a halo but BANDING: with all depth weights collapsed,
    // the quotient is decided by which raw row happens to sit nearest in
    // depth, and on oblique terrain that alternates per output row.
    //
    // So the tolerance is RELATIVE (3% of the centre depth), with a floor so
    // near geometry keeps a usable acceptance band. A grazing slope moves a
    // few percent of depth per raw texel and is accepted; a fence against the
    // ground behind it differs by 10x the tolerance and is still rejected.
    const float depthTolerance = max(centerDepth * 0.03, 4.0);
    if (centerDepth <= 0.0001)
    {
        ssaoBlurTexture[dispatchThreadId.xy] = 1.0;
        return;
    }

    // JOINT BILATERAL UPSAMPLE, not just a blur. The estimator runs at
    // ssaoRawTexture's resolution, which is a fraction of this pass's, so every
    // tap has to land on a raw TEXEL CENTRE and be weighted by the full-res
    // geometry at that same point.
    //
    // Sampling the raw texture at an arbitrary uv instead -- which is what this
    // did when both passes shared a resolution -- makes the hardware bilinearly
    // blend up to four raw texels BEFORE the bilateral weights can reject any of
    // them. At 1:1 that is harmless; at 2x upsample it drags occlusion across
    // every depth discontinuity and haloes the silhouette of every object.
    const float2 rawSize = float2(float(rawW), float(rawH));
    const float2 rawCoordF = (uv * rawSize) - 0.5;
    const int2 baseCoord = int2(floor(rawCoordF));

    float weightedAo = 0.0;
    float weightSum = 0.0;

    // The 2x2 raw texels this output pixel sits between, then the axis cross out
    // to kSsaoBlurRadius from them. Diagonals past the 2x2 are deliberately left
    // out for the same reason as before: at this radius the bilateral weighting
    // already rejects cross-edge contamination, and they cost taps.
    for (int dy = 0; dy <= 1; ++dy)
    {
        for (int dx = 0; dx <= 1; ++dx)
        {
            const int2 c = baseCoord + int2(dx, dy);
            if (c.x < 0 || c.y < 0 || c.x >= int(rawW) || c.y >= int(rawH)) continue;

            // Centre of that raw texel, in this pass's uv space. Sampling the raw
            // texture here hits the texel exactly, so no interpolation happens.
            const float2 suv = (float2(c) + 0.5) / rawSize;
            const float4 sampleNd = normalDepthTexture.SampleLevel(suv, 0.0);
            const float sampleDepth = sampleNd.w;
            if (sampleDepth <= 0.0001) continue;

            // Bilinear footprint weight, so the upsample still interpolates
            // smoothly where the geometry agrees.
            const float2 f = rawCoordF - float2(baseCoord);
            const float bilinear =
                (dx == 0 ? (1.0 - f.x) : f.x) * (dy == 0 ? (1.0 - f.y) : f.y);

            const float3 sampleNormal = decodeViewNormal(sampleNd.xyz);
            const float normalWeight = pow(max(dot(centerNormal, sampleNormal), 0.0), 8.0);
            const float depthWeight = exp(-abs(sampleDepth - centerDepth) / depthTolerance);
            const float weight = max(bilinear, 1e-3) * normalWeight * depthWeight;

            weightedAo += ssaoRawTexture.SampleLevel(suv, 0.0).x * weight;
            weightSum += weight;
        }
    }

    for (int i = 1; i <= kSsaoBlurRadius; ++i)
    {
        const float fi = float(i);
        const float spatialWeight = exp(-(fi * fi) / (2.0 * kSsaoBlurSigma * kSsaoBlurSigma));

        const int2 offsets[4] = {
            int2(i, 0), int2(-i, 0), int2(0, i), int2(0, -i)
        };

        for (int j = 0; j < 4; ++j)
        {
            const int2 c = baseCoord + offsets[j];
            if (c.x < 0 || c.y < 0 || c.x >= int(rawW) || c.y >= int(rawH)) continue;

            const float2 suv = (float2(c) + 0.5) / rawSize;
            const float4 sampleNd = normalDepthTexture.SampleLevel(suv, 0.0);
            const float sampleDepth = sampleNd.w;
            if (sampleDepth <= 0.0001) continue;

            const float3 sampleNormal = decodeViewNormal(sampleNd.xyz);
            const float normalWeight = pow(max(dot(centerNormal, sampleNormal), 0.0), 8.0);
            const float depthWeight = exp(-abs(sampleDepth - centerDepth) / depthTolerance);
            const float weight = spatialWeight * normalWeight * depthWeight;

            weightedAo += ssaoRawTexture.SampleLevel(suv, 0.0).x * weight;
            weightSum += weight;
        }
    }

    // Every tap can be rejected at once -- a lone foreground sliver whose raw
    // neighbours all sit on the background. Unoccluded is the right answer there:
    // it is what an AO-off frame shows, so the failure darkens nothing.
    ssaoBlurTexture[dispatchThreadId.xy] =
        (weightSum > 1e-4) ? clamp(weightedAo / weightSum, 0.0, 1.0) : 1.0;
}
