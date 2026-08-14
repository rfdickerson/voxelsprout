// XeGTAO denoise: edge-aware spatial blur of the AO term.
//
// This is not optional polish, it is half of the algorithm. XeGTAO samples very
// few directions per pixel and decorrelates the error with spatiotemporal blue
// noise, on the explicit assumption that a denoiser spreads that error across
// neighbours before anything reads it. Skipping this pass and relying on the
// joint bilateral UPSAMPLE that follows was measured at 20x the frame-to-frame
// instability of the fixed-sample-count GTAO estimator -- visible as flicker,
// because per-frame noise rotation without denoising is just flicker.
//
// The blur is guided by the packed edges the main pass wrote into the bent
// normal's alpha, so it never averages across a silhouette. That distinction is
// the whole reason a plain gaussian will not do: AO's error is high frequency
// and its signal is low frequency EXCEPT at geometric discontinuities, which are
// exactly where a blind blur produces the halo the effect is meant to avoid.

[[vk::binding(0, 0)]]
Sampler2D<float> aoTermTexture;

// .w carries packEdges()'s four 2-bit connectivity factors; .xyz is the bent
// normal, which this pass does not touch.
[[vk::binding(1, 0)]]
Sampler2D<float4> bentNormalEdgesTexture;

[[vk::binding(2, 0)]]
RWTexture2D<float> aoOutput;

struct PushConstants
{
    uint width;
    uint height;
    // 0 keeps the centre sample only; higher spreads further into neighbours.
    // XeGTAO's default sits near 1.
    float blurAmount;
    float pad;
};

[[vk::push_constant]]
PushConstants pushConstants;

float4 unpackEdges(float packedValue)
{
    const uint packed = uint(packedValue * 255.5);
    float4 edgesLRTB;
    edgesLRTB.x = float((packed >> 6u) & 0x03u) / 3.0;
    edgesLRTB.y = float((packed >> 4u) & 0x03u) / 3.0;
    edgesLRTB.z = float((packed >> 2u) & 0x03u) / 3.0;
    edgesLRTB.w = float((packed >> 0u) & 0x03u) / 3.0;
    return saturate(edgesLRTB);
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= pushConstants.width || dispatchThreadId.y >= pushConstants.height)
    {
        return;
    }
    const int2 pixelCoord = int2(dispatchThreadId.xy);
    const float2 screenSize = float2(float(pushConstants.width), float(pushConstants.height));
    const float2 uv = (float2(pixelCoord) + 0.5) / screenSize;
    const float2 texel = 1.0 / screenSize;

    const float centerAo = aoTermTexture.SampleLevel(uv, 0.0);
    const float4 edges = unpackEdges(bentNormalEdgesTexture.SampleLevel(uv, 0.0).w);

    // Cardinal weights come straight from the edge factors. Diagonals use the
    // MINIMUM of the two cardinals that reach them: a diagonal neighbour is only
    // connected to the centre if BOTH routes to it are, which is what keeps the
    // blur from leaking around the corner of a silhouette.
    const float weightL = edges.x;
    const float weightR = edges.y;
    const float weightT = edges.z;
    const float weightB = edges.w;
    const float weightTL = min(edges.x, edges.z);
    const float weightTR = min(edges.y, edges.z);
    const float weightBL = min(edges.x, edges.w);
    const float weightBR = min(edges.y, edges.w);

    const float blur = max(pushConstants.blurAmount, 0.0);
    // The centre always carries weight 1 so a pixel with every edge closed off
    // (an isolated detail) keeps its own value rather than dividing by zero.
    float sum = centerAo;
    float sumWeight = 1.0;

    const float2 offsets[8] = {
        float2(-1.0, 0.0), float2(1.0, 0.0), float2(0.0, -1.0), float2(0.0, 1.0),
        float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0), float2(1.0, 1.0)
    };
    const float weights[8] = {
        weightL, weightR, weightT, weightB, weightTL, weightTR, weightBL, weightBR
    };

    [unroll]
    for (int i = 0; i < 8; ++i)
    {
        const float weight = weights[i] * blur;
        if (weight <= 0.0)
        {
            continue;
        }
        const float2 sampleUv = saturate(uv + (offsets[i] * texel));
        sum += aoTermTexture.SampleLevel(sampleUv, 0.0) * weight;
        sumWeight += weight;
    }

    aoOutput[pixelCoord] = sum / max(sumWeight, 1e-5);
}
