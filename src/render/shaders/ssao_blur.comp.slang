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

static const int kSsaoBlurRadius = 6;
static const float kSsaoBlurSigma = 3.0;

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

    uint rawW = 0;
    uint rawH = 0;
    ssaoRawTexture.GetDimensions(rawW, rawH);
    const float2 texelSize = 1.0 / float2(float(rawW), float(rawH));

    const float4 centerNd = normalDepthTexture.SampleLevel(uv, 0.0);
    const float3 centerNormal = decodeViewNormal(centerNd.xyz);
    const float centerDepth = centerNd.w;
    if (centerDepth <= 0.0001)
    {
        ssaoBlurTexture[dispatchThreadId.xy] = 1.0;
        return;
    }

    const float centerAo = ssaoRawTexture.SampleLevel(uv, 0.0).x;

    float weightedAo = centerAo;
    float weightSum = 1.0;

    for (int i = 1; i <= kSsaoBlurRadius; ++i)
    {
        const float fi = float(i);
        const float spatialWeight = exp(-(fi * fi) / (2.0 * kSsaoBlurSigma * kSsaoBlurSigma));

        const float2 offsetX = float2(texelSize.x * fi, 0.0);
        const float2 offsetY = float2(0.0, texelSize.y * fi);
        const float2 offsetD1 = float2(texelSize.x * fi, texelSize.y * fi);
        const float2 offsetD2 = float2(texelSize.x * fi, -texelSize.y * fi);

        const float2 sampleUvs[8] = {
            uv + offsetX,
            uv - offsetX,
            uv + offsetY,
            uv - offsetY,
            uv + offsetD1,
            uv - offsetD1,
            uv + offsetD2,
            uv - offsetD2
        };

        for (int j = 0; j < 8; ++j)
        {
            const float2 suv = sampleUvs[j];
            if (suv.x <= 0.0 || suv.x >= 1.0 || suv.y <= 0.0 || suv.y >= 1.0) continue;

            const float4 sampleNd = normalDepthTexture.SampleLevel(suv, 0.0);
            const float3 sampleNormal = decodeViewNormal(sampleNd.xyz);
            const float sampleDepth = sampleNd.w;
            if (sampleDepth <= 0.0001) continue;
            const float sampleAo = ssaoRawTexture.SampleLevel(suv, 0.0).x;

            const float normalWeight = pow(max(dot(centerNormal, sampleNormal), 0.0), 8.0);
            const float depthWeight = exp(-abs(sampleDepth - centerDepth) * 4.0);
            const float weight = spatialWeight * normalWeight * depthWeight;

            weightedAo += sampleAo * weight;
            weightSum += weight;
        }
    }

    ssaoBlurTexture[dispatchThreadId.xy] = clamp(weightedAo / max(weightSum, 1e-4), 0.0, 1.0);
}
