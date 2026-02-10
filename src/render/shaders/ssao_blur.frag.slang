import camera_uniform;

[[vk::binding(6, 0)]]
Sampler2D<float4> normalDepthTexture;
[[vk::binding(8, 0)]]
Sampler2D<float4> ssaoRawTexture;

[[vk::constant_id(2)]]
const int kSsaoBlurRadius = 6;
[[vk::constant_id(3)]]
const float kSsaoBlurSigma = 3.0;

float3 decodeViewNormal(float3 encodedNormal)
{
    return normalize((encodedNormal * 2.0) - 1.0);
}

struct FragmentInput
{
    float2 inUv : TEXCOORD0;
};

float main(FragmentInput input) : SV_Target0
{
    uint rawW = 0;
    uint rawH = 0;
    ssaoRawTexture.GetDimensions(rawW, rawH);
    const float2 texelSize = 1.0 / float2(float(rawW), float(rawH));

    const float4 centerNd = normalDepthTexture.Sample(input.inUv);
    const float3 centerNormal = decodeViewNormal(centerNd.xyz);
    const float centerDepth = centerNd.w;
    if (centerDepth <= 0.0001) return 1.0;

    const float centerAo = ssaoRawTexture.Sample(input.inUv).x;

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
            input.inUv + offsetX,
            input.inUv - offsetX,
            input.inUv + offsetY,
            input.inUv - offsetY,
            input.inUv + offsetD1,
            input.inUv - offsetD1,
            input.inUv + offsetD2,
            input.inUv - offsetD2
        };

        for (int j = 0; j < 8; ++j)
        {
            const float2 suv = sampleUvs[j];
            if (suv.x <= 0.0 || suv.x >= 1.0 || suv.y <= 0.0 || suv.y >= 1.0) continue;

            const float4 sampleNd = normalDepthTexture.Sample(suv);
            const float3 sampleNormal = decodeViewNormal(sampleNd.xyz);
            const float sampleDepth = sampleNd.w;
            if (sampleDepth <= 0.0001) continue;
            const float sampleAo = ssaoRawTexture.Sample(suv).x;

            const float normalWeight = pow(max(dot(centerNormal, sampleNormal), 0.0), 8.0);
            const float depthWeight = exp(-abs(sampleDepth - centerDepth) * 1.5);
            const float weight = spatialWeight * normalWeight * depthWeight;

            weightedAo += sampleAo * weight;
            weightSum += weight;
        }
    }

    return clamp(weightedAo / max(weightSum, 1e-4), 0.0, 1.0);
}
