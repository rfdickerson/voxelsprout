import camera_uniform;

[[vk::binding(1, 0)]]
Sampler2D<float4> normalDepthTexture;

[[vk::binding(4, 0)]]
StructuredBuffer<uint4> halfContactRecords;

[[vk::binding(5, 0)]]
RWStructuredBuffer<uint2> fullContactMask;

struct ContactShadowPushConstants
{
    uint4 extent;
    uint4 dispatch;
    uint4 cluster;
    float4 params;
};

[[vk::push_constant]]
ContactShadowPushConstants pc;

float3 decodeViewNormal(float3 encoded)
{
    return normalize((encoded * 2.0) - 1.0);
}

float3 unpackOctNormal(uint packed)
{
    const uint2 encoded = uint2(packed & 0xffffu, packed >> 16u);
    float2 oct = (float2(encoded) / 65535.0) * 2.0 - 1.0;
    float3 normal = float3(oct, 1.0 - abs(oct.x) - abs(oct.y));
    if (normal.z < 0.0)
    {
        normal.xy = (1.0 - abs(normal.yx)) *
            select(normal.xy >= 0.0, float2(1.0, 1.0), float2(-1.0, -1.0));
    }
    return normalize(normal);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= pc.extent.xy))
    {
        return;
    }
    const uint outputIndex = dispatchThreadId.y * pc.extent.x + dispatchThreadId.x;
    const float2 uv = (float2(dispatchThreadId.xy) + 0.5) / float2(pc.extent.xy);
    const float4 center = normalDepthTexture.SampleLevel(uv, 0.0);
    const float centerDepth = center.w;
    if (centerDepth <= 1e-4)
    {
        fullContactMask[outputIndex] = uint2(0u, 0u);
        return;
    }
    const float3 centerNormal = decodeViewNormal(center.xyz);
    const int2 base = int2(dispatchThreadId.xy / 2u);
    uint2 seenOnce = uint2(0u, 0u);
    uint2 seenTwice = uint2(0u, 0u);
    uint acceptedCount = 0u;
    // A compact 3x3 gather closes the one-half-pixel holes left by the eight
    // jittered trace samples. Bilateral rejection prevents that closing from
    // crossing a depth or normal edge, so contacts become continuous without
    // growing silhouettes.
    [unroll]
    for (int offsetY = -1; offsetY <= 1; ++offsetY)
    {
        [unroll]
        for (int offsetX = -1; offsetX <= 1; ++offsetX)
        {
            const int2 halfPixel = clamp(
                base + int2(offsetX, offsetY), int2(0, 0), int2(pc.extent.zw) - 1);
            const uint index = uint(halfPixel.y) * pc.extent.z + uint(halfPixel.x);
            const uint4 record = halfContactRecords[index];
            const float candidateDepth = asfloat(record.z);
            if (candidateDepth <= 1e-4)
            {
                continue;
            }
            const float3 candidateNormal = unpackOctNormal(record.w);
            const float normalAgreement = dot(centerNormal, candidateNormal);
            const float depthDifference = abs(centerDepth - candidateDepth);
            const float depthLimit = max(4.0, centerDepth * 0.01);
            if (normalAgreement < 0.75 || depthDifference > depthLimit)
            {
                continue;
            }
            // A single half-resolution hit is usually temporal ray jitter or a
            // silhouette discontinuity. Preserve each light bit independently,
            // but require support from a second compatible sample.
            seenTwice |= seenOnce & record.xy;
            seenOnce |= record.xy;
            ++acceptedCount;
        }
    }
    fullContactMask[outputIndex] = acceptedCount >= 2u ? seenTwice : uint2(0u, 0u);
}
