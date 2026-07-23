import camera_uniform;

[[vk::binding(1, 0)]]
Sampler2D<float4> normalDepthTexture;

[[vk::binding(2, 0)]]
RWTexture2D<float> ssaoRawTexture;

struct PushConstants
{
    uint width;
    uint height;
};

[[vk::push_constant]]
PushConstants pushConstants;

static const int kSsaoSampleCount = 32;
static const float kSsaoPower = 1.4;

float hash12(float2 p)
{
    const float3 p3 = frac(float3(p.x, p.y, p.x) * 0.1031);
    const float3 q3 = p3 + dot(p3, float3(p3.y, p3.z, p3.x) + 33.33);
    return frac((q3.x + q3.y) * q3.z);
}

float3 decodeViewNormal(float3 encodedNormal)
{
    return normalize((encodedNormal * 2.0) - 1.0);
}

float3 reconstructViewPosition(float2 uv, float viewDepth)
{
    const float2 ndcXY = (uv * 2.0) - 1.0;
    float projX = camera.proj[0][0];
    float projY = camera.proj[1][1];
    if (abs(projX) < 1e-6) projX = (projX < 0.0) ? -1e-6 : 1e-6;
    if (abs(projY) < 1e-6) projY = (projY < 0.0) ? -1e-6 : 1e-6;

    // Perspective: view-space XY scale with depth (ndc = view * proj / -viewZ).
    // Orthographic: they do not — ndc = view * proj directly. Detect via the
    // projection's [3][3] diagonal (1 for ortho, 0 for perspective; diagonal
    // entries are invariant under the CPU-side transpose).
    const float depthScale = (camera.proj[3][3] > 0.5) ? 1.0 : viewDepth;
    const float viewX = (ndcXY.x * depthScale) / projX;
    const float viewY = (ndcXY.y * depthScale) / projY;
    return float3(viewX, viewY, -viewDepth);
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= pushConstants.width || dispatchThreadId.y >= pushConstants.height)
    {
        return;
    }

    const float2 uv = (float2(dispatchThreadId.xy) + 0.5) / float2(float(pushConstants.width), float(pushConstants.height));

    const float4 centerSample = normalDepthTexture.SampleLevel(uv, 0.0);
    const float centerDepth = centerSample.w;
    if (centerDepth <= 0.0001)
    {
        ssaoRawTexture[dispatchThreadId.xy] = 1.0;
        return;
    }

    const float3 centerNormal = decodeViewNormal(centerSample.xyz);
    const float3 centerViewPos = reconstructViewPosition(uv, centerDepth);

    float3 tangent = normalize(cross(centerNormal, float3(0.0, 1.0, 0.0)));
    if (dot(tangent, tangent) <= 0.0001)
    {
        tangent = normalize(cross(centerNormal, float3(1.0, 0.0, 0.0)));
    }
    const float3 bitangent = normalize(cross(centerNormal, tangent));

    uint texW = 0;
    uint texH = 0;
    normalDepthTexture.GetDimensions(texW, texH);
    const float2 noiseScale = float2(float(texW), float(texH));
    const float randomAngle = hash12(uv * noiseScale) * 6.2831853;
    const float s = sin(randomAngle);
    const float c = cos(randomAngle);
    const float2x2 rot = float2x2(c, -s, s, c);

    const float radius = clamp(camera.shadowConfig2.x, 0.25, 128.0);
    const float bias = clamp(camera.shadowConfig2.y, 0.0, 8.0);
    float occlusion = 0.0;
    float sampleCount = 0.0;
    const float kGoldenAngle = 2.39996323;

    for (int i = 0; i < kSsaoSampleCount; ++i)
    {
        const float fi = float(i);
        const float t = (fi + 0.5) / float(kSsaoSampleCount);
        const float r = sqrt(t);
        const float angle = (fi * kGoldenAngle) + randomAngle;
        const float2 disk = mul(rot, float2(cos(angle), sin(angle)) * r);
        const float z = sqrt(max(1.0 - dot(disk, disk), 0.0));
        float3 k = float3(disk.x, disk.y, z);
        const float scale = lerp(0.15, 1.0, t * t);
        k *= scale;

        const float3 sampleOffset =
            (tangent * k.x) +
            (bitangent * k.y) +
            (centerNormal * abs(k.z));
        const float3 sampleViewPos = centerViewPos + (sampleOffset * radius);

        const float4 sampleClip = mul(camera.proj, float4(sampleViewPos, 1.0));
        if (sampleClip.w <= 1e-4) continue;

        const float3 sampleNdc = sampleClip.xyz / sampleClip.w;
        const float2 sampleUv = (sampleNdc.xy * 0.5) + 0.5;
        if (sampleUv.x <= 0.0 || sampleUv.x >= 1.0 || sampleUv.y <= 0.0 || sampleUv.y >= 1.0) continue;

        const float sampleDepth = normalDepthTexture.SampleLevel(sampleUv, 0.0).w;
        if (sampleDepth <= 0.0001) continue;

        const float3 sampleSceneViewPos = reconstructViewPosition(sampleUv, sampleDepth);
        const float depthDelta = abs(centerViewPos.z - sampleSceneViewPos.z);
        const float rangeWeight = smoothstep(0.0, 1.0, radius / max(depthDelta, 1e-4));
        const float occluded = (sampleSceneViewPos.z > (sampleViewPos.z + bias)) ? 1.0 : 0.0;

        occlusion += occluded * rangeWeight;
        sampleCount += 1.0;
    }

    if (sampleCount <= 0.0)
    {
        ssaoRawTexture[dispatchThreadId.xy] = 1.0;
        return;
    }

    const float rawAo = 1.0 - (occlusion / sampleCount);
    ssaoRawTexture[dispatchThreadId.xy] = pow(clamp(rawAo, 0.0, 1.0), kSsaoPower);
}
