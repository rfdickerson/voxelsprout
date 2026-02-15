import camera_uniform;

[[vk::binding(6, 0)]]
Sampler2D<float4> normalDepthTexture;

[[vk::constant_id(0)]]
const int kSsaoSampleCount = 32;
[[vk::constant_id(1)]]
const float kSsaoPower = 1.4;

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

    const float viewX = (ndcXY.x * viewDepth) / projX;
    const float viewY = (ndcXY.y * viewDepth) / projY;
    return float3(viewX, viewY, -viewDepth);
}

struct FragmentInput
{
    float2 inUv : TEXCOORD0;
};

float main(FragmentInput input) : SV_Target0
{
    const float4 centerSample = normalDepthTexture.Sample(input.inUv);
    const float centerDepth = centerSample.w;
    if (centerDepth <= 0.0001) return 1.0;

    const float3 centerNormal = decodeViewNormal(centerSample.xyz);
    const float3 centerViewPos = reconstructViewPosition(input.inUv, centerDepth);

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
    const float randomAngle = hash12(input.inUv * noiseScale) * 6.2831853;
    const float s = sin(randomAngle);
    const float c = cos(randomAngle);
    const float2x2 rot = float2x2(c, -s, s, c);

    const float radius = clamp(camera.shadowConfig2.x, 0.15, 3.0);
    const float bias = clamp(camera.shadowConfig2.y, 0.0, 0.20);
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

        const float sampleDepth = normalDepthTexture.Sample(sampleUv).w;
        if (sampleDepth <= 0.0001) continue;

        const float3 sampleSceneViewPos = reconstructViewPosition(sampleUv, sampleDepth);
        const float depthDelta = abs(centerViewPos.z - sampleSceneViewPos.z);
        const float rangeWeight = smoothstep(0.0, 1.0, radius / max(depthDelta, 1e-4));
        const float occluded = (sampleSceneViewPos.z > (sampleViewPos.z + bias)) ? 1.0 : 0.0;

        occlusion += occluded * rangeWeight;
        sampleCount += 1.0;
    }

    if (sampleCount <= 0.0) return 1.0;

    const float rawAo = 1.0 - (occlusion / sampleCount);
    return pow(clamp(rawAo, 0.0, 1.0), kSsaoPower);
}
