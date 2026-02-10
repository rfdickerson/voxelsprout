// Compile for Vulkan SPIR-V with column-major matrices:
// slangc -target spirv -entry main -stage fragment -matrix-layout-column-major src/render/shaders/tone_map.frag.slang -o src/render/shaders/tone_map.frag.slang.spv

import camera_uniform;

[[vk::binding(3, 0)]]
Sampler2D<float4> hdrSceneColor;
[[vk::binding(6, 0)]]
Sampler2D<float4> normalDepthTexture;
[[vk::binding(7, 0)]]
Sampler2D<float4> ssaoTexture;

float3 acesFilmTonemap(float3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

float3 applySaturation(float3 color, float saturation)
{
    const float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
    return lerp(float3(luma, luma, luma), color, saturation);
}

float3 decodeViewNormal(float3 encodedNormal)
{
    return normalize((encodedNormal * 2.0) - 1.0);
}

struct FragmentInput
{
    float2 inUv : TEXCOORD0;
};

float4 main(FragmentInput input) : SV_Target0
{
    float3 hdrColor = hdrSceneColor.Sample(input.inUv).xyz;
    const float4 normalDepth = normalDepthTexture.Sample(input.inUv);
    const float viewDepth = normalDepth.w;
    const float geometryMask = step(0.0001, viewDepth);
    const float ssao = clamp(ssaoTexture.Sample(input.inUv).x, 0.0, 1.0);
    const float ssaoMode = camera.shadowVoxelGridSize.w;

    if (ssaoMode > 2.5)
    {
        const float3 viewNormal = decodeViewNormal(normalDepth.xyz);
        const float3 visNormal = (viewNormal * 0.5) + 0.5;
        return float4(visNormal, 1.0);
    }
    if (ssaoMode > 1.5)
    {
        return float4(ssao * geometryMask, ssao * geometryMask, ssao * geometryMask, 1.0);
    }

    const float ssaoEnable = step(0.5, ssaoMode);
    const float ssaoIntensity = clamp(camera.shadowConfig2.z, 0.0, 2.0);
    const float ssaoContribution = lerp(1.0, ssao, ssaoIntensity);
    const float aoFactor = lerp(1.0, ssaoContribution, ssaoEnable);
    hdrColor *= lerp(1.0, aoFactor, geometryMask);

    const float exposure = 0.58;
    float3 toneMapped = acesFilmTonemap(hdrColor * exposure);
    toneMapped = applySaturation(toneMapped, 1.18);
    toneMapped = pow(clamp(toneMapped, 0.0, 1.0), float3(1.06, 1.06, 1.06));
    const float3 ldrSrgb = pow(toneMapped, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
    return float4(ldrSrgb, 1.0);
}
