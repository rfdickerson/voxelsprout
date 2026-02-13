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

float luminance709(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float3 extractBloom(float3 hdrColor, float threshold, float softKnee)
{
    const float luma = max(luminance709(hdrColor), 0.0);
    const float knee = max(threshold * softKnee, 1e-5);
    const float soft = clamp((luma - threshold + knee) / (2.0 * knee), 0.0, 1.0);
    const float contribution = max(luma - threshold, 0.0) + (soft * soft * knee);
    if (contribution <= 1e-5 || luma <= 1e-5)
    {
        return float3(0.0, 0.0, 0.0);
    }
    return hdrColor * (contribution / luma);
}

float3 sampleMipChainBloom(float2 uv, float sunFacing)
{
    const float bloomThreshold = max(camera.skyConfig3.x, 0.0);
    const float bloomSoftKnee = clamp(camera.skyConfig3.y, 0.0, 1.0);
    const float3 bloomMip1 = extractBloom(hdrSceneColor.SampleLevel(uv, 1.0).xyz, bloomThreshold, bloomSoftKnee);
    const float3 bloomMip2 = extractBloom(hdrSceneColor.SampleLevel(uv, 2.0).xyz, bloomThreshold, bloomSoftKnee);
    const float3 bloomMip3 = extractBloom(hdrSceneColor.SampleLevel(uv, 3.0).xyz, bloomThreshold, bloomSoftKnee);
    const float3 bloomMip4 = extractBloom(hdrSceneColor.SampleLevel(uv, 4.0).xyz, bloomThreshold, bloomSoftKnee);
    const float3 bloomMip5 = extractBloom(hdrSceneColor.SampleLevel(uv, 5.0).xyz, bloomThreshold, bloomSoftKnee);

    float3 bloom =
        (bloomMip1 * 0.50) +
        (bloomMip2 * 0.28) +
        (bloomMip3 * 0.14) +
        (bloomMip4 * 0.06) +
        (bloomMip5 * 0.03);
    const float sunFacingScale = lerp(0.85, 1.35, clamp(sunFacing, 0.0, 1.0));
    return bloom * sunFacingScale;
}

float3 analyticSunBloom(float2 uv, float sunFacing)
{
    const float3 toSun = normalize(-camera.sunDirectionIntensity.xyz);
    const float3 sunView = mul((float3x3)camera.view, toSun);
    if (sunView.z >= -1e-4)
    {
        return float3(0.0, 0.0, 0.0);
    }

    const float invDepth = 1.0 / max(-sunView.z, 1e-4);
    const float2 sunNdc = float2(
        sunView.x * camera.proj[0][0] * invDepth,
        sunView.y * camera.proj[1][1] * invDepth
    );
    const float2 sunUv = (sunNdc * 0.5) + 0.5;

    float2 delta = uv - sunUv;
    const float aspect = clamp(
        abs(camera.proj[1][1]) / max(abs(camera.proj[0][0]), 1e-4),
        0.25,
        4.0
    );
    delta.x *= aspect;
    const float radialDistance = length(delta);

    const float core = exp(-radialDistance * 62.0);
    const float halo = exp(-radialDistance * 12.0);
    const float wide = exp(-radialDistance * 3.0);
    const float glow = (1.75 * core) + (0.95 * halo) + (0.35 * wide);
    return camera.sunColorShadow.xyz * (glow * clamp(sunFacing, 0.0, 1.0));
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
    const float giDebugMode = camera.shadowVoxelGridOrigin.z;

    if (giDebugMode > 0.5)
    {
        const float exposure = 0.75;
        float3 toneMapped = acesFilmTonemap(hdrColor * exposure);
        toneMapped = applySaturation(toneMapped, 1.0);
        const float3 ldrSrgb = pow(clamp(toneMapped, 0.0, 1.0), float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
        return float4(ldrSrgb, 1.0);
    }

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

    const float3x3 invViewRotation = transpose((float3x3)camera.view);
    const float3 cameraForward = normalize(mul(invViewRotation, float3(0.0, 0.0, -1.0)));
    const float3 toSun = normalize(-camera.sunDirectionIntensity.xyz);
    const float sunFacing = smoothstep(cos(radians(20.0)), cos(radians(3.0)), dot(cameraForward, toSun));

    const float bloomBaseIntensity = max(camera.skyConfig3.z, 0.0);
    const float bloomSunFacingBoost = max(camera.skyConfig3.w, 0.0);
    const float bloomIntensity = bloomBaseIntensity + (sunFacing * bloomSunFacingBoost);
    if (bloomIntensity > 1e-5)
    {
        const float3 sampledBloom = sampleMipChainBloom(input.inUv, sunFacing);
        const float3 sunBloom = analyticSunBloom(input.inUv, sunFacing);
        hdrColor += (sampledBloom + sunBloom) * bloomIntensity;
    }

    const float exposure = 0.58;
    float3 toneMapped = acesFilmTonemap(hdrColor * exposure);
    toneMapped = applySaturation(toneMapped, 1.18);
    toneMapped = pow(clamp(toneMapped, 0.0, 1.0), float3(1.06, 1.06, 1.06));
    const float3 ldrSrgb = pow(toneMapped, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
    return float4(ldrSrgb, 1.0);
}
