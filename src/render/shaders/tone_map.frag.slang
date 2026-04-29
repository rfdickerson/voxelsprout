// Compile for Vulkan SPIR-V with column-major matrices:
// slangc -target spirv -entry main -stage fragment -matrix-layout-column-major src/render/shaders/tone_map.frag.slang -o src/render/shaders/tone_map.frag.slang.spv

import camera_uniform;

[[vk::binding(2, 0)]]
StructuredBuffer<float4> exposureStateBuffer;
[[vk::binding(3, 0)]]
Sampler2D<float4> hdrSceneColor;
[[vk::binding(6, 0)]]
Sampler2D<float4> normalDepthTexture;
[[vk::binding(7, 0)]]
Sampler2D<float4> ssaoTexture;
[[vk::binding(10, 0)]]
Sampler2D<float4> sunShaftTexture;

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

float3 applyWhiteBalance(float3 color, float3 whiteBalance)
{
    return max(color * max(whiteBalance, float3(0.0, 0.0, 0.0)), float3(0.0, 0.0, 0.0));
}

float3 applyContrastCurve(float3 color, float contrast)
{
    const float c = clamp(contrast, 0.70, 1.40);
    return saturate(((color - 0.5) * c) + 0.5);
}

float3 applyMidtoneContrast(float3 color, float contrast)
{
    const float c = clamp(contrast, 0.80, 1.40);
    const float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
    const float midMask = 1.0 - smoothstep(0.0, 0.60, abs(luma - 0.50) * 2.0);
    return saturate(color + ((color - 0.5) * ((c - 1.0) * midMask)));
}

float3 applyShadowDensity(float3 color, float density)
{
    const float d = clamp(density, 0.70, 1.40);
    const float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
    const float shadowMask = 1.0 - smoothstep(0.10, 0.55, luma);
    const float shadowScale = lerp(1.0, d, shadowMask);
    return saturate(color * shadowScale);
}

float3 applyHighlightRolloff(float3 color, float rolloff)
{
    const float r = clamp(rolloff, 0.70, 1.10);
    const float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
    const float highlightMask = smoothstep(0.55, 1.0, luma);
    const float compression = max(1.0 - r, 0.0);
    const float3 compressed = color / (1.0 + (color * (compression * 1.8)));
    return saturate(lerp(color, compressed, highlightMask));
}

float3 applyVibrance(float3 color, float vibrance)
{
    const float v = clamp(vibrance, -1.0, 1.0);
    const float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
    const float peak = max(color.x, max(color.y, color.z));
    const float valley = min(color.x, min(color.y, color.z));
    const float saturation = peak - valley;
    const float boost = 1.0 + (v * (1.0 - saturation));
    return saturate(lerp(float3(luma, luma, luma), color, boost));
}

float3 preserveBrightColor(float3 color, float amount)
{
    const float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
    const float peak = max(color.x, max(color.y, color.z));
    const float valley = min(color.x, min(color.y, color.z));
    const float saturation = peak - valley;
    const float highlightMask = smoothstep(0.55, 0.95, luma);
    const float boost = 1.0 + (amount * (1.0 - saturation) * highlightMask);
    return saturate(lerp(float3(luma, luma, luma), color, boost));
}

float3 applySplitTone(float3 color, float3 shadowTint, float3 highlightTint)
{
    const float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
    const float pivot = 0.50;
    const float shadowWeight = saturate((pivot - luma) / max(pivot, 1e-3));
    const float highlightWeight = saturate((luma - pivot) / max(1.0 - pivot, 1e-3));
    float3 result = color;
    result *= lerp(float3(1.0, 1.0, 1.0), float3(1.0, 1.0, 1.0) + shadowTint, shadowWeight);
    result *= lerp(float3(1.0, 1.0, 1.0), float3(1.0, 1.0, 1.0) + highlightTint, highlightWeight);
    return saturate(result);
}

float3 decodeViewNormal(float3 encodedNormal)
{
    return normalize((encodedNormal * 2.0) - 1.0);
}

float luminance709(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float bloomBrightness(float3 hdrColor)
{
    const float peakChannel = max(hdrColor.x, max(hdrColor.y, hdrColor.z));
    return max(luminance709(hdrColor), peakChannel);
}

float3 extractBloom(float3 hdrColor, float threshold, float softKnee)
{
    const float brightness = max(bloomBrightness(hdrColor), 0.0);
    const float knee = max(threshold * softKnee, 1e-5);
    const float soft = clamp((brightness - threshold + knee) / (2.0 * knee), 0.0, 1.0);
    const float contribution = max(brightness - threshold, 0.0) + (soft * soft * knee);
    if (contribution <= 1e-5 || brightness <= 1e-5)
    {
        return float3(0.0, 0.0, 0.0);
    }
    return hdrColor * (contribution / brightness);
}

float depthOfFieldRadiusPixels(float viewDepth)
{
    if (camera.dofConfig.x <= 0.5 || viewDepth <= 0.0001)
    {
        return 0.0;
    }

    const float sharpDistance = max(camera.dofConfig.y, 0.5);
    const float falloffDistance = max(camera.dofConfig.z, 0.5);
    const float maxRadiusPixels = max(camera.dofConfig.w, 0.0);
    const float farDistance = max(viewDepth - sharpDistance, 0.0);
    const float farFalloff = smoothstep(0.0, falloffDistance * 2.5, farDistance);
    return farFalloff * maxRadiusPixels * 0.55;
}

float interleavedGradientNoise(float2 pixel)
{
    return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
}

float2 rotate2d(float2 v, float angle)
{
    const float s = sin(angle);
    const float c = cos(angle);
    return float2((v.x * c) - (v.y * s), (v.x * s) + (v.y * c));
}

float2 apertureSampleOffset(int sampleIndex, int sampleCount, float rotation)
{
    const float kGoldenAngle = 2.39996323;
    const float sampleT = (float(sampleIndex) + 0.5) / float(sampleCount);
    const float radius = sqrt(sampleT);
    const float angle = (float(sampleIndex) * kGoldenAngle) + rotation;
    return float2(cos(angle), sin(angle)) * radius;
}

float3 sampleDepthOfField(
    float2 uv,
    float centerDepth,
    float3 centerColor,
    float ssaoEnable,
    float ssaoIntensity
)
{
    const float centerRadiusPixels = depthOfFieldRadiusPixels(centerDepth);
    if (centerRadiusPixels <= 0.25)
    {
        return centerColor;
    }

    uint texW = 0u;
    uint texH = 0u;
    hdrSceneColor.GetDimensions(texW, texH);
    const float2 texelSize = 1.0 / float2(float(max(texW, 1u)), float(max(texH, 1u)));

    const float pixelRotation = interleavedGradientNoise(uv * float2(float(texW), float(texH))) * 6.28318531;
    const float gatherMip = clamp(log2(max(centerRadiusPixels * 0.45, 1.0)), 0.0, 3.5);
    float3 colorSum = centerColor * 1.5;
    float weightSum = 1.5;
    const float depthTolerance = max(camera.dofConfig.z * 0.45, 1.0);
    const int sampleCount = 24;
    for (int i = 0; i < sampleCount; ++i)
    {
        float2 apertureOffset = apertureSampleOffset(i, sampleCount, pixelRotation);
        apertureOffset.x *= 1.08;
        apertureOffset = rotate2d(apertureOffset, 0.18);
        const float2 sampleUv = clamp(
            uv + (apertureOffset * texelSize * centerRadiusPixels),
            float2(0.0, 0.0),
            float2(1.0, 1.0)
        );
        const float4 sampleNormalDepth = normalDepthTexture.SampleLevel(sampleUv, 0.0);
        const float sampleDepth = sampleNormalDepth.w;
        const float sampleGeometry = step(0.0001, sampleDepth);
        const float sampleRadiusPixels = depthOfFieldRadiusPixels(sampleDepth);
        const float sameLayer = 1.0 - smoothstep(depthTolerance, depthTolerance * 2.0, abs(sampleDepth - centerDepth));
        const float blurWeight = max(sameLayer, smoothstep(0.0, centerRadiusPixels, sampleRadiusPixels));
        const float apertureEdge = 1.0 - smoothstep(0.55, 1.05, length(apertureOffset));
        const float weight = lerp(0.18, 1.0, blurWeight) * lerp(0.45, 1.0, sampleGeometry) * lerp(0.65, 1.0, apertureEdge);
        const float sampleSsao = clamp(ssaoTexture.SampleLevel(sampleUv, 0.0).x, 0.0, 1.0);
        const float sampleAo = pow(max(sampleSsao, 0.001), ssaoIntensity);
        const float sampleAoFactor = lerp(1.0, sampleAo, ssaoEnable * sampleGeometry);
        const float3 sampleColor = hdrSceneColor.SampleLevel(sampleUv, gatherMip).xyz * sampleAoFactor;
        const float highlightWeight = 1.0 + (smoothstep(1.0, 8.0, bloomBrightness(sampleColor)) * centerRadiusPixels * 0.035);
        colorSum += sampleColor * weight * highlightWeight;
        weightSum += weight;
    }

    return colorSum / max(weightSum, 1e-5);
}

float resolveExposure()
{
    const float manualExposure = max(camera.skyConfig5.y, 0.001);
    if (camera.skyConfig5.x <= 0.5)
    {
        return manualExposure;
    }
    const float dynamicExposure = max(exposureStateBuffer[0].x, 0.001);
    return dynamicExposure;
}

float3 sampleMipChainBloom(float2 uv, float exposureScale)
{
    const float bloomThreshold = max(camera.skyConfig3.x / max(exposureScale, 1e-4), 0.0);
    const float bloomSoftKnee = clamp(camera.skyConfig3.y, 0.0, 1.0);
    const float3 bloomMip0 = extractBloom(hdrSceneColor.SampleLevel(uv, 0.0).xyz, bloomThreshold, bloomSoftKnee);
    const float3 bloomMip1 = extractBloom(hdrSceneColor.SampleLevel(uv, 1.0).xyz, bloomThreshold, bloomSoftKnee);
    const float3 bloomMip2 = extractBloom(hdrSceneColor.SampleLevel(uv, 2.0).xyz, bloomThreshold, bloomSoftKnee);
    const float3 bloomMip3 = extractBloom(hdrSceneColor.SampleLevel(uv, 3.0).xyz, bloomThreshold, bloomSoftKnee);
    const float3 bloomMip4 = extractBloom(hdrSceneColor.SampleLevel(uv, 4.0).xyz, bloomThreshold, bloomSoftKnee);
    const float3 bloomMip5 = extractBloom(hdrSceneColor.SampleLevel(uv, 5.0).xyz, bloomThreshold, bloomSoftKnee);

    float3 bloom =
        (bloomMip0 * 0.20) +
        (bloomMip1 * 0.40) +
        (bloomMip2 * 0.24) +
        (bloomMip3 * 0.11) +
        (bloomMip4 * 0.05) +
        (bloomMip5 * 0.02);
    return bloom;
}

float sampleSunVisibility(float2 sunUv)
{
    if (any(sunUv < float2(0.0, 0.0)) || any(sunUv > float2(1.0, 1.0)))
    {
        return 0.0;
    }

    uint texW = 0u;
    uint texH = 0u;
    normalDepthTexture.GetDimensions(texW, texH);
    const float2 texelSize = 1.0 / float2(float(max(texW, 1u)), float(max(texH, 1u)));
    const float2 occlusionRadius = texelSize * 2.0;

    float minSkyVisibility = 1.0;
    const float2 offsets[5] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(-1.0, 0.0),
        float2(0.0, 1.0),
        float2(0.0, -1.0)
    };

    for (int i = 0; i < 5; ++i)
    {
        const float2 sampleUv = clamp(sunUv + (offsets[i] * occlusionRadius), float2(0.0, 0.0), float2(1.0, 1.0));
        const float sampleViewDepth = normalDepthTexture.SampleLevel(sampleUv, 0.0).w;
        const float skyVisibility = 1.0 - step(0.0001, sampleViewDepth);
        minSkyVisibility = min(minSkyVisibility, skyVisibility);
    }

    return minSkyVisibility;
}

float sampleSkyMask(float2 uv)
{
    if (any(uv < float2(0.0, 0.0)) || any(uv > float2(1.0, 1.0)))
    {
        return 0.0;
    }
    const float viewDepth = normalDepthTexture.SampleLevel(uv, 0.0).w;
    return 1.0 - step(0.0001, viewDepth);
}

float sampleFogGeometryCoverage(float2 uv)
{
    uint texW = 0u;
    uint texH = 0u;
    normalDepthTexture.GetDimensions(texW, texH);
    const float2 texelSize = 1.0 / float2(float(max(texW, 1u)), float(max(texH, 1u)));

    float coverage = step(0.0001, normalDepthTexture.SampleLevel(uv, 0.0).w) * 4.0;
    coverage += step(0.0001, normalDepthTexture.SampleLevel(uv + float2(texelSize.x, 0.0), 0.0).w);
    coverage += step(0.0001, normalDepthTexture.SampleLevel(uv - float2(texelSize.x, 0.0), 0.0).w);
    coverage += step(0.0001, normalDepthTexture.SampleLevel(uv + float2(0.0, texelSize.y), 0.0).w);
    coverage += step(0.0001, normalDepthTexture.SampleLevel(uv - float2(0.0, texelSize.y), 0.0).w);
    coverage += step(0.0001, normalDepthTexture.SampleLevel(uv + texelSize, 0.0).w) * 0.5;
    coverage += step(0.0001, normalDepthTexture.SampleLevel(uv - texelSize, 0.0).w) * 0.5;
    coverage += step(0.0001, normalDepthTexture.SampleLevel(uv + float2(texelSize.x, -texelSize.y), 0.0).w) * 0.5;
    coverage += step(0.0001, normalDepthTexture.SampleLevel(uv + float2(-texelSize.x, texelSize.y), 0.0).w) * 0.5;
    return saturate(coverage / 10.0);
}

float sampleScreenSpaceSunShafts(float2 uv, float2 sunUv, float sunVisibility, float sunFacing)
{
    if (sunVisibility <= 1e-5 || sunFacing <= 1e-5)
    {
        return 0.0;
    }

    const float2 deltaToSun = sunUv - uv;
    const float radialDistance = length(deltaToSun);
    if (radialDistance <= 1e-5)
    {
        return sunVisibility;
    }

    const float rangeMask = 1.0 - smoothstep(0.03, 0.26, radialDistance);
    if (rangeMask <= 1e-5)
    {
        return 0.0;
    }

    const int sampleCount = 7;
    const float2 stepUv = deltaToSun / float(sampleCount);
    float2 sampleUv = uv + (stepUv * 0.5);
    float decay = 1.0;
    float shaftAccum = 0.0;
    float weightAccum = 0.0;
    for (int i = 0; i < sampleCount; ++i)
    {
        const float skyMask = sampleSkyMask(sampleUv);
        shaftAccum += skyMask * decay;
        weightAccum += decay;
        decay *= 0.86;
        sampleUv += stepUv;
    }

    const float shaftOcclusion = shaftAccum / max(weightAccum, 1e-5);
    const float localSkyMask = sampleSkyMask(uv);
    // Keep shafts mostly in the sky to avoid "image copies" over foreground geometry.
    const float geometryAttenuation = localSkyMask;
    return shaftOcclusion * rangeMask * sunVisibility * geometryAttenuation;
}

float3 analyticSunBloom(float2 uv, float2 sunUv, float sunVisibility)
{
    if (sunVisibility <= 1e-5)
    {
        return float3(0.0, 0.0, 0.0);
    }

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
    return camera.sunColorShadow.xyz * (glow * clamp(sunVisibility, 0.0, 1.0));
}

float hgPhase(float cosTheta, float anisotropy)
{
    const float g = clamp(anisotropy, 0.0, 0.95);
    const float g2 = g * g;
    const float denom = max(pow(1.0 + g2 - (2.0 * g * cosTheta), 1.5), 1e-4);
    return (1.0 - g2) / (4.0 * 3.14159265 * denom);
}

float heightFogDensityAt(float worldHeight, float fogBaseHeight, float heightFalloff)
{
    const float relativeHeight = worldHeight - fogBaseHeight;
    return exp(clamp(-relativeHeight * heightFalloff, -7.0, 7.0));
}

float3 fogExtinctionCoefficient()
{
    const float rayleighT = clamp(camera.skyConfig0.x * 0.25, 0.0, 1.0);
    const float mieT = clamp(camera.skyConfig0.y * 0.25, 0.0, 1.0);
    const float rayleighStrength = lerp(0.80, 1.20, rayleighT);
    const float mieStrength = lerp(0.80, 1.20, mieT);
    const float3 rayleighExtinction = float3(0.025, 0.050, 0.110) * rayleighStrength;
    const float3 mieExtinction = float3(0.145, 0.140, 0.130) * mieStrength;
    return rayleighExtinction + mieExtinction;
}

float3 fogScatteringAlbedo()
{
    return float3(0.98, 0.955, 0.90);
}

float3 volumetricIncidentRadiance(
    float3 worldViewDir,
    float3 toSun,
    float sampleHeight,
    float fogBaseHeight,
    float sunScatterStrength,
    float sunVisibility
)
{
    const float sunAltitude = clamp(toSun.y, -1.0, 1.0);
    const float dayFactor = smoothstep(0.04, 0.70, sunAltitude);
    const float twilightFactor = 1.0 - dayFactor;
    const float skyT = clamp((worldViewDir.y * 0.5) + 0.5, 0.0, 1.0);
    const float valleyT = 1.0 - smoothstep(fogBaseHeight - 16.0, fogBaseHeight + 160.0, sampleHeight);

    const float3 skyHorizon = lerp(float3(0.48, 0.57, 0.70), float3(0.86, 0.52, 0.32), twilightFactor * 0.58);
    const float3 skyZenith = lerp(float3(0.24, 0.34, 0.58), float3(0.30, 0.22, 0.30), twilightFactor * 0.46);
    const float3 groundBounce = lerp(float3(0.54, 0.49, 0.40), float3(0.66, 0.46, 0.30), twilightFactor);
    const float3 skyRadiance = lerp(skyHorizon, skyZenith, pow(skyT, 0.45));
    const float ambientStrength = lerp(0.86, 1.26, dayFactor);
    const float3 ambientRadiance = lerp(skyRadiance, groundBounce, valleyT * 0.10) * ambientStrength;

    const float cosTheta = clamp(dot(worldViewDir, toSun), -1.0, 1.0);
    const float sunPhase = hgPhase(cosTheta, camera.skyConfig0.z);
    const float sunAboveHorizon = smoothstep(-0.08, 0.18, sunAltitude);
    const float sunOcclusion = clamp(sunVisibility, 0.0, 1.0);
    const float3 sunRadiance =
        camera.sunColorShadow.xyz * (sunPhase * sunScatterStrength * sunAboveHorizon * sunOcclusion);
    return ambientRadiance + sunRadiance;
}

float3 applyVolumetricHeightFog(
    float3 litColor,
    float3 cameraWorld,
    float3 worldPosition,
    float travelDistance,
    float3 worldViewDir,
    float3 toSun,
    float sunVisibility
)
{
    const float fogDensity = max(camera.skyConfig4.x, 0.0);
    if (fogDensity <= 1e-6 || travelDistance <= 1e-4)
    {
        return litColor;
    }

    const float heightFalloff = max(camera.skyConfig4.y, 0.0);
    const float fogBaseHeight = camera.skyConfig4.z;
    const float sunScatterStrength = max(camera.skyConfig4.w, 0.0);

    const float3 extinction = fogExtinctionCoefficient();
    const float3 scatteringAlbedo = fogScatteringAlbedo();

    const float horizonGather = 1.0 + (pow(1.0 - saturate(abs(worldViewDir.y)), 2.0) * 0.32);
    const int sampleCount = 6;
    const float stepLength = travelDistance / float(sampleCount);
    float3 transmittance = float3(1.0, 1.0, 1.0);
    float3 inscatter = float3(0.0, 0.0, 0.0);

    [unroll]
    for (int i = 0; i < sampleCount; ++i)
    {
        const float t = (float(i) + 0.5) * stepLength;
        const float3 samplePosition = cameraWorld + (worldViewDir * t);
        const float localDensity =
            fogDensity * heightFogDensityAt(samplePosition.y, fogBaseHeight, heightFalloff) * horizonGather;
        const float opticalDepth = localDensity * stepLength;
        const float3 segmentTransmittance = exp(-(extinction * opticalDepth));
        const float3 incidentRadiance = volumetricIncidentRadiance(
            worldViewDir,
            toSun,
            samplePosition.y,
            fogBaseHeight,
            sunScatterStrength,
            sunVisibility
        );
        inscatter +=
            transmittance *
            (float3(1.0, 1.0, 1.0) - segmentTransmittance) *
            scatteringAlbedo *
            incidentRadiance;
        transmittance *= segmentTransmittance;
    }

    return (litColor * transmittance) + inscatter;
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
    const float ssaoContribution = pow(max(ssao, 0.001), ssaoIntensity);
    const float aoFactor = lerp(1.0, ssaoContribution, ssaoEnable);
    hdrColor *= lerp(1.0, aoFactor, geometryMask);

    if (geometryMask > 0.5)
    {
        hdrColor = sampleDepthOfField(input.inUv, viewDepth, hdrColor, ssaoEnable, ssaoIntensity);
    }

    const float3x3 invViewRotation = transpose((float3x3)camera.view);
    const float3 toSun = normalize(-camera.sunDirectionIntensity.xyz);
    const float3 sunView = mul((float3x3)camera.view, toSun);
    float2 sunUv = float2(-1.0, -1.0);
    float sunVisibility = 0.0;
    if (sunView.z < -1e-4)
    {
        const float invDepth = 1.0 / max(-sunView.z, 1e-4);
        const float2 sunNdc = float2(
            sunView.x * camera.proj[0][0] * invDepth,
            sunView.y * camera.proj[1][1] * invDepth
        );
        sunUv = (sunNdc * 0.5) + 0.5;
        if (!any(sunUv < float2(0.0, 0.0)) && !any(sunUv > float2(1.0, 1.0)))
        {
            sunVisibility = sampleSunVisibility(sunUv);
        }
    }

    if (geometryMask > 0.5)
    {
        const float projX = (abs(camera.proj[0][0]) < 1e-6)
            ? ((camera.proj[0][0] < 0.0) ? -1e-6 : 1e-6)
            : camera.proj[0][0];
        const float projY = (abs(camera.proj[1][1]) < 1e-6)
            ? ((camera.proj[1][1] < 0.0) ? -1e-6 : 1e-6)
            : camera.proj[1][1];
        const float2 ndc = (input.inUv * 2.0) - 1.0;
        const float3 viewRay = normalize(float3(ndc.x / projX, ndc.y / projY, -1.0));
        const float travelDistance = viewDepth / max(-viewRay.z, 1e-4);
        const float3 worldViewDir = normalize(mul(invViewRotation, viewRay));
        const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
        const float3 worldPos = cameraWorld + (worldViewDir * travelDistance);
        const float3 foggedColor = applyVolumetricHeightFog(
            hdrColor,
            cameraWorld,
            worldPos,
            travelDistance,
            worldViewDir,
            toSun,
            sunVisibility
        );
        hdrColor = lerp(hdrColor, foggedColor, sampleFogGeometryCoverage(input.inUv));
    }

    const float3 cameraForward = normalize(mul(invViewRotation, float3(0.0, 0.0, -1.0)));
    const float sunForwardDot = dot(cameraForward, toSun);
    // Wide falloff keeps some bloom when the sun is off-center.
    const float sunFacingWide = smoothstep(cos(radians(72.0)), cos(radians(10.0)), sunForwardDot);
    // Tight falloff keeps a strong kick when looking close to the sun.
    const float sunFacingTight = smoothstep(cos(radians(20.0)), cos(radians(3.0)), sunForwardDot);
    const float sunFacing = max(sunFacingTight, sunFacingWide * 0.65);

    const float sunBloomFactor = sunFacing * sunVisibility;
    const float exposureScale = resolveExposure();
    const float bloomGlobalIntensity = max(camera.skyConfig3.z, 0.0);
    if (bloomGlobalIntensity > 1e-5)
    {
        float3 sampledBloom = sampleMipChainBloom(input.inUv, exposureScale);
        // Prevent residual sun-disk/bloom leakage through occluders near the sun location.
        if (!any(sunUv < float2(0.0, 0.0)) && !any(sunUv > float2(1.0, 1.0)))
        {
            float2 sunDelta = input.inUv - sunUv;
            const float aspect = clamp(
                abs(camera.proj[1][1]) / max(abs(camera.proj[0][0]), 1e-4),
                0.25,
                4.0
            );
            sunDelta.x *= aspect;
            const float sunRegion = 1.0 - smoothstep(0.012, 0.080, length(sunDelta));
            const float blocked = 1.0 - clamp(sunVisibility, 0.0, 1.0);
            const float occlusionAttenuation = 1.0 - (blocked * sunRegion * geometryMask);
            sampledBloom *= clamp(occlusionAttenuation, 0.0, 1.0);
        }
        hdrColor += sampledBloom * bloomGlobalIntensity;
    }

    const float bloomSunFacingBoost = max(camera.skyConfig3.w, 0.0);
    const float sunBloomIntensity = sunBloomFactor * bloomSunFacingBoost;
    if (sunBloomIntensity > 1e-5)
    {
        const float3 sunBloom = analyticSunBloom(input.inUv, sunUv, sunBloomFactor);
        hdrColor += sunBloom * sunBloomIntensity;
    }

    const float shaftFactor = sunShaftTexture.Sample(input.inUv).x;
    if (shaftFactor > 1e-5)
    {
        const float shaftStrength = (camera.skyConfig4.x * 8.0) + (camera.skyConfig4.w * 0.04);
        const float3 shaftTint = lerp(camera.sunColorShadow.xyz, float3(1.0, 0.95, 0.90), 0.25);
        // Let shafts affect geometry too (window beams), but attenuate by scene depth for plausibility.
        const float geometryAttenuation = lerp(1.0, exp(-viewDepth * 0.030), geometryMask);
        hdrColor += shaftTint * (shaftFactor * shaftStrength * geometryAttenuation * clamp(sunVisibility, 0.0, 1.0));
    }

    float3 toneMapped = acesFilmTonemap(hdrColor * exposureScale);
    toneMapped = applyWhiteBalance(
        toneMapped,
        camera.colorGrading0.xyz
    );
    toneMapped = applyContrastCurve(toneMapped, camera.colorGrading0.w);
    toneMapped = applyMidtoneContrast(toneMapped, camera.colorGrading1.z);
    toneMapped = applyShadowDensity(toneMapped, camera.colorGrading1.w);
    toneMapped = applySaturation(toneMapped, camera.colorGrading1.x);
    toneMapped = applyVibrance(toneMapped, camera.colorGrading1.y);
    toneMapped = applyHighlightRolloff(toneMapped, camera.colorGrading2.w);
    toneMapped = preserveBrightColor(toneMapped, 0.35 * (2.0 - camera.colorGrading2.w));
    toneMapped = applySplitTone(
        toneMapped,
        camera.colorGrading2.xyz,
        camera.colorGrading3.xyz
    );
    toneMapped = pow(clamp(toneMapped, 0.0, 1.0), float3(1.06, 1.06, 1.06));
    const float3 ldrSrgb = pow(toneMapped, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
    return float4(ldrSrgb, 1.0);
}
