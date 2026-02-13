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
    return (1.0 - g2) / denom;
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

    const float averageHeight = 0.5 * (cameraWorld.y + worldPosition.y);
    const float heightDensity = exp(clamp((fogBaseHeight - averageHeight) * heightFalloff, -8.0, 8.0));
    const float opticalDepth = fogDensity * travelDistance * heightDensity;

    const float sunAltitude = clamp(toSun.y, -1.0, 1.0);
    const float dayFactor = smoothstep(0.04, 0.70, sunAltitude);
    const float twilightFactor = 1.0 - dayFactor;
    const float skyT = clamp((worldViewDir.y * 0.5) + 0.5, 0.0, 1.0);
    const float3 fogHorizon = lerp(float3(0.46, 0.58, 0.76), float3(0.80, 0.54, 0.48), twilightFactor * 0.62);
    const float3 fogZenith = lerp(float3(0.24, 0.32, 0.48), float3(0.36, 0.25, 0.34), twilightFactor * 0.52);
    float3 ambientFogColor = lerp(fogHorizon, fogZenith, pow(skyT, 0.36));
    const float3 coolAerialHorizon = float3(0.50, 0.62, 0.78);
    const float3 coolAerialZenith = float3(0.34, 0.48, 0.66);
    const float3 coolAerialColor = lerp(coolAerialHorizon, coolAerialZenith, pow(skyT, 0.45));
    ambientFogColor = lerp(ambientFogColor, coolAerialColor, dayFactor * 0.45);

    const float cosTheta = clamp(dot(worldViewDir, toSun), -1.0, 1.0);
    const float sunPhase = hgPhase(cosTheta, camera.skyConfig0.z) * 0.10;
    const float3 sunScatterColor = camera.sunColorShadow.xyz * (sunPhase * sunScatterStrength * clamp(sunVisibility, 0.0, 1.0));
    const float3 fogColor = ambientFogColor + sunScatterColor;

    // RGB extinction gives atmospheric perspective (distance naturally desaturates and cools).
    const float3 sigmaRayleigh = float3(0.18, 0.34, 0.72) * max(camera.skyConfig0.x, 0.01);
    const float sigmaMie = 0.32 * max(camera.skyConfig0.y, 0.01);
    const float3 extinction = (sigmaRayleigh + float3(sigmaMie, sigmaMie, sigmaMie));
    const float3 transmittance = exp(-(extinction * opticalDepth));
    const float luma = dot(litColor, float3(0.2126, 0.7152, 0.0722));
    const float fogAmount = saturate(1.0 - dot(transmittance, float3(0.2126, 0.7152, 0.0722)));
    const float3 litAerial = lerp(litColor, float3(luma, luma, luma), fogAmount * 0.70);
    return (litAerial * transmittance) + (fogColor * (float3(1.0, 1.0, 1.0) - transmittance));
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
        hdrColor = applyVolumetricHeightFog(
            hdrColor,
            cameraWorld,
            worldPos,
            travelDistance,
            worldViewDir,
            toSun,
            sunVisibility
        );
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

    if (geometryMask < 0.5)
    {
        const float shaftFactor = sampleScreenSpaceSunShafts(input.inUv, sunUv, sunVisibility, sunFacing);
        if (shaftFactor > 1e-5)
        {
            const float shaftStrength = (camera.skyConfig4.x * 8.0) + (camera.skyConfig4.w * 0.04);
            const float3 shaftTint = lerp(camera.sunColorShadow.xyz, float3(1.0, 0.95, 0.90), 0.25);
            hdrColor += shaftTint * (shaftFactor * shaftStrength);
        }
    }

    float3 toneMapped = acesFilmTonemap(hdrColor * exposureScale);
    toneMapped = applySaturation(toneMapped, 1.18);
    toneMapped = pow(clamp(toneMapped, 0.0, 1.0), float3(1.06, 1.06, 1.06));
    const float3 ldrSrgb = pow(toneMapped, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
    return float4(ldrSrgb, 1.0);
}
