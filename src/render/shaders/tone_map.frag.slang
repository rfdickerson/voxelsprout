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
// SEPARATE FROM hdrSceneColor, and the reason is the bloom chain below.
//
// Bloom is gathered by sampling MIPS 0-5 of one texture. hdrSceneColor is the
// scene input, and once a temporal upscaler exists that is a single-mip image at
// output resolution -- every "mip" fetch then returns mip0, so bloom becomes six
// copies of the full-resolution scene added on top of itself. Measured as a
// uniform 1.27x brightening of the whole frame, which reads as a washed-out
// image rather than as anything bloom-shaped.
//
// This binding always points at the mip-chained render-resolution HDR target,
// whether or not an upscaler ran. When none did, both bindings are the same
// image and the result is byte-identical to before.
[[vk::binding(14, 0)]]
Sampler2D<float4> bloomSourceColor;

// DebugView in src/render/renderer_types.h; change both together. Only the two
// this pass acts on are named here -- every other mode is handled in
// imported_static.frag.slang and reaches here as "not Off", which is all the
// tonemap bypass needs to know.
static const uint kDebugViewOff = 0u;
static const uint kDebugViewLinearDepth = 10u;

float3 acesFilmTonemap(float3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

// The same fit, renormalized so that `whitePoint` scene-linear maps to display
// white instead of the curve's own asymptote.
//
// The Narkowicz fit approaches 1.0 only as input grows without bound, so with
// auto-exposure holding the scene around middle grey the frame's brightest
// pixels land near 0.65 and the top third of the display range is never
// addressed at all. Dividing by the curve's value AT the white point is the
// standard Hable/Reinhard remedy -- curve(x)/curve(W) -- and it pins a chosen
// scene value to 1.0 without changing the toe, so shadows keep the filmic roll
// they already had.
//
// `shoulder` blends between the raw fit and the normalized one, because pinning
// white exactly is often too much: it makes the sky clip flat. Around 0.6-0.8
// lifts the highlights into the range while leaving the very top soft.
//
// whitePoint <= 0 returns the plain fit BIT-FOR-BIT, which is the default and
// is what keeps every other game in this engine rendering exactly as before.
float3 acesFilmTonemapWhite(float3 color, float whitePoint, float shoulder)
{
    const float3 raw = acesFilmTonemap(color);
    if (whitePoint <= 0.0)
    {
        return raw;
    }
    // Scalar: the fit is applied per channel but the normalizer must not be, or
    // it becomes a per-channel gain and shifts hue in the highlights.
    const float3 white = acesFilmTonemap(float3(whitePoint, whitePoint, whitePoint));
    const float3 normalized = clamp(raw / max(white, 1e-4), 0.0, 1.0);
    return lerp(raw, normalized, clamp(shoulder, 0.0, 1.0));
}

// The tonemap Enhanced Shaders uses (ENB's POSTPROCESS==2), transcribed from
// enbeffect.fx with its own tuned defaults.
//
// Three things it does differently from the ACES fit above, and all three are
// the reason it holds up on Fallout's authored weather colours:
//
//  * The curve is an extended Reinhard, x(1 + x/L)/(x + C), where C is the knee
//    and L damps overbrights. Both are PARAMETERS -- ENB retunes them by time of
//    day (C = 8.0 day, 10.0 night) where the ACES fit is a fixed rational.
//  * Contrast is applied to the MAGNITUDE of the colour and saturation to its
//    DIRECTION, so neither shifts hue. Doing contrast per channel desaturates
//    as a side effect, which is exactly the trap the sky decode fell into.
//  * Exposure divides by an adaptation term rather than multiplying, so the
//    curve sees values in the 1-10 range its constants were chosen for.
//
// tonemapConfig carries the mode and parameters; see frame_run.cc.
// Enhanced Shaders' "Adaptation Day Max/Min" from enbeffect.fx.ini.
static const float kEnbAdaptationMax = 0.13;
static const float kEnbAdaptationMin = 0.08;

float3 enbTonemap(float3 color, float exposureScale, float contrast, float saturation,
                  float curve, float overbrightDampening)
{
    // ENB DIVIDES by an adaptation term of roughly 0.1, which is what puts scene
    // values in the 1-10 range its C=8 knee was chosen for. Multiplying by this
    // engine's exposure scale instead (~0.6) leaves them near 0.2, where the
    // same knee crushes everything to black -- which is exactly what the first
    // version of this function did.
    //
    // ENB's grayadaptation is the ADAPTED SCENE LUMINANCE, not the reciprocal of
    // an exposure multiplier. Auto-exposure solves exposureScale ~ target/avgLum,
    // so recovering the luminance means avgLum ~ target/exposureScale with the
    // usual 0.18 middle grey. Using 1/exposureScale instead inflates the
    // adaptation ~5x, and the extra division is what crushed the shadows to
    // black in the first working version of this.
    const float kMiddleGrey = 0.18;
    const float gray = (exposureScale > 1e-4) ? (kMiddleGrey / exposureScale) : kMiddleGrey;
    const float adaptation = max((gray * kEnbAdaptationMax) + kEnbAdaptationMin, 1e-4);
    color = color / adaptation;
    color += 0.000001;  // ENB does this too: keeps normalize() defined at black

    const float3 direction = normalize(color);
    // color/direction is |color| per channel; ENB writes it this way and the
    // magnitude is what contrast acts on.
    const float3 magnitude = color / direction;
    const float3 shaped = pow(magnitude, contrast) * pow(direction, saturation);

    return (shaped * (1.0 + shaped / max(overbrightDampening, 1e-3))) /
           (shaped + max(curve, 1e-3));
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

    // Behind the subject.
    const float distanceBeyondFocus = max(viewDepth - sharpDistance, 0.0);
    const float farFalloff = smoothstep(0.0, falloffDistance, distanceBeyondFocus);

    // In FRONT of the subject. A lens defocuses on both sides of the focal
    // plane, and leaving this out is very visible on a portrait framing: the
    // background separates but whatever the camera is standing behind -- a
    // fence, a doorway, the ground -- stays razor sharp, which reads as a
    // blurred backdrop rather than as a shallow lens.
    //
    // The near side ramps FASTER than the far side, which is also what real
    // optics do (near depth of field is roughly half the far at portrait
    // distances), and dofConfig2.x is how much faster. It was already uploaded
    // for this and simply never read; 0 disables the near field and restores
    // the far-only behaviour exactly.
    const float nearBlurScale = camera.dofConfig2.x;
    float nearFalloff = 0.0;
    if (nearBlurScale > 0.0001)
    {
        const float distanceBeforeFocus = max(sharpDistance - viewDepth, 0.0);
        nearFalloff =
            smoothstep(0.0, falloffDistance / nearBlurScale, distanceBeforeFocus);
    }

    return max(farFalloff, nearFalloff) * maxRadiusPixels;
}

float3 sampleDepthOfField(
    float2 uv,
    float centerDepth,
    float3 centerColor
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
    const float2 offsets[12] = {
        float2(0.0, 1.0),
        float2(0.866, 0.5),
        float2(0.866, -0.5),
        float2(0.0, -1.0),
        float2(-0.866, -0.5),
        float2(-0.866, 0.5),
        float2(0.35, 0.35),
        float2(0.35, -0.35),
        float2(-0.35, -0.35),
        float2(-0.35, 0.35),
        float2(1.0, 0.0),
        float2(-1.0, 0.0)
    };

    float3 colorSum = centerColor * 2.0;
    float weightSum = 2.0;
    const float depthTolerance = max(camera.dofConfig.z * 0.45, 1.0);
    for (int i = 0; i < 12; ++i)
    {
        const float ringScale = (i < 6) ? 1.0 : 0.52;
        const float2 sampleUv = clamp(
            uv + (offsets[i] * texelSize * centerRadiusPixels * ringScale),
            float2(0.0, 0.0),
            float2(1.0, 1.0)
        );
        const float4 sampleNormalDepth = normalDepthTexture.SampleLevel(sampleUv, 0.0);
        const float sampleDepth = sampleNormalDepth.w;
        const float sampleGeometry = step(0.0001, sampleDepth);
        const float sampleRadiusPixels = depthOfFieldRadiusPixels(sampleDepth);
        const float sameLayer = 1.0 - smoothstep(depthTolerance, depthTolerance * 2.0, abs(sampleDepth - centerDepth));
        const float blurWeight = max(sameLayer, smoothstep(0.0, centerRadiusPixels, sampleRadiusPixels));
        const float weight = lerp(0.35, 1.0, blurWeight) * lerp(0.5, 1.0, sampleGeometry);
        colorSum += hdrSceneColor.SampleLevel(sampleUv, 0.0).xyz * weight;
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
    const float3 bloomMip0 = extractBloom(bloomSourceColor.SampleLevel(uv, 0.0).xyz, bloomThreshold, bloomSoftKnee);
    const float3 bloomMip1 = extractBloom(bloomSourceColor.SampleLevel(uv, 1.0).xyz, bloomThreshold, bloomSoftKnee);
    const float3 bloomMip2 = extractBloom(bloomSourceColor.SampleLevel(uv, 2.0).xyz, bloomThreshold, bloomSoftKnee);
    const float3 bloomMip3 = extractBloom(bloomSourceColor.SampleLevel(uv, 3.0).xyz, bloomThreshold, bloomSoftKnee);
    const float3 bloomMip4 = extractBloom(bloomSourceColor.SampleLevel(uv, 4.0).xyz, bloomThreshold, bloomSoftKnee);
    const float3 bloomMip5 = extractBloom(bloomSourceColor.SampleLevel(uv, 5.0).xyz, bloomThreshold, bloomSoftKnee);

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
    float3 ambientRadiance = lerp(skyRadiance, groundBounce, valleyT * 0.10) * ambientStrength;

    // THE PALETTE ABOVE IS A CLEAR-BLUE-SKY ASSUMPTION. Fog is lit by the sky
    // it sits under, so hardcoding one sky makes the haze disagree with any
    // authored weather: a Mojave dust weather renders an olive sky with
    // blue-grey haze in front of it, which reads as two unrelated atmospheres
    // meeting at the horizon line.
    //
    // A WTHR record already ships the answer -- its Fog channel is the colour
    // the game authored for distance haze under exactly this weather, at
    // exactly this hour -- and it was being uploaded to `weatherFog` and read
    // by nothing at all. Weight 0 (every game but New Vegas) leaves the
    // procedural palette byte-identical.
    const float weatherWeight = saturate(camera.weatherSkyUpper.w);
    if (weatherWeight > 0.0)
    {
        // Fog colour for horizontal and downward rays, which is where geometry
        // is; only a ray heading for the zenith picks up the sky colour, and
        // that one is mostly looking at sky the fog pass never touches.
        const float3 weatherHaze =
            lerp(camera.weatherFog.xyz, camera.weatherSkyUpper.xyz, smoothstep(0.5, 1.0, skyT));
        ambientRadiance = lerp(ambientRadiance, weatherHaze, weatherWeight);
    }

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

    // DebugView in src/render/renderer_types.h. Every mode but LinearDepth is
    // produced by the main pass and has already written a display-ready colour
    // into hdrSceneColor, so the whole exposure/tonemap/grading chain below has
    // to be skipped -- an albedo view shown through ACES and auto-exposure is
    // not albedo. Only the sRGB encode is kept, because the swapchain is UNORM
    // and something has to do it.
    const uint debugView = uint(camera.tonemapConfig2.y + 0.5);
    if (debugView == kDebugViewLinearDepth)
    {
        // Ramped over the far shadow cascade, which is the scene's own sense of
        // "far" rather than an arbitrary constant.
        const float far = max(camera.shadowCascadeSplits.w, 1.0);
        const float t = saturate(viewDepth / far) * geometryMask;
        return float4(pow(float3(t, t, t), float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2)), 1.0);
    }
    if (debugView != kDebugViewOff)
    {
        return float4(
            pow(saturate(hdrColor), float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2)), 1.0);
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

    // SSAO is no longer applied here as a flat multiply on final color (that darkened
    // direct sun along with ambient, muddying sunlit low-poly faces). It's now baked
    // into the ambient term in imported_static.frag.slang's main-pass shading, so
    // hdrSceneColor already carries it for surfaces covered by the normal-depth prepass.
    // `ssao`/`ssaoMode` above remain in use only for the debug visualize branches.
    if (geometryMask > 0.5)
    {
        hdrColor = sampleDepthOfField(input.inUv, viewDepth, hdrColor);
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

    // tonemapConfig.x selects the curve: 0 keeps the ACES fit every other game
    // renders with, so this is inert unless a game asks for the other one.
    float3 toneMapped;
    if (camera.tonemapConfig.x > 0.5)
    {
        toneMapped = enbTonemap(
            hdrColor, exposureScale,
            camera.tonemapConfig.y,    // contrast
            camera.tonemapConfig.z,    // saturation
            camera.tonemapConfig.w,    // curve knee
            camera.tonemapConfig2.x);  // overbright dampening
    }
    else
    {
        toneMapped = acesFilmTonemapWhite(
            hdrColor * exposureScale,
            camera.hdrHighlightConfig.x,
            camera.hdrHighlightConfig.y);
    }
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
