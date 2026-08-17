// Half-resolution planar reflection reconstruction and temporal resolve.
// The raw reflection is stable world geometry; water-normal distortion is
// applied later by imported_water.frag, so accumulating here removes raster
// shimmer without dragging the animated surface behind the camera.

[[vk::binding(0, 0)]]
Sampler2D<float4> currentReflection;
[[vk::binding(1, 0)]]
Sampler2D<float> currentDepth;
[[vk::binding(2, 0)]]
Sampler2D<float4> historyReflection;
[[vk::binding(3, 0)]]
Sampler2D<float> historyDepth;
[[vk::binding(4, 0)]]
RWTexture2D<float4> outputReflection;
[[vk::binding(5, 0)]]
RWTexture2D<float> outputDepth;

struct WaterReflectionResolveUniform
{
    float4x4 invViewProj;
    float4x4 invView;
    float4x4 view;
    float4x4 prevView;
    float4x4 prevViewProj;
    float4 temporalParams;
    float4 jitterParams;
    float4 extentParams;
    float4 projectionParams;
};

[[vk::binding(6, 0)]]
ConstantBuffer<WaterReflectionResolveUniform> resolve;

struct PushConstants
{
    uint width;
    uint height;
    float pad0;
    float pad1;
};

[[vk::push_constant]]
PushConstants pushConstants;

float3 rgbToYCoCg(float3 c)
{
    return float3(
        (0.25 * c.r) + (0.5 * c.g) + (0.25 * c.b),
        (0.5 * c.r) - (0.5 * c.b),
        (-0.25 * c.r) + (0.5 * c.g) - (0.25 * c.b));
}

float3 yCoCgToRgb(float3 c)
{
    const float t = c.x - c.z;
    return float3(t + c.y, c.x + c.z, t - c.y);
}

float3 reconstructWorldFromDeviceDepth(float2 uv, float deviceDepth)
{
    const float2 ndc = ((uv * 2.0) - 1.0) - resolve.jitterParams.xy;
    float4 world = mul(resolve.invViewProj, float4(ndc, deviceDepth, 1.0));
    return world.xyz / max(abs(world.w), 1e-7) * (world.w < 0.0 ? -1.0 : 1.0);
}

float linearDepthAt(float2 uv, float deviceDepth)
{
    if (deviceDepth <= 1e-7)
    {
        return 0.0;
    }
    const float3 world = reconstructWorldFromDeviceDepth(uv, deviceDepth);
    return max(-mul(resolve.view, float4(world, 1.0)).z, 0.0);
}

struct CurrentSample
{
    float3 color;
    float linearDepth;
    float3 momentMean;
    float3 momentSigma;
    float3 neighborhoodMin;
    float3 neighborhoodMax;
};

CurrentSample reconstructCurrent(float2 uv)
{
    const float2 rawExtent = max(resolve.extentParams.xy, float2(1.0, 1.0));
    const float2 rawTexel = 1.0 / rawExtent;
    const float guideDeviceDepth = currentDepth.SampleLevel(uv, 0.0);
    const bool guideSky = guideDeviceDepth <= 1e-7;
    const float guideLinearDepth = linearDepthAt(uv, guideDeviceDepth);

    float3 colorSum = float3(0.0, 0.0, 0.0);
    float depthSum = 0.0;
    float weightSum = 0.0;
    float3 momentSum = float3(0.0, 0.0, 0.0);
    float3 momentSquaredSum = float3(0.0, 0.0, 0.0);
    float3 neighborhoodMin = float3(1e30, 1e30, 1e30);
    float3 neighborhoodMax = float3(-1e30, -1e30, -1e30);
    float momentWeight = 0.0;

    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            const float2 tapUv = clamp(
                uv + (float2(float(x), float(y)) * rawTexel),
                rawTexel * 0.5,
                1.0 - (rawTexel * 0.5));
            const float tapDeviceDepth = currentDepth.SampleLevel(tapUv, 0.0);
            const bool tapSky = tapDeviceDepth <= 1e-7;
            if (tapSky != guideSky)
            {
                continue;
            }
            const float tapLinearDepth = linearDepthAt(tapUv, tapDeviceDepth);
            const float spatialWeight = (x == 0 ? 1.0 : 0.55) * (y == 0 ? 1.0 : 0.55);
            float depthWeight = 1.0;
            if (!guideSky)
            {
                const float depthScale = max(2.0, guideLinearDepth * 0.005);
                depthWeight = exp(-abs(tapLinearDepth - guideLinearDepth) / depthScale);
            }
            const float weight = spatialWeight * depthWeight;
            const float3 tapColor = currentReflection.SampleLevel(tapUv, 0.0).rgb;
            colorSum += tapColor * weight;
            depthSum += tapLinearDepth * weight;
            weightSum += weight;

            const float3 ycocg = rgbToYCoCg(tapColor);
            momentSum += ycocg * spatialWeight;
            momentSquaredSum += ycocg * ycocg * spatialWeight;
            neighborhoodMin = min(neighborhoodMin, ycocg);
            neighborhoodMax = max(neighborhoodMax, ycocg);
            momentWeight += spatialWeight;
        }
    }

    CurrentSample result;
    const float safeWeight = max(weightSum, 1e-5);
    result.color = colorSum / safeWeight;
    result.linearDepth = depthSum / safeWeight;
    const float safeMomentWeight = max(momentWeight, 1e-5);
    result.momentMean = momentSum / safeMomentWeight;
    result.momentSigma = sqrt(max(
        float3(0.0, 0.0, 0.0),
        (momentSquaredSum / safeMomentWeight) - (result.momentMean * result.momentMean)));
    result.neighborhoodMin = neighborhoodMin;
    result.neighborhoodMax = neighborhoodMax;
    return result;
}

float2 reprojectSky(float2 uv)
{
    float projX = resolve.projectionParams.x;
    float projY = resolve.projectionParams.y;
    if (abs(projX) < 1e-6) projX = (projX < 0.0) ? -1e-6 : 1e-6;
    if (abs(projY) < 1e-6) projY = (projY < 0.0) ? -1e-6 : 1e-6;
    const float2 ndc = ((uv * 2.0) - 1.0) - resolve.jitterParams.xy;
    const float3 viewDirection = normalize(float3(ndc.x / projX, ndc.y / projY, -1.0));
    const float3 worldDirection = normalize(mul((float3x3)resolve.invView, viewDirection));
    const float4 previousClip = mul(resolve.prevViewProj, float4(worldDirection, 0.0));
    if (previousClip.w <= 1e-6)
    {
        return float2(-1.0, -1.0);
    }
    const float2 previousNdc = (previousClip.xy / previousClip.w) - resolve.jitterParams.zw;
    return (previousNdc * 0.5) + 0.5;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= pushConstants.width || dispatchThreadId.y >= pushConstants.height)
    {
        return;
    }
    const float2 outputExtent = float2(float(pushConstants.width), float(pushConstants.height));
    const float2 uv = (float2(dispatchThreadId.xy) + 0.5) / outputExtent;
    const CurrentSample current = reconstructCurrent(uv);
    const bool currentSky = current.linearDepth <= 1e-5;
    float3 result = current.color;
    float confidence = 0.0;

    if (resolve.temporalParams.z > 0.5)
    {
        float2 previousUv = float2(-1.0, -1.0);
        float expectedPreviousDepth = 0.0;
        if (currentSky)
        {
            previousUv = reprojectSky(uv);
        }
        else
        {
            const float2 ndc = ((uv * 2.0) - 1.0) - resolve.jitterParams.xy;
            float projX = resolve.projectionParams.x;
            float projY = resolve.projectionParams.y;
            if (abs(projX) < 1e-6) projX = (projX < 0.0) ? -1e-6 : 1e-6;
            if (abs(projY) < 1e-6) projY = (projY < 0.0) ? -1e-6 : 1e-6;
            const float3 viewPosition = float3(
                (ndc.x * current.linearDepth) / projX,
                (ndc.y * current.linearDepth) / projY,
                -current.linearDepth);
            const float3 worldPosition = mul(resolve.invView, float4(viewPosition, 1.0)).xyz;
            const float4 previousClip = mul(resolve.prevViewProj, float4(worldPosition, 1.0));
            if (previousClip.w > 1e-6)
            {
                const float2 previousNdc = (previousClip.xy / previousClip.w) - resolve.jitterParams.zw;
                previousUv = (previousNdc * 0.5) + 0.5;
                expectedPreviousDepth = max(
                    -mul(resolve.prevView, float4(worldPosition, 1.0)).z,
                    0.0);
            }
        }

        if (all(previousUv >= float2(0.0, 0.0)) && all(previousUv <= float2(1.0, 1.0)))
        {
            const float previousDepth = historyDepth.SampleLevel(previousUv, 0.0);
            const bool previousSky = previousDepth <= 1e-5;
            bool accept = previousSky == currentSky;
            if (accept && !currentSky)
            {
                const float threshold = max(2.0, expectedPreviousDepth * 0.005);
                accept = abs(previousDepth - expectedPreviousDepth) <= threshold;
            }
            if (accept)
            {
                float3 history = rgbToYCoCg(
                    historyReflection.SampleLevel(previousUv, 0.0).rgb);
                const float3 clipMin = max(
                    current.momentMean - (1.25 * current.momentSigma),
                    current.neighborhoodMin);
                const float3 clipMax = min(
                    current.momentMean + (1.25 * current.momentSigma),
                    current.neighborhoodMax);
                history = min(max(history, clipMin), clipMax);
                const float historyWeight = currentSky
                    ? resolve.temporalParams.y
                    : resolve.temporalParams.x;
                result = lerp(current.color, yCoCgToRgb(history), historyWeight);
                confidence = historyWeight;
            }
        }
    }

    outputReflection[dispatchThreadId.xy] = float4(result, confidence);
    outputDepth[dispatchThreadId.xy] = current.linearDepth;
}
