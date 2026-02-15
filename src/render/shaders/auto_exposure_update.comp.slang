// Compile for Vulkan SPIR-V with column-major matrices:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/auto_exposure_update.comp.slang -o src/render/shaders/auto_exposure_update.comp.slang.spv

struct AutoExposureUpdatePushConstants
{
    uint totalPixels;
    uint binCount;
    uint resetHistory;
    uint _pad0;
    float minLogLuminance;
    float maxLogLuminance;
    float lowPercentile;
    float highPercentile;
    float keyValue;
    float minExposure;
    float maxExposure;
    float adaptUpRate;
    float adaptDownRate;
    float deltaTimeSeconds;
    float _pad1;
    float _pad2;
};

[[vk::push_constant]]
ConstantBuffer<AutoExposureUpdatePushConstants> exposurePc;

[[vk::binding(1, 0)]]
StructuredBuffer<uint> histogramBins;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float4> exposureState;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x != 0u || dispatchThreadId.y != 0u || dispatchThreadId.z != 0u)
    {
        return;
    }

    const uint binCount = min(max(exposurePc.binCount, 1u), 64u);
    float totalSamples = 0.0;
    for (uint binIndex = 0u; binIndex < binCount; ++binIndex)
    {
        totalSamples += float(histogramBins[binIndex]);
    }

    const float minLogLuminance = exposurePc.minLogLuminance;
    const float logLuminanceRange = max(exposurePc.maxLogLuminance - minLogLuminance, 1e-4);
    const float lowPercentile = clamp(exposurePc.lowPercentile, 0.0, 0.98);
    const float highPercentile = clamp(exposurePc.highPercentile, lowPercentile + 0.01, 1.0);

    float averageLuminance = 1.0;
    if (totalSamples > 0.5)
    {
        const float lowSample = lowPercentile * totalSamples;
        const float highSample = max(highPercentile * totalSamples, lowSample + 1.0);
        float runningCount = 0.0;
        float weightedLogLuminance = 0.0;
        float weightedSamples = 0.0;

        for (uint binIndex = 0u; binIndex < binCount; ++binIndex)
        {
            const float binCountF = float(histogramBins[binIndex]);
            const float binStart = runningCount;
            const float binEnd = runningCount + binCountF;
            const float segmentStart = max(binStart, lowSample);
            const float segmentEnd = min(binEnd, highSample);
            const float segmentWeight = max(segmentEnd - segmentStart, 0.0);
            if (segmentWeight > 0.0)
            {
                const float binT = (float(binIndex) + 0.5) / max(float(binCount - 1u), 1.0);
                const float binLogLuminance = minLogLuminance + (logLuminanceRange * binT);
                weightedLogLuminance += binLogLuminance * segmentWeight;
                weightedSamples += segmentWeight;
            }
            runningCount = binEnd;
        }

        const float averageLogLuminance =
            (weightedSamples > 0.5) ? (weightedLogLuminance / weightedSamples) : (minLogLuminance + (0.5 * logLuminanceRange));
        averageLuminance = max(exp2(averageLogLuminance), 1e-6);
    }

    const float keyValue = max(exposurePc.keyValue, 1e-4);
    const float minExposure = max(exposurePc.minExposure, 1e-4);
    const float maxExposure = max(exposurePc.maxExposure, minExposure);
    const float targetExposure = clamp(keyValue / averageLuminance, minExposure, maxExposure);

    float previousExposure = max(exposureState[0].x, 1e-4);
    if (exposurePc.resetHistory != 0u)
    {
        previousExposure = targetExposure;
    }
    const float adaptUpRate = max(exposurePc.adaptUpRate, 0.05);
    const float adaptDownRate = max(exposurePc.adaptDownRate, 0.05);
    const float adaptRate = (targetExposure > previousExposure) ? adaptUpRate : adaptDownRate;
    const float deltaTime = max(exposurePc.deltaTimeSeconds, 0.0);
    const float blendAlpha = saturate(1.0 - exp(-deltaTime * adaptRate));
    const float exposure = lerp(previousExposure, targetExposure, blendAlpha);
    exposureState[0] = float4(exposure, targetExposure, averageLuminance, 0.0);
}
