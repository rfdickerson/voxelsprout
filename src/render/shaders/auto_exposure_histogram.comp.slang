// Compile for Vulkan SPIR-V with column-major matrices:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/auto_exposure_histogram.comp.slang -o src/render/shaders/auto_exposure_histogram.comp.slang.spv

struct AutoExposureHistogramPushConstants
{
    uint width;
    uint height;
    uint totalPixels;
    uint binCount;
    float minLogLuminance;
    float maxLogLuminance;
    float sourceMipLevel;
    float _pad1;
};

[[vk::push_constant]]
ConstantBuffer<AutoExposureHistogramPushConstants> histogramPc;

[[vk::binding(0, 0)]]
Sampler2D<float4> hdrSceneColor;

[[vk::binding(1, 0)]]
RWStructuredBuffer<uint> histogramBins;

float luminance709(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint width = max(histogramPc.width, 1u);
    const uint height = max(histogramPc.height, 1u);
    if (dispatchThreadId.x >= width || dispatchThreadId.y >= height)
    {
        return;
    }

    const float2 uv = (float2(dispatchThreadId.xy) + 0.5) / float2(width, height);
    const float3 hdrColor = max(
        hdrSceneColor.SampleLevel(uv, histogramPc.sourceMipLevel).xyz,
        float3(0.0, 0.0, 0.0)
    );
    const float luminance = max(luminance709(hdrColor), 1e-6);
    const float logLuminance = log2(luminance);

    const float minLogLuminance = histogramPc.minLogLuminance;
    const float logLuminanceRange = max(histogramPc.maxLogLuminance - minLogLuminance, 1e-4);
    const float normalized = saturate((logLuminance - minLogLuminance) / logLuminanceRange);

    const uint binCount = max(histogramPc.binCount, 1u);
    const uint maxBinIndex = binCount - 1u;
    const uint binIndex = min(maxBinIndex, (uint)floor(normalized * float(maxBinIndex) + 0.5));
    InterlockedAdd(histogramBins[binIndex], 1u);
}
