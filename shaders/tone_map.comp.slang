struct ToneMapPush {
    float exposure;
    float whitePoint;
    float shoulder;
    float gamma;
    float contrast;
    float saturation;
    uint tonemapOperator;
    uint width;
    uint height;
    uint sampleCount;
};

[[vk::push_constant]]
ConstantBuffer<ToneMapPush> gPush;

[[vk::binding(0, 0)]]
RWTexture2D<float4> gAccumulation;
[[vk::binding(1, 0)]]
RWTexture2D<float4> gToneMapOut;

float3 acesApprox(float3 color) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

float3 applySaturation(float3 color, float saturation) {
    const float3 lumaWeights = float3(0.2126f, 0.7152f, 0.0722f);
    float luma = dot(color, lumaWeights);
    return lerp(float3(luma, luma, luma), color, saturation);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    if (dispatchThreadID.x >= gPush.width || dispatchThreadID.y >= gPush.height) {
        return;
    }

    uint2 pixel = dispatchThreadID.xy;
    float3 hdr = gAccumulation[pixel].xyz * gPush.exposure;

    float3 ldr = hdr;
    if (gPush.tonemapOperator == 1u) {
        ldr = hdr / (1.0f + hdr);
    } else if (gPush.tonemapOperator == 2u) {
        ldr = acesApprox(hdr);
    } else {
        ldr = saturate(hdr);
    }

    ldr = applySaturation(ldr, gPush.saturation);
    ldr = saturate(ldr);
    ldr = pow(max(ldr, 0.0f), 1.0f / max(gPush.gamma, 1e-4f));

    gToneMapOut[pixel] = float4(ldr, 1.0f);
}
