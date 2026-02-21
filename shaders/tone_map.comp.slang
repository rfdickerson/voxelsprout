struct ToneMapPush {
    float exposure;
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

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    if (dispatchThreadID.x >= gPush.width || dispatchThreadID.y >= gPush.height) {
        return;
    }

    uint2 pixel = dispatchThreadID.xy;
    float3 hdr = gAccumulation[pixel].xyz * gPush.exposure;
    float3 ldr = acesApprox(hdr);
    ldr = pow(ldr, 1.0f / 2.2f);

    gToneMapOut[pixel] = float4(ldr, 1.0f);
}
