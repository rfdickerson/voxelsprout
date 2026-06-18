struct UiPushConstants
{
    float2 invScreenSize;  // (2/width, 2/height): maps pixel space to NDC.
    float2 pad;
};

[[vk::push_constant]]
ConstantBuffer<UiPushConstants> uiPc;

struct VertexInput
{
    [[vk::location(0)]] float2 inPositionPx : TEXCOORD0;
    [[vk::location(1)]] float2 inUv : TEXCOORD1;
    [[vk::location(2)]] float4 inColor : TEXCOORD2;  // R8G8B8A8_UNORM -> 0..1 straight alpha.
    [[vk::location(3)]] uint inMode : TEXCOORD3;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float4 outColor : TEXCOORD0;
    float2 outUv : TEXCOORD1;
    nointerpolation uint outMode : TEXCOORD2;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    // Pixel space (top-left origin, +Y down) -> Vulkan NDC (+Y down), no flip.
    const float2 ndc = (input.inPositionPx * uiPc.invScreenSize) - float2(1.0, 1.0);
    output.position = float4(ndc, 0.0, 1.0);
    output.outColor = input.inColor;
    output.outUv = input.inUv;
    output.outMode = input.inMode;
    return output;
}
