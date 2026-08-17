import camera_uniform;

struct VertexInput
{
    [[vk::location(0)]] float3 inPosition : TEXCOORD0;
    [[vk::location(1)]] float2 inUv : TEXCOORD1;
    [[vk::location(2)]] uint inNormalTextureSlot : TEXCOORD2;
    [[vk::location(3)]] uint inFlowTextureSlot : TEXCOORD3;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 outWorldPosition : TEXCOORD0;
    float2 outUv : TEXCOORD1;
    nointerpolation uint outNormalTextureSlot : TEXCOORD2;
    nointerpolation uint outFlowTextureSlot : TEXCOORD3;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    output.outWorldPosition = input.inPosition;
    output.outUv = input.inUv;
    output.outNormalTextureSlot = input.inNormalTextureSlot;
    output.outFlowTextureSlot = input.inFlowTextureSlot;
    output.position = mul(camera.mvp, float4(input.inPosition, 1.0));
    return output;
}
