import camera_uniform;

struct VertexInput
{
    [[vk::location(0)]] float3 inPosition : TEXCOORD0;
    [[vk::location(1)]] float2 inUv : TEXCOORD1;
    [[vk::location(2)]] uint inNormalTextureSlot : TEXCOORD2;
    [[vk::location(3)]] uint inFlowTextureSlot : TEXCOORD3;
    [[vk::location(4)]] uint2 inExtraNormals : TEXCOORD4;
    [[vk::location(5)]] float4 inshallow : TEXCOORD5;
    [[vk::location(6)]] float4 indeep : TEXCOORD6;
    [[vk::location(7)]] float4 inreflection : TEXCOORD7;
    [[vk::location(8)]] float4 inoptics : TEXCOORD8;
    [[vk::location(9)]] float4 inlayer0 : TEXCOORD9;
    [[vk::location(10)]] float4 inlayer1 : TEXCOORD10;
    [[vk::location(11)]] float4 inlayer2 : TEXCOORD11;
    [[vk::location(12)]] float4 inlighting : TEXCOORD12;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 outWorldPosition : TEXCOORD0;
    float2 outUv : TEXCOORD1;
    nointerpolation uint outNormalTextureSlot : TEXCOORD2;
    nointerpolation uint outFlowTextureSlot : TEXCOORD3;
    nointerpolation uint2 outExtraNormals : TEXCOORD4;
    nointerpolation float4 outshallow : TEXCOORD5;
    nointerpolation float4 outdeep : TEXCOORD6;
    nointerpolation float4 outreflection : TEXCOORD7;
    nointerpolation float4 outoptics : TEXCOORD8;
    nointerpolation float4 outlayer0 : TEXCOORD9;
    nointerpolation float4 outlayer1 : TEXCOORD10;
    nointerpolation float4 outlayer2 : TEXCOORD11;
    nointerpolation float4 outlighting : TEXCOORD12;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    output.outWorldPosition = input.inPosition;
    output.outUv = input.inUv;
    output.outNormalTextureSlot = input.inNormalTextureSlot;
    output.outFlowTextureSlot = input.inFlowTextureSlot;
    output.outExtraNormals = input.inExtraNormals;
    output.outshallow = input.inshallow;
    output.outdeep = input.indeep;
    output.outreflection = input.inreflection;
    output.outoptics = input.inoptics;
    output.outlayer0 = input.inlayer0;
    output.outlayer1 = input.inlayer1;
    output.outlayer2 = input.inlayer2;
    output.outlighting = input.inlighting;
    output.position = mul(camera.mvp, float4(input.inPosition, 1.0));
    return output;
}
