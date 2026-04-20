import camera_uniform;

struct VertexInput
{
    [[vk::location(0)]] float3 inPosition : TEXCOORD0;
    [[vk::location(1)]] float3 inNormal : TEXCOORD1;
    [[vk::location(2)]] float3 inColor : TEXCOORD2;
    [[vk::location(3)]] float2 inUv : TEXCOORD3;
    [[vk::location(4)]] uint inTextureIndex : TEXCOORD4;
    [[vk::location(5)]] uint inFlags : TEXCOORD5;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 outWorldPosition : TEXCOORD0;
    float3 outWorldNormal : TEXCOORD1;
    float3 outColor : TEXCOORD2;
    float2 outUv : TEXCOORD3;
    nointerpolation uint outTextureIndex : TEXCOORD4;
    nointerpolation uint outFlags : TEXCOORD5;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    output.outWorldPosition = input.inPosition;
    output.outWorldNormal = normalize(input.inNormal);
    output.outColor = input.inColor;
    output.outUv = input.inUv;
    output.outTextureIndex = input.inTextureIndex;
    output.outFlags = input.inFlags;
    output.position = mul(camera.mvp, float4(input.inPosition, 1.0));
    return output;
}
