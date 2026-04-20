import camera_uniform;
import chunk_push_constants;

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
    float2 outUv : TEXCOORD0;
    nointerpolation uint outTextureIndex : TEXCOORD1;
    nointerpolation uint outFlags : TEXCOORD2;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    const int cascadeIndex = clamp(int(chunkPc.cascadeData.x + 0.5), 0, 3);
    output.position = mul(camera.lightViewProj[cascadeIndex], float4(input.inPosition, 1.0));
    output.outUv = input.inUv;
    output.outTextureIndex = input.inTextureIndex;
    output.outFlags = input.inFlags;
    return output;
}
