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
    float3 outLocalDirection : TEXCOORD0;
    float3 outColor : TEXCOORD1;
    float2 outUv : TEXCOORD2;
    nointerpolation uint outTextureIndex : TEXCOORD3;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;

    const float time = camera.skyConfig1.z;
    const float rotation = time * 0.0045;
    const float s = sin(rotation);
    const float c = cos(rotation);

    const float3 localPosition = float3(
        (input.inPosition.x * c) - (input.inPosition.z * s),
        input.inPosition.y,
        (input.inPosition.x * s) + (input.inPosition.z * c));
    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float3 worldPosition = cameraWorld + localPosition;

    float4 clipPosition = mul(camera.proj, mul(camera.view, float4(worldPosition, 1.0)));
    clipPosition.z = 0.0;

    output.position = clipPosition;
    output.outLocalDirection = normalize(localPosition);
    output.outColor = input.inColor;
    output.outUv = input.inUv + float2(time * 0.000035, 0.0);
    output.outTextureIndex = input.inTextureIndex;
    return output;
}
