import camera_uniform;

struct VertexInput
{
    [[vk::location(0)]] float3 inPosition : TEXCOORD0;
    [[vk::location(1)]] float3 inColor : TEXCOORD1;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 outColor : TEXCOORD0;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    output.position = mul(camera.mvp, float4(input.inPosition, 1.0));
    output.outColor = input.inColor;
    return output;
}
