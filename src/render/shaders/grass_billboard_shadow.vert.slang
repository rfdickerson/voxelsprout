import camera_uniform;
import chunk_push_constants;

struct VertexInput
{
    [[vk::location(0)]] float2 inCorner : TEXCOORD0;
    [[vk::location(1)]] float2 inUv : TEXCOORD1;
    [[vk::location(2)]] float inPlane : TEXCOORD2;
    [[vk::location(3)]] float4 inWorldPosYaw : TEXCOORD3;
    [[vk::location(4)]] float4 inColorTint : TEXCOORD4;
};

static const uint kPoppySpriteTileRed = 5u;
static const uint kPoppySpriteTileOrangeRed = 6u;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 outUv : TEXCOORD0;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;

    const float3 basePosition = input.inWorldPosYaw.xyz;
    const float yawRadians = input.inWorldPosYaw.w;
    const uint spriteTile = uint(clamp(input.inColorTint.w, 0.0, 255.0) + 0.5);
    const bool isRedFlower = (spriteTile == kPoppySpriteTileRed) || (spriteTile == kPoppySpriteTileOrangeRed);
    const float scale = isRedFlower ? 0.46 : 1.0;
    const float planeSelect = step(0.5, input.inPlane);
    const float3 baseRight = normalize(lerp(float3(1.0, 0.0, 0.0), float3(0.0, 0.0, 1.0), planeSelect));
    const float s = sin(yawRadians);
    const float c = cos(yawRadians);
    const float3 right = normalize(float3(
        (baseRight.x * c) - (baseRight.z * s),
        0.0,
        (baseRight.x * s) + (baseRight.z * c)
    ));
    const float3 up = float3(0.0, 1.0, 0.0);

    const float3 worldPosition =
        basePosition +
        (right * (input.inCorner.x * scale)) +
        (up * (input.inCorner.y * scale));

    const int cascadeIndex = clamp(int(chunkPc.cascadeData.x + 0.5), 0, 3);
    output.position = mul(camera.lightViewProj[cascadeIndex], float4(worldPosition, 1.0));
    output.outUv = input.inUv;
    return output;
}
