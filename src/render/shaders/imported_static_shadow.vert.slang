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

static const uint kImportedMaterialFlagSiltStrider = 1u << 1u;
static const uint kSiltStriderMaskShift = 8u;
static const uint kSiltStriderMaskMask = 0xffu;
static const uint kSiltStriderSideFlag = 1u << 16u;

float siltStriderLegInfluence(uint flags)
{
    return float((flags >> kSiltStriderMaskShift) & kSiltStriderMaskMask) / 255.0;
}

float3 animateSiltStriderPosition(float3 position, uint flags)
{
    if ((flags & kImportedMaterialFlagSiltStrider) == 0u) {
        return position;
    }

    const float influence = siltStriderLegInfluence(flags);
    if (influence <= 0.001) {
        return position;
    }

    const float side = ((flags & kSiltStriderSideFlag) != 0u) ? 1.0 : -1.0;
    const float phase = (camera.skyConfig1.z * 1.15) + ((side > 0.0) ? 0.0 : 3.14159265) + (position.z * 0.004);
    const float swing = sin(phase);
    const float lift = max(cos(phase), 0.0);
    const float footInfluence = influence * influence;
    position.x += side * swing * 42.0 * influence;
    position.y += lift * 18.0 * footInfluence;
    position.z += cos(phase * 0.73) * 16.0 * influence;
    return position;
}

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    const int cascadeIndex = clamp(int(chunkPc.cascadeData.x + 0.5), 0, 3);
    const float3 animatedPosition = animateSiltStriderPosition(input.inPosition, input.inFlags);
    output.position = mul(camera.lightViewProj[cascadeIndex], float4(animatedPosition, 1.0));
    output.outUv = input.inUv;
    output.outTextureIndex = input.inTextureIndex;
    output.outFlags = input.inFlags;
    return output;
}
