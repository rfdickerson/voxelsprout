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

static const uint kImportedMaterialFlagSiltStrider = 1u << 1u;
static const uint kSiltStriderMaskShift = 8u;
static const uint kSiltStriderMaskMask = 0xffu;
static const uint kSiltStriderSideFlag = 1u << 16u;
static const uint kNpcYawShift = 17u;
static const uint kNpcYawMask = 0x7fu;
static const uint kImportedMaterialFlagNpcGpuTransform = 1u << 24u;

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

float3 rotateNpcVector(float3 value, float yaw)
{
    const float c = cos(yaw);
    const float s = sin(yaw);
    return float3((value.x * c) + (value.z * s), value.y, (-value.x * s) + (value.z * c));
}

float npcYaw(uint flags)
{
    return (float((flags >> kNpcYawShift) & kNpcYawMask) + 0.5) * (6.2831853 / 128.0);
}

float npcWalkPhase(float3 localPosition, float2 worldOffset)
{
    return (camera.skyConfig1.z * 5.4) + dot(worldOffset, float2(0.0017, 0.0023)) + (localPosition.y * 0.012);
}

float3 animateNpcLocalPosition(float3 position, uint flags, float phase)
{
    if ((flags & kImportedMaterialFlagNpcGpuTransform) == 0u) {
        return position;
    }

    const float side = position.x < 0.0 ? -1.0 : 1.0;
    if (position.y < 82.0) {
        const float legWeight = saturate((82.0 - position.y) / 82.0) *
            saturate((abs(position.x) - 2.0) / 18.0);
        const float legSwing = sin(phase + ((side > 0.0) ? 3.14159265 : 0.0)) * 9.5 * legWeight;
        position.z += legSwing;
        position.y += max(cos(phase + ((side > 0.0) ? 3.14159265 : 0.0)), 0.0) * 2.2 * legWeight;
    }
    if (position.y > 42.0 && position.y < 128.0) {
        const float armWeight = saturate((abs(position.x) - 18.0) / 18.0) *
            saturate((128.0 - position.y) / 58.0);
        position.z += sin(phase + ((side > 0.0) ? 0.0 : 3.14159265)) * 7.0 * armWeight;
    }
    return position;
}

float3 transformNpcPosition(float3 position, float3 instanceData, uint flags)
{
    if ((flags & kImportedMaterialFlagNpcGpuTransform) == 0u) {
        return position;
    }
    const float3 worldOffset = instanceData;
    const float yaw = npcYaw(flags);
    float3 local = animateNpcLocalPosition(position, flags, npcWalkPhase(position, worldOffset.xz));
    const float3 rotated = rotateNpcVector(local, yaw);
    return worldOffset + rotated;
}

float3 transformNpcNormal(float3 normal, uint flags)
{
    if ((flags & kImportedMaterialFlagNpcGpuTransform) == 0u) {
        return normal;
    }
    return rotateNpcVector(normal, npcYaw(flags));
}

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    float3 animatedPosition = transformNpcPosition(input.inPosition, input.inColor, input.inFlags);
    animatedPosition = animateSiltStriderPosition(animatedPosition, input.inFlags);
    output.outWorldPosition = animatedPosition;
    output.outWorldNormal = normalize(transformNpcNormal(input.inNormal, input.inFlags));
    output.outColor = ((input.inFlags & kImportedMaterialFlagNpcGpuTransform) != 0u)
        ? float3(0.78, 0.72, 0.62)
        : input.inColor;
    output.outUv = input.inUv;
    output.outTextureIndex = input.inTextureIndex;
    output.outFlags = input.inFlags;
    output.position = mul(camera.mvp, float4(animatedPosition, 1.0));
    return output;
}
