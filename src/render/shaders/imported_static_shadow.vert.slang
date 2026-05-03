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
    [[vk::location(6)]] uint4 inBoneIndices : TEXCOORD6;
    [[vk::location(7)]] float4 inBoneWeights : TEXCOORD7;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float2 outUv : TEXCOORD0;
    nointerpolation uint outTextureIndex : TEXCOORD1;
    nointerpolation uint outFlags : TEXCOORD2;
};

struct ActorBonePaletteMatrix
{
    float4 row0;
    float4 row1;
    float4 row2;
};

[[vk::binding(13, 0)]]
StructuredBuffer<ActorBonePaletteMatrix> actorBonePalette;

static const uint kImportedMaterialFlagSiltStrider = 1u << 1u;
static const uint kSiltStriderMaskShift = 8u;
static const uint kSiltStriderMaskMask = 0xffu;
static const uint kSiltStriderSideFlag = 1u << 16u;
static const uint kImportedMaterialFlagNpcGpuTransform = 1u << 24u;
static const uint kImportedMaterialFlagNpcNoDeform = 1u << 25u;
static const uint kImportedMaterialFlagNpcGpuSkinned = 1u << 26u;

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

float npcWalkPhase(float3 localPosition, float2 worldOffset)
{
    return (camera.skyConfig1.z * 5.4) + dot(worldOffset, float2(0.0017, 0.0023)) + (localPosition.y * 0.012);
}

float3 animateNpcLocalPosition(float3 position, uint flags, float phase)
{
    if ((flags & kImportedMaterialFlagNpcGpuTransform) == 0u ||
        (flags & kImportedMaterialFlagNpcNoDeform) != 0u) {
        return position;
    }

    if (abs(phase) <= 0.001) {
        const float idlePhase = camera.skyConfig1.z * 2.1 + position.y * 0.015;
        const float torsoWeight = saturate((position.y - 54.0) / 42.0) *
            saturate((142.0 - position.y) / 58.0);
        position.y += sin(idlePhase) * 0.8 * torsoWeight;
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

float3 skinNpcLocalPosition(float3 position, uint flags, uint4 boneIndices, float4 boneWeights)
{
    if ((flags & kImportedMaterialFlagNpcGpuSkinned) == 0u) {
        return position;
    }
    if (chunkPc.actorData.y == 0u) {
        return position;
    }
    const float weightSum = boneWeights.x + boneWeights.y + boneWeights.z + boneWeights.w;
    if (weightSum <= 0.001) {
        return position;
    }
    const float4 localPosition = float4(position, 1.0);
    float3 skinned = float3(0.0);
    const uint paletteBase = chunkPc.actorData.x;
    ActorBonePaletteMatrix matrix = actorBonePalette[paletteBase + boneIndices.x];
    skinned += float3(dot(matrix.row0, localPosition), dot(matrix.row1, localPosition), dot(matrix.row2, localPosition)) *
        boneWeights.x;
    matrix = actorBonePalette[paletteBase + boneIndices.y];
    skinned += float3(dot(matrix.row0, localPosition), dot(matrix.row1, localPosition), dot(matrix.row2, localPosition)) *
        boneWeights.y;
    matrix = actorBonePalette[paletteBase + boneIndices.z];
    skinned += float3(dot(matrix.row0, localPosition), dot(matrix.row1, localPosition), dot(matrix.row2, localPosition)) *
        boneWeights.z;
    matrix = actorBonePalette[paletteBase + boneIndices.w];
    skinned += float3(dot(matrix.row0, localPosition), dot(matrix.row1, localPosition), dot(matrix.row2, localPosition)) *
        boneWeights.w;
    return skinned / weightSum;
}

float3 transformNpcPosition(float3 position, uint flags, uint4 boneIndices, float4 boneWeights)
{
    if ((flags & kImportedMaterialFlagNpcGpuTransform) == 0u) {
        return position;
    }
    const float3 worldOffset = chunkPc.chunkOffset.xyz;
    const float yaw = chunkPc.chunkOffset.w;
    float3 local = skinNpcLocalPosition(position, flags, boneIndices, boneWeights);
    if ((flags & kImportedMaterialFlagNpcGpuSkinned) == 0u) {
        local = animateNpcLocalPosition(local, flags, 0.0);
    }
    const float3 rotated = rotateNpcVector(local, yaw);
    return worldOffset + rotated;
}

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    const int cascadeIndex = clamp(int(chunkPc.cascadeData.x + 0.5), 0, 3);
    float3 animatedPosition =
        transformNpcPosition(input.inPosition, input.inFlags, input.inBoneIndices, input.inBoneWeights);
    animatedPosition = animateSiltStriderPosition(animatedPosition, input.inFlags);
    output.position = mul(camera.lightViewProj[cascadeIndex], float4(animatedPosition, 1.0));
    output.outUv = input.inUv;
    output.outTextureIndex = input.inTextureIndex;
    output.outFlags = input.inFlags;
    return output;
}
