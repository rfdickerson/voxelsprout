import camera_uniform;
import chunk_push_constants;

struct VertexInput
{
    // Normal and colour are deliberately absent: this pass writes depth and
    // alpha-tests, nothing more. Declaring them forced every pipeline using this
    // shader to describe attributes it never read, and blocked binding the
    // compact 28-byte shadow stream.
    [[vk::location(0)]] float3 inPosition : TEXCOORD0;
    [[vk::location(3)]] float2 inUv : TEXCOORD3;
    [[vk::location(4)]] uint inTextureIndex : TEXCOORD4;
    [[vk::location(5)]] uint inFlags : TEXCOORD5;
    [[vk::location(7)]] uint inLayerWeights : TEXCOORD7;
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
    float3 worldPosition = input.inPosition;
    static const uint kFoliageWindFlag = 1u << 9u;
    if ((input.inFlags & kFoliageWindFlag) != 0u)
    {
        const float bendWeight = float(input.inLayerWeights & 0xffu) / 255.0;
        const float phase = float((input.inLayerWeights >> 8u) & 0xffu) / 255.0;
        const float wave = sin(
            camera.foliageWind.w * 0.82 + phase * 6.2831853 +
            dot(worldPosition.xz, float2(0.0031, 0.0023)));
        worldPosition.xz += camera.foliageWind.xy *
            (camera.foliageWind.z * bendWeight * wave);
    }
    if (chunkPc.rigidAnimationParams.x > 0.5)
    {
        const float4 p = float4(worldPosition, 1.0);
        worldPosition = float3(
            dot(chunkPc.rigidAnimationTransform[0], p),
            dot(chunkPc.rigidAnimationTransform[1], p),
            dot(chunkPc.rigidAnimationTransform[2], p));
    }
    const bool pointShadowFace = chunkPc.cascadeData.y > 0.5;
    if (pointShadowFace)
    {
        const int faceIndex = clamp(int(chunkPc.cascadeData.x + 0.5), 0, 209);
        output.position = mul(
            camera.interiorPointShadowViewProj[faceIndex],
            float4(worldPosition, 1.0));
    }
    else
    {
        const int cascadeIndex = clamp(int(chunkPc.cascadeData.x + 0.5), 0, 3);
        output.position = mul(camera.lightViewProj[cascadeIndex], float4(worldPosition, 1.0));
    }
    output.outUv = input.inUv;
    output.outTextureIndex = input.inTextureIndex;
    output.outFlags = input.inFlags;
    return output;
}
