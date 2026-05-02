import camera_uniform;
import chunk_push_constants;
import voxel_decode;

struct VertexInput
{
    [[vk::location(0)]] uint inPacked : TEXCOORD0;
    [[vk::location(1)]] float4 inChunkOffset : TEXCOORD1;
};

struct VertexOutput
{
    float4 position : SV_Position;
    nointerpolation uint outFace : TEXCOORD0;
    nointerpolation uint outMaterial : TEXCOORD1;
    float3 outWorldPosition : TEXCOORD2;
    float outVertexAo : TEXCOORD3;
    nointerpolation uint outBaseColorIndex : TEXCOORD4;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;

    const uint x = (input.inPacked >> kShiftX) & 0x1Fu;
    const uint y = (input.inPacked >> kShiftY) & 0x1Fu;
    const uint z = (input.inPacked >> kShiftZ) & 0x1Fu;
    const uint face = (input.inPacked >> kShiftFace) & 0x7u;
    const uint corner = (input.inPacked >> kShiftCorner) & 0x3u;
    const uint ao = (input.inPacked >> kShiftAo) & 0xFu;
    const uint material = (input.inPacked >> kShiftMaterial) & 0x7u;
    const uint baseColorIndex = (input.inPacked >> kShiftBaseColor) & 0x7u;
    const float3 basePosition = float3(float(x), float(y), float(z));
    const float3 geometricWorldPosition =
        basePosition +
        cornerOffset(face, corner) +
        input.inChunkOffset.xyz +
        chunkPc.chunkOffset.xyz;
    const float3 worldPosition = displacedVoxelWorldPosition(material, face, geometricWorldPosition);

    output.outFace = face;
    output.outMaterial = material;
    output.outWorldPosition = worldPosition;
    const float ao01 = saturate(float(ao) / 15.0);
    const float shapedAo = sqrt(ao01);
    output.outVertexAo = lerp(0.52, 1.0, shapedAo);
    output.outBaseColorIndex = baseColorIndex;
    output.position = mul(camera.mvp, float4(worldPosition, 1.0));
    return output;
}
