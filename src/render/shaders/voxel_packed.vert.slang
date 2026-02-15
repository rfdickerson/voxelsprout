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
    const uint ao = (input.inPacked >> kShiftAo) & 0x3u;
    const uint material = (input.inPacked >> kShiftMaterial) & 0xFu;
    const uint baseColorIndex = (input.inPacked >> kShiftBaseColor) & 0xFu;

    const float3 basePosition = float3(float(x), float(y), float(z));
    const float3 worldPosition =
        basePosition +
        cornerOffset(face, corner) +
        input.inChunkOffset.xyz +
        chunkPc.chunkOffset.xyz;

    output.outFace = face;
    output.outMaterial = material;
    output.outWorldPosition = worldPosition;
    output.outVertexAo = lerp(0.45, 1.0, float(ao) / 3.0);
    output.outBaseColorIndex = baseColorIndex;
    output.position = mul(camera.mvp, float4(worldPosition, 1.0));
    return output;
}
