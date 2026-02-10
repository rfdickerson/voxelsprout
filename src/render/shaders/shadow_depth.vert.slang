#include "camera_uniform.slanginc"
#include "chunk_push_constants.slanginc"
#include "voxel_decode.slanginc"

struct VertexInput
{
    [[vk::location(0)]] uint inPacked : TEXCOORD0;
    [[vk::location(1)]] float4 inChunkOffset : TEXCOORD1;
};

float4 main(VertexInput input) : SV_Position
{
    const uint x = (input.inPacked >> kShiftX) & 0x1Fu;
    const uint y = (input.inPacked >> kShiftY) & 0x1Fu;
    const uint z = (input.inPacked >> kShiftZ) & 0x1Fu;
    const uint face = (input.inPacked >> kShiftFace) & 0x7u;
    const uint corner = (input.inPacked >> kShiftCorner) & 0x3u;

    const float3 basePosition = float3(float(x), float(y), float(z));
    const float3 worldPosition =
        basePosition +
        cornerOffset(face, corner) +
        input.inChunkOffset.xyz +
        chunkPc.chunkOffset.xyz;
    const int cascadeIndex = clamp(int(chunkPc.cascadeData.x + 0.5), 0, 3);
    return mul(camera.lightViewProj[cascadeIndex], float4(worldPosition, 1.0));
}
