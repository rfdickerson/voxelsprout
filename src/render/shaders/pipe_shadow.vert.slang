#include "camera_uniform.slanginc"
#include "chunk_push_constants.slanginc"

struct VertexInput
{
    [[vk::location(0)]] float3 inLocalPosition : TEXCOORD0;
    [[vk::location(1)]] float3 inLocalNormal : TEXCOORD1;
    [[vk::location(2)]] float4 inOriginLength : TEXCOORD2;
    [[vk::location(3)]] float4 inAxisRadius : TEXCOORD3;
    [[vk::location(4)]] float4 inTint : TEXCOORD4;
    [[vk::location(5)]] float4 inExtensions : TEXCOORD5;
};

float4 main(VertexInput input) : SV_Position
{
    float3 axis = normalize(input.inAxisRadius.xyz);
    if (dot(axis, axis) <= 0.0001)
    {
        axis = float3(0.0, 1.0, 0.0);
    }

    const float3 fallbackUp = float3(0.0, 1.0, 0.0);
    const float3 fallbackRight = float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(axis, fallbackUp));
    if (dot(tangent, tangent) <= 0.0001)
    {
        tangent = normalize(cross(axis, fallbackRight));
    }
    const float3 bitangent = normalize(cross(axis, tangent));

    const float pipeLength = max(input.inOriginLength.w, 0.05);
    const float pipeRadius = max(input.inAxisRadius.w, 0.02);
    const float startExtension = max(input.inExtensions.x, 0.0);
    const float endExtension = max(input.inExtensions.y, 0.0);
    const float tangentScale = max(input.inExtensions.z, 0.01);
    const float bitangentScale = max(input.inExtensions.w, 0.01);
    const float renderedLength = max(pipeLength + startExtension + endExtension, 0.05);
    const float3 voxelCenter = input.inOriginLength.xyz + float3(0.5, 0.5, 0.5);
    const float3 segmentStart = voxelCenter - (axis * ((pipeLength * 0.5) + startExtension));
    const float alongClamped = clamp(input.inLocalPosition.y, 0.0, 1.0);
    const float alongCapOffset = input.inLocalPosition.y - alongClamped;
    const float worldAlong = (alongClamped * renderedLength) + alongCapOffset;
    const float3 worldPosition =
        segmentStart +
        (axis * worldAlong) +
        (tangent * (input.inLocalPosition.x * pipeRadius * tangentScale)) +
        (bitangent * (input.inLocalPosition.z * pipeRadius * bitangentScale));

    const int cascadeIndex = clamp(int(chunkPc.cascadeData.x + 0.5), 0, 3);
    return mul(camera.lightViewProj[cascadeIndex], float4(worldPosition, 1.0));
}
