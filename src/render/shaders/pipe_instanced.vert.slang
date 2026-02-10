#include "camera_uniform.slanginc"

struct VertexInput
{
    [[vk::location(0)]] float3 inLocalPosition : TEXCOORD0;
    [[vk::location(1)]] float3 inLocalNormal : TEXCOORD1;
    [[vk::location(2)]] float4 inOriginLength : TEXCOORD2;
    [[vk::location(3)]] float4 inAxisRadius : TEXCOORD3;
    [[vk::location(4)]] float4 inTint : TEXCOORD4;
    [[vk::location(5)]] float4 inExtensions : TEXCOORD5;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 outWorldPosition : TEXCOORD0;
    float3 outWorldNormal : TEXCOORD1;
    float3 outTint : TEXCOORD2;
    float outVertexAo : TEXCOORD3;
    float outLocalAlong : TEXCOORD4;
    float outStyle : TEXCOORD5;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;

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

    const float3 localPos = input.inLocalPosition;
    const float3 localNormal = normalize(input.inLocalNormal);
    const float3 voxelCenter = input.inOriginLength.xyz + float3(0.5, 0.5, 0.5);
    const float3 segmentStart = voxelCenter - (axis * ((pipeLength * 0.5) + startExtension));

    const float alongClamped = clamp(localPos.y, 0.0, 1.0);
    const float alongCapOffset = localPos.y - alongClamped;
    const float worldAlong = (alongClamped * renderedLength) + alongCapOffset;

    const float3 worldPosition =
        segmentStart +
        (axis * worldAlong) +
        (tangent * (localPos.x * pipeRadius * tangentScale)) +
        (bitangent * (localPos.z * pipeRadius * bitangentScale));

    const float3 anisotropicLocalNormal = normalize(float3(
        localNormal.x / tangentScale,
        localNormal.y,
        localNormal.z / bitangentScale
    ));
    const float3 worldNormal = normalize(
        (axis * anisotropicLocalNormal.y) +
        (tangent * anisotropicLocalNormal.x) +
        (bitangent * anisotropicLocalNormal.z)
    );

    output.outWorldPosition = worldPosition;
    output.outWorldNormal = worldNormal;
    output.outTint = input.inTint.xyz;
    output.outLocalAlong = localPos.y;
    output.outStyle = input.inTint.w;

    const float sideFactor = 1.0 - abs(localNormal.y);
    const float endFactor = abs((localPos.y * 2.0) - 1.0);
    const float seamOcclusion = sideFactor * endFactor;
    output.outVertexAo = 1.0 - (0.16 * seamOcclusion);
    output.position = mul(camera.mvp, float4(worldPosition, 1.0));
    return output;
}
