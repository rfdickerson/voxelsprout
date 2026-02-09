#version 450

layout(location = 0) in vec3 inLocalPosition;
layout(location = 1) in vec3 inLocalNormal;
layout(location = 2) in vec4 inOriginLength; // xyz origin, w length
layout(location = 3) in vec4 inAxisRadius;   // xyz axis, w radius
layout(location = 4) in vec4 inTint;
layout(location = 5) in vec4 inExtensions;   // x start extension, y end extension, z tangent scale, w bitangent scale

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 mvp;
    mat4 view;
    mat4 proj;
    mat4 lightViewProj[4];
    vec4 shadowCascadeSplits;
    vec4 shadowAtlasUvRects[4];
    vec4 sunDirectionIntensity;
    vec4 sunColorShadow;
    vec4 shIrradiance[9];
    vec4 shadowConfig0;
    vec4 shadowConfig1;
    vec4 shadowConfig2;
    vec4 shadowConfig3;
    vec4 shadowVoxelGridOrigin;
    vec4 shadowVoxelGridSize;
    vec4 skyConfig0;
    vec4 skyConfig1;
} camera;

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outWorldNormal;
layout(location = 2) out vec3 outTint;
layout(location = 3) out float outVertexAo;
layout(location = 4) out float outLocalAlong;
layout(location = 5) out float outStyle;

void main() {
    vec3 axis = normalize(inAxisRadius.xyz);
    if (dot(axis, axis) <= 0.0001) {
        axis = vec3(0.0, 1.0, 0.0);
    }

    const vec3 fallbackUp = vec3(0.0, 1.0, 0.0);
    const vec3 fallbackRight = vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(axis, fallbackUp));
    if (dot(tangent, tangent) <= 0.0001) {
        tangent = normalize(cross(axis, fallbackRight));
    }
    vec3 bitangent = normalize(cross(axis, tangent));

    const float pipeLength = max(inOriginLength.w, 0.05);
    const float pipeRadius = max(inAxisRadius.w, 0.02);
    const float startExtension = max(inExtensions.x, 0.0);
    const float endExtension = max(inExtensions.y, 0.0);
    const float tangentScale = max(inExtensions.z, 0.01);
    const float bitangentScale = max(inExtensions.w, 0.01);
    const float renderedLength = max(pipeLength + startExtension + endExtension, 0.05);

    const vec3 localPos = inLocalPosition;
    const vec3 localNormal = normalize(inLocalNormal);
    const vec3 voxelCenter = inOriginLength.xyz + vec3(0.5);
    const vec3 segmentStart = voxelCenter - (axis * ((pipeLength * 0.5) + startExtension));
    // Keep cap extrusion world-size stable by only scaling the core [0, 1] span by renderedLength.
    const float alongClamped = clamp(localPos.y, 0.0, 1.0);
    const float alongCapOffset = localPos.y - alongClamped;
    const float worldAlong = (alongClamped * renderedLength) + alongCapOffset;
    const vec3 worldPosition =
        segmentStart +
        (axis * worldAlong) +
        (tangent * (localPos.x * pipeRadius * tangentScale)) +
        (bitangent * (localPos.z * pipeRadius * bitangentScale));
    const vec3 anisotropicLocalNormal = normalize(vec3(
        localNormal.x / tangentScale,
        localNormal.y,
        localNormal.z / bitangentScale
    ));
    const vec3 worldNormal = normalize(
        (axis * anisotropicLocalNormal.y) +
        (tangent * anisotropicLocalNormal.x) +
        (bitangent * anisotropicLocalNormal.z)
    );

    outWorldPosition = worldPosition;
    outWorldNormal = worldNormal;
    outTint = inTint.rgb;
    outLocalAlong = localPos.y;
    outStyle = inTint.w;
    const float sideFactor = 1.0 - abs(localNormal.y);
    const float endFactor = abs((localPos.y * 2.0) - 1.0);
    const float seamOcclusion = sideFactor * endFactor;
    outVertexAo = 1.0 - (0.16 * seamOcclusion);
    gl_Position = camera.mvp * vec4(worldPosition, 1.0);
}
