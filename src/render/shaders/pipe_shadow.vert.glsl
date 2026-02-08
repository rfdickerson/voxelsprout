#version 450

layout(location = 0) in vec3 inLocalPosition;
layout(location = 1) in vec3 inLocalNormal;
layout(location = 2) in vec4 inOriginLength; // xyz origin, w length
layout(location = 3) in vec4 inAxisRadius;   // xyz axis, w radius
layout(location = 4) in vec4 inTint;

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 mvp;
    mat4 view;
    mat4 proj;
    mat4 lightViewProj[4];
    vec4 shadowCascadeSplits;
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

layout(push_constant) uniform ChunkPushConstants {
    vec4 chunkOffset;
    vec4 cascadeData;
} chunkPc;

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
    const vec3 voxelCenter = inOriginLength.xyz + vec3(0.5);
    const vec3 segmentStart = voxelCenter - (axis * (pipeLength * 0.5));
    const vec3 worldPosition =
        segmentStart +
        (axis * (inLocalPosition.y * pipeLength)) +
        (tangent * (inLocalPosition.x * pipeRadius)) +
        (bitangent * (inLocalPosition.z * pipeRadius));

    const int cascadeIndex = clamp(int(chunkPc.cascadeData.x + 0.5), 0, 3);
    gl_Position = camera.lightViewProj[cascadeIndex] * vec4(worldPosition, 1.0);
}
