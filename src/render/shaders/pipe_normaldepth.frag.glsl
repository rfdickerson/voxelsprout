#version 450

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec3 inTint;

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

layout(location = 0) out vec4 outNormalDepth;

void main() {
    const vec3 worldNormal = normalize(inWorldNormal);
    const vec3 viewNormal = normalize((camera.view * vec4(worldNormal, 0.0)).xyz);
    const float viewDepth = max(-(camera.view * vec4(inWorldPosition, 1.0)).z, 0.0);

    const vec3 encodedNormal = (viewNormal * 0.5) + 0.5;
    outNormalDepth = vec4(encodedNormal, viewDepth);
}
