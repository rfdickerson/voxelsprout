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

layout(location = 0) out vec4 outColor;

void main() {
    const vec3 normal = normalize(inWorldNormal);
    const vec3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const vec3 sunColor = camera.sunColorShadow.rgb;
    const float ndotl = max(dot(normal, -sunDirection), 0.0);
    const float hemi = clamp((normal.y * 0.5) + 0.5, 0.0, 1.0);
    const vec3 ambient = mix(vec3(0.05, 0.06, 0.07), vec3(0.22, 0.24, 0.26), hemi);
    const vec3 lit = (ambient + (sunColor * ndotl * 0.9)) * inTint;
    outColor = vec4(lit, 1.0);
}
