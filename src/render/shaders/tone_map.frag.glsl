#version 450

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

layout(set = 0, binding = 3) uniform sampler2D hdrSceneColor;
layout(set = 0, binding = 6) uniform sampler2D normalDepthTexture;
layout(set = 0, binding = 7) uniform sampler2D ssaoTexture;

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

vec3 acesFilmTonemap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

vec3 applySaturation(vec3 color, float saturation) {
    const float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(luma), color, saturation);
}

vec3 decodeViewNormal(vec3 encodedNormal) {
    return normalize((encodedNormal * 2.0) - 1.0);
}

void main() {
    vec3 hdrColor = texture(hdrSceneColor, inUv).rgb;
    const vec4 normalDepth = texture(normalDepthTexture, inUv);
    const float viewDepth = normalDepth.a;
    const float geometryMask = step(0.0001, viewDepth);
    const float ssao = clamp(texture(ssaoTexture, inUv).r, 0.0, 1.0);
    const float ssaoMode = camera.shadowVoxelGridSize.w;
    if (ssaoMode > 2.5) {
        const vec3 viewNormal = decodeViewNormal(normalDepth.rgb);
        const vec3 visNormal = (viewNormal * 0.5) + 0.5;
        outColor = vec4(visNormal, 1.0);
        return;
    }
    if (ssaoMode > 1.5) {
        outColor = vec4(vec3(ssao * geometryMask), 1.0);
        return;
    }
    const float ssaoEnable = step(0.5, ssaoMode);
    const float ssaoIntensity = clamp(camera.shadowConfig2.z, 0.0, 2.0);
    const float ssaoContribution = mix(1.0, ssao, ssaoIntensity);
    const float aoFactor = mix(1.0, ssaoContribution, ssaoEnable);
    hdrColor *= mix(1.0, aoFactor, geometryMask);

    const float exposure = 0.58;
    vec3 toneMapped = acesFilmTonemap(hdrColor * exposure);
    toneMapped = applySaturation(toneMapped, 1.18);
    toneMapped = pow(clamp(toneMapped, 0.0, 1.0), vec3(1.06));
    const vec3 ldrSrgb = pow(toneMapped, vec3(1.0 / 2.2));
    outColor = vec4(ldrSrgb, 1.0);
}
