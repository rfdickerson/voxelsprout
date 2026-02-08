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

layout(set = 0, binding = 6) uniform sampler2D normalDepthTexture;
layout(set = 0, binding = 8) uniform sampler2D ssaoRawTexture;

layout(location = 0) in vec2 inUv;
layout(location = 0) out float outAo;

vec3 decodeViewNormal(vec3 encodedNormal) {
    return normalize((encodedNormal * 2.0) - 1.0);
}

void main() {
    const vec2 texelSize = 1.0 / vec2(textureSize(ssaoRawTexture, 0));

    const vec4 centerNd = texture(normalDepthTexture, inUv);
    const vec3 centerNormal = decodeViewNormal(centerNd.rgb);
    const float centerDepth = centerNd.a;
    if (centerDepth <= 0.0001) {
        outAo = 1.0;
        return;
    }
    const float centerAo = texture(ssaoRawTexture, inUv).r;

    float weightedAo = centerAo;
    float weightSum = 1.0;
    const float sigma = 3.0;
    const int blurRadius = 6;

    for (int i = 1; i <= blurRadius; ++i) {
        const float fi = float(i);
        const float spatialWeight = exp(-(fi * fi) / (2.0 * sigma * sigma));

        const vec2 offsetX = vec2(texelSize.x * fi, 0.0);
        const vec2 offsetY = vec2(0.0, texelSize.y * fi);
        const vec2 offsetD1 = vec2(texelSize.x * fi, texelSize.y * fi);
        const vec2 offsetD2 = vec2(texelSize.x * fi, -texelSize.y * fi);

        const vec2 sampleUvs[8] = vec2[](
            inUv + offsetX,
            inUv - offsetX,
            inUv + offsetY,
            inUv - offsetY,
            inUv + offsetD1,
            inUv - offsetD1,
            inUv + offsetD2,
            inUv - offsetD2
        );
        for (int j = 0; j < 8; ++j) {
            const vec2 suv = sampleUvs[j];
            if (suv.x <= 0.0 || suv.x >= 1.0 || suv.y <= 0.0 || suv.y >= 1.0) {
                continue;
            }

            const vec4 sampleNd = texture(normalDepthTexture, suv);
            const vec3 sampleNormal = decodeViewNormal(sampleNd.rgb);
            const float sampleDepth = sampleNd.a;
            if (sampleDepth <= 0.0001) {
                continue;
            }
            const float sampleAo = texture(ssaoRawTexture, suv).r;

            const float normalWeight = pow(max(dot(centerNormal, sampleNormal), 0.0), 8.0);
            const float depthWeight = exp(-abs(sampleDepth - centerDepth) * 1.5);
            const float weight = spatialWeight * normalWeight * depthWeight;

            weightedAo += sampleAo * weight;
            weightSum += weight;
        }
    }

    outAo = clamp(weightedAo / max(weightSum, 1e-4), 0.0, 1.0);
}
