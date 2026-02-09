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

layout(location = 0) in vec2 inUv;
layout(location = 0) out float outAo;

float hash12(vec2 p) {
    const vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    const vec3 q3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((q3.x + q3.y) * q3.z);
}

vec3 decodeViewNormal(vec3 encodedNormal) {
    return normalize((encodedNormal * 2.0) - 1.0);
}

vec3 reconstructViewPosition(vec2 uv, float viewDepth, mat4 invProj) {
    const float a = camera.proj[2][2];
    const float b = camera.proj[3][2];
    const float ndcZ = -a + (b / max(viewDepth, 1e-4));
    const vec2 ndcXY = (uv * 2.0) - 1.0;
    vec4 view = invProj * vec4(ndcXY, ndcZ, 1.0);
    if (abs(view.w) > 1e-4) {
        view.xyz /= view.w;
    }
    return view.xyz;
}

void main() {
    const vec4 centerSample = texture(normalDepthTexture, inUv);
    const float centerDepth = centerSample.a;
    if (centerDepth <= 0.0001) {
        outAo = 1.0;
        return;
    }

    const vec3 centerNormal = decodeViewNormal(centerSample.rgb);
    const mat4 invProj = inverse(camera.proj);
    const vec3 centerViewPos = reconstructViewPosition(inUv, centerDepth, invProj);

    vec3 tangent = normalize(cross(centerNormal, vec3(0.0, 1.0, 0.0)));
    if (dot(tangent, tangent) <= 0.0001) {
        tangent = normalize(cross(centerNormal, vec3(1.0, 0.0, 0.0)));
    }
    const vec3 bitangent = normalize(cross(centerNormal, tangent));

    const vec2 noiseScale = vec2(textureSize(normalDepthTexture, 0));
    const float randomAngle = hash12(inUv * noiseScale) * 6.2831853;
    const float s = sin(randomAngle);
    const float c = cos(randomAngle);
    const mat2 rot = mat2(c, -s, s, c);

    const float radius = clamp(camera.shadowConfig2.x, 0.15, 3.0);
    const float bias = clamp(camera.shadowConfig2.y, 0.0, 0.20);
    float occlusion = 0.0;
    float sampleCount = 0.0;
    const int kSampleCount = 32;
    const float kGoldenAngle = 2.39996323;

    for (int i = 0; i < kSampleCount; ++i) {
        const float fi = float(i);
        const float t = (fi + 0.5) / float(kSampleCount);
        const float r = sqrt(t);
        const float angle = (fi * kGoldenAngle) + randomAngle;
        const vec2 disk = rot * vec2(cos(angle), sin(angle)) * r;
        const float z = sqrt(max(1.0 - dot(disk, disk), 0.0));
        vec3 k = vec3(disk, z);
        const float scale = mix(0.15, 1.0, t * t);
        k *= scale;

        const vec3 sampleOffset =
            (tangent * k.x) +
            (bitangent * k.y) +
            (centerNormal * abs(k.z));
        const vec3 sampleViewPos = centerViewPos + (sampleOffset * radius);

        const vec4 sampleClip = camera.proj * vec4(sampleViewPos, 1.0);
        if (sampleClip.w <= 1e-4) {
            continue;
        }

        const vec3 sampleNdc = sampleClip.xyz / sampleClip.w;
        const vec2 sampleUv = (sampleNdc.xy * 0.5) + 0.5;
        if (sampleUv.x <= 0.0 || sampleUv.x >= 1.0 || sampleUv.y <= 0.0 || sampleUv.y >= 1.0) {
            continue;
        }

        const float sampleDepth = texture(normalDepthTexture, sampleUv).a;
        if (sampleDepth <= 0.0001) {
            continue;
        }

        const vec3 sampleSceneViewPos = reconstructViewPosition(sampleUv, sampleDepth, invProj);
        const float depthDelta = abs(centerViewPos.z - sampleSceneViewPos.z);
        const float rangeWeight = smoothstep(0.0, 1.0, radius / max(depthDelta, 1e-4));
        const float occluded = (sampleSceneViewPos.z > (sampleViewPos.z + bias)) ? 1.0 : 0.0;

        occlusion += occluded * rangeWeight;
        sampleCount += 1.0;
    }

    if (sampleCount <= 0.0) {
        outAo = 1.0;
        return;
    }

    const float rawAo = 1.0 - (occlusion / sampleCount);
    outAo = pow(clamp(rawAo, 0.0, 1.0), 1.4);
}
