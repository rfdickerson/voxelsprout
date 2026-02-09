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

    // Keep the sample pattern deterministic (no per-pixel random rotation) so AO
    // remains stable when the camera moves.
    const float radius = 0.55;
    const float bias = 0.04;
    float occlusionWeight = 0.0;
    float totalWeight = 0.0;
    const int kSampleCount = 20;
    const float kGoldenAngle = 2.39996323;

    for (int i = 0; i < kSampleCount; ++i) {
        const float fi = float(i);
        const float t = (fi + 0.5) / float(kSampleCount);
        const float r = sqrt(t);
        const float angle = fi * kGoldenAngle;
        const vec2 disk = vec2(cos(angle), sin(angle)) * r;
        const float z = sqrt(max(1.0 - dot(disk, disk), 0.0));
        const float sampleRadius = radius * mix(0.20, 1.0, t * t);

        const vec3 sampleOffset =
            (tangent * disk.x) +
            (bitangent * disk.y) +
            (centerNormal * abs(z));
        const vec3 sampleViewPos = centerViewPos + (sampleOffset * sampleRadius);

        const vec4 sampleClip = camera.proj * vec4(sampleViewPos, 1.0);
        if (sampleClip.w <= 1e-4) {
            continue;
        }

        const vec3 sampleNdc = sampleClip.xyz / sampleClip.w;
        const vec2 sampleUv = (sampleNdc.xy * 0.5) + 0.5;
        if (sampleUv.x <= 0.0 || sampleUv.x >= 1.0 || sampleUv.y <= 0.0 || sampleUv.y >= 1.0) {
            continue;
        }

        const vec4 sampleData = texture(normalDepthTexture, sampleUv);
        const float sampleDepth = sampleData.a;
        if (sampleDepth <= 0.0001) {
            continue;
        }
        const vec3 sampleNormal = decodeViewNormal(sampleData.rgb);

        const vec3 sampleSceneViewPos = reconstructViewPosition(sampleUv, sampleDepth, invProj);
        const vec3 sceneDelta = sampleSceneViewPos - centerViewPos;
        const float sampleDistance = length(sceneDelta);
        if (sampleDistance <= 1e-4) {
            continue;
        }

        if (sampleDistance > (radius * 1.25)) {
            continue;
        }
        const float rangeWeight = 1.0 - smoothstep(0.0, radius * 1.25, sampleDistance);

        // Project both expected sample position and fetched scene position onto the center normal.
        // The fetched geometry must be in front of the center point, but still before the expected
        // sample point, otherwise this is likely a self-hit or unrelated depth sample.
        const float sampleProj = dot(sampleViewPos - centerViewPos, centerNormal);
        if (sampleProj <= (bias * 2.0)) {
            continue;
        }
        const float sceneProj = dot(sceneDelta, centerNormal);
        const float occlusionEnter = smoothstep(bias, bias + 0.08, sceneProj);
        const float occlusionExit =
            1.0 - smoothstep(sampleProj - 0.08, sampleProj + 0.02, sceneProj);
        const float occlusionAmount = occlusionEnter * occlusionExit;

        const vec3 sceneDir = sceneDelta / sampleDistance;
        const float directionalWeight = max(dot(centerNormal, sceneDir), 0.0);
        const float normalDot = max(dot(centerNormal, sampleNormal), 0.0);
        if (normalDot <= 0.15) {
            continue;
        }
        const float normalWeight = mix(0.35, 1.0, normalDot * normalDot);
        const float sampleWeight = rangeWeight * directionalWeight * normalWeight;
        if (sampleWeight <= 1e-4) {
            continue;
        }

        totalWeight += sampleWeight;
        occlusionWeight += sampleWeight * occlusionAmount;
    }

    if (totalWeight <= 0.0) {
        outAo = 1.0;
        return;
    }

    const float rawAo = clamp(1.0 - (occlusionWeight / totalWeight), 0.0, 1.0);
    const float softenedAo = mix(1.0, rawAo, 0.65);
    outAo = pow(clamp(softenedAo, 0.0, 1.0), 1.10);
}
