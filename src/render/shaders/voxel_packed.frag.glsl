#version 450
layout(location = 0) in flat uint inFace;
layout(location = 1) in float inAo;
layout(location = 2) in flat uint inMaterial;
layout(location = 3) in vec3 inWorldPosition;

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 mvp;
    mat4 view;
    mat4 proj;
    mat4 lightViewProj;
    vec4 sunDirectionIntensity;
    vec4 sunColorShadow;
} camera;

layout(set = 0, binding = 1) uniform samplerCube irradianceMap;
layout(set = 0, binding = 4) uniform sampler2DShadow shadowMap;

layout(location = 0) out vec4 outColor;

vec3 faceColor(uint face) {
    if (face == 0u) return vec3(0.90, 0.44, 0.35);
    if (face == 1u) return vec3(0.70, 0.34, 0.28);
    if (face == 2u) return vec3(0.40, 0.85, 0.40);
    if (face == 3u) return vec3(0.28, 0.55, 0.28);
    if (face == 4u) return vec3(0.35, 0.55, 0.90);
    return vec3(0.28, 0.42, 0.70);
}

vec3 materialTint(uint material) {
    if (material == 1u) {
        return vec3(1.0, 1.0, 1.0);
    }
    return vec3(0.9, 0.9, 0.9);
}

vec3 faceNormal(uint face) {
    if (face == 0u) return vec3(1.0, 0.0, 0.0);
    if (face == 1u) return vec3(-1.0, 0.0, 0.0);
    if (face == 2u) return vec3(0.0, 1.0, 0.0);
    if (face == 3u) return vec3(0.0, -1.0, 0.0);
    if (face == 4u) return vec3(0.0, 0.0, 1.0);
    return vec3(0.0, 0.0, -1.0);
}

float sampleShadowPcf(vec3 worldPosition, vec3 normal, float ndotl) {
    const vec4 shadowClip = camera.lightViewProj * vec4(worldPosition, 1.0);
    if (shadowClip.w <= 0.0) {
        return 1.0;
    }

    const vec3 shadowNdc = shadowClip.xyz / shadowClip.w;
    const vec2 shadowUv = (shadowNdc.xy * 0.5) + 0.5;
    const float shadowDepthRef = shadowNdc.z;

    if (
        shadowUv.x <= 0.0 || shadowUv.x >= 1.0 ||
        shadowUv.y <= 0.0 || shadowUv.y >= 1.0 ||
        shadowDepthRef <= 0.0 || shadowDepthRef >= 1.0
    ) {
        return 1.0;
    }

    const vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0));
    const float bias = max(0.0018 * (1.0 - ndotl), 0.00025);

    float visibility = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            const vec2 uv = shadowUv + vec2(float(x), float(y)) * texelSize;
            visibility += texture(shadowMap, vec3(uv, shadowDepthRef - bias));
        }
    }
    return visibility / 9.0;
}

void main() {
    const float exposure = 1.35;
    const float aoBrightness = 0.45 + (clamp(inAo, 0.0, 1.0) * 0.55);
    const vec3 baseColor = faceColor(inFace) * materialTint(inMaterial);
    const vec3 normal = faceNormal(inFace);
    const vec3 diffuseIrradiance = texture(irradianceMap, normal).rgb;

    const vec3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const vec3 sunColor = camera.sunColorShadow.rgb;
    const float shadowStrength = clamp(camera.sunColorShadow.w, 0.0, 1.0);
    const float ndotl = max(dot(normal, -sunDirection), 0.0);
    const float shadowVisibility = sampleShadowPcf(inWorldPosition, normal, ndotl);

    const vec3 ambient = diffuseIrradiance * 0.33;
    const vec3 directSun = sunColor * (sunIntensity * ndotl);
    const float directShadowFactor = mix(1.0, shadowVisibility, shadowStrength);
    const vec3 lighting = ambient + (directSun * directShadowFactor);

    outColor = vec4(baseColor * aoBrightness * lighting * exposure, 1.0);
}
