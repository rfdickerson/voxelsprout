#version 450
layout(location = 0) in flat uint inFace;
layout(location = 1) in float inAo;
layout(location = 2) in flat uint inMaterial;
layout(location = 3) in vec3 inWorldPosition;

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 mvp;
    mat4 view;
    mat4 proj;
    mat4 lightViewProj[4];
    vec4 shadowCascadeSplits;
    vec4 sunDirectionIntensity;
    vec4 sunColorShadow;
    vec4 shIrradiance[9];
} camera;

layout(set = 0, binding = 4) uniform sampler2DArrayShadow shadowMap;

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
    if (material == 250u) {
        return vec3(0.30, 0.95, 1.00);
    }
    if (material == 251u) {
        return vec3(1.00, 0.28, 0.22);
    }
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

vec3 evaluateShIrradiance(vec3 normal) {
    const float x = normal.x;
    const float y = normal.y;
    const float z = normal.z;

    float basis[9];
    basis[0] = 0.282095;
    basis[1] = 0.488603 * y;
    basis[2] = 0.488603 * z;
    basis[3] = 0.488603 * x;
    basis[4] = 1.092548 * x * y;
    basis[5] = 1.092548 * y * z;
    basis[6] = 0.315392 * ((3.0 * z * z) - 1.0);
    basis[7] = 1.092548 * x * z;
    basis[8] = 0.546274 * ((x * x) - (y * y));

    vec3 irradiance = vec3(0.0);
    for (int i = 0; i < 9; ++i) {
        irradiance += camera.shIrradiance[i].rgb * basis[i];
    }
    return max(irradiance, vec3(0.0));
}

int chooseShadowCascade(float viewDepth) {
    if (viewDepth <= camera.shadowCascadeSplits.x) {
        return 0;
    }
    if (viewDepth <= camera.shadowCascadeSplits.y) {
        return 1;
    }
    if (viewDepth <= camera.shadowCascadeSplits.z) {
        return 2;
    }
    return 3;
}

float sampleShadowPcf(vec3 worldPosition, int cascadeIndex, float ndotl) {
    const vec4 shadowClip = camera.lightViewProj[cascadeIndex] * vec4(worldPosition, 1.0);
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

    const vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0).xy);
    const float texelScale = max(texelSize.x, texelSize.y);
    const float bias = max(0.38 * texelScale, 2.4 * texelScale * (1.0 - ndotl));

    float visibility = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            const vec2 uv = shadowUv + vec2(float(x), float(y)) * texelSize;
            visibility += texture(shadowMap, vec4(uv, float(cascadeIndex), shadowDepthRef - bias));
        }
    }
    return visibility / 9.0;
}

void main() {
    const vec3 normal = faceNormal(inFace);
    const float aoBrightness = 0.42 + (clamp(inAo, 0.0, 1.0) * 0.58);
    vec3 baseColor = faceColor(inFace) * materialTint(inMaterial);

    const vec3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const vec3 sunColor = camera.sunColorShadow.rgb;
    const float shadowStrength = clamp(camera.sunColorShadow.w, 0.0, 1.0);

    const float ndotl = max(dot(normal, -sunDirection), 0.0);
    const float viewDepth = max(-(camera.view * vec4(inWorldPosition, 1.0)).z, 0.0);
    const int cascadeIndex = chooseShadowCascade(viewDepth);
    const float shadowVisibility = sampleShadowPcf(inWorldPosition, cascadeIndex, ndotl);

    const vec3 ambientIrradiance = evaluateShIrradiance(normal);
    const vec3 ambient = ambientIrradiance * 0.17;
    const vec3 directSun = sunColor * (sunIntensity * ndotl);
    const float directShadowFactor = mix(1.0, shadowVisibility, shadowStrength);
    vec3 lighting = ambient + (directSun * directShadowFactor);

    if (inMaterial == 250u || inMaterial == 251u) {
        const float stripe = 0.5 + 0.5 * sin((inWorldPosition.x + inWorldPosition.z + inWorldPosition.y) * 6.0);
        const vec3 emissive = materialTint(inMaterial) * mix(0.35, 0.85, stripe);
        baseColor = materialTint(inMaterial);
        lighting += emissive;
    }

    outColor = vec4(baseColor * aoBrightness * lighting, 1.0);
}
