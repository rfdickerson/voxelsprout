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
    vec4 shadowConfig0; // normalOffsetNear, normalOffsetFar, baseBiasNearTexel, baseBiasFarTexel
    vec4 shadowConfig1; // slopeBiasNearTexel, slopeBiasFarTexel, blendMin, blendFactor
    vec4 shadowConfig2; // enableRpdb, enableRotatedPoisson, enableHybridNearVoxelRay, poissonSampleCount
    vec4 shadowConfig3; // hybridStep, hybridMaxDistance, rpdbScale, pcfRadius
    vec4 shadowVoxelGridOrigin;
    vec4 shadowVoxelGridSize;
} camera;

layout(set = 0, binding = 4) uniform sampler2DArrayShadow shadowMap;
layout(set = 0, binding = 5) uniform usamplerBuffer shadowVoxelGrid;

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

float hash11(float n) {
    return fract(sin(n) * 43758.5453123);
}

float hash31(vec3 p) {
    return hash11(dot(p, vec3(127.1, 311.7, 74.7)));
}

bool sampleVoxelSolid(ivec3 worldCell) {
    const ivec3 gridOrigin = ivec3(
        int(camera.shadowVoxelGridOrigin.x + 0.5),
        int(camera.shadowVoxelGridOrigin.y + 0.5),
        int(camera.shadowVoxelGridOrigin.z + 0.5)
    );
    const ivec3 gridSize = ivec3(
        int(camera.shadowVoxelGridSize.x + 0.5),
        int(camera.shadowVoxelGridSize.y + 0.5),
        int(camera.shadowVoxelGridSize.z + 0.5)
    );

    if (gridSize.x <= 0 || gridSize.y <= 0 || gridSize.z <= 0) {
        return false;
    }

    const ivec3 local = worldCell - gridOrigin;
    if (
        local.x < 0 || local.x >= gridSize.x ||
        local.y < 0 || local.y >= gridSize.y ||
        local.z < 0 || local.z >= gridSize.z
    ) {
        return false;
    }

    const int linear = local.x + (gridSize.x * (local.z + (gridSize.z * local.y)));
    return texelFetch(shadowVoxelGrid, linear).r > 0u;
}

float sampleVoxelRayShadow(vec3 worldPosition, vec3 normal, float ndotl) {
    if (camera.shadowConfig2.z < 0.5 || ndotl <= 0.0) {
        return 1.0;
    }

    const vec3 rayDir = normalize(-camera.sunDirectionIntensity.xyz);
    const float rayStep = max(camera.shadowConfig3.x, 0.08);
    const float maxDistance = max(camera.shadowConfig3.y, rayStep);
    const float jitter = hash31(worldPosition * 0.73) * rayStep;
    float t = max(0.2, rayStep * 0.8) + jitter;
    const vec3 startPosition = worldPosition + normal * 0.14;

    for (int i = 0; i < 128 && t <= maxDistance; ++i) {
        const vec3 samplePos = startPosition + rayDir * t;
        const ivec3 cell = ivec3(floor(samplePos));
        if (sampleVoxelSolid(cell)) {
            return 0.0;
        }
        t += rayStep;
    }

    return 1.0;
}

float sampleShadowPcf(vec3 worldPosition, vec3 normal, int cascadeIndex, float ndotl) {
    const float cascadeT = float(cascadeIndex) / 3.0;
    const float normalOffset = mix(camera.shadowConfig0.x, camera.shadowConfig0.y, cascadeT) * (1.0 + (1.0 - ndotl));
    const vec3 shadowSamplePosition = worldPosition + (normal * normalOffset);
    const vec4 shadowClip = camera.lightViewProj[cascadeIndex] * vec4(shadowSamplePosition, 1.0);
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

    float bias =
        mix(camera.shadowConfig0.z, camera.shadowConfig0.w, cascadeT) * texelScale +
        mix(camera.shadowConfig1.x, camera.shadowConfig1.y, cascadeT) * texelScale * (1.0 - ndotl);

    if (camera.shadowConfig2.x > 0.5) {
        const float rpdb = max(abs(dFdx(shadowNdc.z)), abs(dFdy(shadowNdc.z)));
        bias += rpdb * camera.shadowConfig3.z;
    }

    const int kernelRadius = clamp(int(camera.shadowConfig3.w + 0.5), 1, 3);
    float visibility = 0.0;
    float weightSum = 0.0;

    if (camera.shadowConfig2.y > 0.5) {
        const vec2 poisson[16] = vec2[](
            vec2(-0.94201624, -0.39906216),
            vec2(0.94558609, -0.76890725),
            vec2(-0.09418410, -0.92938870),
            vec2(0.34495938, 0.29387760),
            vec2(-0.91588581, 0.45771432),
            vec2(-0.81544232, -0.87912464),
            vec2(-0.38277543, 0.27676845),
            vec2(0.97484398, 0.75648379),
            vec2(0.44323325, -0.97511554),
            vec2(0.53742981, -0.47373420),
            vec2(-0.26496911, -0.41893023),
            vec2(0.79197514, 0.19090188),
            vec2(-0.24188840, 0.99706507),
            vec2(-0.81409955, 0.91437590),
            vec2(0.19984126, 0.78641367),
            vec2(0.14383161, -0.14100790)
        );
        const float angle = hash31(worldPosition) * 6.2831853;
        const mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
        const int sampleCount = clamp(int(camera.shadowConfig2.w + 0.5), 4, 16);
        for (int i = 0; i < sampleCount; ++i) {
            const vec2 offset = rot * poisson[i] * float(kernelRadius);
            const vec2 uv = shadowUv + (offset * texelSize);
            visibility += texture(shadowMap, vec4(uv, float(cascadeIndex), shadowDepthRef + bias));
            weightSum += 1.0;
        }
    } else {
        for (int y = -3; y <= 3; ++y) {
            for (int x = -3; x <= 3; ++x) {
                if (abs(x) > kernelRadius || abs(y) > kernelRadius) {
                    continue;
                }
                const vec2 uv = shadowUv + vec2(float(x), float(y)) * texelSize;
                const float wx = float(kernelRadius + 1 - abs(x));
                const float wy = float(kernelRadius + 1 - abs(y));
                const float weight = wx * wy;
                visibility += texture(shadowMap, vec4(uv, float(cascadeIndex), shadowDepthRef + bias)) * weight;
                weightSum += weight;
            }
        }
    }

    return visibility / max(weightSum, 0.0001);
}

float sampleCascadedShadow(vec3 worldPosition, vec3 normal, float viewDepth, float ndotl) {
    const int primaryCascade = chooseShadowCascade(viewDepth);
    float shadow = sampleShadowPcf(worldPosition, normal, primaryCascade, ndotl);

    if (primaryCascade == 0 && camera.shadowConfig2.z > 0.5) {
        shadow *= sampleVoxelRayShadow(worldPosition, normal, ndotl);
    }

    if (primaryCascade >= 3) {
        return shadow;
    }

    const float prevSplit = (primaryCascade == 0) ? 0.0 : camera.shadowCascadeSplits[primaryCascade - 1];
    const float split = camera.shadowCascadeSplits[primaryCascade];
    const float blendRange = max(camera.shadowConfig1.z, (split - prevSplit) * camera.shadowConfig1.w);
    const float blendT = smoothstep(split - blendRange, split, viewDepth);
    if (blendT <= 0.0) {
        return shadow;
    }

    const float nextShadow = sampleShadowPcf(worldPosition, normal, primaryCascade + 1, ndotl);
    return mix(shadow, nextShadow, blendT);
}

void main() {
    const vec3 normal = faceNormal(inFace);
    const float ao = clamp(inAo, 0.0, 1.0);
    const float aoCurve = pow(ao, 1.45);
    const float ambientAo = mix(0.16, 1.0, aoCurve);
    const float directAo = mix(0.34, 1.0, aoCurve);
    vec3 baseColor = faceColor(inFace) * materialTint(inMaterial);

    const vec3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const vec3 sunColor = camera.sunColorShadow.rgb;
    const float shadowStrength = clamp(camera.sunColorShadow.w, 0.0, 1.0);

    const float ndotl = max(dot(normal, -sunDirection), 0.0);
    const float viewDepth = max(-(camera.view * vec4(inWorldPosition, 1.0)).z, 0.0);
    const float shadowVisibility = sampleCascadedShadow(inWorldPosition, normal, viewDepth, ndotl);

    const vec3 ambientIrradiance = evaluateShIrradiance(normal);
    const vec3 ambient = ambientIrradiance * 0.075;
    const vec3 directSun = sunColor * (sunIntensity * ndotl);
    const float directShadowFactor = mix(1.0, shadowVisibility, shadowStrength);
    const float ambientShadowFactor = mix(1.0, shadowVisibility, 0.25 * shadowStrength);
    vec3 lighting =
        (ambient * ambientShadowFactor * ambientAo) +
        (directSun * directShadowFactor * directAo);

    if (inMaterial == 250u || inMaterial == 251u) {
        const float stripe = 0.5 + 0.5 * sin((inWorldPosition.x + inWorldPosition.z + inWorldPosition.y) * 6.0);
        const vec3 emissive = materialTint(inMaterial) * mix(0.35, 0.85, stripe);
        baseColor = materialTint(inMaterial);
        lighting += emissive;
    }

    outColor = vec4(baseColor * lighting, 1.0);
}
