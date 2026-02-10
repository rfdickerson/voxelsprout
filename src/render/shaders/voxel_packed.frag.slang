// Compile for Vulkan SPIR-V with column-major matrices to match Renderer.cpp uniform layout:
// slangc -target spirv -entry main -stage fragment -matrix-layout-column-major src/render/shaders/voxel_packed.frag.slang -o src/render/shaders/voxel_packed.frag.slang.spv

import camera_uniform;

[[vk::binding(4, 0)]]
Sampler2DShadow shadowMap;

[[vk::binding(1, 0)]]
Sampler2D<float4> diffuseAlbedo;

float3 materialTint(uint material)
{
    if (material == 250u) return float3(0.30, 0.95, 1.00);
    if (material == 251u) return float3(1.00, 0.28, 0.22);
    if (material == 1u) return float3(1.0, 1.0, 1.0);
    return float3(0.9, 0.9, 0.9);
}

float2 faceUv(uint face, float3 worldPosition)
{
    float2 uv;
    if (face == 0u || face == 1u) uv = worldPosition.zy;
    else if (face == 2u || face == 3u) uv = worldPosition.xz;
    else uv = worldPosition.xy;
    return frac(uv);
}

float3 faceNormal(uint face)
{
    if (face == 0u) return float3(1.0, 0.0, 0.0);
    if (face == 1u) return float3(-1.0, 0.0, 0.0);
    if (face == 2u) return float3(0.0, 1.0, 0.0);
    if (face == 3u) return float3(0.0, -1.0, 0.0);
    if (face == 4u) return float3(0.0, 0.0, 1.0);
    return float3(0.0, 0.0, -1.0);
}

float3 proceduralSkyNoSunDisk(float3 direction)
{
    const float3 dir = normalize(direction);
    const float rayleighStrength = max(camera.skyConfig0.x, 0.01);
    const float mieStrength = max(camera.skyConfig0.y, 0.01);
    const float skyExposure = max(camera.skyConfig0.w, 0.01);

    const float horizonT = clamp((dir.y * 0.5) + 0.5, 0.0, 1.0);
    const float skyT = pow(horizonT, 0.35);
    const float3 horizonColor =
        (float3(0.55, 0.70, 1.00) * rayleighStrength) +
        (float3(1.00, 0.72, 0.42) * (mieStrength * 0.55));
    const float3 zenithColor =
        (float3(0.06, 0.24, 0.54) * rayleighStrength) +
        (float3(0.22, 0.20, 0.15) * (mieStrength * 0.25));
    const float3 sky = lerp(horizonColor, zenithColor, skyT);

    const float3 groundColor = float3(0.05, 0.06, 0.07);
    const float belowHorizon = clamp(-dir.y, 0.0, 1.0);
    const float3 horizonGroundColor = horizonColor * 0.32;
    const float3 ground = lerp(horizonGroundColor, groundColor, pow(belowHorizon, 0.55));
    const float skyWeight = smoothstep(-0.18, 0.02, dir.y);
    const float3 color = lerp(ground, sky, skyWeight);
    return max(color * skyExposure, float3(0.0, 0.0, 0.0));
}

float3 applyAtmosphericFog(float3 litColor, float3 worldPosition, float viewDepth, float3 sunDirection)
{
    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float3 viewDir = normalize(worldPosition - cameraWorld);
    const float3 skyColor = proceduralSkyNoSunDisk(viewDir);

    const float3 sigmaRayleigh = float3(0.0012, 0.0023, 0.0048) * max(camera.skyConfig0.x, 0.01);
    const float sigmaMie = 0.0038 * max(camera.skyConfig0.y, 0.01);
    const float3 extinction = sigmaRayleigh + float3(sigmaMie);

    const float sunView = max(dot(viewDir, -sunDirection), 0.0);
    const float g = clamp(camera.skyConfig0.z, 0.0, 0.98);
    const float phaseMie = (1.0 - (g * g)) / (4.0 * 3.14159265 * pow(1.0 + (g * g) - (2.0 * g * sunView), 1.5));
    const float phaseBoost = 1.0 + (phaseMie * 1.6);
    const float3 inscatterColor = skyColor * phaseBoost;

    const float3 transmittance = exp(-extinction * max(viewDepth, 0.0));
    return (litColor * transmittance) + (inscatterColor * (float3(1.0, 1.0, 1.0) - transmittance));
}

float3 evaluateShIrradiance(float3 normal)
{
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

    float3 irradiance = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < 9; ++i)
    {
        irradiance += camera.shIrradiance[i].xyz * basis[i];
    }
    return max(irradiance, float3(0.0, 0.0, 0.0));
}

float3 evaluateShHemisphereIrradiance(float3 normal)
{
    const float upT = clamp((normal.y * 0.5) + 0.5, 0.0, 1.0);
    const float3 skyIrradiance = evaluateShIrradiance(float3(0.0, 1.0, 0.0));
    const float3 groundIrradiance = evaluateShIrradiance(float3(0.0, -1.0, 0.0));
    return lerp(groundIrradiance, skyIrradiance, upT);
}

int chooseShadowCascade(float viewDepth)
{
    if (viewDepth <= camera.shadowCascadeSplits.x) return 0;
    if (viewDepth <= camera.shadowCascadeSplits.y) return 1;
    if (viewDepth <= camera.shadowCascadeSplits.z) return 2;
    return 3;
}

float sampleShadowPcf(float3 worldPosition, float3 normal, int cascadeIndex, float ndotl)
{
    const float cascadeT = float(cascadeIndex) / 3.0;
    const float normalOffset = lerp(camera.shadowConfig0.x, camera.shadowConfig0.y, cascadeT) * (1.0 + (1.0 - ndotl));
    const float3 shadowSamplePosition = worldPosition + (normal * normalOffset);
    const float4 shadowClip = mul(camera.lightViewProj[cascadeIndex], float4(shadowSamplePosition, 1.0));
    if (shadowClip.w <= 0.0) return 1.0;

    const float3 shadowNdc = shadowClip.xyz / shadowClip.w;
    const float2 shadowUvLocal = (shadowNdc.xy * 0.5) + 0.5;
    const float4 atlasUvRect = camera.shadowAtlasUvRects[cascadeIndex];
    const float2 shadowUv = atlasUvRect.xy + (shadowUvLocal * atlasUvRect.zw);
    const float shadowDepthRef = shadowNdc.z;

    if (
        shadowUvLocal.x <= 0.0 || shadowUvLocal.x >= 1.0 ||
        shadowUvLocal.y <= 0.0 || shadowUvLocal.y >= 1.0 ||
        shadowDepthRef <= 0.0 || shadowDepthRef >= 1.0)
    {
        return 1.0;
    }

    uint texW = 0;
    uint texH = 0;
    shadowMap.GetDimensions(texW, texH);
    const float2 atlasTexelSize = 1.0 / float2(float(texW), float(texH));
    const float2 texelSize = atlasTexelSize;
    const float texelScale = max(texelSize.x, texelSize.y);

    const float bias =
        lerp(camera.shadowConfig0.z, camera.shadowConfig0.w, cascadeT) * texelScale +
        lerp(camera.shadowConfig1.x, camera.shadowConfig1.y, cascadeT) * texelScale * (1.0 - ndotl);

    const int kernelRadius = clamp(int(camera.shadowConfig3.w + 0.5), 1, 3);
    float visibility = 0.0;
    float weightSum = 0.0;

    for (int y = -3; y <= 3; ++y)
    {
        for (int x = -3; x <= 3; ++x)
        {
            if (abs(x) > kernelRadius || abs(y) > kernelRadius) continue;
            const float2 uv = shadowUv + float2(float(x), float(y)) * texelSize;
            const float wx = float(kernelRadius + 1 - abs(x));
            const float wy = float(kernelRadius + 1 - abs(y));
            const float weight = wx * wy;
            visibility += shadowMap.SampleCmpLevelZero(uv, shadowDepthRef + bias) * weight;
            weightSum += weight;
        }
    }

    return visibility / max(weightSum, 0.0001);
}

float sampleCascadedShadow(float3 worldPosition, float3 normal, float viewDepth, float ndotl)
{
    const int primaryCascade = chooseShadowCascade(viewDepth);
    float shadow = sampleShadowPcf(worldPosition, normal, primaryCascade, ndotl);
    if (primaryCascade >= 3) return shadow;

    const float prevSplit = (primaryCascade == 0) ? 0.0 : camera.shadowCascadeSplits[primaryCascade - 1];
    const float split = camera.shadowCascadeSplits[primaryCascade];
    const float blendRange = max(camera.shadowConfig1.z, (split - prevSplit) * camera.shadowConfig1.w);
    const float blendT = smoothstep(split - blendRange, split, viewDepth);
    if (blendT <= 0.0) return shadow;

    const float nextShadow = sampleShadowPcf(worldPosition, normal, primaryCascade + 1, ndotl);
    return lerp(shadow, nextShadow, blendT);
}

struct FragmentInput
{
    nointerpolation uint inFace : TEXCOORD0;
    nointerpolation uint inMaterial : TEXCOORD1;
    float3 inWorldPosition : TEXCOORD2;
    float inVertexAo : TEXCOORD3;
};

float4 main(FragmentInput input) : SV_Target0
{
    const float3 normal = faceNormal(input.inFace);
    float3 baseColor = diffuseAlbedo.Sample(faceUv(input.inFace, input.inWorldPosition)).xyz;
    if (input.inMaterial == 250u || input.inMaterial == 251u)
    {
        baseColor *= materialTint(input.inMaterial);
    }

    const float3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const float3 sunColor = camera.sunColorShadow.xyz;
    const float shadowStrength = clamp(camera.sunColorShadow.w, 0.0, 1.0);

    const float ndotl = max(dot(normal, -sunDirection), 0.0);
    const float viewDepth = max(-mul(camera.view, float4(input.inWorldPosition, 1.0)).z, 0.0);
    float shadowVisibility = 1.0;
    if (ndotl > 0.0001)
    {
        shadowVisibility = sampleCascadedShadow(input.inWorldPosition, normal, viewDepth, ndotl);
    }

    const float3 shNormalIrradiance = evaluateShIrradiance(normal);
    const float3 shHemisphereIrradiance = evaluateShHemisphereIrradiance(normal);
    const float3 ambientIrradiance = lerp(shNormalIrradiance, shHemisphereIrradiance, 0.70);
    const float vertexAoEnable = clamp(camera.shadowVoxelGridOrigin.w, 0.0, 1.0);
    const float vertexAo = lerp(1.0, clamp(input.inVertexAo, 0.0, 1.0), vertexAoEnable);
    const float3 ambient = ambientIrradiance * (0.26 * vertexAo);
    const float3 directSun = sunColor * (sunIntensity * ndotl);
    const float directShadowFactor = lerp(1.0, shadowVisibility, shadowStrength);
    float3 lighting = ambient + (directSun * directShadowFactor);

    if (input.inMaterial == 250u || input.inMaterial == 251u)
    {
        baseColor = materialTint(input.inMaterial);
    }

    const float3 shaded = baseColor * lighting;
    const float3 fogged = applyAtmosphericFog(shaded, input.inWorldPosition, viewDepth, sunDirection);
    return float4(fogged, 1.0);
}
