// Compile for Vulkan SPIR-V with column-major matrices to match Renderer.cpp uniform layout:
// slangc -target spirv -entry main -stage fragment -matrix-layout-column-major src/render/shaders/voxel_packed.frag.slang -o src/render/shaders/voxel_packed.frag.slang.spv

import camera_uniform;
import sh_lighting;

[[vk::binding(4, 0)]]
Sampler2DShadow shadowMap;

[[vk::binding(9, 0)]]
Sampler3D<float4> voxelGiVolume;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

static const uint kBindlessTextureIndexDiffuseAtlas = 0u;
static const uint kVoxelAtlasTileSize = 16u;
static const uint kMaterialStone = 1u;
static const uint kMaterialDirt = 2u;
static const uint kMaterialGrass = 3u;
static const uint kMaterialWood = 4u;
static const uint kMaterialRed = 5u;
static const uint kMaterialPalette = 6u;

[[vk::constant_id(6)]]
const int kShadowPolicyMode = 2;
[[vk::constant_id(7)]]
const int kAmbientPolicyMode = 2;
[[vk::constant_id(8)]]
const int kForceTintOnly = 0;

float3 materialTint(uint material)
{
    if (material == kMaterialRed) return float3(1.00, 0.28, 0.22);
    return float3(0.9, 0.9, 0.9);
}

float materialSmoothness(uint material)
{
    if (material == kMaterialStone) return 0.18;
    if (material == kMaterialDirt) return 0.05;
    if (material == kMaterialGrass) return 0.08;
    if (material == kMaterialWood) return 0.14;
    if (material == kMaterialRed) return 0.24;
    if (material == kMaterialPalette) return 0.18;
    return 0.12;
}

float3 baseColorFromPalette(uint baseColorIndex)
{
    return camera.voxelBaseColorPalette[baseColorIndex & 0xFu].xyz;
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

bool materialUsesAtlas(uint material)
{
    return
        material == kMaterialStone ||
        material == kMaterialDirt ||
        material == kMaterialGrass ||
        material == kMaterialWood;
}

uint materialAtlasTile(uint material, uint face)
{
    if (material == kMaterialStone) return 0u;
    if (material == kMaterialDirt) return 1u;
    if (material == kMaterialGrass)
    {
        if (face == 3u) return 1u;
        return 2u;
    }
    if (material == kMaterialWood) return 3u;
    return 0u;
}

float3 sampleMaterialAlbedo(uint material, uint face, float3 worldPosition)
{
    if (kForceTintOnly != 0)
    {
        return materialTint(material);
    }

    if (!materialUsesAtlas(material))
    {
        return materialTint(material);
    }

    const uint tileIndex = materialAtlasTile(material, face);
    const float2 uv = faceUv(face, worldPosition);

    uint texW = 0u;
    uint texH = 0u;
    bindlessTextures[kBindlessTextureIndexDiffuseAtlas].GetDimensions(texW, texH);
    const uint tilesX = max(texW / kVoxelAtlasTileSize, 1u);
    const uint tilesY = max(texH / kVoxelAtlasTileSize, 1u);
    const uint tileX = tileIndex % tilesX;
    const uint tileY = tileIndex / tilesX;
    const float2 atlasUv = (float2(float(tileX), float(tileY)) + uv) / float2(float(tilesX), float(tilesY));
    return bindlessTextures[kBindlessTextureIndexDiffuseAtlas].Sample(atlasUv).xyz;
}

float3 sampleVoxelGi(float3 worldPosition, float3 normal)
{
    const float cellSize = max(camera.voxelGiGridOriginCellSize.w, 0.001);
    const float3 gridOrigin = camera.voxelGiGridOriginCellSize.xyz;
    const float3 gridExtent = max(camera.voxelGiGridExtentStrength.xyz, float3(cellSize, cellSize, cellSize));
    const float3 sampleWorld = worldPosition + (normal * (cellSize * 0.5));
    const float3 uvw = (sampleWorld - gridOrigin) / gridExtent;
    if (any(uvw < float3(0.0, 0.0, 0.0)) || any(uvw > float3(1.0, 1.0, 1.0)))
    {
        return float3(0.0, 0.0, 0.0);
    }

    const float3 giRadiance = voxelGiVolume.SampleLevel(uvw, 0.0).xyz;
    return giRadiance * max(camera.voxelGiGridExtentStrength.w, 0.0);
}

float3 voxelGiFalseColor(float t)
{
    const float x = clamp(t, 0.0, 1.0);
    if (x < 0.25) return lerp(float3(0.02, 0.02, 0.10), float3(0.00, 0.42, 1.00), x / 0.25);
    if (x < 0.50) return lerp(float3(0.00, 0.42, 1.00), float3(0.00, 0.95, 0.45), (x - 0.25) / 0.25);
    if (x < 0.75) return lerp(float3(0.00, 0.95, 0.45), float3(1.00, 0.92, 0.12), (x - 0.50) / 0.25);
    return lerp(float3(1.00, 0.92, 0.12), float3(1.00, 0.16, 0.12), (x - 0.75) / 0.25);
}

float3 proceduralSkyNoSunDisk(float3 direction)
{
    const float3 dir = normalize(direction);
    const float3 toSun = -normalize(camera.sunDirectionIntensity.xyz);
    const float rayleighStrength = max(camera.skyConfig0.x, 0.01);
    const float mieStrength = max(camera.skyConfig0.y, 0.01);
    const float skyExposure = max(camera.skyConfig0.w, 0.01);
    const float sunAltitude = clamp(toSun.y, -1.0, 1.0);
    const float dayFactor = smoothstep(0.05, 0.65, sunAltitude);
    const float twilightFactor = 1.0 - dayFactor;
    const float horizonBand = clamp(1.0 - (abs(sunAltitude) / 0.35), 0.0, 1.0);
    const float warmAmount = twilightFactor * pow(horizonBand, 1.2);
    const float pinkAmount = warmAmount * clamp((0.10 - sunAltitude) / 0.30, 0.0, 1.0);

    const float horizonT = clamp((dir.y * 0.5) + 0.5, 0.0, 1.0);
    const float skyT = pow(horizonT, 0.35);

    float3 horizonRayleigh = lerp(float3(0.54, 0.70, 1.00), float3(0.74, 0.44, 0.52), warmAmount);
    float3 horizonMie = lerp(float3(1.00, 0.74, 0.42), float3(1.18, 0.54, 0.30), warmAmount);
    horizonRayleigh = lerp(horizonRayleigh, float3(0.70, 0.36, 0.68), pinkAmount * 0.70);
    horizonMie = lerp(horizonMie, float3(1.08, 0.46, 0.72), pinkAmount * 0.85);

    const float zenithWarm = twilightFactor * 0.58;
    const float3 zenithRayleigh = lerp(float3(0.06, 0.24, 0.54), float3(0.16, 0.12, 0.30), zenithWarm);
    const float3 zenithMie = lerp(float3(0.22, 0.20, 0.15), float3(0.30, 0.18, 0.24), zenithWarm);

    const float3 horizonColor =
        (horizonRayleigh * rayleighStrength) +
        (horizonMie * (mieStrength * 0.58));
    const float3 zenithColor =
        (zenithRayleigh * rayleighStrength) +
        (zenithMie * (mieStrength * 0.25));
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
    const float phaseBoost = 1.0 + (phaseMie * 1.6 * sunView);
    const float3 inscatterColor = skyColor * phaseBoost;

    const float3 transmittance = exp(-extinction * max(viewDepth, 0.0));
    return (litColor * transmittance) + (inscatterColor * (float3(1.0, 1.0, 1.0) - transmittance));
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

interface IShadowPolicy
{
    float visibility(float3 worldPosition, float3 normal, float viewDepth, float ndotl);
};

struct NoShadowPolicy : IShadowPolicy
{
    float visibility(float3 worldPosition, float3 normal, float viewDepth, float ndotl)
    {
        return 1.0;
    }
};

struct PrimaryCascadePcfShadowPolicy : IShadowPolicy
{
    float visibility(float3 worldPosition, float3 normal, float viewDepth, float ndotl)
    {
        const int cascadeIndex = chooseShadowCascade(viewDepth);
        return sampleShadowPcf(worldPosition, normal, cascadeIndex, ndotl);
    }
};

struct CascadedBlendPcfShadowPolicy : IShadowPolicy
{
    float visibility(float3 worldPosition, float3 normal, float viewDepth, float ndotl)
    {
        return sampleCascadedShadow(worldPosition, normal, viewDepth, ndotl);
    }
};

float evaluateShadow<TShadowPolicy : IShadowPolicy>(
    TShadowPolicy policy,
    float3 worldPosition,
    float3 normal,
    float viewDepth,
    float ndotl
)
{
    return policy.visibility(worldPosition, normal, viewDepth, ndotl);
}

interface IAmbientPolicy
{
    float3 ambient(float3 normal, float vertexAo);
};

struct ShOnlyAmbientPolicy : IAmbientPolicy
{
    float3 ambient(float3 normal, float vertexAo)
    {
        const float3 shIrradiance = evaluateShIrradiance(normal);
        return shIrradiance * 0.26;
    }
};

struct ShHemisphereAmbientPolicy : IAmbientPolicy
{
    float3 ambient(float3 normal, float vertexAo)
    {
        const float3 shNormalIrradiance = evaluateShIrradiance(normal);
        const float3 shHemisphereIrradiance = evaluateShHemisphereIrradiance(normal);
        const float3 ambientIrradiance = lerp(shNormalIrradiance, shHemisphereIrradiance, 0.70);
        return ambientIrradiance * 0.26;
    }
};

struct ShHemisphereVertexAoAmbientPolicy : IAmbientPolicy
{
    float3 ambient(float3 normal, float vertexAo)
    {
        const float3 shNormalIrradiance = evaluateShIrradiance(normal);
        const float3 shHemisphereIrradiance = evaluateShHemisphereIrradiance(normal);
        const float3 ambientIrradiance = lerp(shNormalIrradiance, shHemisphereIrradiance, 0.70);
        return ambientIrradiance * (0.26 * vertexAo);
    }
};

float3 evaluateAmbient<TAmbientPolicy : IAmbientPolicy>(
    TAmbientPolicy policy,
    float3 normal,
    float vertexAo
)
{
    return policy.ambient(normal, vertexAo);
}

float selectShadowVisibility(float3 worldPosition, float3 normal, float viewDepth, float ndotl)
{
    if (kShadowPolicyMode <= 0)
    {
        return evaluateShadow(NoShadowPolicy(), worldPosition, normal, viewDepth, ndotl);
    }
    if (kShadowPolicyMode == 1)
    {
        return evaluateShadow(PrimaryCascadePcfShadowPolicy(), worldPosition, normal, viewDepth, ndotl);
    }
    return evaluateShadow(CascadedBlendPcfShadowPolicy(), worldPosition, normal, viewDepth, ndotl);
}

float3 selectAmbient(float3 normal, float vertexAo)
{
    if (kAmbientPolicyMode <= 0)
    {
        return evaluateAmbient(ShOnlyAmbientPolicy(), normal, vertexAo);
    }
    if (kAmbientPolicyMode == 1)
    {
        return evaluateAmbient(ShHemisphereAmbientPolicy(), normal, vertexAo);
    }
    return evaluateAmbient(ShHemisphereVertexAoAmbientPolicy(), normal, vertexAo);
}

struct FragmentInput
{
    nointerpolation uint inFace : TEXCOORD0;
    nointerpolation uint inMaterial : TEXCOORD1;
    float3 inWorldPosition : TEXCOORD2;
    float inVertexAo : TEXCOORD3;
    nointerpolation uint inBaseColorIndex : TEXCOORD4;
};

float4 main(FragmentInput input) : SV_Target0
{
    const float3 normal = faceNormal(input.inFace);
    float3 baseColor = sampleMaterialAlbedo(input.inMaterial, input.inFace, input.inWorldPosition);
    if (input.inMaterial == kMaterialPalette)
    {
        baseColor = baseColorFromPalette(input.inBaseColorIndex);
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
        shadowVisibility = selectShadowVisibility(input.inWorldPosition, normal, viewDepth, ndotl);
    }

    const float vertexAoEnable = clamp(camera.shadowVoxelGridOrigin.w, 0.0, 1.0);
    const float vertexAo = lerp(1.0, clamp(input.inVertexAo, 0.0, 1.0), vertexAoEnable);
    const float3 voxelGi = sampleVoxelGi(input.inWorldPosition, normal) * vertexAo;
    const int voxelGiDebugMode = int(camera.shadowVoxelGridOrigin.z + 0.5);
    if (voxelGiDebugMode == 1)
    {
        const float3 giClamped = max(voxelGi, float3(0.0, 0.0, 0.0));
        const float peak = max(max(giClamped.x, giClamped.y), giClamped.z);
        const float3 hue = (peak > 0.0001) ? (giClamped / peak) : float3(0.0, 0.0, 0.0);
        const float intensity = clamp(log2(1.0 + (peak * 8.0)) / 4.0, 0.0, 1.0);
        return float4(hue * intensity, 1.0);
    }
    if (voxelGiDebugMode == 2)
    {
        const float giLuma = dot(voxelGi, float3(0.2126, 0.7152, 0.0722));
        const float giMapped = clamp(log2(1.0 + (giLuma * 6.0)) / 3.0, 0.0, 1.0);
        return float4(voxelGiFalseColor(giMapped), 1.0);
    }

    const float voxelGiLuma = dot(voxelGi, float3(0.2126, 0.7152, 0.0722));
    const float ambientRebalanceStrength = max(camera.shadowVoxelGridOrigin.x, 0.0);
    const float ambientFloor = clamp(camera.shadowVoxelGridOrigin.y, 0.0, 1.0);
    const float ambientScale = clamp(1.0 / (1.0 + (voxelGiLuma * ambientRebalanceStrength)), ambientFloor, 1.0);
    const float3 ambient = selectAmbient(normal, vertexAo) * ambientScale;
    const float3 directSun = sunColor * (sunIntensity * ndotl);
    const float directShadowFactor = lerp(1.0, shadowVisibility, shadowStrength);
    float3 lighting = ambient + voxelGi + (directSun * directShadowFactor);

    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float3 viewDir = normalize(cameraWorld - input.inWorldPosition);
    const float3 reflectDir = reflect(-viewDir, normal);
    const float nDotV = clamp(dot(normal, viewDir), 0.0, 1.0);
    const float fresnel = pow(1.0 - nDotV, 5.0);
    const float smoothness = materialSmoothness(input.inMaterial);
    const float f0 = 0.028;
    const float fresnelTerm = f0 + ((1.0 - f0) * fresnel);
    const float specularStrength = lerp(0.03, 0.14, smoothness);
    const float sunSpecExponent = lerp(20.0, 144.0, smoothness);
    const float sunGlint = pow(max(dot(reflectDir, -sunDirection), 0.0), sunSpecExponent) * step(0.0001, ndotl);
    const float sunSpecVisibility = directShadowFactor;
    const float3 skyReflection = proceduralSkyNoSunDisk(reflectDir);
    const float3 specular =
        (skyReflection * (fresnelTerm * specularStrength * 0.55)) +
        (sunColor * (sunIntensity * sunGlint * fresnelTerm * specularStrength * 0.20 * sunSpecVisibility));

    const float3 shaded = (baseColor * lighting) + specular;
    const float3 fogged = applyAtmosphericFog(shaded, input.inWorldPosition, viewDepth, sunDirection);
    return float4(fogged, 1.0);
}
