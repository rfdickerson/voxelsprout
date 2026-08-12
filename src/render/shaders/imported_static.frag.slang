import camera_uniform;
import chunk_push_constants;
import pbr;
import sh_lighting;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

// Fixed bindless slot for the fog-of-war R8 texture.
static const int kBindlessIndexFogMap = 10;

[[vk::binding(4, 0)]]
Sampler2DShadow shadowMap;

[[vk::binding(7, 0)]]
Sampler2D<float4> ssaoBlurTexture;

[[vk::binding(9, 0)]]
Sampler3D<float4> voxelGiVolume;

[[vk::binding(11, 0)]]
Sampler3D<float4> voxelGiOccupancyVolume;

#ifdef ODAI_RT_SHADOWS
[[vk::binding(12, 0)]]
RaytracingAccelerationStructure shadowScene;
#endif

// Mirrors the packed material layout in src/import/imported_scene.h; change
// both together. Geometry cooked before materials existed has flags == 0, so
// the PBR bit stays clear and shading falls back to the legacy diffuse chain.
static const uint kImportedMaterialFlagAlphaTest = 1u;
static const uint kImportedMaterialFlagPbr = 4u;
static const uint kImportedMaterialFlagVertexColorTint = 8u;
static const uint kImportedMaterialFlagTerrainLayers = 16u;
// kImportedSceneMaterialFlagUnlit -- BSShaderNoLightingProperty.
static const uint kImportedMaterialFlagUnlit = 128u;
static const uint kImportedMaterialRoughnessShift = 8u;
static const uint kImportedMaterialMetallicShift = 16u;
static const uint kImportedMaterialChannelMask = 0xffu;
static const uint kInvalidImportedTextureIndex = 0xffffffffu;

// Named material library. Bits 24-31 of the flags word index this table; 0 means
// "no library material" and the legacy bits 8-23 are used instead. Mirrored from
// src/import/imported_material.h -- the struct below must stay 32 bytes and keep
// this field order, because the CPU side memcpys the whole table in.
//
// Slot 0 is a reserved sentinel and is never read: it exists so the index stored
// in the vertex flags and the index into this array are the same number, with no
// -1 to reproduce identically here, in the packer and in the loader.
static const uint kImportedMaterialIndexShift = 24u;
static const uint kImportedMaterialIndexMask = 0xffu;

struct GpuImportedMaterial
{
    float4 baseColorMetallic;  // rgb = base color tint, w = metallic
    float4 emissiveRoughness;  // rgb = emissive radiance, w = roughness
};

[[vk::binding(13, 0)]]
StructuredBuffer<GpuImportedMaterial> importedMaterials;

uint importedMaterialIndex(uint flags)
{
    return (flags >> kImportedMaterialIndexShift) & kImportedMaterialIndexMask;
}

bool importedMaterialHasPbr(uint flags)
{
    return (flags & kImportedMaterialFlagPbr) != 0u;
}

PbrSurface unpackImportedMaterial(uint flags)
{
    PbrSurface surface;
    surface.metallic = 0.0;
    surface.roughness = 1.0;
    if (!importedMaterialHasPbr(flags))
    {
        return surface;
    }
    surface.roughness =
        float((flags >> kImportedMaterialRoughnessShift) & kImportedMaterialChannelMask) / 255.0;
    surface.metallic =
        float((flags >> kImportedMaterialMetallicShift) & kImportedMaterialChannelMask) / 255.0;
    return surface;
}

// Resolves a surface through the material library when the flags carry an index,
// and through the legacy per-vertex bits otherwise. The packer forces the PBR bit
// whenever it writes a nonzero index, so callers can keep testing
// importedMaterialHasPbr() as the single "is this PBR" condition.
PbrSurface resolveImportedMaterial(uint flags, out float3 baseColorTint, out float3 emissive)
{
    baseColorTint = float3(1.0, 1.0, 1.0);
    emissive = float3(0.0, 0.0, 0.0);

    const uint index = importedMaterialIndex(flags);
    if (index == 0u)
    {
        return unpackImportedMaterial(flags);
    }

    const GpuImportedMaterial entry = importedMaterials[index];
    baseColorTint = entry.baseColorMetallic.rgb;
    emissive = entry.emissiveRoughness.rgb;

    PbrSurface surface;
    surface.metallic = saturate(entry.baseColorMetallic.w);
    surface.roughness = saturate(entry.emissiveRoughness.w);
    return surface;
}

struct FragmentInput
{
    float4 inScreenPosition : SV_Position;
    float3 inWorldPosition : TEXCOORD0;
    float3 inWorldNormal : TEXCOORD1;
    float3 inColor : TEXCOORD2;
    float2 inUv : TEXCOORD3;
    nointerpolation uint inTextureIndex : TEXCOORD4;
    nointerpolation uint inFlags : TEXCOORD5;
    nointerpolation uint4 inLayerTextures : TEXCOORD6;
    float4 inLayerWeights : TEXCOORD7;
};

// Screen-space AO is baked at half resolution (see ao.ssaoBlur in swapchain_targets.cc);
// this mirrors waterScreenUv() in imported_water.frag.slang so the two shaders derive the
// same full-res-normalized UV from SV_Position without needing the extent as a uniform.
float sampleSsaoAmbientFactor(float4 screenPosition)
{
    uint texW = 0u;
    uint texH = 0u;
    ssaoBlurTexture.GetDimensions(texW, texH);
    const float2 fullResolutionExtent = float2(float(max(texW, 1u)), float(max(texH, 1u))) * 2.0;
    const float2 ssaoUv = clamp(screenPosition.xy / fullResolutionExtent, float2(0.0, 0.0), float2(1.0, 1.0));
    const float ssaoRaw = clamp(ssaoBlurTexture.SampleLevel(ssaoUv, 0.0).x, 0.0, 1.0);
    const float ssaoIntensity = clamp(camera.shadowConfig2.z, 0.0, 2.0);
    const float ssaoEnable = step(0.5, camera.shadowVoxelGridSize.w);
    return lerp(1.0, pow(max(ssaoRaw, 0.001), ssaoIntensity), ssaoEnable);
}

// Aerial perspective: distance haze, so a landscape reads as deep instead of
// flat.
//
// Not a fog constant. Extinction is wavelength-dependent (blue survives least,
// so distant terrain shifts toward the sky) and the inscattered light warms
// toward the sun, which is what makes a hill lit from behind glow at its edge.
// Both fall straight out of the same sun colour and sky strengths the skybox
// shades with, so the horizon and the terrain in front of it agree instead of
// meeting at a visible seam.
//
// Exterior only: an interior has no kilometres of air between the surface and
// the eye, and applying it there washes rooms out at their far wall.
// Value noise for breaking up terrain blend boundaries. Hash-based rather than a
// texture: this is sampled once per fragment and a dependent texture read would
// cost more than the arithmetic.
float terrainBlendHash(float2 p)
{
    float2 wrapped = frac(p * float2(123.34, 345.45));
    wrapped += dot(wrapped, wrapped + 34.345);
    return frac(wrapped.x * wrapped.y);
}

float terrainBlendNoise(float2 p)
{
    const float2 cell = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0 - (2.0 * f));
    const float a = terrainBlendHash(cell);
    const float b = terrainBlendHash(cell + float2(1.0, 0.0));
    const float c = terrainBlendHash(cell + float2(0.0, 1.0));
    const float d = terrainBlendHash(cell + float2(1.0, 1.0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

float3 applyAerialPerspective(float3 shaded, float3 worldPosition, float3 viewDirection)
{
    // Distance at which roughly 1/e of the surface's own colour survives. ~15000
    // Bethesda units is a little over 200 m: far enough that near geometry is
    // untouched, close enough that the far ridges of a cooked region sit back.
    const float kAerialEFoldDistance = 15000.0;
    const float3 cameraPosition = camera.shadowVoxelGridSize.xyz;
    const float viewDistance = length(worldPosition - cameraPosition);
    const float3 toSun = -normalize(camera.sunDirectionIntensity.xyz);

    // Blue extinguishes least, so it dominates what comes back from the air.
    const float3 extinctionPerUnit =
        float3(1.35, 1.12, 0.85) / kAerialEFoldDistance;
    const float3 transmittance = exp(-viewDistance * extinctionPerUnit);

    // Forward scattering: looking toward the sun through air brightens and warms
    // it. pow(,6) keeps the effect close to the sun rather than tinting the
    // whole sky.
    const float sunAlignment = saturate(dot(viewDirection, toSun));
    const float3 skyInscatter = float3(0.40, 0.56, 0.82);
    const float3 sunInscatter = float3(1.00, 0.86, 0.66);
    const float3 inscatterColor =
        lerp(skyInscatter, sunInscatter, pow(sunAlignment, 6.0) * 0.85);

    // Scaled by the sun's own colour and the sky's rayleigh strength so dusk
    // haze goes warm and night haze goes dark, instead of staying daylight blue.
    const float skyStrength = max(camera.skyConfig0.x, 0.01);
    const float sunLuminance = dot(camera.sunColorShadow.xyz, float3(0.30, 0.59, 0.11));
    const float3 inscatter =
        inscatterColor * (skyStrength * 0.42) * max(sunLuminance, 0.03);

    return (shaded * transmittance) + (inscatter * (float3(1.0, 1.0, 1.0) - transmittance));
}

float4 sampleImportedAlbedo(FragmentInput input)
{
    const bool disableTextures = chunkPc.cascadeData.z > 0.5;
    if (input.inTextureIndex == kInvalidImportedTextureIndex)
    {
        return float4(input.inColor, 1.0);
    }
    float4 sampled = bindlessTextures[input.inTextureIndex].Sample(input.inUv);
    // Fallout's NiAlphaProperty test mode is GREATER: a fragment survives when
    // its alpha is strictly above the authored threshold, so discard on <=.
    if ((input.inFlags & kImportedMaterialFlagAlphaTest) != 0u &&
        sampled.a <= chunkPc.materialParams.x)
    {
        // ODAI_DEBUG_ALPHATEST_HIGHLIGHT paints what would be cut away instead
        // of cutting it. ODAI_DEBUG_NO_ALPHATEST answers "is alpha test doing
        // this"; this answers "WHICH surface", which is the harder half when a
        // scene draws 490 alpha-tested shapes and only one of them is wrong.
        if (chunkPc.materialParams.y > 0.5)
        {
            return float4(1.0, 0.0, 1.0, 1.0);
        }
        discard;
    }
    // Fallout's landscape is a base texture per quadrant with additional layers
    // alpha-blended over it (ATXT/VTXT) -- roads, gravel, and the transitions
    // between ground types. Without this the terrain is one flat texture per
    // 2048-unit quadrant with a hard seam at every quadrant boundary.
    //
    // Layers composite in order, each a straight lerp by its own weight, which
    // is what the game's own layer stack does. The weights are independent
    // opacities, NOT a partition of unity, so they must not be normalized: a
    // post where two layers are both fully opaque means the later one wins, not
    // that they are each half strength.
    if ((input.inFlags & kImportedMaterialFlagTerrainLayers) != 0u)
    {
        // VTXT authors one opacity per landscape post -- a value every 128 units
        // -- and interpolating that linearly across the triangles makes every
        // blend boundary a straight, grid-aligned edge. That is the blockiness,
        // and it is in the interpolation, not the data: the same weights can
        // produce an organic boundary.
        //
        // Two corrections, one noise sample shared by all four layers because
        // the noise perturbs WHERE the boundary falls, not which layer it
        // belongs to.
        //
        //   - Perturb the weight by value noise, so the boundary wanders across
        //     the lattice instead of following it. Two octaves: the coarse one
        //     breaks up long straight runs, the fine one gives the edge grain.
        //   - Sharpen with smoothstep. A linear ramp spreads every transition
        //     across a full 128-unit cell, which reads as a soft rectangle;
        //     tightening it puts the transition where the noise put it.
        //
        // Both are no-ops where a layer is fully present or fully absent, so
        // large single-texture areas are untouched.
        const float2 blendNoiseUv = input.inWorldPosition.xz;
        const float blendNoise =
            (terrainBlendNoise(blendNoiseUv * (1.0 / 220.0)) * 0.65) +
            (terrainBlendNoise(blendNoiseUv * (1.0 / 70.0)) * 0.35);
        const float blendNoiseSigned = blendNoise - 0.5;

        for (uint layer = 0u; layer < 4u; ++layer)
        {
            const uint layerTexture = input.inLayerTextures[layer];
            const float rawWeight = saturate(input.inLayerWeights[layer]);
            if (layerTexture == kInvalidImportedTextureIndex || rawWeight <= 0.0)
            {
                continue;
            }
            // Scaled by how mid-transition the weight is, so the noise cannot
            // punch holes in a fully opaque layer or speckle a fully absent one.
            const float transitionAmount = 1.0 - abs((rawWeight * 2.0) - 1.0);
            const float perturbed = saturate(rawWeight + (blendNoiseSigned * 0.55 * transitionAmount));
            const float layerWeight = smoothstep(0.22, 0.78, perturbed);
            if (layerWeight <= 0.0)
            {
                continue;
            }
            const float4 layerSample = bindlessTextures[layerTexture].Sample(input.inUv);
            sampled = float4(lerp(sampled.rgb, layerSample.rgb, layerWeight), sampled.a);
        }
    }
    if (disableTextures)
    {
        return float4(input.inColor, 1.0);
    }
    // Vertex color modulates a textured surface only where the cooker says the
    // color is authored data (kImportedSceneMaterialFlagVertexColorTint). For
    // Fallout terrain that is LAND's VCLR: the baked ambient and regional
    // palette the game itself shades landscape with.
    //
    // The bit is what makes this safe to do unconditionally. This used to blend
    // 25% toward `sampled.rgb * inColor * 1.6` for *everything* textured, back
    // when the only colors reaching here were
    // buildImportedScenePackedRenderData's fallbacks — a hash of the model path
    // for instanced statics, a height ramp for terrain. Both are noise, both
    // biased dark with blue lowest, which is what made cooked scenes read dark,
    // warm and over-saturated with a per-model hue cast. Those fallbacks still
    // exist and still must not tint anything; they just no longer set the bit.
    //
    // Untextured geometry is unaffected either way: it returns inColor as its
    // albedo at the early-out above.
    if ((input.inFlags & kImportedMaterialFlagVertexColorTint) != 0u)
    {
        return float4(sampled.rgb * input.inColor, sampled.a);
    }
    return sampled;
}

float3 sampleImportedVoxelGi(float3 worldPosition, float3 normal)
{
    const float cellSize = max(camera.voxelGiGridOriginCellSize.w, 0.001);
    const float3 gridOrigin = camera.voxelGiGridOriginCellSize.xyz;
    const float3 gridExtent = max(camera.voxelGiGridExtentStrength.xyz, float3(cellSize, cellSize, cellSize));
    const float3 uvwCenter = (worldPosition - gridOrigin) / gridExtent;
    if (any(uvwCenter < float3(0.0, 0.0, 0.0)) || any(uvwCenter > float3(1.0, 1.0, 1.0)))
    {
        return float3(0.0, 0.0, 0.0);
    }

    const float3 uvwForward = (worldPosition + (normal * (cellSize * 0.55)) - gridOrigin) / gridExtent;
    const float3 uvwInterior = (worldPosition - (normal * (cellSize * 0.35)) - gridOrigin) / gridExtent;
    const float3 giCenter = voxelGiVolume.SampleLevel(uvwCenter, 0.0).xyz;
    const float3 giForward =
        (any(uvwForward < float3(0.0, 0.0, 0.0)) || any(uvwForward > float3(1.0, 1.0, 1.0)))
            ? giCenter
            : voxelGiVolume.SampleLevel(uvwForward, 0.0).xyz;

    const float interiorOccupancy =
        (any(uvwInterior < float3(0.0, 0.0, 0.0)) || any(uvwInterior > float3(1.0, 1.0, 1.0)))
            ? 0.0
            : voxelGiOccupancyVolume.SampleLevel(uvwInterior, 0.0).x;
    const float oneSidedBlend = (interiorOccupancy > 0.5) ? 1.0 : 0.0;
    const float3 baseBlend = (giCenter * 0.70) + (giForward * 0.30);
    const float3 oneSidedBlendTarget = (giCenter * 0.20) + (giForward * 0.80);
    return lerp(baseBlend, oneSidedBlendTarget, oneSidedBlend) *
        max(camera.voxelGiGridExtentStrength.w, 0.0);
}

float giLumaFromColor(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float3 sampleImportedVoxelGiDebug(float3 worldPosition, float3 normal)
{
    const float cellSize = max(camera.voxelGiGridOriginCellSize.w, 0.001);
    const float3 gridOrigin = camera.voxelGiGridOriginCellSize.xyz;
    const float3 gridExtent = max(camera.voxelGiGridExtentStrength.xyz, float3(cellSize, cellSize, cellSize));
    const float3 uvwCenter = (worldPosition - gridOrigin) / gridExtent;
    if (any(uvwCenter < float3(0.0, 0.0, 0.0)) || any(uvwCenter > float3(1.0, 1.0, 1.0)))
    {
        return float3(0.0, 0.0, 0.0);
    }

    const float occupancyCenter = voxelGiOccupancyVolume.SampleLevel(uvwCenter, 0.0).x;
    if (occupancyCenter > 0.5)
    {
        return sampleImportedVoxelGi(worldPosition + (normal * (cellSize * 0.90)), normal);
    }
    return sampleImportedVoxelGi(worldPosition, normal);
}

float3 sampleImportedVoxelGiOccupancyAlbedo(float3 worldPosition, float3 normal)
{
    const float cellSize = max(camera.voxelGiGridOriginCellSize.w, 0.001);
    const float3 gridOrigin = camera.voxelGiGridOriginCellSize.xyz;
    const float3 gridExtent = max(camera.voxelGiGridExtentStrength.xyz, float3(cellSize, cellSize, cellSize));
    const float3 uvwCenter = (worldPosition - gridOrigin) / gridExtent;
    if (any(uvwCenter < float3(0.0, 0.0, 0.0)) || any(uvwCenter > float3(1.0, 1.0, 1.0)))
    {
        return float3(0.0, 0.0, 0.0);
    }

    const float probeDistance = cellSize * 0.90;
    const float3 uvwForward = (worldPosition + (normal * probeDistance) - gridOrigin) / gridExtent;
    const float3 uvwBack = (worldPosition - (normal * probeDistance) - gridOrigin) / gridExtent;
    const float4 center = voxelGiOccupancyVolume.SampleLevel(uvwCenter, 0.0);
    const float4 forward =
        (any(uvwForward < float3(0.0, 0.0, 0.0)) || any(uvwForward > float3(1.0, 1.0, 1.0)))
            ? center
            : voxelGiOccupancyVolume.SampleLevel(uvwForward, 0.0);
    const float4 back =
        (any(uvwBack < float3(0.0, 0.0, 0.0)) || any(uvwBack > float3(1.0, 1.0, 1.0)))
            ? center
            : voxelGiOccupancyVolume.SampleLevel(uvwBack, 0.0);

    float4 best = center;
    if (forward.x > best.x)
    {
        best = forward;
    }
    if (back.x > best.x)
    {
        best = back;
    }
    return (best.x > 0.5) ? saturate(best.yzw) : float3(0.0, 0.0, 0.0);
}

float3 voxelGiFalseColor(float t)
{
    const float x = clamp(t, 0.0, 1.0);
    if (x < 0.25) return lerp(float3(0.02, 0.02, 0.10), float3(0.00, 0.42, 1.00), x / 0.25);
    if (x < 0.50) return lerp(float3(0.00, 0.42, 1.00), float3(0.00, 0.95, 0.45), (x - 0.25) / 0.25);
    if (x < 0.75) return lerp(float3(0.00, 0.95, 0.45), float3(1.00, 0.92, 0.12), (x - 0.50) / 0.25);
    return lerp(float3(1.00, 0.92, 0.12), float3(1.00, 0.16, 0.12), (x - 0.75) / 0.25);
}

#ifdef ODAI_RT_SHADOWS
static const float kPi = 3.14159265;
static const float kGoldenAngle = 2.39996323;

float2 rtDiskSampleOffset(int sampleIndex, int sampleCount)
{
    const float count = max(float(sampleCount), 1.0);
    const float radius = sqrt((float(sampleIndex) + 0.5) / count);
    const float angle = (float(sampleIndex) + 1.0) * kGoldenAngle;
    return float2(cos(angle), sin(angle)) * radius;
}

float hashRtShadowRotation(float3 worldPosition)
{
    return frac(sin(dot(worldPosition, float3(12.9898, 78.233, 37.719))) * 43758.5453);
}

float2 rotateRtDiskOffset(float2 offset, float angle)
{
    const float s = sin(angle);
    const float c = cos(angle);
    return float2((offset.x * c) - (offset.y * s), (offset.x * s) + (offset.y * c));
}

float sampleRayTracedShadowRay(float3 origin, float3 direction, float tMin, float tMax)
{
    if (tMax <= tMin)
    {
        return 1.0;
    }

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> rayQuery;
    rayQuery.TraceRayInline(shadowScene, 0, 0xFF, ray);
    while (rayQuery.Proceed()) {}
    return (rayQuery.CommittedStatus() == COMMITTED_NOTHING) ? 1.0 : 0.0;
}

float sampleRayTracedPointLightShadow(
    float3 worldPosition,
    float3 normal,
    float3 lightDirection,
    float lightDistance,
    float ndotl)
{
    const float originBias = lerp(0.45, 0.18, saturate(ndotl));
    const float3 origin = worldPosition + (normal * originBias) + (lightDirection * 0.12);
    return sampleRayTracedShadowRay(origin, lightDirection, 0.03, max(lightDistance - 0.30, 0.03));
}

float sampleRayTracedDirectionalShadow(float3 worldPosition, float3 normal, float ndotl)
{
    const float shadowMapReceiverOffset =
        lerp(camera.shadowConfig0.x, camera.shadowConfig0.y, 0.35) * (1.0 + (1.0 - ndotl));
    const float normalOffset = clamp(shadowMapReceiverOffset * 4.0, 0.18, 0.85);
    const float3 toSun = -normalize(camera.sunDirectionIntensity.xyz);
    const float3 shadowSamplePosition = worldPosition + (normal * normalOffset) + (toSun * 0.12);
    const float tMin = 0.03;
    const float tMax = max(camera.shadowCascadeSplits.w * 8.0, 4096.0);
    const int sampleCount = clamp(int(camera.shadowConfig4.x + 0.5), 1, 8);
    const float sunAngularRadiusDegrees = max(camera.shadowConfig4.y, 0.0);
    const float coneRadius = tan(sunAngularRadiusDegrees * (kPi / 180.0));

    if (sampleCount <= 1 || coneRadius <= 1e-5)
    {
        return sampleRayTracedShadowRay(shadowSamplePosition, toSun, tMin, tMax);
    }

    const float3 basisUp = (abs(toSun.y) < 0.999) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 tangent = normalize(cross(basisUp, toSun));
    const float3 bitangent = cross(toSun, tangent);
    const float rotationAngle = hashRtShadowRotation(worldPosition) * (2.0 * kPi);

    float visibility = 0.0;
    [unroll]
    for (int sampleIndex = 0; sampleIndex < 8; ++sampleIndex)
    {
        if (sampleIndex >= sampleCount) break;
        const float2 diskOffset = rotateRtDiskOffset(rtDiskSampleOffset(sampleIndex, sampleCount), rotationAngle);
        const float3 rayDirection = normalize(
            toSun +
            ((tangent * diskOffset.x) + (bitangent * diskOffset.y)) * coneRadius
        );
        visibility += sampleRayTracedShadowRay(shadowSamplePosition, rayDirection, tMin, tMax);
    }
    return visibility / float(sampleCount);
}
#endif

// Diffuse contribution of the local point lights, and — for surfaces that opted
// into a PBR material — their specular highlights accumulated into outSpecular.
// Non-PBR surfaces pass applySpecular=false and get exactly the pre-PBR result.
float3 evaluateImportedLocalLights(
    float3 worldPosition,
    float3 normal,
    float3 viewDirection,
    float3 albedo,
    PbrSurface surface,
    bool applySpecular,
    inout float3 outSpecular)
{
    const int lightCount = clamp(int(camera.importedLightConfig.x + 0.5), 0, 64);
    const float globalIntensity = max(camera.importedLightConfig.y, 0.0);
    if (lightCount <= 0 || globalIntensity <= 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }

    float3 lighting = float3(0.0, 0.0, 0.0);
    [loop]
    for (int lightIndex = 0; lightIndex < 64; ++lightIndex)
    {
        if (lightIndex >= lightCount) break;
        const float4 positionRadius = camera.importedLightPositionRadius[lightIndex];
        const float4 colorIntensity = camera.importedLightColorIntensity[lightIndex];
        const float3 toLight = positionRadius.xyz - worldPosition;
        const float distanceSquared = dot(toLight, toLight);
        const float radius = max(positionRadius.w, 1.0);
        const float radiusSquared = radius * radius;
        if (distanceSquared >= radiusSquared)
        {
            continue;
        }

        const float distanceToLight = sqrt(max(distanceSquared, 1e-4));
        const float3 lightDirection = toLight / distanceToLight;
        const float ndotl = saturate(dot(normal, lightDirection));
        if (ndotl <= 0.0001)
        {
            continue;
        }
        const float normalizedDistance = saturate(distanceToLight / radius);
        const float falloff = saturate(1.0 - (normalizedDistance * normalizedDistance));
        const float attenuation = falloff * falloff;
        float visibility = 1.0;
#ifdef ODAI_RT_SHADOWS
        visibility = sampleRayTracedPointLightShadow(
            worldPosition,
            normal,
            lightDirection,
            distanceToLight,
            ndotl);
#endif
        // Left exactly as it was pre-PBR, factoring included: this expression is
        // the legacy path for every surface without a material, and a reassociation
        // here would shift existing scenes by an ULP for no reason.
        lighting += colorIntensity.rgb * (colorIntensity.w * globalIntensity * ndotl * attenuation * visibility);
        if (applySpecular)
        {
            const float3 radiance =
                colorIntensity.rgb * (colorIntensity.w * globalIntensity * attenuation * visibility);
            // kPbrDiffuseConventionScale: pbrDirectSpecular returns a normalized
            // BRDF, the diffuse term two lines up does not. See pbr.slang.
            outSpecular +=
                radiance * kPbrDiffuseConventionScale *
                pbrDirectSpecular(surface, albedo, normal, viewDirection, lightDirection);
        }
    }
    return lighting;
}

float3 evaluateImportedLocalLightDebug(float3 worldPosition)
{
    const int lightCount = clamp(int(camera.importedLightConfig.x + 0.5), 0, 64);
    const float globalIntensity = max(camera.importedLightConfig.y, 0.0);
    if (lightCount <= 0 || globalIntensity <= 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }

    float3 lighting = float3(0.0, 0.0, 0.0);
    [loop]
    for (int lightIndex = 0; lightIndex < 64; ++lightIndex)
    {
        if (lightIndex >= lightCount) break;
        const float4 positionRadius = camera.importedLightPositionRadius[lightIndex];
        const float4 colorIntensity = camera.importedLightColorIntensity[lightIndex];
        const float3 toLight = positionRadius.xyz - worldPosition;
        const float distanceSquared = dot(toLight, toLight);
        const float radius = max(positionRadius.w, 1.0);
        if (distanceSquared >= radius * radius)
        {
            continue;
        }

        const float distanceToLight = sqrt(max(distanceSquared, 1e-4));
        const float normalizedDistance = saturate(distanceToLight / radius);
        const float falloff = saturate(1.0 - (normalizedDistance * normalizedDistance));
        lighting += colorIntensity.rgb * (colorIntensity.w * globalIntensity * falloff * falloff);
    }
    return lighting;
}

int chooseShadowCascade(float viewDepth)
{
    if (viewDepth <= camera.shadowCascadeSplits.x) return 0;
    if (viewDepth <= camera.shadowCascadeSplits.y) return 1;
    if (viewDepth <= camera.shadowCascadeSplits.z) return 2;
    return 3;
}

float sampleImportedShadowPcf(float3 worldPosition, float3 normal, int cascadeIndex, float ndotl)
{
    const float cascadeT = float(cascadeIndex) / 3.0;
    const float normalOffset =
        lerp(camera.shadowConfig0.x, camera.shadowConfig0.y, cascadeT) * (1.0 + (1.0 - ndotl));
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
    const float2 texelSize = 1.0 / float2(float(texW), float(texH));
    const float texelScale = max(texelSize.x, texelSize.y);
    const float bias =
        lerp(camera.shadowConfig0.z, camera.shadowConfig0.w, cascadeT) * texelScale +
        lerp(camera.shadowConfig1.x, camera.shadowConfig1.y, cascadeT) * texelScale * (1.0 - ndotl);
    const int kernelRadius = clamp(int(camera.shadowConfig3.w + 0.5), 1, 3);

    float visibility = 0.0;
    float weightSum = 0.0;
    // Two paths, and the split is measured rather than stylistic.
    //
    // This used to iterate -3..3 (49 steps) and `continue` past everything
    // outside kernelRadius, which is 1 by default -- 40 of 49 iterations per
    // fragment branching and doing nothing, and the cascade blend region runs
    // the whole thing twice. Constant bounds let the compiler unroll to exactly
    // nine taps: 7.42 -> 6.49 ms of main pass at identical output. Making the
    // bounds follow kernelRadius dynamically instead only reached 7.11, because
    // a runtime trip count blocks the unroll -- so the default radius gets a
    // fixed 3x3 and the wider kernels, which only the debug UI selects, take the
    // general path.
    if (kernelRadius <= 1)
    {
        [unroll]
        for (int y = -1; y <= 1; ++y)
        {
            [unroll]
            for (int x = -1; x <= 1; ++x)
            {
                const float2 uv = shadowUv + float2(float(x), float(y)) * texelSize;
                const float weight = float(2 - abs(x)) * float(2 - abs(y));
                visibility += shadowMap.SampleCmpLevelZero(uv, shadowDepthRef + bias) * weight;
                weightSum += weight;
            }
        }
    }
    else
    {
        [loop]
        for (int y = -kernelRadius; y <= kernelRadius; ++y)
        {
            [loop]
            for (int x = -kernelRadius; x <= kernelRadius; ++x)
            {
                const float2 uv = shadowUv + float2(float(x), float(y)) * texelSize;
                const float wx = float(kernelRadius + 1 - abs(x));
                const float wy = float(kernelRadius + 1 - abs(y));
                const float weight = wx * wy;
                visibility += shadowMap.SampleCmpLevelZero(uv, shadowDepthRef + bias) * weight;
                weightSum += weight;
            }
        }
    }
    return visibility / max(weightSum, 0.0001);
}

float sampleImportedCascadedShadow(float3 worldPosition, float3 normal, float viewDepth, float ndotl)
{
    const int primaryCascade = chooseShadowCascade(viewDepth);
    float shadow = sampleImportedShadowPcf(worldPosition, normal, primaryCascade, ndotl);
    if (primaryCascade >= 3) return shadow;

    const float prevSplit = (primaryCascade == 0) ? 0.0 : camera.shadowCascadeSplits[primaryCascade - 1];
    const float split = camera.shadowCascadeSplits[primaryCascade];
    const float blendRange = max(camera.shadowConfig1.z, (split - prevSplit) * camera.shadowConfig1.w);
    const float blendT = smoothstep(split - blendRange, split, viewDepth);
    if (blendT <= 0.0) return shadow;

    const float nextShadow = sampleImportedShadowPcf(worldPosition, normal, primaryCascade + 1, ndotl);
    return lerp(shadow, nextShadow, blendT);
}

float4 main(FragmentInput input) : SV_Target0
{
    const float4 albedoFull = sampleImportedAlbedo(input);
    if (chunkPc.cascadeData.w > 0.5) {
        return float4(saturate(albedoFull.rgb * 1.25), 1.0);
    }
    // Self-lit surfaces bypass shading entirely: a CRT screen or a neon tube
    // is a light source, and running it through the diffuse chain leaves it as
    // dark as whatever happens to be falling on it. Placed before the normal is
    // even read, because none of the shading below applies to one.
    //
    // The albedo is authored display-referred, like every other Fallout
    // texture, so it is scaled toward the exposure the rest of the scene sits
    // at rather than emitted at 1.0 -- which on an auto-exposed HDR frame keyed
    // to a sunlit desert would read as a dim grey card, not a lit screen.
    if ((input.inFlags & kImportedMaterialFlagUnlit) != 0u)
    {
        return float4(albedoFull.rgb * 3.0, albedoFull.a);
    }
    const float3 normal = normalize(input.inWorldNormal);
    // Slope-based rock/cliff blending. Steep faces (terrain skirts, cliff rims)
    // read as exposed rock or lighter cliff stone; flat faces are unaffected.
    // The welded terrain's smooth normals make this correct at elevation transitions.
    const float slope = saturate(1.0 - normal.y);

    // The material is resolved here rather than down in the shading block,
    // because the library's base-color tint modulates the *sampled* albedo and
    // so has to land before the terrain slope lerps below — those deliberately
    // override albedo on cliff faces, and tinting afterwards would fight them.
    float3 materialTint = float3(1.0, 1.0, 1.0);
    float3 materialEmissive = float3(0.0, 0.0, 0.0);
    const PbrSurface surface =
        resolveImportedMaterial(input.inFlags, materialTint, materialEmissive);
    const bool usePbr = importedMaterialHasPbr(input.inFlags);

    float3 albedo = albedoFull.rgb * materialTint;
    // Stylized slope tint (docs/stylized_low_poly.md): steep faces blend toward
    // flat rock greys so untextured low-poly terrain reads as cliff rather than
    // as a coloured ramp.
    //
    // Applied ONLY to untextured surfaces. It used to run unconditionally, which
    // meant it overwrote real albedo on anything sloped: on a cooked Fallout
    // scene a rock face measured (97,91,90) and a Joshua tree trunk (164,169,166)
    // -- near-perfectly neutral grey -- while the flat ground beside them kept
    // its warm (74,72,50). That is the whole "grey materials instead of textures"
    // symptom, and it is this lerp, not the texture pipeline.
    //
    // Gating on textured-ness rather than on
    // kImportedSceneMaterialFlagTerrainSlopeBlend keeps every existing scene
    // pixel-identical: the geometry this was authored for (the hex strategy map,
    // procedural low-poly) is untextured and still gets it. The reserved flag is
    // the right control if a TEXTURED surface ever wants the effect too.
    if (input.inTextureIndex == kInvalidImportedTextureIndex)
    {
        albedo = lerp(albedo, float3(0.40, 0.39, 0.36), smoothstep(0.35, 0.66, slope) * 0.6);
        albedo = lerp(albedo, float3(0.52, 0.51, 0.47), smoothstep(0.66, 0.92, slope) * 0.5);
    }
    const float3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const float3 shNormalIrradiance = evaluateShIrradiance(normal);
    const float3 shHemisphereIrradiance = evaluateShHemisphereIrradiance(normal);
    const float3 ambientIrradiance = lerp(shNormalIrradiance, shHemisphereIrradiance, 0.6);
    const float ndotl = max(dot(normal, -sunDirection), 0.0);
    const float wrappedNdotL = saturate((ndotl * 0.65) + 0.35);
    const float viewDepth = max(-mul(camera.view, float4(input.inWorldPosition, 1.0)).z, 0.0);
    float shadowVisibility = 1.0;
    if (ndotl > 0.0001)
    {
#ifdef ODAI_RT_SHADOWS
        shadowVisibility = sampleRayTracedDirectionalShadow(input.inWorldPosition, normal, ndotl);
#else
        shadowVisibility = sampleImportedCascadedShadow(input.inWorldPosition, normal, viewDepth, ndotl);
#endif
    }
    const bool interiorMode = chunkPc.cascadeData.y > 0.5;
    const float3 skyProbeNormal = float3(0.0, 1.0, 0.0);
    const float skyProbeNdotL = max(dot(skyProbeNormal, -sunDirection), 0.25);
    float skyVisibility = 1.0;
    if (interiorMode)
    {
#ifdef ODAI_RT_SHADOWS
        skyVisibility = sampleRayTracedDirectionalShadow(input.inWorldPosition, skyProbeNormal, skyProbeNdotL);
#else
        skyVisibility = sampleImportedCascadedShadow(input.inWorldPosition, skyProbeNormal, viewDepth, skyProbeNdotL);
#endif
    }
    // Interior sky-visibility term stays as-is; screen-space AO layers on top and is the
    // only source of contact/corner darkening for outdoor surfaces (previously hardcoded
    // to 1.0 here — see docs/stylized_low_poly.md). Scoped to ambient only, not direct
    // sun, so sunlit faces stay clean instead of muddying under an AO halo.
    const float ambientOcclusion =
        (interiorMode ? lerp(0.28, 1.0, skyVisibility) : 1.0) * sampleSsaoAmbientFactor(input.inScreenPosition);
    const float3 ambient =
        (interiorMode ? float3(0.045, 0.047, 0.052) : float3(0.18, 0.19, 0.21)) +
        (ambientIrradiance * float3(0.92, 0.95, 1.00) * (interiorMode ? 0.22 : 0.58));
    const float3 voxelGi = interiorMode
        ? sampleImportedVoxelGi(input.inWorldPosition, normal) * 1.35
        : float3(0.0, 0.0, 0.0);
    const float3 voxelGiDebug = sampleImportedVoxelGiDebug(input.inWorldPosition, normal);
    const int voxelGiDebugMode = int(camera.shadowVoxelGridOrigin.z + 0.5);
    if (voxelGiDebugMode == 1)
    {
        const float3 giClamped = max(voxelGiDebug, float3(0.0, 0.0, 0.0));
        const float peak = max(max(giClamped.x, giClamped.y), giClamped.z);
        const float3 hue = (peak > 0.0001) ? (giClamped / peak) : float3(0.0, 0.0, 0.0);
        const float intensity = clamp(log2(1.0 + (peak * 8.0)) / 4.0, 0.0, 1.0);
        return float4(hue * intensity, 1.0);
    }
    if (voxelGiDebugMode == 2)
    {
        const float giLuma = giLumaFromColor(voxelGiDebug);
        const float giMapped = clamp(log2(1.0 + (giLuma * 6.0)) / 3.0, 0.0, 1.0);
        return float4(voxelGiFalseColor(giMapped), 1.0);
    }
    if (voxelGiDebugMode == 3)
    {
        const float giLuma = giLumaFromColor(voxelGiDebug);
        const float giMapped = clamp(log2(1.0 + (giLuma * 4.0)) / 3.0, 0.0, 1.0);
        return float4(giMapped, giMapped, giMapped, 1.0);
    }
    if (voxelGiDebugMode == 4)
    {
        const float3 occupancyAlbedo = sampleImportedVoxelGiOccupancyAlbedo(input.inWorldPosition, normal);
        return float4(occupancyAlbedo, 1.0);
    }
    if (voxelGiDebugMode == 5)
    {
        const float3 lightDebug = max(evaluateImportedLocalLightDebug(input.inWorldPosition), float3(0.0, 0.0, 0.0));
        const float peak = max(max(lightDebug.x, lightDebug.y), lightDebug.z);
        const float3 hue = (peak > 0.0001) ? (lightDebug / peak) : float3(0.0, 0.0, 0.0);
        const float intensity = clamp(log2(1.0 + (peak * 2.0)) / 4.0, 0.0, 1.0);
        return float4(hue * intensity, 1.0);
    }
    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float3 viewDir = normalize(cameraWorld - input.inWorldPosition);

    // Metallic-roughness material, opt-in per vertex. Surfaces without the PBR
    // flag decode to a rough dielectric and take the branch-free legacy path:
    // specular stays zero, diffuse keeps its full weight, and the old fake-rim
    // term is preserved. See docs/ROADMAP.md (Core Rendering / PBR).
    // surface / usePbr are resolved up at the albedo site above, so the library
    // tint can be applied before the terrain slope lerps.

    float3 specular = float3(0.0, 0.0, 0.0);
    const float3 importedLights =
        evaluateImportedLocalLights(
            input.inWorldPosition,
            normal,
            viewDir,
            albedo,
            surface,
            usePbr,
            specular) *
        (interiorMode ? 1.25 : 1.0);
    const float directShadowFactor = lerp(1.0, shadowVisibility, clamp(camera.sunColorShadow.w, 0.0, 1.0));

    // wrappedNdotL is ndotl*0.65 + 0.35: a true-cosine core plus a constant
    // fill that keeps faces turned away from the sun from going black. That
    // fill is a stand-in for indirect bounce, NOT direct sunlight -- so it is
    // the part ambient occlusion should darken.
    //
    // Leaving it unoccluded is why AO read as too subtle outdoors: the Mojave
    // at midday is dominated by the sun term, AO only ever multiplied `ambient`,
    // and every crevice still received the full 0.35 fill. Occluding the fill
    // (and not the cosine core) darkens contact regions without muddying
    // surfaces that genuinely face the sun.
    const float directCore = ndotl * 0.65;
    const float directFill = max(wrappedNdotL - directCore, 0.0);
    const float occludedWrap = directCore + (directFill * ambientOcclusion);
    const float3 direct =
        camera.sunColorShadow.xyz *
        (sunIntensity * occludedWrap * directShadowFactor * (interiorMode ? 0.24 : 0.62));
    const float3 diffuseLight = (ambient * ambientOcclusion) + voxelGi + direct + importedLights;

    if (usePbr)
    {
        // Sun specular uses the same radiance and shadow term the diffuse sun
        // does, minus the wrapped-NdotL softening (that wrap is a diffuse
        // stylization; a specular lobe wrapped past the terminator reads wrong).
        const float3 sunRadiance =
            camera.sunColorShadow.xyz *
            (sunIntensity * directShadowFactor * (interiorMode ? 0.24 : 0.62));
        // kPbrDiffuseConventionScale converts pbrDirectSpecular's normalized BRDF
        // into the same units as the `direct` diffuse term above, which omits the
        // Lambert 1/pi. Without it a metal loses all of its diffuse to
        // pbrDiffuseScale() and gets back only 1/pi of the specular that should
        // replace it, so it renders ~3x dark. See pbr.slang for the derivation.
        specular += sunRadiance * kPbrDiffuseConventionScale *
                    pbrDirectSpecular(surface, albedo, normal, viewDir, -sunDirection);

        // Ambient/GI specular so metals still read as metal with no analytic
        // light on them. Fed by the same irradiance the diffuse term uses --
        // which is exactly why this one takes NO kPbrDiffuseConventionScale: the
        // split-sum factor is a reflectance, and being handed irradiance instead
        // of radiance already carries the pi. See pbr.slang.
        const float ndotv = saturate(dot(normal, viewDir));
        specular += pbrAmbientSpecular(
            surface,
            albedo,
            ((ambient * ambientOcclusion) + voxelGi),
            ndotv);
    }

    const float rim = pow(1.0 - saturate(dot(normal, viewDir)), 2.0);
    // The rim term is a stand-in Fresnel for surfaces with no real material;
    // PBR surfaces get a physical one from pbrFresnelSchlick instead.
    const float rimWeight = usePbr ? 0.0 : (interiorMode ? 0.010 : 0.022);
    // Emissive is added after the lighting sum, not folded into albedo: it is
    // radiance the surface produces, so it must not be scaled by incoming light
    // or by pbrDiffuseScale (a fully metallic emitter still glows).
    float3 shaded =
        ((diffuseLight * albedo * pbrDiffuseScale(surface.metallic)) + specular +
         (float3(1.0, 0.98, 0.94) * (rim * rimWeight)) + materialEmissive);

    if (!interiorMode)
    {
        shaded = applyAerialPerspective(shaded, input.inWorldPosition, -viewDir);
    }

    // Fog-of-war visibility blend. Only applied when the fog map is active
    // (strategy map mode). The fog texture is a blurred per-tile R8 map where
    // 0=hidden, ~0.39=explored, 1=visible. Bilinear sampling gives smooth
    // gradients at tile boundaries. Animated noise at mid-values creates a
    // wispy, rolling-clouds effect at the explored/hidden boundary.
    if (camera.fogMapConfig.w > 0.5)
    {
        const float2 fogUv = float2(
            input.inWorldPosition.x * camera.fogMapConfig.x,
            input.inWorldPosition.z * camera.fogMapConfig.y
        );
        const float fogRaw = bindlessTextures[kBindlessIndexFogMap].Sample(fogUv).x;

        // Animated noise: strongest at the mid-range boundary zone.
        const float time = camera.skyConfig1.z;
        const float wx = input.inWorldPosition.x;
        const float wz = input.inWorldPosition.z;
        const float noise =
            sin(wx * 0.017 + time * 0.008) * cos(wz * 0.015 - time * 0.006) * 0.5
            + sin(wx * 0.031 - time * 0.005) * cos(wz * 0.027 + time * 0.007) * 0.5;
        // Mask noise to the transitional zone (roughly 0.1–0.7 in fogRaw).
        const float boundaryMask =
            smoothstep(0.05, 0.45, fogRaw) * (1.0 - smoothstep(0.45, 0.80, fogRaw));
        const float fogFactor = saturate(fogRaw + boundaryMask * noise * 0.22);

        // Color zones:
        //   fogFactor ≈ 0.00 → dark animated shroud (hidden, incl. terrain near boundary)
        //   fogFactor ≈ 0.39 → cool desaturated blue-grey (explored, not visible)
        //   fogFactor ≈ 1.00 → full shaded terrain color (visible)
        const float grayscale = dot(shaded, float3(0.299, 0.587, 0.114));
        const float3 exploredColor =
            float3(grayscale * 0.50, grayscale * 0.55, grayscale * 0.68) * 0.50;
        const float3 shroudColor = float3(0.030, 0.030, 0.048);

        float3 foggedColor;
        if (fogFactor < 0.30)
        {
            // Hidden zone: blend shroud to explored.
            foggedColor = lerp(shroudColor, exploredColor, fogFactor / 0.30);
        }
        else if (fogFactor < 0.72)
        {
            // Explored zone: blend explored to full visible.
            foggedColor = lerp(exploredColor, shaded, (fogFactor - 0.30) / 0.42);
        }
        else
        {
            foggedColor = shaded;
        }
        return float4(foggedColor, albedoFull.a);
    }

    return float4(shaded, albedoFull.a);
}
