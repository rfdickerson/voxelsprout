module camera_uniform;

public struct CameraUniform
{
    public float4x4 mvp;
    public float4x4 view;
    public float4x4 proj;
    public float4x4 lightViewProj[4];
    public float4x4 invLightViewProj[4];
    public float4 shadowCascadeSplits;
    public float4 shadowAtlasUvRects[4];
    public float4 sunDirectionIntensity;
    public float4 sunColorShadow;
    public float4 shIrradiance[9];
    public float4 shadowConfig0;
    public float4 shadowConfig1;
    public float4 shadowConfig2;
    public float4 shadowConfig3;
    public float4 shadowConfig4;
    public float4 shadowVoxelGridOrigin;
    public float4 shadowVoxelGridSize;
    public float4 skyConfig0;
    public float4 skyConfig1;
    public float4 skyConfig2;
    public float4 skyConfig3;
    public float4 skyConfig4;
    public float4 skyConfig5;
    public float4 voxelGiRestirConfig0;
    public float4 voxelGiRestirConfig1;
    public float4 colorGrading0;
    public float4 colorGrading1;
    public float4 colorGrading2;
    public float4 colorGrading3;
    public float4 dofConfig;
    public float4 dofConfig2;
    public float4 waterConfig;
    public float4 importedLightPositionRadius[64];
    public float4 importedLightColorIntensity[64];
    public float4 importedLightConfig;
    public float4 voxelBaseColorPalette[16];
    public float4 voxelGiGridOriginCellSize;
    public float4 voxelGiGridExtentStrength;
    public float4 fogMapConfig; // x=invExtentX, y=invExtentZ, z=unused, w=enabled
    // Authored sky from a Fallout WTHR record; w of weatherSkyUpper is the
    // blend weight over the procedural sky, and 0 means "procedural only".
    // Mirrored in MvpUniform in src/render/renderer_shared.h.
    public float4 weatherSkyUpper;
    public float4 weatherSkyLower;
    // w: the weather's sun-glare scale, 1 when no weather is published.
    public float4 weatherHorizon;
    public float4 weatherFog;
    // Four cloud layers: tint.rgb + opacity in w, then
    // (bindless slot, drift u, scale, drift v), then
    // (band low dir.y, band high dir.y, mapping mode, unused).
    public float4 weatherCloudTint[4];
    public float4 weatherCloudParams[4];
    public float4 weatherCloudBand[4];
    // (mode, contrast, saturation, curve knee) and (overbright dampening, ...).
    public float4 tonemapConfig;
    public float4 tonemapConfig2;
    // Terrain layer-blend shaping; see renderer_shared.h for the channel map.
    // (sharpness, coarse noise units, fine noise units, noise amount).
    public float4 terrainBlendConfig;
    // (white point, shoulder strength, unused, unused); see renderer_shared.h.
    public float4 hdrHighlightConfig;
    // Clustered light culling; see light_clusters.slang.
    // (grid x, grid y, grid z, tile size px) and (slice scale, slice bias, 0, 0).
    public float4 lightClusterConfig0;
    public float4 lightClusterConfig1;
    // (one atlas texel in UV, unused x3). Replaces shadowMap.GetDimensions() at
    // every PCF site -- see renderer_shared.h for the measurement.
    public float4 shadowAtlasConfig;
    // Authored imported-interior CELL lighting; see renderer_shared.h.
    public float4 interiorAmbient;
    public float4 interiorDirectional;
    public float4 interiorFog;
    public float4 interiorFogRange;
    public float4x4 interiorPointShadowViewProj[210];
    public float4 interiorPointShadowLightIndices[16];
    // x = active slot count, y = face UV scale, z = atlas texel size,
    // w = enabled.
    public float4 interiorPointShadowParams;
    public float4x4 contactShadowInvView;
    // x/y = full-resolution contact mask extent, z = enabled, w = phase.
    public float4 contactShadowConfig;
    // x/y = quarter-resolution record extent, z = enabled, w = bounce scale.
    public float4 screenSpaceGiConfig;
    // x/y = legacy object/terrain roughness, z = metallic, w = enabled.
    public float4 importedPbrConfig;
    // x = dominant horizontal water plane in world Y, y = planar target valid.
    public float4 waterReflectionConfig;
};

[[vk::binding(0, 0)]]
public ConstantBuffer<CameraUniform> camera;
