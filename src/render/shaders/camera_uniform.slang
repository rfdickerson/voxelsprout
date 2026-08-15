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
    public float4 weatherHorizon;
    public float4 weatherFog;
    // Four planar cloud layers: tint.rgb + opacity in w, then
    // (bindless slot, scroll speed, tiling, unused).
    public float4 weatherCloudTint[4];
    public float4 weatherCloudParams[4];
    // (mode, contrast, saturation, curve knee) and (overbright dampening, ...).
    public float4 tonemapConfig;
    public float4 tonemapConfig2;
    // Terrain layer-blend shaping; see renderer_shared.h for the channel map.
    // (sharpness, coarse noise units, fine noise units, noise amount).
    public float4 terrainBlendConfig;
    // (white point, shoulder strength, unused, unused); see renderer_shared.h.
    public float4 hdrHighlightConfig;
};

[[vk::binding(0, 0)]]
public ConstantBuffer<CameraUniform> camera;
