module camera_uniform;

public struct CameraUniform
{
    public float4x4 mvp;
    public float4x4 view;
    public float4x4 proj;
    public float4x4 lightViewProj[4];
    public float4 shadowCascadeSplits;
    public float4 shadowAtlasUvRects[4];
    public float4 sunDirectionIntensity;
    public float4 sunColorShadow;
    public float4 shIrradiance[9];
    public float4 shadowConfig0;
    public float4 shadowConfig1;
    public float4 shadowConfig2;
    public float4 shadowConfig3;
    public float4 shadowVoxelGridOrigin;
    public float4 shadowVoxelGridSize;
    public float4 skyConfig0;
    public float4 skyConfig1;
    public float4 skyConfig2;
    public float4 skyConfig3;
    public float4 skyConfig4;
    public float4 skyConfig5;
    public float4 colorGrading0;
    public float4 colorGrading1;
    public float4 voxelBaseColorPalette[16];
    public float4 voxelGiGridOriginCellSize;
    public float4 voxelGiGridExtentStrength;
};

[[vk::binding(0, 0)]]
public ConstantBuffer<CameraUniform> camera;
