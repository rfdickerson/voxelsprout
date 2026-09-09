struct GpuImportedMaterial
{
    float4 baseColorMetallic;  // rgb = base color tint, w = metallic
    float4 emissiveRoughness;  // rgb = emissive radiance, w = roughness
    float4 environment;
    float4 specularStrength;
    uint4 textures;
    uint4 sourceFlags;
    float4 animationUv;
    float4 animationColor;
    float4 animationPalette;
    uint4 animationState;
};

[[vk::binding(13, 0)]]
StructuredBuffer<GpuImportedMaterial> importedMaterials;

