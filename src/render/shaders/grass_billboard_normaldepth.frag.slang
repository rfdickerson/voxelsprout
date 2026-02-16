import camera_uniform;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

static const uint kBindlessTextureIndexDiffuseAtlas = 6u;
static const uint kVoxelAtlasTileSize = 16u;
static const uint kGrassSpriteTile = 4u;

struct FragmentInput
{
    float2 inUv : TEXCOORD0;
    float4 inColorTint : TEXCOORD1;
    float3 inWorldPosition : TEXCOORD2;
    float3 inQuadNormal : TEXCOORD3;
};

float4 sampleGrassSprite(float2 uv, uint tileIndex)
{
    uint texW = 0u;
    uint texH = 0u;
    bindlessTextures[kBindlessTextureIndexDiffuseAtlas].GetDimensions(texW, texH);
    const uint tilesX = max(texW / kVoxelAtlasTileSize, 1u);
    const uint tilesY = max(texH / kVoxelAtlasTileSize, 1u);
    const uint tileCount = max(tilesX * tilesY, 1u);
    const uint clampedTile = min(tileIndex, tileCount - 1u);
    const uint tileX = clampedTile % tilesX;
    const uint tileY = clampedTile / tilesX;
    const float inset = 0.5 / float(kVoxelAtlasTileSize);
    const float2 uvInset = lerp(float2(inset, inset), float2(1.0 - inset, 1.0 - inset), saturate(uv));
    const float2 atlasUv = (float2(float(tileX), float(tileY)) + uvInset) / float2(float(tilesX), float(tilesY));
    return bindlessTextures[kBindlessTextureIndexDiffuseAtlas].Sample(atlasUv);
}

float4 main(FragmentInput input) : SV_Target0
{
    const uint spriteTile = uint(clamp(input.inColorTint.w, 0.0, 255.0) + 0.5);
    const float4 sprite = sampleGrassSprite(input.inUv, (spriteTile == 0u) ? kGrassSpriteTile : spriteTile);
    if (sprite.a < 0.50)
    {
        discard;
    }

    const float3 worldNormal = normalize(input.inQuadNormal);
    const float3 viewNormal = normalize(mul(camera.view, float4(worldNormal, 0.0)).xyz);
    // Treat cutout foliage as non-solid for SSAO accumulation/combine.
    const float viewDepth = 0.0;
    const float3 encodedNormal = (viewNormal * 0.5) + 0.5;
    return float4(encodedNormal, viewDepth);
}
