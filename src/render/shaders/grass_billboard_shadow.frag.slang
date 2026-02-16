[[vk::binding(1, 0)]]
Sampler2D<float4> diffuseTexture;

static const uint kVoxelAtlasTileSize = 16u;
static const uint kGrassSpriteTile = 4u;

struct FragmentInput
{
    float2 inUv : TEXCOORD0;
};

float sampleGrassAlpha(float2 uv)
{
    uint texW = 0u;
    uint texH = 0u;
    diffuseTexture.GetDimensions(texW, texH);
    const uint tilesX = max(texW / kVoxelAtlasTileSize, 1u);
    const uint tilesY = max(texH / kVoxelAtlasTileSize, 1u);
    const uint tileX = kGrassSpriteTile % tilesX;
    const uint tileY = kGrassSpriteTile / tilesX;
    const float inset = 0.5 / float(kVoxelAtlasTileSize);
    const float2 uvInset = lerp(float2(inset, inset), float2(1.0 - inset, 1.0 - inset), saturate(uv));
    const float2 atlasUv = (float2(float(tileX), float(tileY)) + uvInset) / float2(float(tilesX), float(tilesY));
    return diffuseTexture.Sample(atlasUv).a;
}

void main(FragmentInput input)
{
    if (sampleGrassAlpha(input.inUv) < 0.50)
    {
        discard;
    }
}
