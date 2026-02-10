import camera_uniform;
import sh_lighting;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

static const uint kBindlessTextureIndexDiffuseAtlas = 0u;
static const uint kVoxelAtlasTileSize = 16u;
static const uint kGrassSpriteTile = 4u;

struct FragmentInput
{
    float2 inUv : TEXCOORD0;
    float3 inWorldPosition : TEXCOORD1;
};

float4 sampleGrassSprite(float2 uv)
{
    uint texW = 0u;
    uint texH = 0u;
    bindlessTextures[kBindlessTextureIndexDiffuseAtlas].GetDimensions(texW, texH);
    const uint tilesX = max(texW / kVoxelAtlasTileSize, 1u);
    const uint tilesY = max(texH / kVoxelAtlasTileSize, 1u);
    const uint tileX = kGrassSpriteTile % tilesX;
    const uint tileY = kGrassSpriteTile / tilesX;
    const float inset = 0.5 / float(kVoxelAtlasTileSize);
    const float2 uvInset = lerp(float2(inset, inset), float2(1.0 - inset, 1.0 - inset), saturate(uv));
    const float2 atlasUv = (float2(float(tileX), float(tileY)) + uvInset) / float2(float(tilesX), float(tilesY));
    return bindlessTextures[kBindlessTextureIndexDiffuseAtlas].Sample(atlasUv);
}

float4 main(FragmentInput input) : SV_Target0
{
    const float4 sprite = sampleGrassSprite(input.inUv);
    if (sprite.a < 0.50)
    {
        discard;
    }

    const float3 normal = normalize(float3(0.12, 0.98, 0.15));
    const float3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const float3 sunColor = camera.sunColorShadow.xyz;
    const float ndotl = max(dot(normal, -sunDirection), 0.0);
    const float3 ambient = evaluateShHemisphereIrradiance(normal) * 0.30;
    const float3 direct = sunColor * (sunIntensity * ndotl);

    const float3 lit = sprite.rgb * (ambient + direct);
    return float4(lit, 1.0);
}
