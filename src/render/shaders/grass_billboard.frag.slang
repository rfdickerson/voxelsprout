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
    float3 inColorTint : TEXCOORD1;
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

    const float3 plantNormal = float3(0.0, 1.0, 0.0);
    const float3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const float3 sunColor = camera.sunColorShadow.xyz;
    const float wrappedNdotL = saturate((dot(plantNormal, -sunDirection) + 0.40) / 1.40);
    const float transmission = saturate(dot(plantNormal, sunDirection)) * 0.18;
    const float3 ambient = evaluateShIrradiance(plantNormal) * 0.30;
    const float3 direct = sunColor * (sunIntensity * (wrappedNdotL + transmission) * 0.70);

    const float3 variedSprite = sprite.rgb * clamp(input.inColorTint, float3(0.7, 0.7, 0.7), float3(1.3, 1.3, 1.3));
    const float3 lit = variedSprite * (ambient + direct);
    return float4(lit, 1.0);
}
