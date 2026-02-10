import camera_uniform;
import sh_lighting;

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
    const float phaseBoost = 1.0 + (phaseMie * 1.6);
    const float3 inscatterColor = skyColor * phaseBoost;

    const float3 transmittance = exp(-extinction * max(viewDepth, 0.0));
    return (litColor * transmittance) + (inscatterColor * (float3(1.0, 1.0, 1.0) - transmittance));
}

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

    const float3 plantNormal = float3(0.0, 1.0, 0.0);
    const float3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const float3 sunColor = camera.sunColorShadow.xyz;
    float3 quadNormal = normalize(input.inQuadNormal);
    if (dot(quadNormal, -sunDirection) < 0.0) {
        quadNormal = -quadNormal;
    }
    const float directionalWeight = clamp(camera.skyConfig2.z, 0.0, 1.0);
    const float3 lightingNormal = normalize(lerp(plantNormal, quadNormal, directionalWeight));
    const float wrappedNdotL = saturate((dot(lightingNormal, -sunDirection) + 0.40) / 1.40);
    const float transmission =
        (saturate(dot(plantNormal, sunDirection)) * 0.12) +
        (saturate(dot(-quadNormal, sunDirection)) * 0.10 * directionalWeight);
    const float canopyT = smoothstep(0.05, 0.95, saturate(input.inUv.y));
    const float3 skyAmbient = evaluateShIrradiance(float3(0.0, 1.0, 0.0)) * 0.33;
    const float3 groundAmbient = evaluateShIrradiance(float3(0.0, -1.0, 0.0)) * 0.14;
    const float3 ambient = lerp(groundAmbient, skyAmbient, canopyT);
    const float3 direct = sunColor * (sunIntensity * (wrappedNdotL + transmission) * 0.70);

    const float3 tint = clamp(input.inColorTint.xyz, float3(0.10, 0.10, 0.10), float3(2.50, 2.50, 2.50));
    const bool isFlower = spriteTile >= 5u;
    const float flowerTintStrength = isFlower ? 1.0 : 0.68;
    const float3 variedSprite = lerp(sprite.rgb, sprite.rgb * tint, flowerTintStrength);
    const float3 lit = variedSprite * (ambient + direct);
    const float viewDepth = max(-mul(camera.view, float4(input.inWorldPosition, 1.0)).z, 0.0);
    const float3 fogged = applyAtmosphericFog(lit, input.inWorldPosition, viewDepth, sunDirection);
    return float4(fogged, 1.0);
}
