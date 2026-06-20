// Hex-terrain fragment stage. v1 shades from the terrain-class palette (matching the
// map's existing vertex-color look) plus a slope rock-blend, then standard sun + SH
// ambient lighting. Outputs linear color into the HDR target; the tonemap pass does
// the sRGB encode (same convention as terrain_heightmap.frag / imported_static.frag).
// (Sampling the terrain_*.png textures by class is a deferred enhancement.)
import camera_uniform;
import sh_lighting;

struct FragmentInput
{
    float3 inWorldPosition : TEXCOORD0;
    float3 inWorldNormal : TEXCOORD1;
    nointerpolation uint inClassFlags : TEXCOORD2;
    float inSlope : TEXCOORD3;
};

// Mirrors odai::game::terrainColor (strategy_map.cc) so the displaced terrain keeps
// the established palette.
float3 terrainPalette(uint terrain)
{
    switch (terrain)
    {
        case 0u:  return float3(0.10, 0.24, 0.50);  // Ocean
        case 1u:  return float3(0.20, 0.46, 0.68);  // Coast
        case 2u:  return float3(0.32, 0.54, 0.24);  // Grassland
        case 3u:  return float3(0.56, 0.60, 0.30);  // Plains
        case 4u:  return float3(0.18, 0.38, 0.20);  // Forest
        case 5u:  return float3(0.13, 0.45, 0.16);  // Jungle
        case 6u:  return float3(0.46, 0.42, 0.26);  // Hills
        case 7u:  return float3(0.46, 0.44, 0.43);  // Mountains
        case 8u:  return float3(0.82, 0.73, 0.44);  // Desert
        case 9u:  return float3(0.54, 0.57, 0.50);  // Tundra
        case 10u: return float3(0.90, 0.92, 0.96);  // Snow
        default:  return float3(1.0, 0.0, 1.0);
    }
}

float4 main(FragmentInput input) : SV_Target0
{
    const uint terrain = input.inClassFlags & 0xFFu;
    const uint flags = (input.inClassFlags >> 8u) & 0xFFu;

    float3 albedo = terrainPalette(terrain);
    // River/road overlays, matching the mesher's tileTopColor blends.
    if ((flags & 0x1u) != 0u)  // TileFlag_River
    {
        albedo = lerp(albedo, float3(0.18, 0.42, 0.74), 0.55);
    }
    if ((flags & 0x2u) != 0u)  // TileFlag_Road
    {
        albedo = lerp(albedo, float3(0.50, 0.38, 0.24), 0.55);
    }

    // Steep faces read as exposed rock; near-vertical as lighter cliff stone.
    const float3 rock = float3(0.40, 0.39, 0.36);
    const float3 cliff = float3(0.52, 0.51, 0.47);
    albedo = lerp(albedo, rock, smoothstep(0.30, 0.62, input.inSlope) * 0.65);
    albedo = lerp(albedo, cliff, smoothstep(0.62, 0.90, input.inSlope) * 0.5);

    const float3 normal = normalize(input.inWorldNormal);
    const float3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const float3 sunColor = camera.sunColorShadow.xyz * float3(1.05, 0.95, 0.88);
    const float ndotl = max(dot(normal, -sunDirection), 0.0);
    const float3 shAmbient = evaluateShHemisphereIrradiance(normal) * float3(0.78, 0.86, 1.08) * 0.72;
    const float3 direct = sunColor * (sunIntensity * ndotl);

    const float3 color = albedo * (shAmbient + direct);
    return float4(color, 1.0);
}
