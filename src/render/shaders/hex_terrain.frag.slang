// Hex-terrain fragment stage. Samples the per-terrain ground texture (bindless slot
// packed into classFlags bits 16-31; palette fallback when 0xFFFF), blends exposed rock
// on steep slopes, darkens a thin band along each hex edge so borders ride the displaced
// surface, then applies sun + SH ambient lighting. Outputs linear color into the HDR
// target; the tonemap pass does the sRGB encode (same convention as the other passes).
import camera_uniform;
import sh_lighting;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

static const uint kHexNoTexture = 0xFFFFu;
static const float kHexTexTiling = 1.0 / 48.0;  // world units per texture repeat
static const float kHexBorderWidth = 2.0;       // world units of the dark edge band

struct FragmentInput
{
    float3 inWorldPosition : TEXCOORD0;
    float3 inWorldNormal : TEXCOORD1;
    nointerpolation uint inClassFlags : TEXCOORD2;
    float inSlope : TEXCOORD3;
    float inEdgeDistance : TEXCOORD4;
    nointerpolation uint inNeighborTerrainPacked : TEXCOORD5;
    nointerpolation float2 inCenterXZ : TEXCOORD6;
    nointerpolation float inHexSize : TEXCOORD7;
};

// Mirrors odai::game::terrainColor (strategy_map.cc); used when a tile has no texture.
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
    const uint texSlot = (input.inClassFlags >> 16u) & 0xFFFFu;

    float3 albedo = terrainPalette(terrain);
    if (texSlot != kHexNoTexture)
    {
        const float2 uv = input.inWorldPosition.xz * kHexTexTiling;
        // Sample with an explicit distance-based gradient rather than Sample()'s implicit
        // screen-space derivatives. At the hex fan's centre vertex six tessellated wedges
        // meet; the rasteriser's 2x2-quad derivatives degenerate there (sub-pixel
        // triangles), producing a garbage mip LOD that sampled the texture as black -- the
        // dark dot at every tile centre. An always-finite gradient keeps mip selection
        // stable (and is plenty for a near-top-down strategy view).
        const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
        const float viewDist = max(length(cameraWorld - input.inWorldPosition), 1.0);
        const float worldPerPixel = viewDist * 0.00077;  // ~1080p / 45-deg-FOV heuristic
        const float2 du = float2(worldPerPixel * kHexTexTiling, 0.0);
        const float2 dv = float2(0.0, worldPerPixel * kHexTexTiling);
        const float3 sampled = bindlessTextures[texSlot].SampleGrad(uv, du, dv).rgb;
        // Texture supplies the surface detail; keep a touch of palette identity so the
        // terrain type still reads at a glance.
        albedo = lerp(sampled, sampled * terrainPalette(terrain) * 1.6, 0.25);
    }

    // Blend toward the nearest neighbor's terrain color near each hex edge. This
    // eliminates the hard color cut between terrain types (Civ 6-style splatting).
    // The blend only affects color; geometry stays crack-free via the continuous base.
    {
        const float2 local = input.inWorldPosition.xz - input.inCenterXZ;
        int nearestEdge = 0;
        float maxProj = dot(local, float2(1.0, 0.0));  // edge 0 = angle 0°
        [unroll]
        for (int i = 1; i < 6; ++i)
        {
            const float ang = 1.0471975512 * float(i);  // i * (pi/3)
            const float p = dot(local, float2(cos(ang), sin(ang)));
            nearestEdge = (p > maxProj) ? i : nearestEdge;
            maxProj = max(maxProj, p);
        }
        const uint neighborType = (input.inNeighborTerrainPacked >> (uint(nearestEdge) * 4u)) & 0xFu;
        const float3 neighborAlbedo = terrainPalette(neighborType);
        const float hexInradius = input.inHexSize * 0.8660254;
        const float blendT = 1.0 - smoothstep(hexInradius * 0.72, hexInradius * 0.92, input.inEdgeDistance);
        albedo = lerp(albedo, neighborAlbedo, blendT * 0.5);
    }

    // River/road overlays, matching the mesher's tileTopColor blends.
    if ((flags & 0x1u) != 0u)  // TileFlag_River
    {
        albedo = lerp(albedo, float3(0.18, 0.42, 0.74), 0.45);
    }
    if ((flags & 0x2u) != 0u)  // TileFlag_Road
    {
        albedo = lerp(albedo, float3(0.50, 0.38, 0.24), 0.45);
    }

    // Steep faces read as exposed rock; near-vertical as lighter cliff stone.
    const float3 rock = float3(0.40, 0.39, 0.36);
    const float3 cliff = float3(0.52, 0.51, 0.47);
    albedo = lerp(albedo, rock, smoothstep(0.35, 0.66, input.inSlope) * 0.6);
    albedo = lerp(albedo, cliff, smoothstep(0.66, 0.92, input.inSlope) * 0.5);

    // Territory border: a thin dark band only on tiles flagged as border tiles.
    // Natural terrain transitions are handled by the neighbor blend above, not a ring.
    if ((flags & 0x4u) != 0u)  // TileFlag_Border
    {
        const float borderMask = 1.0 - smoothstep(0.0, kHexBorderWidth, max(input.inEdgeDistance, 0.0));
        albedo *= lerp(1.0, 0.55, borderMask);
    }

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
