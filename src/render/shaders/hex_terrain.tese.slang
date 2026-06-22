// Hex-terrain tessellation evaluation (domain) stage. Interpolates the continuous
// base surface (planar within each fan sector, from the control points' base heights)
// and adds smooth per-tile detail (rolling fbm or stepped strip-mine terraces),
// windowed to the hex interior so it reaches 0 before any shared edge. The base plane
// normal comes from the patch geometry; the (gently sampled) detail gradient perturbs
// it. Also emits a distance-to-hex-edge so the fragment can draw borders that ride the
// displaced surface.
import camera_uniform;
import noise;

struct HexControl
{
    float3 worldPos : TEXCOORD0;
    float2 centerXZ : TEXCOORD1;
    float4 detailParams : TEXCOORD2;
    uint classFlags : TEXCOORD3;
    float hexSize : TEXCOORD4;
    uint neighborTerrainPacked : TEXCOORD5;
};

struct PatchConstants
{
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
};

struct DomainOutput
{
    float4 position : SV_Position;
    float3 outWorldPosition : TEXCOORD0;
    float3 outWorldNormal : TEXCOORD1;
    nointerpolation uint outClassFlags : TEXCOORD2;
    float outSlope : TEXCOORD3;
    float outEdgeDistance : TEXCOORD4;  // world-space distance to the nearest hex edge
    nointerpolation uint outNeighborTerrainPacked : TEXCOORD5;
    nointerpolation float2 outCenterXZ : TEXCOORD6;  // tile center for local-coord computation in frag
    nointerpolation float outHexSize : TEXCOORD7;
};

// Noise primitives (hashNoise / valueNoise / fbm2) live in the shared `noise`
// module so this displacement path and the CPU prop-placement twin stay in sync.

// Distance from a local-XZ point (relative to the tile center) to the nearest hex edge.
// A pointy-top hexagon is the intersection of six half-planes whose outward normals
// point through the edge midpoints at 60-degree steps; the apothem is hexSize*sqrt3/2.
float hexEdgeDistance(float2 local, float hexSize)
{
    const float apothem = hexSize * 0.8660254;
    float maxProj = -1e9;
    [unroll]
    for (int i = 0; i < 6; ++i)
    {
        const float ang = radians(60.0 * float(i));
        maxProj = max(maxProj, dot(local, float2(cos(ang), sin(ang))));
    }
    return apothem - maxProj;  // >= 0 inside, 0 on an edge
}

// Per-tile detail, windowed so it is exactly 0 by detailParams.z (< inradius) from the
// tile center. A pure function of world XZ in the interior, so it stays continuous.
float detailHeight(float2 worldXz, float2 centerXz, float4 detailParams)
{
    const float windowEnd = detailParams.z;
    if (windowEnd <= 0.0)
    {
        return 0.0;
    }
    const float dist = length(worldXz - centerXz);
    const float window = 1.0 - smoothstep(windowEnd * 0.35, windowEnd, dist);
    const uint feature = (uint)(detailParams.x + 0.5);
    const float amplitude = detailParams.y;
    const float noiseFreq = detailParams.w;

    float displacement;
    if (feature == 2u)  // HexDetail_StripMine: concentric descending terraces.
    {
        const float r = saturate(dist / windowEnd);
        const float steps = 5.0;
        const float terrace = floor((1.0 - r) * steps) / steps;
        displacement = -terrace * amplitude;
    }
    else  // HexDetail_Rough: smooth low-frequency fbm.
    {
        displacement = fbm2(worldXz * noiseFreq) * amplitude;
    }
    return displacement * window;
}

// --- Cliff profile reshaping -------------------------------------------------
// By default the continuous base within a fan sector is a linear barycentric blend
// from the tile-centre height to the (neighbour-averaged) corner heights, so any
// drop to a lower neighbour reads as a gentle ramp across the whole tile. Reshaping
// warps the RADIAL profile: the height clings to the centre value across the tile
// interior, then falls within a narrow band at the rim, turning that ramp into a
// defined escarpment. Crack-free is preserved exactly -- the height on the shared
// outer edge (radial parameter t == 1) is untouched, and the rim steepness is a
// per-corner quantity (from each shared corner's height delta) so adjacent sectors
// agree along their shared radial edge. At a flat boundary the exponent is 1, which
// reproduces the original linear blend, so flat terrain is unchanged.

static const float kCliffMaxExponent = 2.0;  // radial-profile pow() exponent at a full cliff
static const float kCliffDeltaScale = 0.90;  // |centre-corner| / (hexSize*this) -> full strength

// Map a centre->corner height delta to the radial-profile exponent (1 = linear ramp,
// higher = flatter plateau top with a steeper rim).
float cliffExponent(float heightDelta, float hexSize)
{
    const float strength = saturate(abs(heightDelta) / max(hexSize * kCliffDeltaScale, 1e-3));
    return lerp(1.0, kCliffMaxExponent, strength);
}

struct SectorBasis
{
    float2 centerXZ;
    float centerH;
    float2 v1XZ;
    float v1H;
    float2 v2XZ;
    float v2H;
    float hexSize;
};

// Warped continuous base height at world-XZ point p inside one fan sector triangle
// (centre, v1, v2). A pure function of p given the sector, so it can be central-
// differenced for the surface normal just like detailHeight().
float cliffBaseHeight(SectorBasis sb, float2 p)
{
    // Barycentric of p: p = centre + b1*(v1-centre) + b2*(v2-centre).
    const float2 e1 = sb.v1XZ - sb.centerXZ;
    const float2 e2 = sb.v2XZ - sb.centerXZ;
    const float2 ep = p - sb.centerXZ;
    const float den = (e1.x * e2.y) - (e1.y * e2.x);
    if (abs(den) < 1e-6)
    {
        return sb.centerH;
    }
    const float b1 = ((ep.x * e2.y) - (ep.y * e2.x)) / den;
    const float b2 = ((e1.x * ep.y) - (e1.y * ep.x)) / den;

    const float t = saturate(b1 + b2);  // 0 at the centre, 1 on the outer (shared) edge
    if (t < 1e-5)
    {
        return sb.centerH;
    }
    const float s = saturate(b2 / t);   // angular position between v1 and v2 along the rim

    const float edgeH = lerp(sb.v1H, sb.v2H, s);
    // Per-corner steepness so the two sectors sharing a radial edge agree on it there.
    const float k1 = cliffExponent(sb.centerH - sb.v1H, sb.hexSize);
    const float k2 = cliffExponent(sb.centerH - sb.v2H, sb.hexSize);
    const float k = lerp(k1, k2, s);
    const float w = pow(t, k);          // w(1) == 1 for any k -> shared-edge height preserved
    return lerp(sb.centerH, edgeH, w);
}

[domain("tri")]
[partitioning("fractional_odd")]
DomainOutput main(
    PatchConstants patchConstants,
    const OutputPatch<HexControl, 3> patch,
    float3 bary : SV_DomainLocation)
{
    // Continuous base: barycentric blend of the control points' (XZ, baseHeight).
    const float3 basePos =
        (patch[0].worldPos * bary.x) + (patch[1].worldPos * bary.y) + (patch[2].worldPos * bary.z);
    const float2 centerXz = patch[0].centerXZ;          // uniform across the patch
    const float4 detailParams = patch[0].detailParams;  // uniform across the patch
    const float hexSize = max(patch[0].hexSize, 1.0);

    // Warped continuous base height (reshaped so cliffs read as defined rims rather
    // than gentle ramps) and its gradient via central differences. A finer step than
    // the detail uses so the steep rim band keeps a crisp normal.
    SectorBasis sb;
    sb.centerXZ = patch[0].worldPos.xz;
    sb.centerH = patch[0].worldPos.y;
    sb.v1XZ = patch[1].worldPos.xz;
    sb.v1H = patch[1].worldPos.y;
    sb.v2XZ = patch[2].worldPos.xz;
    sb.v2H = patch[2].worldPos.y;
    sb.hexSize = hexSize;

    const float baseH = cliffBaseHeight(sb, basePos.xz);
    const float bs = hexSize * 0.06;
    const float baseGradX =
        (cliffBaseHeight(sb, basePos.xz + float2(bs, 0.0)) -
         cliffBaseHeight(sb, basePos.xz - float2(bs, 0.0))) / (2.0 * bs);
    const float baseGradZ =
        (cliffBaseHeight(sb, basePos.xz + float2(0.0, bs)) -
         cliffBaseHeight(sb, basePos.xz - float2(0.0, bs))) / (2.0 * bs);

    // Detail displacement and its slope via wide central differences (a step near the
    // noise wavelength keeps normals smooth instead of aliasing into jagged spikes).
    const float detail = detailHeight(basePos.xz, centerXz, detailParams);
    const float s = hexSize * 0.14;
    const float detailGradX =
        (detailHeight(basePos.xz + float2(s, 0.0), centerXz, detailParams) -
         detailHeight(basePos.xz - float2(s, 0.0), centerXz, detailParams)) / (2.0 * s);
    const float detailGradZ =
        (detailHeight(basePos.xz + float2(0.0, s), centerXz, detailParams) -
         detailHeight(basePos.xz - float2(0.0, s), centerXz, detailParams)) / (2.0 * s);

    float3 normal = normalize(float3(
        -(baseGradX + detailGradX), 1.0, -(baseGradZ + detailGradZ)));
    // The six fan triangles meet at the tile centre, and each derives its base-height
    // gradient from its OWN wedge's two corners, so the normal handed to the fragment is
    // genuinely multivalued at the apex -- the dark dot at every tile centre. (It is NOT
    // SSAO: hex terrain is not in the normal-depth prepass, so AO is forced to 1.0 over
    // land.) A narrow blend-to-up only moved the discontinuity into a ring/seam. Instead,
    // shade the whole tile INTERIOR with a single stable up normal and reserve the gradient
    // (sloped) normal for the rim: the strategy map is near-top-down, so flat interior
    // lighting reads cleanly and cliffs still show at tile edges. This removes the apex
    // singularity as a shaded location entirely -- every fragment near the centre is up.
    const float distFromCenter = length(basePos.xz - centerXz);
    const float rimT = smoothstep(hexSize * 0.55, hexSize * 0.85, distFromCenter);
    normal = normalize(lerp(float3(0.0, 1.0, 0.0), normal, rimT));
    const float3 worldPosition = float3(basePos.x, baseH + detail, basePos.z);

    DomainOutput output;
    output.outWorldPosition = worldPosition;
    output.outWorldNormal = normal;
    output.outClassFlags = patch[0].classFlags;
    output.outSlope = saturate(1.0 - normal.y);
    output.outEdgeDistance = hexEdgeDistance(basePos.xz - centerXz, hexSize);
    output.outNeighborTerrainPacked = patch[0].neighborTerrainPacked;
    output.outCenterXZ = centerXz;
    output.outHexSize = hexSize;
    output.position = mul(camera.mvp, float4(worldPosition, 1.0));
    return output;
}
