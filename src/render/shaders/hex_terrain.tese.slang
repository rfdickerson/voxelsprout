// Hex-terrain tessellation evaluation (domain) stage. Interpolates the continuous
// base surface (planar within each fan sector, from the control points' base heights)
// and adds per-tile detail (fbm roughness or stepped strip-mine terraces), windowed to
// the hex interior so it reaches 0 before any shared edge. The base plane normal comes
// from the patch geometry; the detail gradient perturbs it.
import camera_uniform;

struct HexControl
{
    float3 worldPos : TEXCOORD0;
    float2 centerXZ : TEXCOORD1;
    float4 detailParams : TEXCOORD2;
    uint classFlags : TEXCOORD3;
    float hexSize : TEXCOORD4;
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
};

float hashNoise(float2 p)
{
    return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

float valueNoise(float2 p)
{
    const float2 i = floor(p);
    const float2 f = frac(p);
    const float2 u = f * f * (3.0 - (2.0 * f));
    const float a = hashNoise(i);
    const float b = hashNoise(i + float2(1.0, 0.0));
    const float c = hashNoise(i + float2(0.0, 1.0));
    const float d = hashNoise(i + float2(1.0, 1.0));
    return (lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y) * 2.0) - 1.0;
}

float fbm(float2 p)
{
    return
        (valueNoise(p) * 0.52) +
        (valueNoise((p * 2.03) + float2(19.3, -8.7)) * 0.30) +
        (valueNoise((p * 4.11) + float2(-31.1, 24.6)) * 0.18);
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
    else  // HexDetail_Rough: global fbm so it stays continuous across the interior.
    {
        displacement = fbm(worldXz * noiseFreq) * amplitude;
    }
    return displacement * window;
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

    // Base plane normal from the patch geometry (the base is planar within a sector).
    float3 baseNormal = normalize(cross(
        patch[1].worldPos - patch[0].worldPos,
        patch[2].worldPos - patch[0].worldPos));
    if (baseNormal.y < 0.0)
    {
        baseNormal = -baseNormal;
    }
    const float baseGradX = (baseNormal.y > 1e-4) ? (-baseNormal.x / baseNormal.y) : 0.0;
    const float baseGradZ = (baseNormal.y > 1e-4) ? (-baseNormal.z / baseNormal.y) : 0.0;

    // Detail displacement and its slope via central differences.
    const float detail = detailHeight(basePos.xz, centerXz, detailParams);
    const float s = hexSize * 0.05;
    const float detailGradX =
        (detailHeight(basePos.xz + float2(s, 0.0), centerXz, detailParams) -
         detailHeight(basePos.xz - float2(s, 0.0), centerXz, detailParams)) / (2.0 * s);
    const float detailGradZ =
        (detailHeight(basePos.xz + float2(0.0, s), centerXz, detailParams) -
         detailHeight(basePos.xz - float2(0.0, s), centerXz, detailParams)) / (2.0 * s);

    const float3 normal = normalize(float3(
        -(baseGradX + detailGradX), 1.0, -(baseGradZ + detailGradZ)));
    const float3 worldPosition = float3(basePos.x, basePos.y + detail, basePos.z);

    DomainOutput output;
    output.outWorldPosition = worldPosition;
    output.outWorldNormal = normal;
    output.outClassFlags = patch[0].classFlags;
    output.outSlope = saturate(1.0 - normal.y);
    output.position = mul(camera.mvp, float4(worldPosition, 1.0));
    return output;
}
