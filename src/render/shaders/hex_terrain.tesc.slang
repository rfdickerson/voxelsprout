// Hex-terrain tessellation control (hull) stage. Triangle patches; distance-based
// LOD. Crack-free across tiles: each edge factor is computed only from that edge's two
// world-space endpoints, so two tiles sharing a hex edge derive an identical factor.
import camera_uniform;

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

float edgeFactor(float3 a, float3 b, float hexSize)
{
    const float hs = max(hexSize, 1.0);
    const float2 mid = (a.xz + b.xz) * 0.5;
    const float3 cam = camera.shadowVoxelGridSize.xyz;
    const float dist = length(float2(mid.x - cam.x, mid.y - cam.z));
    const float nearTess = 10.0;
    const float farTess = 1.0;
    const float t = smoothstep(hs * 3.0, hs * 30.0, dist);
    return clamp(lerp(nearTess, farTess, t), farTess, nearTess);
}

PatchConstants patchConstants(InputPatch<HexControl, 3> patch)
{
    const float hexSize = patch[0].hexSize;
    PatchConstants constants;
    // HLSL triangle convention: edge i lies opposite control point i.
    constants.edges[0] = edgeFactor(patch[1].worldPos, patch[2].worldPos, hexSize);
    constants.edges[1] = edgeFactor(patch[2].worldPos, patch[0].worldPos, hexSize);
    constants.edges[2] = edgeFactor(patch[0].worldPos, patch[1].worldPos, hexSize);
    constants.inside = max(max(constants.edges[0], constants.edges[1]), constants.edges[2]);
    return constants;
}

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("patchConstants")]
HexControl main(InputPatch<HexControl, 3> patch, uint controlPointId : SV_OutputControlPointID)
{
    return patch[controlPointId];
}
