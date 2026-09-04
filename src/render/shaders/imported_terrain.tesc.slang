import camera_uniform;

// Hull stage for imported (Bethesda LAND) terrain. The control points are
// imported_static.vert's outputs verbatim -- the vertex shader is shared with
// the untessellated pipeline, and its SV_Position is simply recomputed by the
// domain shader after displacement.
//
// Terrain posts are 128 world units apart, which at Bethesda's ~70 units per
// metre puts one triangle across nearly two metres of ground: close up, every
// ridge line is visibly a polyline. Tessellating near patches and smoothing
// them (Phong, in the domain shader) rounds those silhouettes without
// touching the meshes or the cache.
struct ControlPoint
{
    float4 position : SV_Position;
    float3 worldPosition : TEXCOORD0;
    float3 worldNormal : TEXCOORD1;
    // .rgb tint, .a authored vertex alpha. Must match
    // imported_static.vert's VertexOutput -- this stage sits between it and
    // the fragment shaders, so a narrower type here drops the alpha the
    // alpha test needs.
    float4 color : TEXCOORD2;
    float2 uv : TEXCOORD3;
    nointerpolation uint textureIndex : TEXCOORD4;
    nointerpolation uint flags : TEXCOORD5;
    nointerpolation uint4 layerTextures : TEXCOORD6;
    float4 layerWeights : TEXCOORD7;
};

struct PatchConstants
{
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
};

// Per EDGE, from that edge's own two endpoints and nothing else: the adjacent
// triangle computes the identical factor for the shared edge, which is what
// keeps the tessellation watertight. A per-patch factor would crack every
// boundary between a near patch and a far one.
static const uint kDistantLodTessellation = (1u << 10);

float edgeTessFactor(float3 a, float3 b, bool distantLod)
{
    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float distance = length(((a + b) * 0.5) - cameraWorld);
    // Full detail inside ~20 m, back to a passthrough triangle by ~140 m --
    // past that the original 128-unit grid is already subpixel. Factor 4, not
    // 6: triangle count grows with the SQUARE of the factor and this stage
    // runs twice per frame (merged prepass + main), and on the LNL iGPU the
    // step from 4 to 6 measured +2.5 ms for a difference the eye cannot find.
    const float nearTess = distantLod ? 5.0 : 4.0;
    const float farTess = 1.0;
    const float fadeStart = distantLod ? 9000.0 : 1500.0;
    const float fadeEnd = distantLod ? 48000.0 : 10000.0;
    return clamp(
        lerp(nearTess, farTess, smoothstep(fadeStart, fadeEnd, distance)), farTess, nearTess);
}

PatchConstants patchConstants(InputPatch<ControlPoint, 3> patch)
{
    PatchConstants constants;
    const bool distantLod = (patch[0].flags & kDistantLodTessellation) != 0u;
    // Edge i is opposite control point i (D3D/Vulkan triangle convention):
    // edge 0 runs between CPs 1 and 2.
    constants.edges[0] = edgeTessFactor(patch[1].worldPosition, patch[2].worldPosition, distantLod);
    constants.edges[1] = edgeTessFactor(patch[2].worldPosition, patch[0].worldPosition, distantLod);
    constants.edges[2] = edgeTessFactor(patch[0].worldPosition, patch[1].worldPosition, distantLod);
    constants.inside = max(max(constants.edges[0], constants.edges[1]), constants.edges[2]);
    return constants;
}

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("patchConstants")]
ControlPoint main(InputPatch<ControlPoint, 3> patch, uint controlPointId : SV_OutputControlPointID)
{
    return patch[controlPointId];
}
