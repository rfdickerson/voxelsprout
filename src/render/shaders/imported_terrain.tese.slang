import camera_uniform;

// Domain stage for imported terrain: Phong tessellation plus a small
// deterministic detail bump, then the projection imported_static.vert would
// have done.
//
// PHONG, NOT PN TRIANGLES, ON PURPOSE. Phong needs only the per-vertex
// normals already in the stream (project the flat point onto each corner's
// tangent plane and blend), costs three dot products, and its C1-ish rounding
// is exactly what a heightfield silhouette needs. PN triangles buy slightly
// nicer interior curvature for a bezier patch construction this data cannot
// justify -- LAND normals are themselves derived from the height grid.
//
// EVERYTHING HERE MUST BE A PURE FUNCTION OF THE PATCH AND THE CAMERA UBO.
// The merged depth prepass and the main pass both run this shader, and main
// depth-tests EQUAL against the prepass's depth: any input the two passes do
// not share (frame counters, jitter) would make them disagree by a few ulps
// and punch holes in the ground.
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

struct DomainOutput
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

// Same construction as imported_static.frag's terrainBlendNoise, kept separate
// because the fragment shader's operates in UV space and this one in world
// space -- sharing them would couple the bump field to texture tiling.
float terrainDetailHash(float2 p)
{
    return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453123);
}

float terrainDetailNoise(float2 p)
{
    const float2 cell = floor(p);
    const float2 f = frac(p);
    const float2 u = f * f * (3.0 - (2.0 * f));
    const float a = terrainDetailHash(cell);
    const float b = terrainDetailHash(cell + float2(1.0, 0.0));
    const float c = terrainDetailHash(cell + float2(0.0, 1.0));
    const float d = terrainDetailHash(cell + float2(1.0, 1.0));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

[domain("tri")]
DomainOutput main(
    PatchConstants constants,
    float3 bary : SV_DomainLocation,
    const OutputPatch<ControlPoint, 3> patch)
{
    DomainOutput output;

    const float3 flatPosition =
        (patch[0].worldPosition * bary.x) +
        (patch[1].worldPosition * bary.y) +
        (patch[2].worldPosition * bary.z);
    const float3 n0 = normalize(patch[0].worldNormal);
    const float3 n1 = normalize(patch[1].worldNormal);
    const float3 n2 = normalize(patch[2].worldNormal);

    // Phong: the flat point projected onto each corner's tangent plane, blended
    // by the same barycentrics.
    const float3 projected0 = flatPosition - (dot(flatPosition - patch[0].worldPosition, n0) * n0);
    const float3 projected1 = flatPosition - (dot(flatPosition - patch[1].worldPosition, n1) * n1);
    const float3 projected2 = flatPosition - (dot(flatPosition - patch[2].worldPosition, n2) * n2);
    const float3 phongPosition =
        (projected0 * bary.x) + (projected1 * bary.y) + (projected2 * bary.z);
    // 0.5: enough to round a ridge, not enough to pull the surface visibly off
    // the collision heights the CPU walks on.
    const float kPhongAlpha = 0.5;
    float3 worldPosition = lerp(flatPosition, phongPosition, kPhongAlpha);

    const float3 worldNormal = normalize((n0 * bary.x) + (n1 * bary.y) + (n2 * bary.z));

    // Detail bump: two octaves of value noise in world XZ, displaced along the
    // smoothed normal. Amplitude fades with the same distance ramp the hull
    // stage uses for its factors, so geometry that tessellates back to a single
    // triangle also carries no displacement -- otherwise the transition pops.
    // At corners (bary hits a control point) the noise fades to zero so patch
    // edges shared with an untessellated neighbour stay watertight.
    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float viewDistance = length(worldPosition - cameraWorld);
    const float distanceFade = 1.0 - smoothstep(2000.0, 10000.0, viewDistance);
    const float edgeFade = saturate(3.0 * min(bary.x, min(bary.y, bary.z)));
    if (distanceFade > 0.0 && edgeFade > 0.0)
    {
        const float coarse = terrainDetailNoise(worldPosition.xz * (1.0 / 96.0));
        const float fine = terrainDetailNoise(worldPosition.xz * (1.0 / 34.0));
        const float bump = ((coarse - 0.5) * 0.7) + ((fine - 0.5) * 0.3);
        // bump spans roughly [-0.5, 0.5], so this is +/-9 units -- about
        // 13 cm at Bethesda's ~70 units per metre. Enough to break up a flat
        // plain up close, small enough that an actor standing on the CPU
        // collision heights never visibly floats or sinks.
        const float kBumpAmplitude = 18.0;
        worldPosition += worldNormal * (bump * kBumpAmplitude * distanceFade * edgeFade);
    }

    output.worldPosition = worldPosition;
    output.worldNormal = worldNormal;
    output.color =
        (patch[0].color * bary.x) + (patch[1].color * bary.y) + (patch[2].color * bary.z);
    output.uv = (patch[0].uv * bary.x) + (patch[1].uv * bary.y) + (patch[2].uv * bary.z);
    // Flat attributes take control point 0, matching the rasterizer's default
    // first-vertex provoking convention on the untessellated path.
    output.textureIndex = patch[0].textureIndex;
    output.flags = patch[0].flags;
    output.layerTextures = patch[0].layerTextures;
    output.layerWeights =
        (patch[0].layerWeights * bary.x) +
        (patch[1].layerWeights * bary.y) +
        (patch[2].layerWeights * bary.z);
    output.position = mul(camera.mvp, float4(worldPosition, 1.0));
    return output;
}
