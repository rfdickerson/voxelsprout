// Depth-only prewrite for the imported-static main pass.
//
// The main pass clears its own depth and rejects occluded fragments using only
// what it has already drawn, which means every surface that ends up hidden is
// still shaded by a forward shader carrying cascaded PCF shadows, a 64-light
// loop, a four-layer terrain blend and PBR. This pass lays the depth first so
// the hardware kills those fragments before any of that runs.
//
// It is NOT the SSAO prepass. That one renders view normals and linear depth
// into its own AO-resolution target with storeOp DONT_CARE, so nothing it
// writes reaches the main depth buffer -- which is exactly why the main pass
// used to get no early-Z out of it.
//
// Deliberately shares imported_static.vert with the shading pass rather than
// using the compact 28-byte shadow stream. A different vertex shader can
// compute a clip position that differs in the last bit, and against a
// GREATER_OR_EQUAL test with depth writes off that renders as holes in exactly
// the surfaces this is supposed to be accelerating. Same vertex shader, same
// vertex layout, bit-identical depth.
//
// The alpha test is the same one imported_static_normaldepth.frag runs and must
// stay that way: a cutout that discards here but not in the shading pass would
// pre-write depth for a texel that is meant to be see-through.

import chunk_push_constants;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

static const uint kImportedMaterialFlagAlphaTest = 1u;
static const uint kInvalidImportedTextureIndex = 0xffffffffu;

struct FragmentInput
{
    float3 inWorldPosition : TEXCOORD0;
    float3 inWorldNormal : TEXCOORD1;
    // .a is authored vertex alpha; see imported_static.frag.slang.
    float4 inColor : TEXCOORD2;
    float2 inUv : TEXCOORD3;
    nointerpolation uint inTextureIndex : TEXCOORD4;
    nointerpolation uint inFlags : TEXCOORD5;
};

// Writes no colour. The pipeline binds the main pass's colour attachment with a
// zero write mask, so this returning nothing is what keeps the HDR target
// untouched while depth is written.
void main(FragmentInput input)
{
    if (input.inTextureIndex != kInvalidImportedTextureIndex &&
        (input.inFlags & kImportedMaterialFlagAlphaTest) != 0u &&
        (bindlessTextures[input.inTextureIndex].Sample(input.inUv).a * input.inColor.a) <=
            chunkPc.materialParams.x)
    {
        discard;
    }
}
