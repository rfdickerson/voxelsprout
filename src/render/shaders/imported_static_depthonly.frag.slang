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
// Match the main pass's cutout coverage, including its MSAA alpha ramp, so the
// depth prewrite cannot occlude samples the shading pass leaves transparent.

import chunk_push_constants;
import camera_uniform;

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

// The zero colour write mask preserves HDR, but alpha must still be exported:
// alpha-to-coverage consumes it when producing the depth sample mask.
float4 main(FragmentInput input) : SV_Target0
{
    if (chunkPc.cascadeData.x > 0.5 &&
        input.inWorldPosition.y < chunkPc.materialParams.w - 0.25)
    {
        discard;
    }
    float coverage = 1.0;
    if ((input.inFlags & kImportedMaterialFlagAlphaTest) != 0u)
    {
        float alpha = input.inColor.a;
        if (input.inTextureIndex != kInvalidImportedTextureIndex)
            alpha *= bindlessTextures[input.inTextureIndex].Sample(input.inUv).a;
        if (camera.importedExteriorConfig.w > 1.5)
        {
            coverage = saturate((alpha - chunkPc.materialParams.x) /
                max(fwidth(alpha), 1.0e-4) + 0.5);
        }
        else if (alpha <= chunkPc.materialParams.x)
        {
            discard;
        }
    }
    return float4(0.0, 0.0, 0.0, coverage);
}
