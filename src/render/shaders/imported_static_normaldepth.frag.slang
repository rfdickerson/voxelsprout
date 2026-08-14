import camera_uniform;
import chunk_push_constants;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

static const uint kImportedMaterialFlagAlphaTest = 1u;
static const uint kImportedMaterialFlagTwoSided = 64u;
static const uint kInvalidImportedTextureIndex = 0xffffffffu;

struct FragmentInput
{
    float3 inWorldPosition : TEXCOORD0;
    float3 inWorldNormal : TEXCOORD1;
    float3 inColor : TEXCOORD2;
    float2 inUv : TEXCOORD3;
    nointerpolation uint inTextureIndex : TEXCOORD4;
    nointerpolation uint inFlags : TEXCOORD5;
};

float4 main(FragmentInput input, bool isFrontFace : SV_IsFrontFace) : SV_Target0
{
    if (input.inTextureIndex != kInvalidImportedTextureIndex &&
        (input.inFlags & kImportedMaterialFlagAlphaTest) != 0u &&
        bindlessTextures[input.inTextureIndex].Sample(input.inUv).a <= chunkPc.materialParams.x)
    {
        discard;
    }
    // Same backface flip the main pass does (see resolveShadingNormal in
    // imported_static.frag.slang). This buffer is what the AO estimator reads,
    // so leaving it unflipped would have AO and shading disagree about which
    // way every two-sided surface faces -- and the disagreement is worse than
    // either choice, because AO then occludes against a hemisphere the lighting
    // never used.
    float3 worldNormal = normalize(input.inWorldNormal);
    if (!isFrontFace && (input.inFlags & kImportedMaterialFlagTwoSided) != 0u)
    {
        worldNormal = -worldNormal;
    }
    const float3 viewNormal = normalize(mul(camera.view, float4(worldNormal, 0.0)).xyz);
    const float viewDepth = max(-mul(camera.view, float4(input.inWorldPosition, 1.0)).z, 0.0);
    return float4((viewNormal * 0.5) + 0.5, viewDepth);
}
