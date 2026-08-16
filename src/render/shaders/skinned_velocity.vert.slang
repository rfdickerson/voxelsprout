// Per-pixel motion vectors for GPU-skinned actors.
//
// WHY ONLY ACTORS. This world is static geometry plus a moving camera, and TAA
// already recovers motion for all of it by reprojecting depth through
// prevViewProj (taa.comp.slang, which says so in its header: "over a STATIC
// world"). The one thing that reprojection cannot know about is geometry that
// moved independently of the camera, and in this engine that is exactly the
// GPU-skinned actors -- Victor, the settlers, anything posed from a .kf clip.
// So this pass fills in the missing half rather than duplicating the half that
// already works, and it draws a handful of actors instead of the whole world.
//
// WHY RE-SKIN RATHER THAN KEEP LAST FRAME'S POSITIONS. The skinning compute pass
// writes ImportedMeshVertex, which is what the main pass draws -- and it carries
// no bone indices, so the previous frame's OUTPUT cannot be re-posed and
// double-buffering it would mean per-frame vertex buffer handles threaded
// through the baked draw list. Re-skinning the REST POSE twice, once with this
// frame's bone matrices and once with last frame's, needs neither: it touches
// only this pipeline.
//
// The rest pose is bound as a VERTEX BUFFER with explicit attribute offsets, not
// as a StructuredBuffer. See skinning.comp.slang's header for why that matters:
// GpuSkinnedVertexIn is 84 tightly packed bytes on the CPU, and std430 would put
// a float3-declared member at a 16-byte alignment and read every field after the
// first from the wrong offset. Vertex attributes carry their offsets explicitly,
// so the layout cannot drift.

struct PushConstants
{
    // Last frame's world -> clip, including last frame's jitter.
    float4x4 prevViewProj;
    // This frame's world -> clip, including this frame's jitter.
    float4x4 viewProj;
    // Both frames' sub-pixel jitter in NDC. Motion vectors must describe where
    // the SURFACE went, not where the sampling grid went, so both jitters are
    // removed before the difference is taken -- leaving them in would write the
    // jitter pattern into the motion field and TAA would chase its own dither.
    float4 jitterCurrPrev;
    uint boneCount;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]]
PushConstants pushConstants;

// Bone matrices are uploaded TRANSPOSED (see skinning_resources.cc), matching
// the camera-MVP convention for this -matrix-layout-column-major compile.
[[vk::binding(0, 0)]]
StructuredBuffer<float4x4> currentBoneMatrices;

[[vk::binding(1, 0)]]
StructuredBuffer<float4x4> previousBoneMatrices;

struct VertexInput
{
    float3 position : POSITION;          // offset 0
    uint4 boneIndices : TEXCOORD0;       // offset 52
    float4 boneWeights : TEXCOORD1;      // offset 68
};

struct VertexOutput
{
    float4 clipPosition : SV_Position;
    // Carried to the fragment shader rather than differenced here: the
    // perspective divide has to happen per PIXEL, after interpolation, or the
    // motion vector is wrong everywhere a triangle spans any depth range.
    float4 currentClip : TEXCOORD0;
    float4 previousClip : TEXCOORD1;
};

float3 skinPosition(float3 restPosition, uint4 boneIndices, float4 boneWeights,
                    StructuredBuffer<float4x4> matrices)
{
    float3 skinned = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;
    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        const float weight = boneWeights[i];
        if (weight <= 0.0)
        {
            continue;
        }
        const uint boneIndex = min(boneIndices[i], pushConstants.boneCount - 1u);
        skinned += mul(matrices[boneIndex], float4(restPosition, 1.0)).xyz * weight;
        weightSum += weight;
    }
    // An unweighted vertex passes through at its authored position, matching
    // what the skinning compute pass does with it. Renormalizing instead would
    // put this pass's idea of the surface somewhere the main pass never drew.
    return (weightSum > 1e-5) ? (skinned / weightSum) : restPosition;
}

VertexOutput main(VertexInput input)
{
    const float3 currentWorld =
        skinPosition(input.position, input.boneIndices, input.boneWeights, currentBoneMatrices);
    const float3 previousWorld =
        skinPosition(input.position, input.boneIndices, input.boneWeights, previousBoneMatrices);

    VertexOutput output;
    output.clipPosition = mul(pushConstants.viewProj, float4(currentWorld, 1.0));
    output.currentClip = output.clipPosition;
    output.previousClip = mul(pushConstants.prevViewProj, float4(previousWorld, 1.0));
    return output;
}
