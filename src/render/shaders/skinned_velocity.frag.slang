// Writes the per-pixel motion vector for a skinned actor.
//
// Output is float4(motionNdcXY, 1, 0). The .z = 1 is a VALIDITY FLAG, and it is
// what makes a partial velocity buffer usable: the target is cleared to zero, so
// a consumer reads .z to distinguish "this pixel has a real dynamic motion
// vector" from "nothing wrote here, fall back to reprojecting depth through
// prevViewProj". A zero motion vector is a perfectly legal value for a stationary
// actor, so the flag cannot be inferred from the xy being zero.

struct FragmentInput
{
    float4 clipPosition : SV_Position;
    float4 currentClip : TEXCOORD0;
    float4 previousClip : TEXCOORD1;
};

struct PushConstants
{
    float4x4 prevViewProj;
    float4x4 viewProj;
    float4 jitterCurrPrev;  // xy = this frame's NDC jitter, zw = last frame's
    uint boneCount;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]]
PushConstants pushConstants;

float4 main(FragmentInput input) : SV_Target0
{
    // Perspective divide per pixel, not per vertex -- see the note in the vertex
    // shader. Guarded because a vertex behind the eye can interpolate through
    // w == 0 along an edge that crosses the near plane.
    const float currentW = (abs(input.currentClip.w) < 1e-6) ? 1e-6 : input.currentClip.w;
    const float previousW = (abs(input.previousClip.w) < 1e-6) ? 1e-6 : input.previousClip.w;

    // Both jitters removed so the vector describes surface motion rather than
    // the sampling grid's motion.
    const float2 currentNdc = (input.currentClip.xy / currentW) - pushConstants.jitterCurrPrev.xy;
    const float2 previousNdc = (input.previousClip.xy / previousW) - pushConstants.jitterCurrPrev.zw;

    // Points BACKWARD, from where the surface is now to where it was: that is
    // what a history lookup needs, and it is the convention taa.comp.slang
    // already uses for its depth-derived reprojection.
    return float4(previousNdc - currentNdc, 1.0, 0.0);
}
