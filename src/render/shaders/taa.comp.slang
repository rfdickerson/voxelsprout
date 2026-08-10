// Compile for Vulkan SPIR-V with column-major matrices:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/taa.comp.slang -o src/render/shaders/taa.comp.slang.spv
//
// Temporal anti-aliasing by camera reprojection, over a STATIC world.
//
// Why this exists, with the measurement that motivated it: with the camera
// nearly still (2 units/s, fixed timestep), consecutive frames differed by
// more than 8/255 on 3.9% of all pixels -- and with textures replaced by flat
// vertex color that dropped 13x, to 0.30%. The instability is texture-content
// resampling: high-contrast texel-scale detail (wood grain, BC-compressed
// noise) re-lands on a slightly different sampling grid every frame, and no
// static filter fixes that -- mips, trilinear and 8x anisotropy were all
// verified present and correctly configured before this pass was written.
// Averaging across frames is the only tool that touches it, which is what
// every modern engine ships TAA for.
//
// Reprojection is CAMERA-ONLY: history UV comes from re-projecting the
// reconstructed world position through last frame's view-projection. There are
// no per-object motion vectors, which is correct for everything this renderer
// currently draws through the imported path -- the Mojave's geometry is
// static. When skinned actors reach the screen they will ghost under this
// scheme; that is the point at which this pass needs a velocity input, not
// before.
//
// No projection jitter yet either. Jitter buys sub-pixel EDGE refinement;
// the measured artifact is texture shimmer, which history accumulation fixes
// with or without jitter. Adding jitter later only touches the projection
// setup and this file's unjitter of the current sample.
//
// Ghosting control is a 3x3 neighborhood clamp in RGB: history is clamped to
// the min/max of the current frame's local neighborhood, so stale colors
// (disocclusions, lighting changes like the day cycle) converge in a few
// frames instead of smearing.

import camera_uniform;

[[vk::binding(1, 0)]]
Sampler2D<float4> currentColorTexture;

[[vk::binding(2, 0)]]
Sampler2D<float4> historyColorTexture;

// Half-resolution prepass output; .w is view-space depth, cleared to 0 on sky.
// Half res means silhouette pixels can reproject slightly wrong; the
// neighborhood clamp bounds the damage to "converges like a disocclusion".
[[vk::binding(3, 0)]]
Sampler2D<float4> normalDepthTexture;

[[vk::binding(4, 0)]]
RWTexture2D<float4> outputTexture;

struct TaaUniform
{
    float4x4 invView;       // view space -> world space
    float4x4 prevViewProj;  // last frame's world -> clip
    // x = history weight (0..1), y = history valid (0 or 1), zw unused.
    float4 params;
};

[[vk::binding(5, 0)]]
ConstantBuffer<TaaUniform> taa;

struct PushConstants
{
    uint width;
    uint height;
    float pad0;
    float pad1;
};

[[vk::push_constant]]
PushConstants pushConstants;

// Same reconstruction ssao.comp.slang uses, kept textually close on purpose:
// the two must agree about what .w means or TAA reprojects a different world
// than AO shaded.
float3 reconstructViewPosition(float2 uv, float viewDepth)
{
    const float2 ndcXY = (uv * 2.0) - 1.0;
    float projX = camera.proj[0][0];
    float projY = camera.proj[1][1];
    if (abs(projX) < 1e-6) projX = (projX < 0.0) ? -1e-6 : 1e-6;
    if (abs(projY) < 1e-6) projY = (projY < 0.0) ? -1e-6 : 1e-6;
    const float depthScale = (camera.proj[3][3] > 0.5) ? 1.0 : viewDepth;
    const float viewX = (ndcXY.x * depthScale) / projX;
    const float viewY = (ndcXY.y * depthScale) / projY;
    return float3(viewX, viewY, -viewDepth);
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= pushConstants.width || dispatchThreadId.y >= pushConstants.height)
    {
        return;
    }
    const float2 texelSize =
        float2(1.0 / float(pushConstants.width), 1.0 / float(pushConstants.height));
    const float2 uv = (float2(dispatchThreadId.xy) + 0.5) * texelSize;

    const float3 current = currentColorTexture.SampleLevel(uv, 0.0).rgb;
    float3 result = current;

    const float historyWeight = taa.params.x * taa.params.y;
    const float viewDepth = normalDepthTexture.SampleLevel(uv, 0.0).w;
    // Sky (depth cleared to 0) takes no history: it was measured stable
    // without TAA, and reprojecting it needs a rotation-only path this pass
    // does not carry.
    if (historyWeight > 0.0 && viewDepth > 0.0001)
    {
        const float3 viewPosition = reconstructViewPosition(uv, viewDepth);
        const float3 worldPosition = mul(taa.invView, float4(viewPosition, 1.0)).xyz;
        const float4 previousClip = mul(taa.prevViewProj, float4(worldPosition, 1.0));
        if (previousClip.w > 0.0)
        {
            const float2 previousUv = ((previousClip.xy / previousClip.w) * 0.5) + 0.5;
            if (all(previousUv >= float2(0.0, 0.0)) && all(previousUv <= float2(1.0, 1.0)))
            {
                float3 history = historyColorTexture.SampleLevel(previousUv, 0.0).rgb;

                // Neighborhood clamp. min/max rather than clamp() so a NaN in
                // history resolves toward the current frame instead of
                // propagating -- IEEE min/max return the non-NaN operand.
                float3 neighborhoodMin = current;
                float3 neighborhoodMax = current;
                [unroll]
                for (int offsetY = -1; offsetY <= 1; ++offsetY)
                {
                    [unroll]
                    for (int offsetX = -1; offsetX <= 1; ++offsetX)
                    {
                        if (offsetX == 0 && offsetY == 0)
                        {
                            continue;
                        }
                        const float2 neighborUv = uv + (float2(offsetX, offsetY) * texelSize);
                        const float3 neighbor =
                            currentColorTexture.SampleLevel(neighborUv, 0.0).rgb;
                        neighborhoodMin = min(neighborhoodMin, neighbor);
                        neighborhoodMax = max(neighborhoodMax, neighbor);
                    }
                }
                history = min(max(history, neighborhoodMin), neighborhoodMax);
                result = lerp(current, history, historyWeight);
            }
        }
    }

    outputTexture[dispatchThreadId.xy] = float4(result, 1.0);
}
