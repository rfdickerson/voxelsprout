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
    // x = history weight (0..1), y = history valid (0 or 1),
    // zw = LAST frame's sub-pixel jitter in NDC, which prevViewProj carries and
    // the history image does not.
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
// RGB <-> YCoCg. The neighbourhood bound below is built in YCoCg because its
// luma axis is the one the eye judges an edge on: a box fitted in RGB is a cube
// straddling all three channels and is far looser than the actual distribution
// of a local neighbourhood, so it lets stale colour through as ghosting. One
// axis of chroma slack costs nothing visually and tightens luma a lot.
float3 rgbToYCoCg(float3 c)
{
    return float3(
        (0.25 * c.r) + (0.5 * c.g) + (0.25 * c.b),
        (0.5 * c.r) - (0.5 * c.b),
        (-0.25 * c.r) + (0.5 * c.g) - (0.25 * c.b));
}

float3 yCoCgToRgb(float3 c)
{
    const float t = c.x - c.z;
    return float3(t + c.y, c.x + c.z, t - c.y);
}

float3 reconstructViewPosition(float2 uv, float viewDepth)
{
    // Take the sub-pixel jitter back out. The colour and depth in front of us
    // were rasterized through a jittered projection, so the NDC that a texel
    // centre corresponds to is offset by exactly that jitter -- and this
    // function's whole job is to invert the projection. Skipping it reprojects
    // every pixel about half a pixel wrong, every frame, in a direction that
    // changes each frame: TAA then fights itself and the image softens instead
    // of sharpening.
    const float2 ndcXY = ((uv * 2.0) - 1.0) - camera.tonemapConfig2.zw;
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
            // Un-jitter with LAST frame's offset, not this one's: prevViewProj
            // is the matrix that frame rendered with, so its NDC still carries
            // that frame's jitter, while the history image it addresses is the
            // resolved result on the fixed output grid and carries none.
            const float2 previousNdc = (previousClip.xy / previousClip.w) - taa.params.zw;
            const float2 previousUv = (previousNdc * 0.5) + 0.5;
            if (all(previousUv >= float2(0.0, 0.0)) && all(previousUv <= float2(1.0, 1.0)))
            {
                float3 history = historyColorTexture.SampleLevel(previousUv, 0.0).rgb;

                // VARIANCE CLIPPING (Salvi 2016) in YCoCg, replacing a min/max
                // box in RGB. The box is the mean of the 3x3 neighbourhood plus
                // or minus a multiple of its standard deviation, intersected
                // with the true min/max so a flat neighbourhood still bounds
                // tightly.
                //
                // This matters more than it used to. A min/max box is the
                // LOOSEST bound the neighbourhood admits -- one bright texel
                // opens it wide enough to accept nearly any history -- and the
                // town now walks. Without per-object velocity a moving actor's
                // history reprojects to where the BACKGROUND was, and only this
                // bound stands between that and a smear trailing every settler.
                // Jitter also raises the stakes: it deliberately makes the
                // current frame differ sub-pixel each frame, which widens a
                // min/max box further still.
                float3 momentSum = float3(0.0, 0.0, 0.0);
                float3 momentSquaredSum = float3(0.0, 0.0, 0.0);
                float3 neighborhoodMin = float3(1e30, 1e30, 1e30);
                float3 neighborhoodMax = float3(-1e30, -1e30, -1e30);
                [unroll]
                for (int offsetY = -1; offsetY <= 1; ++offsetY)
                {
                    [unroll]
                    for (int offsetX = -1; offsetX <= 1; ++offsetX)
                    {
                        const float2 neighborUv = uv + (float2(offsetX, offsetY) * texelSize);
                        const float3 neighbor = rgbToYCoCg(
                            currentColorTexture.SampleLevel(neighborUv, 0.0).rgb);
                        momentSum += neighbor;
                        momentSquaredSum += neighbor * neighbor;
                        neighborhoodMin = min(neighborhoodMin, neighbor);
                        neighborhoodMax = max(neighborhoodMax, neighbor);
                    }
                }
                const float3 mean = momentSum / 9.0;
                // max(0) before the sqrt: the two sums are accumulated in float
                // and a flat neighbourhood can land the variance a hair below
                // zero, which would make this NaN and poison the pixel forever
                // through the history.
                const float3 sigma =
                    sqrt(max(float3(0.0, 0.0, 0.0), (momentSquaredSum / 9.0) - (mean * mean)));
                // 1.25 sigma. Lower ghosts less and flickers more; this is the
                // usual compromise and is the first number to try if the
                // walking actors trail.
                const float3 clipMin = max(mean - (1.25 * sigma), neighborhoodMin);
                const float3 clipMax = min(mean + (1.25 * sigma), neighborhoodMax);
                // min/max rather than clamp() so a NaN in history resolves
                // toward the current frame instead of propagating -- IEEE
                // min/max return the non-NaN operand.
                const float3 historyYCoCg =
                    min(max(rgbToYCoCg(history), clipMin), clipMax);
                history = yCoCgToRgb(historyYCoCg);
                result = lerp(current, history, historyWeight);
            }
        }
    }

    outputTexture[dispatchThreadId.xy] = float4(result, 1.0);
}
