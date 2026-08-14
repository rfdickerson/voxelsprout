// Temporal upscaler: reconstructs a target-resolution image from a jittered
// low-resolution render plus an accumulated history.
//
// This is the engine's own UpscalerBackend::Temporal. It is a separate shader
// from taa.comp.slang rather than a mode of it because the two differ in the one
// thing that structures the whole pass: TAA's input and output share a grid, so
// "the current sample" is a texel fetch, while here the input grid is coarser
// AND offset by a per-frame jitter, so the current sample has to be
// RECONSTRUCTED from several low-res taps weighted by where they actually landed
// in output space. Everything downstream of that -- history weight, clamping,
// disocclusion -- then behaves differently enough that one shader with a branch
// would be two shaders sharing a name.
//
// Structure and the parts worth keeping are lifted from taa.comp.slang, which
// documents them at length: variance clipping in YCoCg rather than a min/max box
// in RGB, and un-jittering with the PREVIOUS frame's offset when addressing
// history. Both apply unchanged here.
//
// The one genuinely new input is per-object motion. taa.comp.slang says outright
// that "without per-object velocity a moving actor's history reprojects to where
// the BACKGROUND was", and leans on the neighbourhood clamp to bound the damage.
// The velocity buffer (frame_pass_velocity.cc) removes that guess for skinned
// actors: where it says a pixel has real motion, this uses it, and the clamp
// goes back to being a safety net instead of the primary defence.
//
// Design follows Intel's XeSS-SR developer guide where it applies to any
// temporal upscaler -- jitter in [-0.5, 0.5] pixels on a Halton sequence whose
// length scales with the upscale ratio, HDR linear input, upscale before
// tone-mapping -- since those constraints come from the sampling problem rather
// than from their network.

import camera_uniform;

// Low-resolution jittered render, at input extent.
[[vk::binding(1, 0)]]
Sampler2D<float4> currentColorTexture;

// Accumulated history, at OUTPUT extent.
[[vk::binding(2, 0)]]
Sampler2D<float4> historyColorTexture;

// Normal + view depth, at AO extent. .w is view depth, 0 on sky.
[[vk::binding(3, 0)]]
Sampler2D<float4> normalDepthTexture;

// Output, at OUTPUT extent.
[[vk::binding(4, 0)]]
RWTexture2D<float4> outputTexture;

struct TemporalUpscaleUniform
{
    float4x4 invView;       // view -> world
    float4x4 prevViewProj;  // last frame's world -> clip
    // x = history weight, y = history valid (0/1),
    // zw = LAST frame's sub-pixel jitter in NDC.
    float4 params;
    // xy = THIS frame's jitter in input PIXELS (the value added to the
    // projection, converted back), zw = input extent in pixels.
    float4 jitterAndInputExtent;
    float4 upscaleTuning;
};

[[vk::binding(5, 0)]]
ConstantBuffer<TemporalUpscaleUniform> upscale;

// Per-pixel motion for geometry that moved independently of the camera, at
// INPUT extent. .xy = NDC motion (previous - current), .z = 1 where a dynamic
// object wrote it. See frame_pass_velocity.cc.
[[vk::binding(6, 0)]]
Sampler2D<float4> velocityTexture;

struct PushConstants
{
    uint width;   // output
    uint height;  // output
    float pad0;
    float pad1;
};

[[vk::push_constant]]
PushConstants pushConstants;

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
    // Jitter removed for the same reason taa.comp.slang removes it: the depth in
    // front of us was rasterized through a jittered projection, so inverting
    // that projection has to undo the jitter or every pixel reprojects about
    // half a pixel wrong in a direction that changes every frame.
    const float2 ndcXY = ((uv * 2.0) - 1.0) - camera.tonemapConfig2.zw;
    float projX = camera.proj[0][0];
    float projY = camera.proj[1][1];
    if (abs(projX) < 1e-6) projX = (projX < 0.0) ? -1e-6 : 1e-6;
    if (abs(projY) < 1e-6) projY = (projY < 0.0) ? -1e-6 : 1e-6;
    const float depthScale = (camera.proj[3][3] > 0.5) ? 1.0 : viewDepth;
    return float3((ndcXY.x * depthScale) / projX, (ndcXY.y * depthScale) / projY, -viewDepth);
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= pushConstants.width || dispatchThreadId.y >= pushConstants.height)
    {
        return;
    }
    const float2 outputExtent = float2(float(pushConstants.width), float(pushConstants.height));
    const float2 inputExtent = upscale.jitterAndInputExtent.zw;
    const float2 uv = (float2(dispatchThreadId.xy) + 0.5) / outputExtent;

    // Where this output pixel sits on the INPUT grid, in input pixels.
    const float2 inputCoord = uv * inputExtent;

    // SAMPLE RECONSTRUCTION.
    //
    // Each low-res texel centre was displaced by this frame's jitter, so its
    // true position in the scene is (texelCentre - jitter). Gathering the 3x3
    // neighbourhood and weighting each tap by its distance to THIS output
    // pixel is what turns a sequence of jittered low-res frames into a
    // high-resolution signal -- take the nearest tap instead and the extra
    // samples are thrown away, which is a blurry upscale with a history buffer
    // attached rather than an upscaler.
    //
    // The kernel is TIGHT on purpose, and this is the whole difference between
    // an upscaler and an expensive blur.
    //
    // A wide per-frame filter pre-blurs the current sample, and averaging
    // pre-blurred images stays blurred no matter how many jitter phases are
    // accumulated -- measured: with a sigma-0.5 gather this scored identically
    // to a plain bilinear stretch (RMS 12.69 vs 12.64 against a native-resolution
    // reference) and the history weight changed 0.39% of pixels. The detail has
    // to come from ACCUMULATION across phases, not from filtering within a
    // frame, so the per-frame reconstruction stays near-nearest-neighbour and
    // the blend weight below decides how much of it to take.
    const float2 jitterPixels = upscale.jitterAndInputExtent.xy;
    const int2 baseTexel = int2(floor(inputCoord - 0.5));

    float3 currentSum = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;
    float3 momentSum = float3(0.0, 0.0, 0.0);
    float3 momentSquaredSum = float3(0.0, 0.0, 0.0);
    float3 neighborhoodMin = float3(1e30, 1e30, 1e30);
    float3 neighborhoodMax = float3(-1e30, -1e30, -1e30);
    float neighborhoodCount = 0.0;
    // How well THIS frame sampled THIS output pixel: the closest tap's weight.
    // 1 means a jittered sample landed on the pixel centre, ~0 means this
    // frame's sample grid missed it entirely and only history knows what is
    // here.
    float bestTapWeight = 0.0;
    // Convergence of THIS pixel, read from the history's accumulated weight
    // before the gather so the kernel can be chosen from it. Sampled at the
    // pixel's own location rather than the reprojected one: this is about how
    // much evidence has landed here, and a cheap approximation is right because
    // it only selects a filter width.
    const float priorAccum = historyColorTexture.SampleLevel(uv, 0.0).a;
    const float kernelSharpness = saturate(priorAccum / 4.0);

    [unroll]
    for (int offsetY = 0; offsetY <= 2; ++offsetY)
    {
        [unroll]
        for (int offsetX = 0; offsetX <= 2; ++offsetX)
        {
            const int2 texel = baseTexel + int2(offsetX, offsetY);
            const float2 tapUv = (float2(texel) + 0.5) / inputExtent;
            const float3 tap = currentColorTexture.SampleLevel(saturate(tapUv), 0.0).rgb;

            // Un-jitter the tap's position: the jitter moved the CAMERA, so the
            // scene point this texel saw is offset the other way.
            const float2 tapPosition = float2(texel) + 0.5 - jitterPixels;
            const float2 delta = tapPosition - inputCoord;
            // TWO kernels, blended by how converged this pixel is.
            //
            // A tight kernel (sigma ~0.35 input pixels) is what lets detail
            // accumulate rather than being pre-blurred away -- but on a pixel
            // that has not accumulated yet it is very nearly nearest-neighbour,
            // and nearest-neighbour at a 2x upscale is visibly BLOCKY. That is
            // worst exactly where it is most noticeable: distant geometry, whose
            // high-frequency detail never converges because a sub-pixel sliver
            // of mountain is a different surface every jitter phase.
            //
            // So the wide kernel is the floor, not a compromise: an unconverged
            // pixel should look like a smooth upscale, which is what the eye
            // reads as "soft" rather than "broken". As accumulated weight grows
            // the kernel tightens and the real detail comes through.
            const float tightWeight = exp(-4.0 * dot(delta, delta));
            const float wideWeight = exp(-1.0 * dot(delta, delta));
            const float weight = lerp(wideWeight, tightWeight, kernelSharpness);

            currentSum += tap * weight;
            weightSum += weight;

            const float3 tapYCoCg = rgbToYCoCg(tap);
            momentSum += tapYCoCg;
            momentSquaredSum += tapYCoCg * tapYCoCg;
            neighborhoodMin = min(neighborhoodMin, tapYCoCg);
            neighborhoodMax = max(neighborhoodMax, tapYCoCg);
            neighborhoodCount += 1.0;
            bestTapWeight = max(bestTapWeight, weight);
        }
    }
    const float3 current = currentSum / max(weightSum, 1e-5);
    float3 result = current;
    // Accumulated weight written alongside the colour. A fresh pixel starts at
    // this frame's own sample weight, so the next frame blends against it
    // proportionally rather than treating it as fully converged.
    float resultAccum = max(bestTapWeight, 0.02);

    const float historyWeight = upscale.params.x * upscale.params.y;
    const float viewDepth = normalDepthTexture.SampleLevel(uv, 0.0).w;
    // Sky takes no history, matching taa.comp.slang: depth is cleared to 0
    // there and reprojecting it needs a rotation-only path this pass does not
    // carry.
    if (historyWeight > 0.0 && viewDepth > 0.0001)
    {
        // WHERE THIS PIXEL WAS LAST FRAME.
        //
        // Two sources, and the dynamic one wins where it exists. The velocity
        // buffer covers geometry that moved independently of the camera; the
        // depth reprojection below covers the static world, which is everything
        // else. Preferring velocity is the whole reason it was built.
        float2 previousUv = float2(-1.0, -1.0);
        const float4 velocity = velocityTexture.SampleLevel(uv, 0.0);
        if (velocity.z > 0.5)
        {
            // NDC delta -> UV delta. NDC spans 2 across the screen and UV spans
            // 1, so the factor is 0.5.
            previousUv = uv + (velocity.xy * 0.5);
        }
        else
        {
            const float3 viewPosition = reconstructViewPosition(uv, viewDepth);
            const float3 worldPosition = mul(upscale.invView, float4(viewPosition, 1.0)).xyz;
            const float4 previousClip = mul(upscale.prevViewProj, float4(worldPosition, 1.0));
            if (previousClip.w > 0.0)
            {
                // Un-jitter with LAST frame's offset: prevViewProj is the matrix
                // that frame rendered with, so its NDC carries that frame's
                // jitter, while the history image is on the fixed output grid
                // and carries none.
                const float2 previousNdc = (previousClip.xy / previousClip.w) - upscale.params.zw;
                previousUv = (previousNdc * 0.5) + 0.5;
            }
        }

        if (all(previousUv >= float2(0.0, 0.0)) && all(previousUv <= float2(1.0, 1.0)))
        {
            const float4 historySample = historyColorTexture.SampleLevel(previousUv, 0.0);
            float3 history = historySample.rgb;
            // .a carries ACCUMULATED SAMPLE WEIGHT, not opacity. See the blend
            // below for why a running weighted average needs it.
            float historyAccum = max(historySample.a, 0.0);

            // VARIANCE CLIPPING in YCoCg (Salvi 2016), as taa.comp.slang does
            // and for the reasons it gives: a min/max box in RGB is the loosest
            // bound the neighbourhood admits, and one bright texel opens it wide
            // enough to accept nearly any history.
            //
            // The neighbourhood here is the LOW-RES 3x3 gathered above, which is
            // the right set: it is the actual evidence this frame has about what
            // colours belong at this location, and building a bound from the
            // upscaled result instead would just be bounding history against
            // history.
            const float3 mean = momentSum / neighborhoodCount;
            const float3 variance =
                max((momentSquaredSum / neighborhoodCount) - (mean * mean), float3(0.0, 0.0, 0.0));
            const float3 sigma = sqrt(variance);

            // CLAMP WIDTH SCALES WITH MOTION, and this is the difference between
            // an upscaler and an expensive blur.
            //
            // The bound is built from the LOW-RES neighbourhood -- it is the only
            // evidence this frame has -- but that neighbourhood is by definition
            // blurrier than the output being reconstructed. Clamping hard to it
            // every frame throws away exactly the high-frequency detail the
            // accumulation spent 32 jitter phases building, so the result
            // converges to the low-res image scaled up. Measured with a fixed
            // 1.25 sigma: error against a native-resolution reference was
            // identical to a plain bilinear stretch, and the history weight made
            // no visible difference at all.
            //
            // Where nothing is moving, history is trustworthy and the clamp
            // should barely bind. Where a pixel is moving fast, reprojection and
            // the motion vector are both least reliable and the clamp is the
            // only defence against ghosting -- so it tightens. Motion is measured
            // in OUTPUT pixels of reprojection distance, which is the quantity
            // that actually predicts whether history belongs to this surface.
            const float2 motionPixels = (previousUv - uv) * outputExtent;
            const float motionLength = length(motionPixels);
            // 0 px -> 4 sigma (accumulate freely), 8 px -> 1.25 sigma (defend).
            const float clampWidth = lerp(
                upscale.upscaleTuning.x, upscale.upscaleTuning.y,
                saturate(motionLength / 8.0));
            const float3 clipMin = max(mean - (clampWidth * sigma), neighborhoodMin);
            const float3 clipMax = min(mean + (clampWidth * sigma), neighborhoodMax);
            history = yCoCgToRgb(clamp(rgbToYCoCg(history), clipMin, clipMax));

            // RUNNING WEIGHTED AVERAGE, not a lerp.
            //
            // The per-frame update has to be weighted by how well this frame
            // sampled this pixel -- that is what builds detail rather than
            // averaging back to the low-resolution image. But feeding a varying
            // weight into a plain lerp means neighbouring output pixels are
            // updated by different amounts every frame, and because which pixels
            // get a close sample is determined by the jitter phase, that pattern
            // is locked to the input pixel grid. It shows up as dithering over
            // the whole image, which is exactly what a fixed lerp produced here.
            //
            // Carrying the accumulated weight per pixel fixes it at the root: a
            // pixel that has received few close samples simply has a low total
            // weight, and still converges to the correct average rather than to
            // whatever its update schedule happened to be. Every pixel converges
            // to the same answer at a different rate instead of to different
            // answers.
            //
            // The cap is what keeps it responsive -- an uncapped sum would make
            // an old pixel effectively frozen, so a lighting change or a slow
            // pan would take thousands of frames to appear.
            const float maxAccum = max(historyWeight / max(1.0 - historyWeight, 1e-3), 1.0);
            const float currentWeight = max(bestTapWeight, 0.02);
            const float totalWeight = historyAccum + currentWeight;
            result = ((history * historyAccum) + (current * currentWeight)) / max(totalWeight, 1e-5);
            resultAccum = min(totalWeight, maxAccum);

            // SPATIALLY VARYING BLEND -- superseded by the weighted average
            // above; kept only as the shape of what it replaced.
            //
            // A single global history weight updates every output pixel by the
            // same amount every frame, so a pixel that this frame's sample grid
            // missed gets polluted by a neighbour's colour just as strongly as
            // one the grid hit. Over the jitter sequence that averages back to
            // the low-resolution image, which is exactly the failure measured
            // before this: accumulation on, and no better than bilinear.
            //
            // Weighting the update by how close this frame's nearest sample
            // landed means each output pixel takes a large update only on the
            // frames when a sample genuinely lands on it, and holds its
            // accumulated value otherwise. Across a Halton sequence whose length
            // scales with the ratio, every output pixel gets hit repeatedly.
            //
        }
    }

    outputTexture[dispatchThreadId.xy] = float4(result, resultAccum);
}
