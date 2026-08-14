// XeGTAO main pass (Jimenez et al. 2016 integral, Intel's XeGTAO formulation).
//
// This is a separate shader from ssao.comp.slang rather than a fourth mode in
// it, because almost nothing is shared: that one marches a full-resolution
// normal-depth texture with fixed step counts and an interleaved-gradient
// dither, and every part of that is what this replaces.
//
// Four things here are specifically about running on an Xe iGPU:
//
//  * MIP SELECTION. Each step picks a pyramid level from its own offset length,
//    so the texel footprint tracks the step length and neighbouring threads read
//    overlapping cache lines. See xegtao_prefilter.comp.slang for why this is the
//    dominant cost on this hardware.
//
//  * ADAPTIVE SAMPLE COUNTS. Slices and steps scale with the screen-space radius
//    instead of being fixed. A surface 60 m away whose effect radius covers three
//    pixels gets the minimum; a wall at contact range gets the maximum. Fixed
//    counts spend the same work on both and the distant one is entirely wasted --
//    it is sampling inside its own texel.
//
//  * SPATIOTEMPORAL BLUE NOISE. A Hilbert-curve index into an R2 sequence,
//    advanced per frame. Blue noise puts the sampling error at high spatial
//    frequency where the denoiser removes it; the temporal advance means the
//    error is also decorrelated frame to frame, so TAA integrates it away rather
//    than locking in a static dither pattern.
//
//  * POINT SAMPLING. The pyramid is read with a point sampler on purpose.
//    Linear filtering blends neighbouring depths on the same MIP, which invents
//    surfaces between real ones and puts false horizons into the march.
//
// Bent normals come out alongside the AO term: the average unoccluded direction,
// which is what lets an ambient term be fetched along where light can actually
// arrive from instead of along the geometric normal.

import camera_uniform;

[[vk::binding(1, 0)]]
Sampler2D<float4> normalDepthTexture;

// The prefiltered viewspace depth pyramid, as five separate images rather than
// one mip-chained one. Same cache property -- a large step reads a small image --
// and it needs no per-mip storage views. Nothing is lost: XeGTAO point-samples
// the pyramid on purpose (see the header), so hardware trilinear between levels
// was never going to be used.
//
// Sampled with a POINT sampler.
[[vk::binding(2, 0)]]
Sampler2D<float> depthPyramid[5];

// AO term (visibility, 1 = unoccluded).
[[vk::binding(3, 0)]]
RWTexture2D<float> aoOutput;

// Bent normal, view space, encoded to [0,1]. .w is unused.
[[vk::binding(4, 0)]]
RWTexture2D<float4> bentNormalOutput;

struct PushConstants
{
    uint width;
    uint height;
    float effectRadius;
    float falloffRange;
    // Non-linear step distribution along the slice: >1 clusters samples near the
    // centre, which is where contact occlusion lives.
    float sampleDistributionPower;
    // Widens the falloff along view depth so a thin object does not occlude as
    // if it were a solid volume extending away from the camera.
    float thinOccluderCompensation;
    float finalValuePower;
    // Frame counter for the temporal half of the noise.
    uint temporalIndex;
};

[[vk::push_constant]]
PushConstants pushConstants;

static const float kPi = 3.14159265358979323846;
static const float kPiHalf = 1.5707963267948966;
static const float kDepthMipLevels = 5.0;
// Offset applied to log2(offsetLength) when choosing a MIP. XeGTAO's default of
// 3.30 keeps the first few steps on MIP0 where the detail actually matters.
static const float kDepthMipSamplingOffset = 3.30;

// Cheap acos, max error ~0.0004 rad. Used per sample, so the difference against
// a real acos is measurable and the error is far below what AO resolves.
float fastAcos(float x)
{
    const float absX = abs(x);
    float result = ((-0.156583 * absX) + kPiHalf) * sqrt(max(1.0 - absX, 0.0));
    result = (x < 0.0) ? (kPi - result) : result;
    return result;
}

// Hilbert curve index over a 64x64 tile. The Hilbert ordering is what makes the
// R2 sequence below read as blue noise in 2D rather than as a scanline gradient:
// consecutive indices stay spatially adjacent, so consecutive R2 values (which
// are maximally spread in 1D) land on neighbouring pixels.
uint hilbertIndex(uint posX, uint posY)
{
    uint index = 0u;
    for (uint level = 32u; level > 0u; level /= 2u)
    {
        const uint regionX = ((posX & level) > 0u) ? 1u : 0u;
        const uint regionY = ((posY & level) > 0u) ? 1u : 0u;
        index += level * level * ((3u * regionX) ^ regionY);
        if (regionY == 0u)
        {
            if (regionX == 1u)
            {
                posX = 63u - posX;
                posY = 63u - posY;
            }
            const uint temp = posX;
            posX = posY;
            posY = temp;
        }
    }
    return index;
}

// R2 low-discrepancy sequence, offset per frame. x drives the slice angle, y the
// step offset within a slice.
float2 spatioTemporalNoise(uint2 pixelCoord, uint temporalIndex)
{
    uint index = hilbertIndex(pixelCoord.x & 63u, pixelCoord.y & 63u);
    index += 288u * (temporalIndex % 64u);
    return frac(0.5 + float2(float(index) * 0.75487766624669276005,
                             float(index) * 0.56984029099805326591));
}

float3 decodeViewNormal(float3 encodedNormal)
{
    return normalize((encodedNormal * 2.0) - 1.0);
}

// Matches ssao.comp.slang's reconstruction, including the TAA jitter removal:
// the normal-depth source was rasterized through the jittered projection, so a
// texel centre is not at uv*2-1, and the AO march has to agree with TAA about
// where a texel is.
float3 viewPositionFromUv(float2 uv, float viewDepth)
{
    const float2 ndcXY = ((uv * 2.0) - 1.0) - camera.tonemapConfig2.zw;
    float projX = camera.proj[0][0];
    float projY = camera.proj[1][1];
    if (abs(projX) < 1e-6) projX = (projX < 0.0) ? -1e-6 : 1e-6;
    if (abs(projY) < 1e-6) projY = (projY < 0.0) ? -1e-6 : 1e-6;
    const float depthScale = (camera.proj[3][3] > 0.5) ? 1.0 : viewDepth;
    return float3((ndcXY.x / projX) * depthScale, (ndcXY.y / projY) * depthScale, -viewDepth);
}

// Rotation taking `from` to `to`, both unit. Used to lift the slice-local bent
// normal back into view space.
float3x3 rotFromTo(float3 from, float3 to)
{
    const float cosTheta = dot(from, to);
    const float3 axis = cross(from, to);
    const float k = 1.0 / (1.0 + cosTheta);
    float3x3 result;
    result[0] = float3((axis.x * axis.x * k) + cosTheta,
                       (axis.y * axis.x * k) - axis.z,
                       (axis.z * axis.x * k) + axis.y);
    result[1] = float3((axis.x * axis.y * k) + axis.z,
                       (axis.y * axis.y * k) + cosTheta,
                       (axis.z * axis.y * k) - axis.x);
    result[2] = float3((axis.x * axis.z * k) - axis.y,
                       (axis.y * axis.z * k) + axis.x,
                       (axis.z * axis.z * k) + cosTheta);
    return result;
}

// Geometric edges for the denoiser, from depth discontinuities around this pixel.
//
// The slope adjustment is what stops a steeply-angled but CONTINUOUS surface --
// a road running away from the camera -- from reading as an edge. Raw depth
// deltas across such a surface are large; subtracting the local linear slope
// leaves only the part that is not explained by the surface tilting, which is
// what an actual silhouette looks like.
float4 calculateEdges(float centerZ, float leftZ, float rightZ, float topZ, float bottomZ)
{
    float4 edgesLRTB = float4(leftZ, rightZ, topZ, bottomZ) - centerZ;
    const float slopeLR = (edgesLRTB.y - edgesLRTB.x) * 0.5;
    const float slopeTB = (edgesLRTB.w - edgesLRTB.z) * 0.5;
    const float4 adjusted = edgesLRTB + float4(slopeLR, -slopeLR, slopeTB, -slopeTB);
    edgesLRTB = min(abs(edgesLRTB), abs(adjusted));
    // Tolerance scales with depth: a 1cm step matters up close and is noise at
    // 200 m, and a fixed threshold would edge-detect the whole distance field.
    return saturate(1.25 - (edgesLRTB / (centerZ * 0.011 + 1e-5)));
}

// Four edge factors into one 8-bit channel, 2 bits each. The denoiser only needs
// "how connected is this neighbour", and 4 levels is enough for a blur weight --
// which is what lets the whole thing ride in the bent normal's unused alpha
// rather than costing another render target.
float packEdges(float4 edgesLRTB)
{
    const float4 quantized = round(saturate(edgesLRTB) * 2.9);
    return dot(quantized, float4(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= pushConstants.width || dispatchThreadId.y >= pushConstants.height)
    {
        return;
    }
    const int2 pixelCoord = int2(dispatchThreadId.xy);
    const float2 screenSize = float2(float(pushConstants.width), float(pushConstants.height));
    const float2 uv = (float2(pixelCoord) + 0.5) / screenSize;

    const float4 centerSample = normalDepthTexture.SampleLevel(uv, 0.0);
    const float centerDepth = centerSample.w;
    if (centerDepth <= 1e-4)
    {
        // Nothing rendered here: fully unoccluded, bent normal facing the viewer.
        aoOutput[pixelCoord] = 1.0;
        // Edges 0 = fully disconnected, so the denoiser never pulls sky into a
        // neighbouring surface.
        bentNormalOutput[pixelCoord] = float4(0.5, 0.5, 1.0, 0.0);
        return;
    }

    const float3 viewNormal = decodeViewNormal(centerSample.xyz);
    const float3 centerPos = viewPositionFromUv(uv, centerDepth);
    const float3 viewVec = normalize(-centerPos);

    // Neighbour depths come from pyramid MIP0 rather than the normal-depth
    // texture: it is the same values, already point-sampled, and the four taps
    // land in cache lines this thread's march is about to read anyway.
    const float2 texel = 1.0 / screenSize;
    const float leftZ = depthPyramid[0].SampleLevel(saturate(uv - float2(texel.x, 0.0)), 0.0);
    const float rightZ = depthPyramid[0].SampleLevel(saturate(uv + float2(texel.x, 0.0)), 0.0);
    const float topZ = depthPyramid[0].SampleLevel(saturate(uv - float2(0.0, texel.y)), 0.0);
    const float bottomZ = depthPyramid[0].SampleLevel(saturate(uv + float2(0.0, texel.y)), 0.0);
    const float packedEdges =
        packEdges(calculateEdges(centerDepth, leftZ, rightZ, topZ, bottomZ));

    const float effectRadius = max(pushConstants.effectRadius, 1e-4);
    const float falloffRange = max(pushConstants.falloffRange * effectRadius, 1e-6);
    const float falloffFrom = effectRadius * (1.0 - pushConstants.falloffRange);
    const float falloffMul = -1.0 / falloffRange;
    const float falloffAdd = (falloffFrom / falloffRange) + 1.0;

    // Screen-space radius of the effect at this depth, in MIP0 texels. Everything
    // adaptive below is driven by this.
    const float projScale = abs(camera.proj[0][0]) * 0.5 * screenSize.x;
    const float screenspaceRadius =
        (camera.proj[3][3] > 0.5) ? (effectRadius * projScale)
                                  : (effectRadius * projScale / max(centerDepth, 1e-4));

    // Below a few pixels the march is sampling inside its own texel and cannot
    // resolve anything; fade to unoccluded rather than spending samples on it.
    if (screenspaceRadius < 1.5)
    {
        aoOutput[pixelCoord] = 1.0;
        bentNormalOutput[pixelCoord] = float4((viewNormal * 0.5) + 0.5, packedEdges);
        return;
    }

    // ADAPTIVE SAMPLE COUNTS. Both scale with the log of the screen-space radius:
    // doubling the covered area is one more step, not twice as many. A 3-pixel
    // radius gets 1 slice / 2 steps, a 100-pixel one gets 3 slices / 6 steps.
    // ADAPTIVE SAMPLE COUNTS, with a FLOOR that is not negotiable.
    //
    // Both scale with the screen-space radius, so a surface whose effect radius
    // covers a few pixels does not pay for samples it cannot resolve. But the
    // floor was 1 slice / 2 steps and that is measurably wrong here: at four
    // samples the raw AO is noisy enough that rotating the noise per frame turns
    // spatial error into visible temporal flicker. Measured frame-to-frame
    // instability at the old floor was 1.25% of pixels against plain GTAO's
    // 0.06% -- a 20x regression -- because the spatiotemporal noise assumes a
    // denoiser strong enough to absorb it and this path reuses the joint
    // bilateral upsample, which is an upsample first and a denoiser second.
    //
    // 2 slices / 4 steps is 16 samples, which is XeGTAO's Low preset.
    const float radiusLog = log2(max(screenspaceRadius, 2.0));
    const int sliceCount = int(clamp(radiusLog - 0.5, 2.0, 3.0));
    const int stepsPerSlice = int(clamp(radiusLog, 4.0, 6.0));

    const float2 noise = spatioTemporalNoise(uint2(pixelCoord), pushConstants.temporalIndex);

    // Skip the centre texel: it carries no horizon information about itself.
    const float pixelTooCloseThreshold = 1.3;
    const float minS = pixelTooCloseThreshold / screenspaceRadius;

    float visibility = 0.0;
    float3 bentNormal = float3(0.0, 0.0, 0.0);

    for (int slice = 0; slice < sliceCount; ++slice)
    {
        const float sliceK = (float(slice) + noise.x) / float(sliceCount);
        const float phi = sliceK * kPi;
        const float cosPhi = cos(phi);
        const float sinPhi = sin(phi);
        // Screen-space march direction, scaled to the effect radius in UV.
        const float2 omega = float2(cosPhi, -sinPhi) * (screenspaceRadius / screenSize.x);

        const float3 directionVec = float3(cosPhi, sinPhi, 0.0);
        const float3 orthoDirectionVec = directionVec - (dot(directionVec, viewVec) * viewVec);
        const float3 axisVec = normalize(cross(orthoDirectionVec, viewVec));
        const float3 projectedNormalVec = viewNormal - (axisVec * dot(viewNormal, axisVec));
        const float projectedNormalVecLength = length(projectedNormalVec);
        if (projectedNormalVecLength < 1e-4)
        {
            continue;
        }

        const float signNorm = (dot(orthoDirectionVec, projectedNormalVec) < 0.0) ? -1.0 : 1.0;
        const float cosNorm = saturate(dot(projectedNormalVec, viewVec) / projectedNormalVecLength);
        const float n = signNorm * fastAcos(cosNorm);

        // Horizons start at the surface plane, so a flat surface self-occludes to
        // exactly zero rather than to some biased fraction.
        const float lowHorizonCos0 = cos(n + kPiHalf);
        const float lowHorizonCos1 = cos(n - kPiHalf);
        float horizonCos0 = lowHorizonCos0;
        float horizonCos1 = lowHorizonCos1;

        for (int step = 0; step < stepsPerSlice; ++step)
        {
            // Golden-ratio offset per step on top of the per-pixel noise, so two
            // slices of the same pixel do not sample the same distances.
            const float stepBaseNoise = float(slice + (step * stepsPerSlice)) * 0.6180339887498948482;
            const float stepNoise = frac(noise.y + stepBaseNoise);
            float s = (float(step) + stepNoise) / float(stepsPerSlice);
            s = pow(s, pushConstants.sampleDistributionPower);
            s += minS;

            const float2 sampleOffset = s * omega;
            const float sampleOffsetLength = length(sampleOffset * screenSize);
            // MIP whose texel footprint matches this step's length.
            const float mipLevel =
                clamp(log2(max(sampleOffsetLength, 1.0)) - kDepthMipSamplingOffset,
                      0.0, kDepthMipLevels - 1.0);

            const float2 uv0 = uv + sampleOffset;
            const float2 uv1 = uv - sampleOffset;
            // Uniform across the wave in practice (neighbouring pixels take the
            // same step), so this indexes without divergence.
            const int mipIndex = int(mipLevel);
            const float depth0 = depthPyramid[mipIndex].SampleLevel(saturate(uv0), 0.0);
            const float depth1 = depthPyramid[mipIndex].SampleLevel(saturate(uv1), 0.0);

            const float3 samplePos0 = viewPositionFromUv(uv0, depth0);
            const float3 samplePos1 = viewPositionFromUv(uv1, depth1);
            const float3 sampleDelta0 = samplePos0 - centerPos;
            const float3 sampleDelta1 = samplePos1 - centerPos;
            const float sampleDist0 = max(length(sampleDelta0), 1e-5);
            const float sampleDist1 = max(length(sampleDelta1), 1e-5);

            // Thin-occluder compensation: stretching the falloff along view depth
            // makes a sample that is far BEHIND the centre fall off sooner, so a
            // fence post stops occluding as though it were a wall extending away
            // from the camera.
            const float thin = 1.0 + pushConstants.thinOccluderCompensation;
            const float falloffBase0 =
                length(float3(sampleDelta0.x, sampleDelta0.y, sampleDelta0.z * thin));
            const float falloffBase1 =
                length(float3(sampleDelta1.x, sampleDelta1.y, sampleDelta1.z * thin));
            const float weight0 = saturate((falloffBase0 * falloffMul) + falloffAdd);
            const float weight1 = saturate((falloffBase1 * falloffMul) + falloffAdd);

            float shc0 = dot(sampleDelta0 / sampleDist0, viewVec);
            float shc1 = dot(sampleDelta1 / sampleDist1, viewVec);
            shc0 = lerp(lowHorizonCos0, shc0, weight0);
            shc1 = lerp(lowHorizonCos1, shc1, weight1);

            horizonCos0 = max(horizonCos0, shc0);
            horizonCos1 = max(horizonCos1, shc1);
        }

        const float h0 = -fastAcos(clamp(horizonCos1, -1.0, 1.0));
        const float h1 = fastAcos(clamp(horizonCos0, -1.0, 1.0));
        const float sinN = sin(n);
        const float iarc0 = (cosNorm + (2.0 * h0 * sinN) - cos((2.0 * h0) - n)) * 0.25;
        const float iarc1 = (cosNorm + (2.0 * h1 * sinN) - cos((2.0 * h1) - n)) * 0.25;
        visibility += projectedNormalVecLength * (iarc0 + iarc1);

        // Bent normal for this slice: the centroid of the unoccluded arc, lifted
        // out of the slice frame and weighted the same way visibility is.
        const float t0 = ((6.0 * sin(h0 - n)) - sin((3.0 * h0) - n) +
                          (6.0 * sin(h1 - n)) - sin((3.0 * h1) - n) +
                          (16.0 * sinN) - (3.0 * (sin(h0 + n) + sin(h1 + n)))) * 0.08333333333;
        const float t1 = (-cos((3.0 * h0) - n) - cos((3.0 * h1) - n) +
                          (8.0 * cos(n)) - (3.0 * (cos(h0 + n) + cos(h1 + n)))) * 0.08333333333;
        const float3 localBentNormal = float3(directionVec.x * t0, directionVec.y * t0, -t1);
        bentNormal +=
            mul(rotFromTo(float3(0.0, 0.0, -1.0), viewVec), localBentNormal) *
            projectedNormalVecLength;
    }

    visibility = saturate(visibility / float(sliceCount));
    visibility = pow(max(visibility, 1e-4), pushConstants.finalValuePower);

    bentNormal = normalize(bentNormal);
    // Degenerate slices (all of them clipped) leave a zero vector; fall back to
    // the geometric normal rather than writing a NaN the ambient term would
    // spread across the frame.
    if (any(isnan(bentNormal)) || dot(bentNormal, bentNormal) < 1e-6)
    {
        bentNormal = viewNormal;
    }

    aoOutput[pixelCoord] = visibility;
    bentNormalOutput[pixelCoord] = float4((bentNormal * 0.5) + 0.5, packedEdges);
}
