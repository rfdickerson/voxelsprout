// Screen-space ambient occlusion. One source, three estimators selected at compile
// time by ODAI_AO_MODE, each built into its own pipeline (see add_slang_shader_variant
// in CMakeLists.txt) so the sample loop carries no uniform branch:
//
//   0 (default) SSAO  -- normal-oriented hemisphere point sampling. Cheapest, and the
//                        look this project shipped with.
//   1           HBAO  -- horizon-based (Bavoil 2008). Marches directions in screen space
//                        and integrates the elevation angle of the highest occluder, so
//                        it captures contact darkening a point sampler misses.
//   2           GTAO  -- ground-truth AO (Jimenez 2016). Same horizon search as HBAO but
//                        with the correct cosine-weighted visibility integral and a
//                        projected-normal frame, so it converges on ray-traced AO
//                        instead of approximating it.
//
// All three share the scaffolding below (normal/position reconstruction, the noise, the
// radius/bias uniforms and the write-out); only estimateOcclusion() differs.
#ifndef ODAI_AO_MODE
#define ODAI_AO_MODE 0
#endif

#define ODAI_AO_MODE_SSAO 0
#define ODAI_AO_MODE_HBAO 1
#define ODAI_AO_MODE_GTAO 2

import camera_uniform;

[[vk::binding(1, 0)]]
Sampler2D<float4> normalDepthTexture;

[[vk::binding(2, 0)]]
RWTexture2D<float> ssaoRawTexture;

struct PushConstants
{
    uint width;
    uint height;
};

[[vk::push_constant]]
PushConstants pushConstants;

static const float kSsaoPower = 1.4;

// Interleaved gradient noise (Jimenez 2014): a blue-noise-like dither with a much
// less correlated spatial spectrum than a per-pixel hash, so the residual SSAO
// dither reads as fine grain instead of the diagonal streaking hash12 left behind
// after the bilateral blur softened (but didn't remove) its structure.
float interleavedGradientNoise(float2 pixelCoord)
{
    const float3 magic = float3(0.06711056, 0.00583715, 52.9829189);
    return frac(magic.z * frac(dot(pixelCoord, magic.xy)));
}

float3 decodeViewNormal(float3 encodedNormal)
{
    return normalize((encodedNormal * 2.0) - 1.0);
}

float3 reconstructViewPosition(float2 uv, float viewDepth)
{
    const float2 ndcXY = (uv * 2.0) - 1.0;
    float projX = camera.proj[0][0];
    float projY = camera.proj[1][1];
    if (abs(projX) < 1e-6) projX = (projX < 0.0) ? -1e-6 : 1e-6;
    if (abs(projY) < 1e-6) projY = (projY < 0.0) ? -1e-6 : 1e-6;

    // Perspective: view-space XY scale with depth (ndc = view * proj / -viewZ).
    // Orthographic: they do not — ndc = view * proj directly. Detect via the
    // projection's [3][3] diagonal (1 for ortho, 0 for perspective; diagonal
    // entries are invariant under the CPU-side transpose).
    const float depthScale = (camera.proj[3][3] > 0.5) ? 1.0 : viewDepth;
    const float viewX = (ndcXY.x * depthScale) / projX;
    const float viewY = (ndcXY.y * depthScale) / projY;
    return float3(viewX, viewY, -viewDepth);
}

// Fetches a neighbour's view position. Returns false on sky/background (the prepass
// clears depth to 0), so estimators can reject those instead of treating the clear
// value as geometry sitting at the near plane.
bool viewPositionAt(float2 uv, out float3 outViewPosition)
{
    const float depth = normalDepthTexture.SampleLevel(uv, 0.0).w;
    if (depth <= 0.0001)
    {
        outViewPosition = float3(0.0, 0.0, 0.0);
        return false;
    }
    outViewPosition = reconstructViewPosition(uv, depth);
    return true;
}

// Screen-space length, in pixels, of a world-space radius at this depth. Perspective
// shrinks it with distance; orthographic does not (same [3][3] test as above).
float radiusInPixels(float radius, float viewDepth, float2 screenSize)
{
    const float projX = max(abs(camera.proj[0][0]), 1e-6);
    const float depthScale = (camera.proj[3][3] > 0.5) ? 1.0 : max(viewDepth, 1e-4);
    return (radius * projX * 0.5 * screenSize.x) / depthScale;
}

#if ODAI_AO_MODE == ODAI_AO_MODE_SSAO

// Normal-oriented hemisphere point sampling: scatter points inside the hemisphere,
// reproject each, and count the ones the depth buffer says are buried.
float estimateVisibility(float2 uv, float2 screenSize, float3 centerViewPos,
                         float3 centerNormal, float randomAngle, float radius, float bias)
{
    static const int kSampleCount = 32;
    static const float kGoldenAngle = 2.39996323;

    float3 tangent = normalize(cross(centerNormal, float3(0.0, 1.0, 0.0)));
    if (dot(tangent, tangent) <= 0.0001)
    {
        tangent = normalize(cross(centerNormal, float3(1.0, 0.0, 0.0)));
    }
    const float3 bitangent = normalize(cross(centerNormal, tangent));

    const float s = sin(randomAngle);
    const float c = cos(randomAngle);
    const float2x2 rot = float2x2(c, -s, s, c);

    float occlusion = 0.0;
    float sampleCount = 0.0;
    for (int i = 0; i < kSampleCount; ++i)
    {
        const float fi = float(i);
        const float t = (fi + 0.5) / float(kSampleCount);
        const float r = sqrt(t);
        const float angle = (fi * kGoldenAngle) + randomAngle;
        const float2 disk = mul(rot, float2(cos(angle), sin(angle)) * r);
        const float z = sqrt(max(1.0 - dot(disk, disk), 0.0));
        float3 k = float3(disk.x, disk.y, z);
        k *= lerp(0.15, 1.0, t * t);

        const float3 sampleOffset =
            (tangent * k.x) + (bitangent * k.y) + (centerNormal * abs(k.z));
        const float3 sampleViewPos = centerViewPos + (sampleOffset * radius);

        const float4 sampleClip = mul(camera.proj, float4(sampleViewPos, 1.0));
        if (sampleClip.w <= 1e-4) continue;

        const float3 sampleNdc = sampleClip.xyz / sampleClip.w;
        const float2 sampleUv = (sampleNdc.xy * 0.5) + 0.5;
        if (sampleUv.x <= 0.0 || sampleUv.x >= 1.0 || sampleUv.y <= 0.0 || sampleUv.y >= 1.0) continue;

        float3 sceneViewPos;
        if (!viewPositionAt(sampleUv, sceneViewPos)) continue;

        const float depthDelta = abs(centerViewPos.z - sceneViewPos.z);
        const float rangeWeight = smoothstep(0.0, 1.0, radius / max(depthDelta, 1e-4));
        const float occluded = (sceneViewPos.z > (sampleViewPos.z + bias)) ? 1.0 : 0.0;

        occlusion += occluded * rangeWeight;
        sampleCount += 1.0;
    }

    if (sampleCount <= 0.0) return 1.0;
    return saturate(1.0 - (occlusion / sampleCount));
}

#else  // HBAO and GTAO share the horizon search; only the integral differs.

static const int kDirectionCount = 4;
static const int kStepCount = 6;
static const float kPi = 3.14159265359;

// Distance falloff shared by both horizon estimators: occluders fade out as they
// approach the radius so the effect ends smoothly instead of popping at the edge.
float horizonFalloff(float distance, float radius)
{
    return saturate(1.0 - (distance * distance) / max(radius * radius, 1e-6));
}

#if ODAI_AO_MODE == ODAI_AO_MODE_HBAO

// Horizon-based AO (Bavoil 2008). Per direction, march outward and track the highest
// elevation above the tangent plane; the sine of that horizon angle is the occluded
// fraction of the hemisphere in that direction.
float estimateVisibility(float2 uv, float2 screenSize, float3 centerViewPos,
                         float3 centerNormal, float randomAngle, float radius, float bias)
{
    const float radiusPx = radiusInPixels(radius, -centerViewPos.z, screenSize);
    if (radiusPx < 1.0) return 1.0;
    const float stepPx = radiusPx / float(kStepCount);

    float occlusion = 0.0;
    for (int d = 0; d < kDirectionCount; ++d)
    {
        const float phi = (float(d) * kPi / float(kDirectionCount)) + randomAngle;
        const float2 dir = float2(cos(phi), sin(phi));

        // Both sides of the slice are marched as independent directions, which is
        // what makes 4 slices cover 8 directions.
        for (int side = 0; side < 2; ++side)
        {
            const float2 marchDir = (side == 0) ? dir : -dir;
            float sinHorizon = 0.0;   // tangent plane is the horizon floor
            float weight = 0.0;
            for (int s = 1; s <= kStepCount; ++s)
            {
                const float2 sampleUv = uv + (marchDir * (stepPx * float(s))) / screenSize;
                if (sampleUv.x <= 0.0 || sampleUv.x >= 1.0 || sampleUv.y <= 0.0 || sampleUv.y >= 1.0) break;

                float3 sampleViewPos;
                if (!viewPositionAt(sampleUv, sampleViewPos)) continue;

                const float3 delta = sampleViewPos - centerViewPos;
                const float dist = length(delta);
                if (dist < 1e-5 || dist > radius) continue;

                // dot(delta/|delta|, N) is the sine of the elevation angle measured
                // from the tangent plane, so no trigonometry is needed to find it.
                const float sinElevation = dot(delta / dist, centerNormal) - bias;
                if (sinElevation > sinHorizon)
                {
                    sinHorizon = sinElevation;
                    weight = horizonFalloff(dist, radius);
                }
            }
            occlusion += saturate(sinHorizon) * weight;
        }
    }

    return saturate(1.0 - (occlusion / float(kDirectionCount * 2)));
}

#else  // ODAI_AO_MODE_GTAO

// Ground-truth AO (Jimenez et al. 2016). Same horizon search, but the two horizons of
// each slice are fed through the cosine-weighted arc integral and the normal is
// projected into the slice plane -- which is what makes this converge on ray-traced
// AO rather than approximate it the way HBAO's sine term does.
float estimateVisibility(float2 uv, float2 screenSize, float3 centerViewPos,
                         float3 centerNormal, float randomAngle, float radius, float bias)
{
    const float radiusPx = radiusInPixels(radius, -centerViewPos.z, screenSize);
    if (radiusPx < 1.0) return 1.0;
    const float stepPx = radiusPx / float(kStepCount);
    const float3 viewVec = normalize(-centerViewPos);

    float visibility = 0.0;
    for (int d = 0; d < kDirectionCount; ++d)
    {
        const float phi = (float(d) * kPi / float(kDirectionCount)) + randomAngle;
        const float2 dir = float2(cos(phi), sin(phi));

        // Largest cosine found on each side = smallest angle from the view vector.
        float cosH1 = -1.0;
        float cosH2 = -1.0;
        for (int s = 1; s <= kStepCount; ++s)
        {
            const float2 offset = (dir * (stepPx * float(s))) / screenSize;

            float3 p1;
            const float2 uv1 = uv + offset;
            if (uv1.x > 0.0 && uv1.x < 1.0 && uv1.y > 0.0 && uv1.y < 1.0 && viewPositionAt(uv1, p1))
            {
                const float3 delta = p1 - centerViewPos;
                const float dist = length(delta);
                if (dist > 1e-5)
                {
                    const float c = dot(delta / dist, viewVec);
                    cosH1 = max(cosH1, lerp(-1.0, c, horizonFalloff(dist, radius)));
                }
            }

            float3 p2;
            const float2 uv2 = uv - offset;
            if (uv2.x > 0.0 && uv2.x < 1.0 && uv2.y > 0.0 && uv2.y < 1.0 && viewPositionAt(uv2, p2))
            {
                const float3 delta = p2 - centerViewPos;
                const float dist = length(delta);
                if (dist > 1e-5)
                {
                    const float c = dot(delta / dist, viewVec);
                    cosH2 = max(cosH2, lerp(-1.0, c, horizonFalloff(dist, radius)));
                }
            }
        }

        // Slice frame: the normal projected onto the plane spanned by the view vector
        // and the march direction. gamma is its signed angle from the view vector.
        const float3 directionVec = float3(dir, 0.0);
        const float3 orthoDirectionVec = directionVec - (dot(directionVec, viewVec) * viewVec);
        const float3 axisVec = normalize(cross(directionVec, viewVec));
        const float3 projNormal = centerNormal - (axisVec * dot(centerNormal, axisVec));
        const float projNormalLen = length(projNormal);
        if (projNormalLen < 1e-4) continue;

        const float cosGamma = clamp(dot(projNormal, viewVec) / projNormalLen, -1.0, 1.0);
        const float sgn = (dot(orthoDirectionVec, projNormal) < 0.0) ? -1.0 : 1.0;
        const float gamma = sgn * acos(cosGamma);
        const float sinGamma = sin(gamma);

        // Horizons, clamped into the hemisphere around the projected normal, with the
        // bias pulling them in slightly to keep flat surfaces from self-occluding.
        float h1 = -acos(clamp(cosH1, -1.0, 1.0));
        float h2 = acos(clamp(cosH2, -1.0, 1.0));
        h1 = gamma + max(h1 - gamma, -kPi * 0.5 + bias);
        h2 = gamma + min(h2 - gamma, kPi * 0.5 - bias);

        const float arc =
            0.25 * (-cos((2.0 * h1) - gamma) + cosGamma + ((2.0 * h1) * sinGamma)) +
            0.25 * (-cos((2.0 * h2) - gamma) + cosGamma + ((2.0 * h2) * sinGamma));
        visibility += projNormalLen * arc;
    }

    return saturate(visibility / float(kDirectionCount));
}

#endif  // HBAO / GTAO
#endif  // estimator selection

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= pushConstants.width || dispatchThreadId.y >= pushConstants.height)
    {
        return;
    }

    const float2 screenSize = float2(float(pushConstants.width), float(pushConstants.height));
    const float2 uv = (float2(dispatchThreadId.xy) + 0.5) / screenSize;

    const float4 centerSample = normalDepthTexture.SampleLevel(uv, 0.0);
    const float centerDepth = centerSample.w;
    if (centerDepth <= 0.0001)
    {
        ssaoRawTexture[dispatchThreadId.xy] = 1.0;
        return;
    }

    const float3 centerNormal = decodeViewNormal(centerSample.xyz);
    const float3 centerViewPos = reconstructViewPosition(uv, centerDepth);
    const float randomAngle = interleavedGradientNoise(float2(dispatchThreadId.xy)) * 6.2831853;

    const float radius = clamp(camera.shadowConfig2.x, 0.25, 128.0);
    const float bias = clamp(camera.shadowConfig2.y, 0.0, 8.0);

    const float visibility =
        estimateVisibility(uv, screenSize, centerViewPos, centerNormal, randomAngle, radius, bias);
    ssaoRawTexture[dispatchThreadId.xy] = pow(saturate(visibility), kSsaoPower);
}
