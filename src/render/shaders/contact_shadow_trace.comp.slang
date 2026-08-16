import camera_uniform;
import light_clusters;

[[vk::binding(1, 0)]]
Sampler2D<float4> normalDepthTexture;

[[vk::binding(2, 0)]]
StructuredBuffer<uint2> clusterLightMask;

[[vk::binding(3, 0)]]
StructuredBuffer<float2> depthHierarchy;

[[vk::binding(4, 0)]]
RWStructuredBuffer<uint4> halfContactRecords;

struct ContactShadowPushConstants
{
    uint4 extent;
    uint4 dispatch;     // mip count, unused, unused, temporal phase
    uint4 cluster;
    float4 params;      // slice scale/bias, max ray distance, unused
};

[[vk::push_constant]]
ContactShadowPushConstants pc;

float3 decodeViewNormal(float3 encoded)
{
    return normalize((encoded * 2.0) - 1.0);
}

float3 viewPositionFromUv(float2 uv, float viewDepth)
{
    const float2 ndc = ((uv * 2.0) - 1.0) - camera.tonemapConfig2.zw;
    const float projX = abs(camera.proj[0][0]) < 1e-6 ? 1e-6 : camera.proj[0][0];
    const float projY = abs(camera.proj[1][1]) < 1e-6 ? 1e-6 : camera.proj[1][1];
    const float scale = camera.proj[3][3] > 0.5 ? 1.0 : viewDepth;
    return float3(ndc.x * scale / projX, ndc.y * scale / projY, -viewDepth);
}

uint hierarchyOffset(uint mip)
{
    uint offset = 0u;
    uint2 extent = max(pc.extent.zw, uint2(1u, 1u));
    for (uint i = 0u; i < mip; ++i)
    {
        offset += extent.x * extent.y;
        extent = max((extent + 1u) / 2u, uint2(1u, 1u));
    }
    return offset;
}

uint2 hierarchyExtent(uint mip)
{
    uint2 extent = max(pc.extent.zw, uint2(1u, 1u));
    for (uint i = 0u; i < mip; ++i)
    {
        extent = max((extent + 1u) / 2u, uint2(1u, 1u));
    }
    return extent;
}

float2 sampleDepthRange(float2 uv, uint mip)
{
    const uint2 extent = hierarchyExtent(mip);
    const uint2 pixel = min(uint2(saturate(uv) * float2(extent)), extent - 1u);
    return depthHierarchy[hierarchyOffset(mip) + pixel.y * extent.x + pixel.x];
}

bool projectWorld(float3 worldPosition, out float2 uv, out float viewDepth)
{
    const float4 clip = mul(camera.mvp, float4(worldPosition, 1.0));
    if (clip.w <= 1e-5)
    {
        uv = 0.0;
        viewDepth = 0.0;
        return false;
    }
    uv = ((clip.xy / clip.w) * 0.5) + 0.5;
    viewDepth = max(-mul(camera.view, float4(worldPosition, 1.0)).z, 0.0);
    return all(uv > float2(0.0, 0.0)) && all(uv < float2(1.0, 1.0));
}

float contactNoise(uint2 pixel, uint phase)
{
    uint value = pixel.x * 1973u + pixel.y * 9277u + (phase & 3u) * 26699u + 911u;
    value = (value << 13u) ^ value;
    value *= value * value * 15731u + 789221u;
    return float(value & 0x00ffffffu) / 16777216.0;
}

bool traceContactShadow(
    float3 receiverWorld,
    float3 receiverNormal,
    float3 lightPosition,
    float lightRadius,
    float ndotl,
    uint2 pixel)
{
    const float receiverOffset = max(1.5, lightRadius * 0.0025) * (1.0 + (1.0 - ndotl));
    const float3 start = receiverWorld + receiverNormal * receiverOffset;
    const float3 toLight = lightPosition - start;
    const float lightDistance = length(toLight);
    const float maxDistance = min(min(lightDistance - 1.0, pc.params.z), 160.0);
    if (maxDistance <= 3.0)
    {
        return false;
    }
    const float3 rayDirection = toLight / max(lightDistance, 1e-4);
    const float jitter = contactNoise(pixel, pc.dispatch.w);
    float previousT = 2.0;
    float2 previousUv = 0.0;
    float previousRayDepth = 0.0;
    bool previousProjected =
        projectWorld(start + rayDirection * previousT, previousUv, previousRayDepth);

    [unroll]
    for (uint step = 0u; step < 8u; ++step)
    {
        const float stepFraction = (float(step) + 0.35 + jitter * 0.3) / 8.0;
        const float t = lerp(2.0, maxDistance, stepFraction);
        float2 uv;
        float rayDepth;
        if (!projectWorld(start + rayDirection * t, uv, rayDepth))
        {
            return false;
        }

        const float pixelTravel = previousProjected
            ? length((uv - previousUv) * float2(pc.extent.zw))
            : 1.0;
        const uint mip = min(uint(max(floor(log2(max(pixelTravel, 1.0))) - 1.0, 0.0)),
                             max(pc.dispatch.x, 1u) - 1u);
        const float2 depthRange = sampleDepthRange(uv, mip);
        if (depthRange.y > 0.0 && rayDepth >= depthRange.x)
        {
            float low = previousT;
            float high = t;
            float finalDelta = 1e30;
            [unroll]
            for (uint refine = 0u; refine < 3u; ++refine)
            {
                const float middle = (low + high) * 0.5;
                float2 middleUv;
                float middleRayDepth;
                if (!projectWorld(start + rayDirection * middle, middleUv, middleRayDepth))
                {
                    return false;
                }
                const float sceneDepth = normalDepthTexture.SampleLevel(middleUv, 0.0).w;
                finalDelta = sceneDepth > 1e-4 ? middleRayDepth - sceneDepth : -1.0;
                if (finalDelta >= 0.0)
                {
                    high = middle;
                }
                else
                {
                    low = middle;
                }
            }
            float2 hitUv;
            float hitRayDepth;
            if (!projectWorld(start + rayDirection * high, hitUv, hitRayDepth))
            {
                return false;
            }
            const float sceneDepth = normalDepthTexture.SampleLevel(hitUv, 0.0).w;
            const float delta = sceneDepth > 1e-4 ? hitRayDepth - sceneDepth : -1.0;
            const float thickness = clamp(2.0 + hitRayDepth * 0.001, 2.0, 8.0);
            if (delta >= 0.0 && delta <= thickness)
            {
                return true;
            }
        }
        previousT = t;
        previousUv = uv;
        previousProjected = true;
    }
    return false;
}

uint packOctNormal(float3 normal)
{
    normal /= abs(normal.x) + abs(normal.y) + abs(normal.z) + 1e-6;
    float2 oct = normal.xy;
    if (normal.z < 0.0)
    {
        oct = (1.0 - abs(oct.yx)) *
            select(oct >= 0.0, float2(1.0, 1.0), float2(-1.0, -1.0));
    }
    const uint2 encoded = uint2(round(saturate(oct * 0.5 + 0.5) * 65535.0));
    return encoded.x | (encoded.y << 16u);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= pc.extent.zw))
    {
        return;
    }
    const uint outputIndex = dispatchThreadId.y * pc.extent.z + dispatchThreadId.x;
    // Pick the centre of the lower-right full-resolution sample in this 2x2
    // footprint.  The old +1.0 coordinate landed exactly on the boundary
    // between full-resolution texels, blending unrelated depth and normals at
    // silhouette edges and turning those discontinuities into contact acne.
    const float2 fullPixel = float2(dispatchThreadId.xy * 2u + 1u) + 0.5;
    const float2 uv = fullPixel / float2(pc.extent.xy);
    const float4 normalDepth = normalDepthTexture.SampleLevel(uv, 0.0);
    const float viewDepth = normalDepth.w;
    if (viewDepth <= 1e-4 || !lightClustersActive())
    {
        halfContactRecords[outputIndex] = uint4(0u, 0u, asuint(viewDepth), 0u);
        return;
    }

    const float3 viewNormal = decodeViewNormal(normalDepth.xyz);
    const float3 viewPosition = viewPositionFromUv(uv, viewDepth);
    const float3 worldPosition = mul(camera.contactShadowInvView, float4(viewPosition, 1.0)).xyz;
    const float3 worldNormal = normalize(mul((float3x3)camera.contactShadowInvView, viewNormal));
    uint2 candidates = clusterLightMask[lightClusterIndexForFragment(fullPixel, viewDepth)];

    float bestScore[4] = { -1.0, -1.0, -1.0, -1.0 };
    int bestIndex[4] = { -1, -1, -1, -1 };
    while ((candidates.x | candidates.y) != 0u)
    {
        int lightIndex;
        if (candidates.x != 0u)
        {
            lightIndex = int(firstbitlow(candidates.x));
            candidates.x &= candidates.x - 1u;
        }
        else
        {
            lightIndex = 32 + int(firstbitlow(candidates.y));
            candidates.y &= candidates.y - 1u;
        }
        const float4 positionRadius = camera.importedLightPositionRadius[lightIndex];
        const float4 colorIntensity = camera.importedLightColorIntensity[lightIndex];
        const float3 toLight = positionRadius.xyz - worldPosition;
        const float distanceSquared = dot(toLight, toLight);
        const float radius = max(positionRadius.w, 1.0);
        if (distanceSquared >= radius * radius)
        {
            continue;
        }
        const float distanceToLight = sqrt(max(distanceSquared, 1e-4));
        const float ndotl = saturate(dot(worldNormal, toLight / distanceToLight));
        if (ndotl <= 1e-4)
        {
            continue;
        }
        const float normalizedDistance = saturate(distanceToLight / radius);
        const float falloff = saturate(1.0 - normalizedDistance * normalizedDistance);
        const float luminance = dot(colorIntensity.rgb, float3(0.2126, 0.7152, 0.0722));
        const float score = luminance * colorIntensity.w * ndotl * falloff * falloff;
        for (int slot = 0; slot < 4; ++slot)
        {
            if (score > bestScore[slot])
            {
                for (int move = 3; move > slot; --move)
                {
                    bestScore[move] = bestScore[move - 1];
                    bestIndex[move] = bestIndex[move - 1];
                }
                bestScore[slot] = score;
                bestIndex[slot] = lightIndex;
                break;
            }
        }
    }

    uint2 occluded = uint2(0u, 0u);
    [unroll]
    for (int slot = 0; slot < 4; ++slot)
    {
        const int lightIndex = bestIndex[slot];
        if (lightIndex < 0)
        {
            continue;
        }
        const float4 positionRadius = camera.importedLightPositionRadius[lightIndex];
        const float3 toLight = positionRadius.xyz - worldPosition;
        const float distanceToLight = max(length(toLight), 1e-4);
        const float ndotl = saturate(dot(worldNormal, toLight / distanceToLight));
        if (traceContactShadow(worldPosition, worldNormal, positionRadius.xyz,
                               positionRadius.w, ndotl, dispatchThreadId.xy))
        {
            if (lightIndex < 32)
            {
                occluded.x |= 1u << uint(lightIndex);
            }
            else
            {
                occluded.y |= 1u << uint(lightIndex - 32);
            }
        }
    }
    halfContactRecords[outputIndex] =
        uint4(occluded.x, occluded.y, asuint(viewDepth), packOctNormal(viewNormal));
}
