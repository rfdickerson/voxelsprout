import camera_uniform;

[[vk::binding(1, 0)]]
Sampler2D<float4> normalDepthTexture;

[[vk::binding(2, 0)]]
Sampler2D<float4> previousHdrTexture;

[[vk::binding(3, 0)]]
StructuredBuffer<float2> depthHierarchy;

[[vk::binding(4, 0)]]
StructuredBuffer<uint4> previousRecords;

[[vk::binding(5, 0)]]
RWStructuredBuffer<uint4> currentRecords;

struct ScreenSpaceGiPushConstants
{
    float4x4 prevViewProj;
    uint4 extent;       // full width/height, quarter width/height
    uint4 dispatch;     // hierarchy mip count, eight-frame phase, history valid, unused
    float4 params;      // max distance, source luma clamp, previous jitter NDC xy
};

[[vk::push_constant]]
ScreenSpaceGiPushConstants pc;

float3 decodeViewNormal(float3 encoded)
{
    return normalize((encoded * 2.0) - 1.0);
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

float3 unpackOctNormal(uint packed)
{
    const uint2 encoded = uint2(packed & 0xffffu, packed >> 16u);
    float2 oct = (float2(encoded) / 65535.0) * 2.0 - 1.0;
    float3 normal = float3(oct, 1.0 - abs(oct.x) - abs(oct.y));
    if (normal.z < 0.0)
    {
        normal.xy = (1.0 - abs(normal.yx)) *
            select(normal.xy >= 0.0, float2(1.0, 1.0), float2(-1.0, -1.0));
    }
    return normalize(normal);
}

uint2 packIrradiance(float3 irradiance)
{
    return uint2(
        (f32tof16(irradiance.x) & 0xffffu) | (f32tof16(irradiance.y) << 16u),
        f32tof16(irradiance.z) & 0xffffu);
}

float3 viewPositionFromUv(float2 uv, float viewDepth)
{
    const float2 ndc = ((uv * 2.0) - 1.0) - camera.tonemapConfig2.zw;
    const float projX = abs(camera.proj[0][0]) < 1e-6 ? 1e-6 : camera.proj[0][0];
    const float projY = abs(camera.proj[1][1]) < 1e-6 ? 1e-6 : camera.proj[1][1];
    return float3(ndc.x * viewDepth / projX, ndc.y * viewDepth / projY, -viewDepth);
}

uint2 hierarchyExtent(uint mip)
{
    uint2 result = max((pc.extent.xy + 1u) / 2u, uint2(1u, 1u));
    for (uint i = 0u; i < mip; ++i)
    {
        result = max((result + 1u) / 2u, uint2(1u, 1u));
    }
    return result;
}

uint hierarchyOffset(uint mip)
{
    uint offset = 0u;
    uint2 mipSize = max((pc.extent.xy + 1u) / 2u, uint2(1u, 1u));
    for (uint i = 0u; i < mip; ++i)
    {
        offset += mipSize.x * mipSize.y;
        mipSize = max((mipSize + 1u) / 2u, uint2(1u, 1u));
    }
    return offset;
}

float2 sampleDepthRange(float2 uv, uint mip)
{
    const uint2 mipSize = hierarchyExtent(mip);
    const uint2 pixel = min(uint2(saturate(uv) * float2(mipSize)), mipSize - 1u);
    return depthHierarchy[hierarchyOffset(mip) + pixel.y * mipSize.x + pixel.x];
}

bool projectCurrent(float3 worldPosition, out float2 uv, out float viewDepth)
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

float random01(uint2 pixel, uint phase, uint salt)
{
    uint value = pixel.x * 1973u + pixel.y * 9277u + phase * 26699u + salt * 104729u;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return float(value & 0x00ffffffu) / 16777216.0;
}

float3 cosineHemisphereDirection(float3 normal, uint2 pixel, uint phase)
{
    const float u = random01(pixel, phase, 1u);
    const float v = random01(pixel, phase, 2u);
    const float radius = sqrt(u);
    const float angle = v * 6.28318530718;
    const float3 local = float3(radius * cos(angle), radius * sin(angle), sqrt(1.0 - u));
    const float3 helper = abs(normal.z) < 0.999
        ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    const float3 tangent = normalize(cross(helper, normal));
    const float3 bitangent = cross(normal, tangent);
    return normalize((tangent * local.x) + (bitangent * local.y) + (normal * local.z));
}

bool traceDiffuseRay(
    float3 start,
    float3 direction,
    uint2 pixel,
    out float2 hitUv,
    out float3 hitWorld,
    out float hitDistance)
{
    hitUv = float2(0.0, 0.0);
    hitWorld = float3(0.0, 0.0, 0.0);
    hitDistance = 0.0;
    const float phaseJitter = random01(pixel, pc.dispatch.y, 3u);
    float previousT = 4.0;
    float2 previousUv = 0.0;
    bool previousProjected = false;
    [unroll]
    // The original 8/3 trace exceeded the 2 ms p95 target under sustained
    // load on the LNL iGPU. Keep the quarter-resolution receiver density (the
    // part that preserves small bounce details) and take the policy's ordered
    // fallback instead: six hierarchy steps and two binary refinements.
    for (uint step = 0u; step < 6u; ++step)
    {
        const float fraction = saturate((float(step) + 0.35 + phaseJitter * 0.3) / 6.0);
        const float t = lerp(4.0, pc.params.x, fraction * fraction);
        float2 uv;
        float rayDepth;
        if (!projectCurrent(start + direction * t, uv, rayDepth))
        {
            return false;
        }
        const float pixelTravel = previousProjected
            ? length((uv - previousUv) * float2(pc.extent.xy) * 0.5)
            : 1.0;
        const uint mip = min(
            uint(max(floor(log2(max(pixelTravel, 1.0))) - 1.0, 0.0)),
            max(pc.dispatch.x, 1u) - 1u);
        const float2 range = sampleDepthRange(uv, mip);
        if (range.y > 0.0 && rayDepth >= range.x)
        {
            float low = previousT;
            float high = t;
            [unroll]
            for (uint refine = 0u; refine < 2u; ++refine)
            {
                const float middle = (low + high) * 0.5;
                float2 middleUv;
                float middleDepth;
                if (!projectCurrent(start + direction * middle, middleUv, middleDepth))
                {
                    return false;
                }
                const float sceneDepth = normalDepthTexture.SampleLevel(middleUv, 0.0).w;
                if (sceneDepth > 1e-4 && middleDepth >= sceneDepth)
                {
                    high = middle;
                }
                else
                {
                    low = middle;
                }
            }
            float rayDepthAtHit;
            if (!projectCurrent(start + direction * high, hitUv, rayDepthAtHit))
            {
                return false;
            }
            const float sceneDepth = normalDepthTexture.SampleLevel(hitUv, 0.0).w;
            const float delta = sceneDepth > 1e-4 ? rayDepthAtHit - sceneDepth : -1.0;
            const float thickness = clamp(4.0 + rayDepthAtHit * 0.0015, 4.0, 16.0);
            if (delta >= 0.0 && delta <= thickness)
            {
                const float3 hitView = viewPositionFromUv(hitUv, sceneDepth);
                hitWorld = mul(camera.contactShadowInvView, float4(hitView, 1.0)).xyz;
                hitDistance = high;
                return true;
            }
        }
        previousT = t;
        previousUv = uv;
        previousProjected = true;
    }
    return false;
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
    const uint2 fullPixel = min(dispatchThreadId.xy * 4u + 2u, pc.extent.xy - 1u);
    const float2 uv = (float2(fullPixel) + 0.5) / float2(pc.extent.xy);
    const float4 normalDepth = normalDepthTexture.SampleLevel(uv, 0.0);
    const float viewDepth = normalDepth.w;
    if (viewDepth <= 1e-4)
    {
        currentRecords[outputIndex] = uint4(0u, 0u, 0u, 0u);
        return;
    }

    const float3 viewNormal = decodeViewNormal(normalDepth.xyz);
    const float3 worldNormal = normalize(
        mul((float3x3)camera.contactShadowInvView, viewNormal));
    const uint guideNormal = packOctNormal(worldNormal);
    float3 incident = float3(0.0, 0.0, 0.0);

    if (pc.dispatch.z != 0u)
    {
        const float3 viewPosition = viewPositionFromUv(uv, viewDepth);
        const float3 worldPosition =
            mul(camera.contactShadowInvView, float4(viewPosition, 1.0)).xyz;
        const float originBias = clamp(viewDepth * 0.0015, 4.0, 16.0);
        const float3 rayDirection = cosineHemisphereDirection(
            worldNormal, dispatchThreadId.xy, pc.dispatch.y & 7u);
        float2 hitUv;
        float3 hitWorld;
        float hitDistance;
        if (traceDiffuseRay(
                worldPosition + worldNormal * originBias,
                rayDirection,
                dispatchThreadId.xy,
                hitUv,
                hitWorld,
                hitDistance))
        {
            const float4 hitNormalDepth = normalDepthTexture.SampleLevel(hitUv, 0.0);
            const float3 hitViewNormal = decodeViewNormal(hitNormalDepth.xyz);
            const float3 hitWorldNormal = normalize(
                mul((float3x3)camera.contactShadowInvView, hitViewNormal));
            const float facing = saturate(dot(hitWorldNormal, -rayDirection));
            const float4 previousClip = mul(pc.prevViewProj, float4(hitWorld, 1.0));
            if (facing > 1e-4 && previousClip.w > 1e-4)
            {
                const float2 previousNdc =
                    (previousClip.xy / previousClip.w) - pc.params.zw;
                const float2 previousUv = previousNdc * 0.5 + 0.5;
                if (all(previousUv > float2(0.0, 0.0)) &&
                    all(previousUv < float2(1.0, 1.0)))
                {
                    const uint2 previousPixel = min(
                        uint2(previousUv * float2(pc.extent.zw)), pc.extent.zw - 1u);
                    const uint4 previousGuide = previousRecords[
                        previousPixel.y * pc.extent.z + previousPixel.x];
                    const float previousDepth = asfloat(previousGuide.z);
                    const float3 previousNormal = unpackOctNormal(previousGuide.w);
                    const float expectedDepth = previousClip.w;
                    const float depthLimit = max(12.0, expectedDepth * 0.02);
                    if (previousDepth > 1e-4 &&
                        abs(previousDepth - expectedDepth) <= depthLimit &&
                        dot(previousNormal, hitWorldNormal) >= 0.5)
                    {
                        float3 source = max(
                            previousHdrTexture.SampleLevel(previousUv, 0.0).rgb,
                            float3(0.0, 0.0, 0.0));
                        const float sourceLuma = dot(
                            source, float3(0.2126, 0.7152, 0.0722));
                        source *= min(1.0, pc.params.y / max(sourceLuma, 1e-5));
                        const float distanceFade =
                            pow(saturate(1.0 - hitDistance / pc.params.x), 2.0);
                        incident = source * facing * distanceFade;
                    }
                }
            }
        }
    }

    const uint2 packedIncident = packIrradiance(incident);
    currentRecords[outputIndex] =
        uint4(packedIncident.x, packedIncident.y, asuint(viewDepth), guideNormal);
}
