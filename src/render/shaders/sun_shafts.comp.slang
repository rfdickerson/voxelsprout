import camera_uniform;

[[vk::binding(1, 0)]]
Sampler2D<float4> normalDepthTexture;

[[vk::binding(2, 0)]]
Sampler2DShadow shadowMap;

[[vk::binding(3, 0)]]
RWTexture2D<float4> sunShaftTexture;

struct PushConstants
{
    uint width;
    uint height;
    uint sampleCount;
    uint _pad0;
};

[[vk::push_constant]]
PushConstants pushConstants;

int chooseShadowCascade(float viewDepth)
{
    if (viewDepth <= camera.shadowCascadeSplits.x) return 0;
    if (viewDepth <= camera.shadowCascadeSplits.y) return 1;
    if (viewDepth <= camera.shadowCascadeSplits.z) return 2;
    return 3;
}

float sampleShadowVisibility(float3 worldPosition)
{
    const float viewDepth = max(-mul(camera.view, float4(worldPosition, 1.0)).z, 0.0);
    const int cascadeIndex = chooseShadowCascade(viewDepth);
    const float4 shadowClip = mul(camera.lightViewProj[cascadeIndex], float4(worldPosition, 1.0));
    if (shadowClip.w <= 0.0) return 1.0;

    const float3 shadowNdc = shadowClip.xyz / shadowClip.w;
    const float2 shadowUvLocal = (shadowNdc.xy * 0.5) + 0.5;
    if (any(shadowUvLocal <= float2(0.0, 0.0)) || any(shadowUvLocal >= float2(1.0, 1.0)))
    {
        return 1.0;
    }

    const float4 atlasRect = camera.shadowAtlasUvRects[cascadeIndex];
    const float2 shadowUv = atlasRect.xy + (shadowUvLocal * atlasRect.zw);
    const float shadowDepthRef = shadowNdc.z;
    if (shadowDepthRef <= 0.0 || shadowDepthRef >= 1.0)
    {
        return 1.0;
    }

    uint texW = 0u;
    uint texH = 0u;
    shadowMap.GetDimensions(texW, texH);
    const float2 atlasTexelSize = float2(
        1.0 / max(float(texW), 1.0),
        1.0 / max(float(texH), 1.0)
    );
    const float2 localTexelSize = atlasTexelSize / max(atlasRect.zw, float2(1e-6, 1e-6));
    const float cascadeT = float(cascadeIndex) / 3.0;
    const float baseBias = lerp(camera.shadowConfig0.z, camera.shadowConfig0.w, cascadeT);
    const float bias = baseBias * max(localTexelSize.x, localTexelSize.y);
    return shadowMap.SampleCmpLevelZero(shadowUv, shadowDepthRef + bias);
}

float hgPhase(float cosTheta, float anisotropy)
{
    const float g = clamp(anisotropy, 0.0, 0.95);
    const float g2 = g * g;
    const float denom = max(pow(1.0 + g2 - (2.0 * g * cosTheta), 1.5), 1e-4);
    return (1.0 - g2) / (4.0 * 3.14159265 * denom);
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= pushConstants.width || dispatchThreadId.y >= pushConstants.height)
    {
        return;
    }

    const float2 uv = (float2(dispatchThreadId.xy) + 0.5) / float2(float(pushConstants.width), float(pushConstants.height));
    const float sceneDepth = normalDepthTexture.SampleLevel(uv, 0.0).w;
    const bool hasGeometry = sceneDepth > 0.0001;

    const float2 ndc = (uv * 2.0) - 1.0;
    const float3 viewDir = normalize(float3(ndc.x / camera.proj[0][0], ndc.y / camera.proj[1][1], -1.0));
    const float3x3 invViewRotation = transpose((float3x3)camera.view);
    const float3 worldDir = normalize(mul(invViewRotation, viewDir));
    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;

    const float maxDistance = hasGeometry ? min(sceneDepth, 96.0) : 96.0;
    if (maxDistance <= 0.05)
    {
        sunShaftTexture[dispatchThreadId.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float3 toSun = normalize(-camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    if (sunIntensity <= 1e-4)
    {
        sunShaftTexture[dispatchThreadId.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float fogDensityBase = max(camera.skyConfig4.x, 0.0);
    const float fogHeightFalloff = max(camera.skyConfig4.y, 0.0);
    const float fogBaseHeight = camera.skyConfig4.z;
    const float sunScatter = max(camera.skyConfig4.w, 0.0);
    const uint sampleCount = max(pushConstants.sampleCount, 1u);
    const float stepLength = maxDistance / float(sampleCount);
    const float phase = hgPhase(dot(worldDir, toSun), 0.72);

    float transmittance = 1.0;
    float inscatter = 0.0;
    for (uint i = 0; i < sampleCount; ++i)
    {
        const float t = (float(i) + 0.5) * stepLength;
        const float3 worldPos = cameraWorld + (worldDir * t);
        const float relativeHeight = worldPos.y - fogBaseHeight;
        const float localDensity = fogDensityBase * exp(-relativeHeight * fogHeightFalloff);
        if (localDensity <= 1e-6)
        {
            continue;
        }

        const float shadowVisibility = sampleShadowVisibility(worldPos);
        const float segmentExtinction = localDensity * stepLength;
        const float segmentTransmittance = exp(-segmentExtinction);
        const float segmentInscatter =
            transmittance * (1.0 - segmentTransmittance) * shadowVisibility * phase * sunScatter;
        inscatter += segmentInscatter;
        transmittance *= segmentTransmittance;
    }

    const float shaft = max(inscatter * sunIntensity, 0.0);
    sunShaftTexture[dispatchThreadId.xy] = float4(shaft, shaft, shaft, 1.0);
}
