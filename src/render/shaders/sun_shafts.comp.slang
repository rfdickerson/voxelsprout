import camera_uniform;

[[vk::binding(1, 0)]]
Sampler2D<float4> normalDepthTexture;

[[vk::binding(2, 0)]]
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

float sampleSkyMask(float2 uv)
{
    if (any(uv < float2(0.0, 0.0)) || any(uv > float2(1.0, 1.0)))
    {
        return 0.0;
    }
    const float viewDepth = normalDepthTexture.SampleLevel(uv, 0.0).w;
    return 1.0 - step(0.0001, viewDepth);
}

float sampleSunVisibility(float2 sunUv)
{
    if (any(sunUv < float2(0.0, 0.0)) || any(sunUv > float2(1.0, 1.0)))
    {
        return 0.0;
    }

    uint texW = 0u;
    uint texH = 0u;
    normalDepthTexture.GetDimensions(texW, texH);
    const float2 texelSize = 1.0 / float2(float(max(texW, 1u)), float(max(texH, 1u)));
    const float2 occlusionRadius = texelSize * 2.0;

    float minSkyVisibility = 1.0;
    const float2 offsets[5] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(-1.0, 0.0),
        float2(0.0, 1.0),
        float2(0.0, -1.0)
    };

    for (int i = 0; i < 5; ++i)
    {
        const float2 sampleUv = clamp(sunUv + (offsets[i] * occlusionRadius), float2(0.0, 0.0), float2(1.0, 1.0));
        const float sampleViewDepth = normalDepthTexture.SampleLevel(sampleUv, 0.0).w;
        const float skyVisibility = 1.0 - step(0.0001, sampleViewDepth);
        minSkyVisibility = min(minSkyVisibility, skyVisibility);
    }

    return minSkyVisibility;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= pushConstants.width || dispatchThreadId.y >= pushConstants.height)
    {
        return;
    }

    const float2 uv = (float2(dispatchThreadId.xy) + 0.5) / float2(float(pushConstants.width), float(pushConstants.height));

    const float3x3 invViewRotation = transpose((float3x3)camera.view);
    const float3 cameraForward = normalize(mul(invViewRotation, float3(0.0, 0.0, -1.0)));
    const float3 toSun = normalize(-camera.sunDirectionIntensity.xyz);
    const float sunForwardDot = dot(cameraForward, toSun);
    const float sunFacingWide = smoothstep(cos(radians(72.0)), cos(radians(10.0)), sunForwardDot);
    const float sunFacingTight = smoothstep(cos(radians(20.0)), cos(radians(3.0)), sunForwardDot);
    const float sunFacing = max(sunFacingTight, sunFacingWide * 0.65);
    if (sunFacing <= 1e-5)
    {
        sunShaftTexture[dispatchThreadId.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float3 sunView = mul((float3x3)camera.view, toSun);
    if (sunView.z >= -1e-4)
    {
        sunShaftTexture[dispatchThreadId.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float invDepth = 1.0 / max(-sunView.z, 1e-4);
    const float2 sunNdc = float2(
        sunView.x * camera.proj[0][0] * invDepth,
        sunView.y * camera.proj[1][1] * invDepth
    );
    const float2 sunUv = (sunNdc * 0.5) + 0.5;
    const float sunVisibility = sampleSunVisibility(sunUv);
    if (sunVisibility <= 1e-5)
    {
        sunShaftTexture[dispatchThreadId.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float2 deltaToSun = sunUv - uv;
    const float radialDistance = length(deltaToSun);
    const float rangeMask = 1.0 - smoothstep(0.03, 0.26, radialDistance);
    if (rangeMask <= 1e-5)
    {
        sunShaftTexture[dispatchThreadId.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const uint sampleCount = max(pushConstants.sampleCount, 1u);
    const float2 stepUv = deltaToSun / float(sampleCount);
    float2 sampleUv = uv + (stepUv * 0.5);
    float decay = 1.0;
    float shaftAccum = 0.0;
    float weightAccum = 0.0;
    for (uint i = 0; i < sampleCount; ++i)
    {
        const float skyMask = sampleSkyMask(sampleUv);
        shaftAccum += skyMask * decay;
        weightAccum += decay;
        decay *= 0.86;
        sampleUv += stepUv;
    }

    const float shaftOcclusion = shaftAccum / max(weightAccum, 1e-5);
    const float localSkyMask = sampleSkyMask(uv);
    const float shaft = shaftOcclusion * rangeMask * sunVisibility * sunFacing * localSkyMask;
    sunShaftTexture[dispatchThreadId.xy] = float4(shaft, shaft, shaft, 1.0);
}
