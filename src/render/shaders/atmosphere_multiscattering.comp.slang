import camera_uniform;
import atmosphere_common;

[[vk::binding(2, 0)]]
RWTexture2D<float4> atmosphereMultiScatteringOut;
[[vk::binding(5, 0)]]
Sampler2D<float4> atmosphereTransmittanceLut;

float3 sampleTransmittanceFromLut(float3 positionKm, float3 dir, AtmosphereParameters params)
{
    return atmosphereTransmittanceLut.SampleLevel(atmosphereLookupUv(positionKm, dir, params), 0.0).xyz;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint width = 0u;
    uint height = 0u;
    atmosphereMultiScatteringOut.GetDimensions(width, height);
    if (dispatchThreadID.x >= width || dispatchThreadID.y >= height)
    {
        return;
    }

    const AtmosphereParameters params = loadAtmosphereParameters();
    const float2 uv = (float2(dispatchThreadID.xy) + 0.5) / float2(float(width), float(height));
    const float sunMu = (uv.x * 2.0) - 1.0;
    const float altitudeKm = uv.y * max(params.topRadiusKm - params.bottomRadiusKm, 0.001);
    const float sunRadial = sqrt(max(1.0 - (sunMu * sunMu), 0.0));
    const float3 sunDir = normalize(float3(sunRadial, sunMu, 0.0));
    const float3 originKm = float3(0.0, params.bottomRadiusKm + altitudeKm, 0.0);

    float3 ambient = float3(0.0);
    const uint sampleCount = 8u;
    for (uint i = 0u; i < sampleCount; ++i)
    {
        const float phi = ((float(i) + 0.5) / float(sampleCount)) * kTwoPi;
        const float mu = lerp(-0.2, 1.0, (float(i) + 0.5) / float(sampleCount));
        const float radial = sqrt(max(1.0 - (mu * mu), 0.0));
        const float3 dir = normalize(float3(cos(phi) * radial, mu, sin(phi) * radial));

        bool hitGround = false;
        const float maxDistanceKm = distanceToAtmosphereBoundaryKm(originKm, dir, params, hitGround);
        const float stepKm = maxDistanceKm / 8.0;
        float3 viewTransmittance = float3(1.0);
        float3 scatteringSum = float3(0.0);
        for (uint stepIndex = 0u; stepIndex < 8u; ++stepIndex)
        {
            const float t = (float(stepIndex) + 0.5) * stepKm;
            const float3 samplePos = originKm + (dir * t);
            const AtmosphereMediumSample medium = sampleAtmosphereMedium(samplePos, params);
            const float3 sunTransmittance = sampleTransmittanceFromLut(samplePos, sunDir, params);
            scatteringSum += viewTransmittance * medium.scattering * sunTransmittance * stepKm;
            viewTransmittance *= exp(-medium.extinction * stepKm);
        }

        ambient += scatteringSum;
    }

    ambient *= params.sunIlluminance / float(sampleCount);
    ambient *= 0.08 + (params.groundAlbedo * 0.12);
    atmosphereMultiScatteringOut[dispatchThreadID.xy] = float4(max(ambient, float3(0.0)), 1.0);
}
