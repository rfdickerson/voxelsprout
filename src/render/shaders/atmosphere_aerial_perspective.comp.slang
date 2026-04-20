import camera_uniform;
import atmosphere_common;

[[vk::binding(4, 0)]]
RWTexture3D<float4> atmosphereAerialPerspectiveOut;
[[vk::binding(5, 0)]]
Sampler2D<float4> atmosphereTransmittanceLut;
[[vk::binding(6, 0)]]
Sampler2D<float4> atmosphereMultiScatteringLut;

float3 sampleTransmittanceFromLut(float3 positionKm, float3 dir, AtmosphereParameters params)
{
    return atmosphereTransmittanceLut.SampleLevel(atmosphereLookupUv(positionKm, dir, params), 0.0).xyz;
}

float3 sampleMultiScatteringFromLut(float3 positionKm, float3 sunDir, AtmosphereParameters params)
{
    const float sunMu = clamp(dot(normalize(positionKm), sunDir), -1.0, 1.0);
    const float heightT = saturate01(heightKm(positionKm, params) / max(params.topRadiusKm - params.bottomRadiusKm, 0.001));
    return atmosphereMultiScatteringLut.SampleLevel(float2((sunMu * 0.5) + 0.5, heightT), 0.0).xyz;
}

[numthreads(4, 4, 4)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint width = 0u;
    uint height = 0u;
    uint depth = 0u;
    atmosphereAerialPerspectiveOut.GetDimensions(width, height, depth);
    if (dispatchThreadID.x >= width || dispatchThreadID.y >= height || dispatchThreadID.z >= depth)
    {
        return;
    }

    const AtmosphereParameters params = loadAtmosphereParameters();
    const float3 uvw =
        (float3(dispatchThreadID) + 0.5) /
        float3(float(width), float(height), float(depth));
    const float3 viewDirCamera = skyViewDirection(uvw.xy);
    const float3x3 invViewRotation = transpose((float3x3)camera.view);
    const float3 viewDirWorld = normalize(mul(invViewRotation, viewDirCamera));
    const float3 toSunWorld = normalize(-camera.sunDirectionIntensity.xyz);
    const float3 originKm = worldToPlanetPositionKm(cameraWorldPosition(), params);
    const float targetDistanceKm = uvw.z * max(params.aerialPerspectiveDistanceKm, 0.001);
    const uint stepCount = 24u;
    const float stepKm = targetDistanceKm / float(stepCount);

    float3 viewTransmittance = float3(1.0);
    float3 inscatter = float3(0.0);
    for (uint stepIndex = 0u; stepIndex < stepCount; ++stepIndex)
    {
        const float t = (float(stepIndex) + 0.5) * stepKm;
        const float3 samplePos = originKm + (viewDirWorld * t);
        const AtmosphereMediumSample medium = sampleAtmosphereMedium(samplePos, params);
        const float cosTheta = dot(viewDirWorld, toSunWorld);
        const float phase =
            rayleighPhase(cosTheta) +
            miePhase(cosTheta, params.mieAnisotropy);
        const float3 sunTransmittance = sampleTransmittanceFromLut(samplePos, toSunWorld, params);
        const float3 multiScattering = sampleMultiScatteringFromLut(samplePos, toSunWorld, params);
        const float3 inScatter =
            (medium.scattering * phase * sunTransmittance * params.sunIlluminance) +
            (multiScattering * 0.30);
        inscatter += viewTransmittance * inScatter * stepKm;
        viewTransmittance *= exp(-medium.extinction * stepKm);
    }

    const float transmittanceScalar = dot(viewTransmittance, float3(0.2126, 0.7152, 0.0722));
    atmosphereAerialPerspectiveOut[dispatchThreadID] = float4(max(inscatter, float3(0.0)), saturate01(transmittanceScalar));
}
