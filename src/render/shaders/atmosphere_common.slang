module atmosphere_common;
import camera_uniform;

public static const float kPi = 3.14159265359;
public static const float kTwoPi = 6.28318530718;

public struct AtmosphereParameters
{
    public float bottomRadiusKm;
    public float topRadiusKm;
    public float groundAlbedo;
    public float sunIlluminance;
    public float3 rayleighScattering;
    public float rayleighScaleHeightKm;
    public float3 mieScattering;
    public float mieScaleHeightKm;
    public float3 mieExtinction;
    public float mieAnisotropy;
    public float3 ozoneAbsorption;
    public float ozoneCenterHeightKm;
    public float ozoneWidthKm;
    public float worldUnitToKm;
    public float aerialPerspectiveDistanceKm;
};

public struct AtmosphereMediumSample
{
    public float3 scattering;
    public float3 extinction;
};

public float saturate01(float value)
{
    return clamp(value, 0.0, 1.0);
}

public float3 saturate01(float3 value)
{
    return clamp(value, float3(0.0), float3(1.0));
}

public AtmosphereParameters loadAtmosphereParameters()
{
    AtmosphereParameters params;
    params.bottomRadiusKm = camera.atmosphereConfig0.x;
    params.topRadiusKm = camera.atmosphereConfig0.x + camera.atmosphereConfig0.y;
    params.groundAlbedo = camera.atmosphereConfig0.z;
    params.sunIlluminance = camera.atmosphereConfig0.w;
    params.rayleighScattering = camera.atmosphereConfig1.xyz;
    params.rayleighScaleHeightKm = camera.atmosphereConfig1.w;
    params.mieScattering = camera.atmosphereConfig2.xyz;
    params.mieScaleHeightKm = camera.atmosphereConfig2.w;
    params.mieExtinction = camera.atmosphereConfig3.xyz;
    params.mieAnisotropy = camera.atmosphereConfig3.w;
    params.ozoneAbsorption = camera.atmosphereConfig4.xyz;
    params.ozoneCenterHeightKm = camera.atmosphereConfig4.w;
    params.ozoneWidthKm = camera.atmosphereConfig5.x;
    params.worldUnitToKm = camera.atmosphereConfig5.z;
    params.aerialPerspectiveDistanceKm = camera.atmosphereConfig5.w;
    return params;
}

public float3 cameraWorldPosition()
{
    return camera.shadowVoxelGridSize.xyz;
}

public float3 worldToPlanetPositionKm(float3 worldPosition, AtmosphereParameters params)
{
    return float3(
        worldPosition.x * params.worldUnitToKm,
        params.bottomRadiusKm + max(worldPosition.y * params.worldUnitToKm, 0.0),
        worldPosition.z * params.worldUnitToKm
    );
}

public float heightKm(float3 planetPositionKm, AtmosphereParameters params)
{
    return max(length(planetPositionKm) - params.bottomRadiusKm, 0.0);
}

public bool raySphereIntersection(float3 origin, float3 dir, float radius, out float tNear, out float tFar)
{
    const float b = dot(origin, dir);
    const float c = dot(origin, origin) - (radius * radius);
    const float discriminant = (b * b) - c;
    if (discriminant < 0.0)
    {
        tNear = 0.0;
        tFar = 0.0;
        return false;
    }

    const float root = sqrt(discriminant);
    tNear = -b - root;
    tFar = -b + root;
    return tFar >= 0.0;
}

public float distanceToAtmosphereBoundaryKm(
    float3 originKm,
    float3 dir,
    AtmosphereParameters params,
    out bool hitGround)
{
    float tNearTop = 0.0;
    float tFarTop = 0.0;
    if (!raySphereIntersection(originKm, dir, params.topRadiusKm, tNearTop, tFarTop))
    {
        hitGround = false;
        return 0.0;
    }

    float distanceKm = max(tFarTop, 0.0);
    hitGround = false;
    float tNearGround = 0.0;
    float tFarGround = 0.0;
    if (raySphereIntersection(originKm, dir, params.bottomRadiusKm, tNearGround, tFarGround))
    {
        if (tNearGround > 0.0)
        {
            distanceKm = min(distanceKm, tNearGround);
            hitGround = true;
        }
    }
    return max(distanceKm, 0.0);
}

public AtmosphereMediumSample sampleAtmosphereMedium(float3 planetPositionKm, AtmosphereParameters params)
{
    const float h = heightKm(planetPositionKm, params);
    const float rayleighDensity = exp(-h / max(params.rayleighScaleHeightKm, 0.001));
    const float mieDensity = exp(-h / max(params.mieScaleHeightKm, 0.001));
    const float ozoneDistance = abs(h - params.ozoneCenterHeightKm);
    const float ozoneDensity = saturate01(1.0 - (ozoneDistance / max(params.ozoneWidthKm, 0.001)));

    AtmosphereMediumSample medium;
    medium.scattering = (params.rayleighScattering * rayleighDensity) + (params.mieScattering * mieDensity);
    medium.extinction =
        (params.rayleighScattering * rayleighDensity) +
        (params.mieExtinction * mieDensity) +
        (params.ozoneAbsorption * ozoneDensity);
    return medium;
}

public float rayleighPhase(float cosTheta)
{
    return (3.0 / (16.0 * kPi)) * (1.0 + (cosTheta * cosTheta));
}

public float miePhase(float cosTheta, float anisotropy)
{
    const float g = clamp(anisotropy, 0.0, 0.98);
    const float g2 = g * g;
    const float denom = max(pow(1.0 + g2 - (2.0 * g * cosTheta), 1.5), 1e-4);
    return (3.0 / (8.0 * kPi)) * ((1.0 - g2) * (1.0 + (cosTheta * cosTheta))) / ((2.0 + g2) * denom);
}

public float2 atmosphereLookupUv(float3 planetPositionKm, float3 dir, AtmosphereParameters params)
{
    const float mu = clamp(dot(normalize(planetPositionKm), dir), -1.0, 1.0);
    const float h = saturate01(heightKm(planetPositionKm, params) / max(params.topRadiusKm - params.bottomRadiusKm, 0.001));
    return float2((mu * 0.5) + 0.5, h);
}

public float2 skyViewUv(float3 viewDirCamera)
{
    const float3 dir = normalize(viewDirCamera);
    const float azimuth = atan2(dir.x, -dir.z);
    return float2(
        frac((azimuth / kTwoPi) + 1.0),
        saturate01((dir.y * 0.5) + 0.5)
    );
}

public float3 skyViewDirection(float2 uv)
{
    const float azimuth = ((uv.x - 0.5) * kTwoPi);
    const float y = clamp((uv.y * 2.0) - 1.0, -1.0, 1.0);
    const float radial = sqrt(max(1.0 - (y * y), 0.0));
    return normalize(float3(sin(azimuth) * radial, y, -cos(azimuth) * radial));
}

public float3 sampleTransmittanceToBoundary(
    float3 originKm,
    float3 dir,
    AtmosphereParameters params,
    uint sampleCount)
{
    bool hitGround = false;
    const float distanceKm = distanceToAtmosphereBoundaryKm(originKm, dir, params, hitGround);
    if (distanceKm <= 1e-4)
    {
        return float3(1.0);
    }

    float3 opticalDepth = float3(0.0);
    const float stepKm = distanceKm / float(max(sampleCount, 1u));
    for (uint i = 0u; i < sampleCount; ++i)
    {
        const float t = (float(i) + 0.5) * stepKm;
        const float3 samplePos = originKm + (dir * t);
        const AtmosphereMediumSample medium = sampleAtmosphereMedium(samplePos, params);
        opticalDepth += medium.extinction * stepKm;
    }
    return exp(-opticalDepth);
}

public float4 sampleAerialPerspectiveLut(
    Sampler3D<float4> aerialPerspectiveLut,
    float3 viewDirCamera,
    float distanceKm,
    AtmosphereParameters params)
{
    const float2 dirUv = skyViewUv(viewDirCamera);
    const float distanceUv = saturate01(distanceKm / max(params.aerialPerspectiveDistanceKm, 0.001));
    return aerialPerspectiveLut.SampleLevel(float3(dirUv, distanceUv), 0.0);
}
