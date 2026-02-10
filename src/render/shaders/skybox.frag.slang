import camera_uniform;

float saturate01(float value)
{
    return clamp(value, 0.0, 1.0);
}

float smoothStep(float edge0, float edge1, float x)
{
    const float t = saturate01((x - edge0) / max(edge1 - edge0, 1e-6));
    return t * t * (3.0 - (2.0 * t));
}

float3 proceduralSky(float3 direction, float3 sunDirection)
{
    const float3 dir = normalize(direction);
    const float3 toSun = -normalize(sunDirection);
    const float rayleighStrength = max(camera.skyConfig0.x, 0.01);
    const float mieStrength = max(camera.skyConfig0.y, 0.01);
    const float mieG = clamp(camera.skyConfig0.z, 0.0, 0.98);
    const float skyExposure = max(camera.skyConfig0.w, 0.01);

    const float sunAltitude = clamp(toSun.y, -1.0, 1.0);
    const float dayFactor = smoothStep(0.05, 0.65, sunAltitude);
    const float twilightFactor = 1.0 - dayFactor;
    const float horizonBand = saturate01(1.0 - (abs(sunAltitude) / 0.35));
    const float warmAmount = twilightFactor * pow(horizonBand, 1.2);
    const float pinkAmount = warmAmount * saturate01((0.10 - sunAltitude) / 0.30);

    const float horizonT = saturate01((dir.y * 0.5) + 0.5);
    const float skyT = pow(horizonT, 0.35);

    float3 horizonRayleigh = lerp(float3(0.54, 0.70, 1.00), float3(0.74, 0.44, 0.52), warmAmount);
    float3 horizonMie = lerp(float3(1.00, 0.74, 0.42), float3(1.18, 0.54, 0.30), warmAmount);
    horizonRayleigh = lerp(horizonRayleigh, float3(0.70, 0.36, 0.68), pinkAmount * 0.70);
    horizonMie = lerp(horizonMie, float3(1.08, 0.46, 0.72), pinkAmount * 0.85);

    const float zenithWarm = twilightFactor * 0.58;
    const float3 zenithRayleigh = lerp(float3(0.06, 0.24, 0.54), float3(0.16, 0.12, 0.30), zenithWarm);
    const float3 zenithMie = lerp(float3(0.22, 0.20, 0.15), float3(0.30, 0.18, 0.24), zenithWarm);

    const float3 horizonColor =
        (horizonRayleigh * rayleighStrength) +
        (horizonMie * (mieStrength * 0.58));
    const float3 zenithColor =
        (zenithRayleigh * rayleighStrength) +
        (zenithMie * (mieStrength * 0.25));
    float3 sky = lerp(horizonColor, zenithColor, skyT);

    const float sunDotRaw = clamp(dot(dir, toSun), -1.0, 1.0);
    const float sunDot = max(sunDotRaw, 0.0);
    const float3 sunColor = camera.sunColorShadow.xyz;
    const float phaseRayleigh = 0.0596831 * (1.0 + (sunDot * sunDot));
    const float phaseMie = 0.0795775 * (1.0 - (mieG * mieG)) /
        max(0.001, pow(1.0 + (mieG * mieG) - (2.0 * mieG * sunDot), 1.5));
    const float phaseBoost = (phaseRayleigh * rayleighStrength) + (phaseMie * mieStrength * 1.4);

    // Analytic sun disk in angular space with altitude-dependent haze bloom.
    // Low sun: larger, hazier bloom. High sun: tighter, more discrete disk.
    const float sunAngle = acos(sunDotRaw);
    const float diskSizeScale = clamp(camera.skyConfig2.x, 0.5, 6.0);
    const float hazeFalloffScale = clamp(camera.skyConfig2.y, 0.10, 1.20);
    const float lowSunHaze =
        (1.0 - smoothStep(0.08, 0.55, sunAltitude)) *
        smoothStep(-0.18, 0.12, sunAltitude);
    const float highSunCrisp = smoothStep(0.35, 0.85, sunAltitude);

    const float radiusDay = 0.0056;
    const float radiusTwilight = 0.0108;
    const float sunRadius = lerp(radiusTwilight, radiusDay, dayFactor) * diskSizeScale;
    const float sunAaBase = max(fwidth(sunAngle), 0.00010);
    const float sunEdgeWidth = sunAaBase * lerp(2.30, 0.85, highSunCrisp);
    const float diskMask = 1.0 - smoothStep(sunRadius - sunEdgeWidth, sunRadius + sunEdgeWidth, sunAngle);
    const float centerT = saturate01(1.0 - (sunAngle / max(sunRadius, 1e-5)));
    const float limbDarkening = 0.52 + (0.48 * pow(centerT, lerp(0.50, 0.24, highSunCrisp)));

    const float diskIntensity = max(camera.skyConfig1.x / 1150.0, 0.01);
    const float disk = diskMask * limbDarkening * (6.8 * diskIntensity) * lerp(0.95, 1.15, highSunCrisp);

    const float haloIntensity = max(camera.skyConfig1.y / 22.0, 0.01);

    // Mid/near halo component.
    const float haloRadius = sunRadius * lerp(18.0, 6.0, highSunCrisp);
    const float haloFalloff = max(haloRadius * hazeFalloffScale * lerp(1.45, 0.55, highSunCrisp), 1e-5);
    const float haloCore = exp(-sunAngle / haloFalloff);
    const float haloMask = 1.0 - smoothStep(haloRadius * lerp(0.45, 0.72, highSunCrisp), haloRadius, sunAngle);
    const float halo = haloCore * haloMask * (1.2 * haloIntensity) * (0.35 + (1.35 * lowSunHaze));

    // Broad low-sun haze bloom component.
    const float hazeRadius = sunRadius * lerp(34.0, 10.0, highSunCrisp);
    const float hazeFalloff = max(hazeRadius * hazeFalloffScale * 0.85, 1e-5);
    const float hazeCore = exp(-sunAngle / hazeFalloff);
    const float hazeMask = 1.0 - smoothStep(hazeRadius * 0.55, hazeRadius, sunAngle);
    const float hazeBloom = hazeCore * hazeMask * (1.6 * haloIntensity * lowSunHaze);

    sky += sunColor * (disk + halo + hazeBloom) * (1.0 + phaseBoost);

    const float3 groundColor = float3(0.05, 0.06, 0.07);
    const float belowHorizon = saturate01(-dir.y);
    const float3 horizonGroundColor = horizonColor * 0.32;
    const float3 ground = lerp(horizonGroundColor, groundColor, pow(belowHorizon, 0.55));
    const float skyWeight = smoothstep(-0.18, 0.02, dir.y);
    const float3 color = lerp(ground, sky, skyWeight);
    return max(color * skyExposure, float3(0.0, 0.0, 0.0));
}

struct FragmentInput
{
    float2 inNdc : TEXCOORD0;
};

float4 main(FragmentInput input) : SV_Target0
{
    float projX = camera.proj[0][0];
    float projY = camera.proj[1][1];
    if (abs(projX) < 1e-6)
    {
        projX = (projX < 0.0) ? -1e-6 : 1e-6;
    }
    if (abs(projY) < 1e-6)
    {
        projY = (projY < 0.0) ? -1e-6 : 1e-6;
    }
    const float3 viewDir = normalize(float3(input.inNdc.x / projX, input.inNdc.y / projY, -1.0));
    const float3x3 invViewRotation = transpose((float3x3)camera.view);
    const float3 worldDir = normalize(mul(invViewRotation, viewDir));

    return float4(proceduralSky(worldDir, camera.sunDirectionIntensity.xyz), 1.0);
}
