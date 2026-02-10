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

    const float sunDot = max(dot(dir, toSun), 0.0);
    const float sunDisk = pow(sunDot, max(camera.skyConfig1.x, 1.0));
    const float sunGlow = pow(sunDot, max(camera.skyConfig1.y, 1.0));
    const float3 sunColor = camera.sunColorShadow.xyz;
    const float phaseRayleigh = 0.0596831 * (1.0 + (sunDot * sunDot));
    const float phaseMie = 0.0795775 * (1.0 - (mieG * mieG)) /
        max(0.001, pow(1.0 + (mieG * mieG) - (2.0 * mieG * sunDot), 1.5));
    const float phaseBoost = (phaseRayleigh * rayleighStrength) + (phaseMie * mieStrength * 1.4);

    sky += sunColor * ((sunDisk * 6.5) + (sunGlow * 1.3)) * (1.0 + phaseBoost);

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
