#include "camera_uniform.slanginc"

float3 proceduralSky(float3 direction, float3 sunDirection)
{
    const float3 dir = normalize(direction);
    const float3 toSun = -normalize(sunDirection);
    const float rayleighStrength = max(camera.skyConfig0.x, 0.01);
    const float mieStrength = max(camera.skyConfig0.y, 0.01);
    const float mieG = clamp(camera.skyConfig0.z, 0.0, 0.98);
    const float skyExposure = max(camera.skyConfig0.w, 0.01);

    const float horizonT = clamp((dir.y * 0.5) + 0.5, 0.0, 1.0);
    const float skyT = pow(horizonT, 0.35);
    const float3 horizonColor =
        (float3(0.55, 0.70, 1.00) * rayleighStrength) +
        (float3(1.00, 0.72, 0.42) * (mieStrength * 0.55));
    const float3 zenithColor =
        (float3(0.06, 0.24, 0.54) * rayleighStrength) +
        (float3(0.22, 0.20, 0.15) * (mieStrength * 0.25));
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
    const float belowHorizon = clamp(-dir.y, 0.0, 1.0);
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
