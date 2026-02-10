#version 450

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 mvp;
    mat4 view;
    mat4 proj;
    mat4 lightViewProj[4];
    vec4 shadowCascadeSplits;
    vec4 shadowAtlasUvRects[4];
    vec4 sunDirectionIntensity;
    vec4 sunColorShadow;
    vec4 shIrradiance[9];
    vec4 shadowConfig0;
    vec4 shadowConfig1;
    vec4 shadowConfig2;
    vec4 shadowConfig3;
    vec4 shadowVoxelGridOrigin;
    vec4 shadowVoxelGridSize;
    vec4 skyConfig0;
    vec4 skyConfig1;
} camera;

layout(location = 0) in vec2 inNdc;
layout(location = 0) out vec4 outColor;

vec3 proceduralSky(vec3 direction, vec3 sunDirection) {
    const vec3 dir = normalize(direction);
    const vec3 toSun = -normalize(sunDirection);
    const float rayleighStrength = max(camera.skyConfig0.x, 0.01);
    const float mieStrength = max(camera.skyConfig0.y, 0.01);
    const float mieG = clamp(camera.skyConfig0.z, 0.0, 0.98);
    const float skyExposure = max(camera.skyConfig0.w, 0.01);

    const float horizonT = clamp((dir.y * 0.5) + 0.5, 0.0, 1.0);
    const float skyT = pow(horizonT, 0.35);
    const vec3 horizonColor =
        (vec3(0.55, 0.70, 1.00) * rayleighStrength) +
        (vec3(1.00, 0.72, 0.42) * (mieStrength * 0.55));
    const vec3 zenithColor =
        (vec3(0.06, 0.24, 0.54) * rayleighStrength) +
        (vec3(0.22, 0.20, 0.15) * (mieStrength * 0.25));
    vec3 sky = mix(horizonColor, zenithColor, skyT);

    const float sunDot = max(dot(dir, toSun), 0.0);
    const float sunDisk = pow(sunDot, max(camera.skyConfig1.x, 1.0));
    const float sunGlow = pow(sunDot, max(camera.skyConfig1.y, 1.0));
    const vec3 sunColor = camera.sunColorShadow.rgb;
    const float phaseRayleigh = 0.0596831 * (1.0 + (sunDot * sunDot));
    const float phaseMie = 0.0795775 * (1.0 - (mieG * mieG)) /
        max(0.001, pow(1.0 + (mieG * mieG) - (2.0 * mieG * sunDot), 1.5));
    const float phaseBoost = (phaseRayleigh * rayleighStrength) + (phaseMie * mieStrength * 1.4);

    sky += sunColor * ((sunDisk * 6.5) + (sunGlow * 1.3)) * (1.0 + phaseBoost);

    const vec3 groundColor = vec3(0.05, 0.06, 0.07);
    const float belowHorizon = clamp(-dir.y, 0.0, 1.0);
    const vec3 horizonGroundColor = horizonColor * 0.32;
    const vec3 ground = mix(horizonGroundColor, groundColor, pow(belowHorizon, 0.55));
    const float skyWeight = smoothstep(-0.18, 0.02, dir.y);
    const vec3 color = mix(ground, sky, skyWeight);
    return max(color * skyExposure, vec3(0.0));
}

void main() {
    float projX = camera.proj[0][0];
    float projY = camera.proj[1][1];
    if (abs(projX) < 1e-6) {
        projX = (projX < 0.0) ? -1e-6 : 1e-6;
    }
    if (abs(projY) < 1e-6) {
        projY = (projY < 0.0) ? -1e-6 : 1e-6;
    }
    const vec3 viewDir = normalize(vec3(inNdc.x / projX, inNdc.y / projY, -1.0));
    const mat3 invViewRotation = transpose(mat3(camera.view));
    const vec3 worldDir = normalize(invViewRotation * viewDir);

    outColor = vec4(proceduralSky(worldDir, camera.sunDirectionIntensity.xyz), 1.0);
}
