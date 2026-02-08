#version 450

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 mvp;
    mat4 view;
    mat4 proj;
    mat4 lightViewProj[4];
    vec4 shadowCascadeSplits;
    vec4 sunDirectionIntensity;
    vec4 sunColorShadow;
    vec4 shIrradiance[9];
} camera;

layout(location = 0) in vec2 inNdc;
layout(location = 0) out vec4 outColor;

vec3 proceduralSky(vec3 direction, vec3 sunDirection) {
    const vec3 dir = normalize(direction);
    const vec3 toSun = -normalize(sunDirection);

    const float horizonT = clamp((dir.y * 0.5) + 0.5, 0.0, 1.0);
    const float skyT = pow(horizonT, 0.35);

    const vec3 horizonColor = vec3(0.83, 0.68, 0.46);
    const vec3 zenithColor = vec3(0.11, 0.30, 0.62);
    vec3 sky = mix(horizonColor, zenithColor, skyT);

    const float sunDot = max(dot(dir, toSun), 0.0);
    const float sunDisk = pow(sunDot, 1150.0);
    const float sunGlow = pow(sunDot, 22.0);
    const vec3 sunColor = camera.sunColorShadow.rgb;

    sky += sunColor * ((sunDisk * 6.5) + (sunGlow * 1.3));

    const vec3 groundColor = vec3(0.05, 0.06, 0.07);
    const vec3 color = (dir.y >= 0.0) ? sky : (groundColor * (0.45 + (0.55 * (-dir.y))));
    return max(color, vec3(0.0));
}

void main() {
    const vec4 viewSpace = inverse(camera.proj) * vec4(inNdc, 1.0, 1.0);
    const vec3 viewDir = normalize(viewSpace.xyz / max(abs(viewSpace.w), 1e-6));
    const mat3 invViewRotation = transpose(mat3(camera.view));
    const vec3 worldDir = normalize(invViewRotation * viewDir);

    outColor = vec4(proceduralSky(worldDir, camera.sunDirectionIntensity.xyz), 1.0);
}
