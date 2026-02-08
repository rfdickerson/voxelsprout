#version 450

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec3 inTint;

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 mvp;
    mat4 view;
    mat4 proj;
    mat4 lightViewProj[4];
    vec4 shadowCascadeSplits;
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
layout(set = 0, binding = 7) uniform sampler2D ssaoTexture;

layout(location = 0) out vec4 outColor;

vec3 evaluateShIrradiance(vec3 normal) {
    const float x = normal.x;
    const float y = normal.y;
    const float z = normal.z;

    float basis[9];
    basis[0] = 0.282095;
    basis[1] = 0.488603 * y;
    basis[2] = 0.488603 * z;
    basis[3] = 0.488603 * x;
    basis[4] = 1.092548 * x * y;
    basis[5] = 1.092548 * y * z;
    basis[6] = 0.315392 * ((3.0 * z * z) - 1.0);
    basis[7] = 1.092548 * x * z;
    basis[8] = 0.546274 * ((x * x) - (y * y));

    vec3 irradiance = vec3(0.0);
    for (int i = 0; i < 9; ++i) {
        irradiance += camera.shIrradiance[i].rgb * basis[i];
    }
    return max(irradiance, vec3(0.0));
}

vec3 evaluateShHemisphereIrradiance(vec3 normal) {
    const float upT = clamp((normal.y * 0.5) + 0.5, 0.0, 1.0);
    const vec3 skyIrradiance = evaluateShIrradiance(vec3(0.0, 1.0, 0.0));
    const vec3 groundIrradiance = evaluateShIrradiance(vec3(0.0, -1.0, 0.0));
    return mix(groundIrradiance, skyIrradiance, upT);
}

void main() {
    const vec3 normal = normalize(inWorldNormal);
    const vec3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const vec3 sunColor = camera.sunColorShadow.rgb;
    const float ndotl = max(dot(normal, -sunDirection), 0.0);

    const vec3 shNormalIrradiance = evaluateShIrradiance(normal);
    const vec3 shHemisphereIrradiance = evaluateShHemisphereIrradiance(normal);
    const vec3 ambientIrradiance = mix(shNormalIrradiance, shHemisphereIrradiance, 0.70);
    const vec2 ssaoUv = gl_FragCoord.xy / vec2(textureSize(ssaoTexture, 0));
    const float ssao = clamp(texture(ssaoTexture, ssaoUv).r, 0.0, 1.0);
    const vec3 ambient = ambientIrradiance * (0.26 * ssao);
    const vec3 directSun = sunColor * (sunIntensity * ndotl);

    const vec3 lit = (ambient + directSun) * inTint;
    outColor = vec4(lit, 1.0);
}
