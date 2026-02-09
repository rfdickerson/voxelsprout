#version 450

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec3 inTint;
layout(location = 3) in float inVertexAo;
layout(location = 4) in float inLocalAlong;
layout(location = 5) in float inStyle;

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
    const float vertexAoEnable = clamp(camera.shadowVoxelGridOrigin.w, 0.0, 1.0);
    const float vertexAo = mix(1.0, clamp(inVertexAo, 0.0, 1.0), vertexAoEnable);
    const vec3 ambient = ambientIrradiance * (0.26 * vertexAo);
    const vec3 directSun = sunColor * (sunIntensity * ndotl);

    const float flowTime = camera.skyConfig1.z;
    const float flowSpeed = max(camera.skyConfig1.w, 0.0);
    const float flowCoord = (inLocalAlong * 6.5) - (flowTime * flowSpeed);
    const float flowBand = 1.0 - abs((fract(flowCoord) * 2.0) - 1.0);
    const float flowHighlight = pow(clamp(flowBand, 0.0, 1.0), 3.0);

    const float stylePipe = 1.0 - step(0.5, inStyle);
    const float styleConveyor = step(0.5, inStyle) * (1.0 - step(1.5, inStyle));
    const float styleTrack = step(1.5, inStyle);

    // Pipe style: bright moving liquid through the center with endcaps left as solid shell.
    const float transferStart = 0.26;
    const float transferEnd = 0.74;
    const float transferEdge = 0.03;
    const float pipeLiquidMask =
        smoothstep(transferStart, transferStart + transferEdge, inLocalAlong) *
        (1.0 - smoothstep(transferEnd - transferEdge, transferEnd, inLocalAlong));
    const vec3 liquidBaseColor = vec3(1.0, 0.08, 0.78);
    const vec3 liquidColor = liquidBaseColor * (0.82 + (0.48 * flowHighlight));
    const float liquidMask = pipeLiquidMask * stylePipe;

    // Conveyor style: simple moving highlight stripe along belt direction.
    const float conveyorBand = 1.0 - abs((fract((inLocalAlong * 4.0) - (flowTime * flowSpeed * 0.8)) * 2.0) - 1.0);
    const float conveyorHighlight = pow(clamp(conveyorBand, 0.0, 1.0), 2.2) * styleConveyor;
    const vec3 conveyorColor = inTint * (1.0 + (0.30 * conveyorHighlight));

    // Track style: subtle tie rhythm.
    const float tiePulse = step(0.68, fract(inLocalAlong * 6.0));
    const float tieDarken = mix(1.0, 0.82, tiePulse * styleTrack);
    const vec3 trackColor = inTint * tieDarken;

    vec3 surfaceTint = inTint;
    surfaceTint = mix(surfaceTint, conveyorColor, styleConveyor);
    surfaceTint = mix(surfaceTint, trackColor, styleTrack);
    surfaceTint = mix(surfaceTint, liquidColor, liquidMask);

    const vec3 lit =
        ((ambient + directSun) * surfaceTint) +
        (liquidColor * ((0.24 + (0.42 * flowHighlight)) * liquidMask));
    outColor = vec4(lit, 1.0);
}
