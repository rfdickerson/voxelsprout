#version 450
layout(location = 0) in flat uint inFace;
layout(location = 1) in float inAo;
layout(location = 2) in flat uint inMaterial;

layout(set = 0, binding = 1) uniform samplerCube irradianceMap;

layout(location = 0) out vec4 outColor;

vec3 faceColor(uint face) {
    if (face == 0u) return vec3(0.90, 0.44, 0.35);
    if (face == 1u) return vec3(0.70, 0.34, 0.28);
    if (face == 2u) return vec3(0.40, 0.85, 0.40);
    if (face == 3u) return vec3(0.28, 0.55, 0.28);
    if (face == 4u) return vec3(0.35, 0.55, 0.90);
    return vec3(0.28, 0.42, 0.70);
}

vec3 materialTint(uint material) {
    if (material == 1u) {
        return vec3(1.0, 1.0, 1.0);
    }
    return vec3(0.9, 0.9, 0.9);
}

vec3 faceNormal(uint face) {
    if (face == 0u) return vec3(1.0, 0.0, 0.0);
    if (face == 1u) return vec3(-1.0, 0.0, 0.0);
    if (face == 2u) return vec3(0.0, 1.0, 0.0);
    if (face == 3u) return vec3(0.0, -1.0, 0.0);
    if (face == 4u) return vec3(0.0, 0.0, 1.0);
    return vec3(0.0, 0.0, -1.0);
}

void main() {
    const float exposure = 1.35;
    const float aoBrightness = 0.45 + (clamp(inAo, 0.0, 1.0) * 0.55);
    const vec3 baseColor = faceColor(inFace) * materialTint(inMaterial);
    const vec3 diffuseIrradiance = texture(irradianceMap, faceNormal(inFace)).rgb;
    outColor = vec4(baseColor * aoBrightness * diffuseIrradiance * exposure, 1.0);
}
