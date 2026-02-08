#version 450
layout(location = 0) in uint inPacked;

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

layout(push_constant) uniform ChunkPushConstants {
    vec4 chunkOffset;
    vec4 cascadeData;
} chunkPc;

layout(location = 0) out flat uint outFace;
layout(location = 1) out float outAo;
layout(location = 2) out flat uint outMaterial;
layout(location = 3) out vec3 outWorldPosition;

const uint kShiftX = 0u;
const uint kShiftY = 5u;
const uint kShiftZ = 10u;
const uint kShiftFace = 15u;
const uint kShiftCorner = 18u;
const uint kShiftAo = 20u;
const uint kShiftMaterial = 22u;

vec3 cornerOffset(uint face, uint corner) {
    if (face == 0u) {
        if (corner == 0u) return vec3(1.0, 0.0, 0.0);
        if (corner == 1u) return vec3(1.0, 1.0, 0.0);
        if (corner == 2u) return vec3(1.0, 1.0, 1.0);
        return vec3(1.0, 0.0, 1.0);
    }
    if (face == 1u) {
        if (corner == 0u) return vec3(0.0, 0.0, 1.0);
        if (corner == 1u) return vec3(0.0, 1.0, 1.0);
        if (corner == 2u) return vec3(0.0, 1.0, 0.0);
        return vec3(0.0, 0.0, 0.0);
    }
    if (face == 2u) {
        if (corner == 0u) return vec3(0.0, 1.0, 0.0);
        if (corner == 1u) return vec3(0.0, 1.0, 1.0);
        if (corner == 2u) return vec3(1.0, 1.0, 1.0);
        return vec3(1.0, 1.0, 0.0);
    }
    if (face == 3u) {
        if (corner == 0u) return vec3(0.0, 0.0, 1.0);
        if (corner == 1u) return vec3(0.0, 0.0, 0.0);
        if (corner == 2u) return vec3(1.0, 0.0, 0.0);
        return vec3(1.0, 0.0, 1.0);
    }
    if (face == 4u) {
        if (corner == 0u) return vec3(1.0, 0.0, 1.0);
        if (corner == 1u) return vec3(1.0, 1.0, 1.0);
        if (corner == 2u) return vec3(0.0, 1.0, 1.0);
        return vec3(0.0, 0.0, 1.0);
    }

    if (corner == 0u) return vec3(0.0, 0.0, 0.0);
    if (corner == 1u) return vec3(0.0, 1.0, 0.0);
    if (corner == 2u) return vec3(1.0, 1.0, 0.0);
    return vec3(1.0, 0.0, 0.0);
}

void main() {
    const uint x = (inPacked >> kShiftX) & 0x1Fu;
    const uint y = (inPacked >> kShiftY) & 0x1Fu;
    const uint z = (inPacked >> kShiftZ) & 0x1Fu;
    const uint face = (inPacked >> kShiftFace) & 0x7u;
    const uint corner = (inPacked >> kShiftCorner) & 0x3u;
    const uint ao = (inPacked >> kShiftAo) & 0x3u;
    const uint material = (inPacked >> kShiftMaterial) & 0xFFu;

    const vec3 basePosition = vec3(float(x), float(y), float(z));
    const vec3 worldPosition = basePosition + cornerOffset(face, corner) + chunkPc.chunkOffset.xyz;

    outFace = face;
    outAo = float(ao) / 3.0;
    outMaterial = material;
    outWorldPosition = worldPosition;
    gl_Position = camera.mvp * vec4(worldPosition, 1.0);
}
