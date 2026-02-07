#version 450

layout(set = 0, binding = 3) uniform sampler2D hdrSceneColor;

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

vec3 acesFilmTonemap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

void main() {
    const vec3 hdrColor = texture(hdrSceneColor, inUv).rgb;
    const vec3 toneMapped = acesFilmTonemap(hdrColor);
    const vec3 ldrSrgb = pow(toneMapped, vec3(1.0 / 2.2));
    outColor = vec4(ldrSrgb, 1.0);
}
