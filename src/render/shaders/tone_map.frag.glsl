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

vec3 applySaturation(vec3 color, float saturation) {
    const float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(luma), color, saturation);
}

void main() {
    const vec3 hdrColor = texture(hdrSceneColor, inUv).rgb;
    const float exposure = 0.58;
    vec3 toneMapped = acesFilmTonemap(hdrColor * exposure);
    toneMapped = applySaturation(toneMapped, 1.18);
    toneMapped = pow(clamp(toneMapped, 0.0, 1.0), vec3(1.06));
    const vec3 ldrSrgb = pow(toneMapped, vec3(1.0 / 2.2));
    outColor = vec4(ldrSrgb, 1.0);
}
