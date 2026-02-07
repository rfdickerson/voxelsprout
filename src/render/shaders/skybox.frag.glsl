#version 450

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 mvp;
    mat4 view;
    mat4 proj;
} camera;

layout(set = 0, binding = 2) uniform samplerCube skyboxMap;

layout(location = 0) in vec2 inNdc;
layout(location = 0) out vec4 outColor;

void main() {
    const vec4 viewSpace = inverse(camera.proj) * vec4(inNdc, 1.0, 1.0);
    const vec3 viewDir = normalize(viewSpace.xyz / max(abs(viewSpace.w), 1e-6));
    const mat3 invViewRotation = transpose(mat3(camera.view));
    const vec3 worldDir = normalize(invViewRotation * viewDir);
    outColor = vec4(texture(skyboxMap, worldDir).rgb, 1.0);
}
