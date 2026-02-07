#version 450

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 mvp;
    mat4 view;
    mat4 proj;
} camera;

layout(location = 0) out vec2 outNdc;

void main() {
    vec2 ndcPosition;
    if (gl_VertexIndex == 0) {
        ndcPosition = vec2(-1.0, -1.0);
    } else if (gl_VertexIndex == 1) {
        ndcPosition = vec2(3.0, -1.0);
    } else {
        ndcPosition = vec2(-1.0, 3.0);
    }

    outNdc = ndcPosition;
    gl_Position = vec4(ndcPosition, 1.0, 1.0);
}
