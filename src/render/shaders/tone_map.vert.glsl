#version 450

layout(location = 0) out vec2 outUv;

void main() {
    vec2 ndcPosition;
    if (gl_VertexIndex == 0) {
        ndcPosition = vec2(-1.0, -1.0);
    } else if (gl_VertexIndex == 1) {
        ndcPosition = vec2(3.0, -1.0);
    } else {
        ndcPosition = vec2(-1.0, 3.0);
    }

    outUv = (ndcPosition * 0.5) + 0.5;
    gl_Position = vec4(ndcPosition, 0.0, 1.0);
}
