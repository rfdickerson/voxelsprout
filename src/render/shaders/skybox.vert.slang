import fullscreen_triangle;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 outNdc : TEXCOORD0;
};

VertexOutput main(uint vertexId : SV_VertexID)
{
    VertexOutput output;
    const float2 ndc = fullscreenTriangleNdc(vertexId);
    output.outNdc = ndc;
    // Reversed-Z far depth so sky can be depth-tested as background-only when drawn last.
    output.position = float4(ndc, 0.0, 1.0);
    return output;
}
