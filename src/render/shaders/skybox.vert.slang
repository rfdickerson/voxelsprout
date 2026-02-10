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
    output.position = float4(ndc, 1.0, 1.0);
    return output;
}
