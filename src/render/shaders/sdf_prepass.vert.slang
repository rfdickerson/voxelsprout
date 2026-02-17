import fullscreen_triangle;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 outUv : TEXCOORD0;
};

VertexOutput main(uint vertexId : SV_VertexID)
{
    VertexOutput output;
    const float2 ndc = fullscreenTriangleNdc(vertexId);
    output.position = float4(ndc, 0.0, 1.0);
    output.outUv = (ndc * 0.5) + 0.5;
    return output;
}

