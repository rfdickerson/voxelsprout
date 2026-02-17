import fullscreen_triangle;
import chunk_push_constants;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 outUv : TEXCOORD0;
    nointerpolation int cascadeIndex : TEXCOORD1;
};

VertexOutput main(uint vertexId : SV_VertexID)
{
    VertexOutput output;
    const float2 ndc = fullscreenTriangleNdc(vertexId);
    output.position = float4(ndc, 0.0, 1.0);
    output.outUv = (ndc * 0.5) + 0.5;
    output.cascadeIndex = clamp(int(chunkPc.cascadeData.x + 0.5), 0, 3);
    return output;
}

