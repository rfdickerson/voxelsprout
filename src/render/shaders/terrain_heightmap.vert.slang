struct VertexOutput
{
    float2 patchMin : TEXCOORD0;
    float2 patchMax : TEXCOORD1;
    float2 cornerUv : TEXCOORD2;
};

static const uint kPatchGridSize = 16u;
static const float kTerrainExtent = 512.0;
static const float kPatchSize = kTerrainExtent / float(kPatchGridSize);
static const float kTerrainMin = -kTerrainExtent * 0.5;

VertexOutput main(uint vertexId : SV_VertexID)
{
    const uint patchVertex = vertexId & 3u;
    const uint patchIndex = vertexId >> 2u;
    const uint patchX = patchIndex % kPatchGridSize;
    const uint patchZ = patchIndex / kPatchGridSize;

    const float2 patchMin =
        float2(kTerrainMin, kTerrainMin) +
        float2(float(patchX), float(patchZ)) * kPatchSize;
    const float2 patchMax = patchMin + float2(kPatchSize, kPatchSize);

    VertexOutput output;
    output.patchMin = patchMin;
    output.patchMax = patchMax;
    output.cornerUv = float2((patchVertex == 1u || patchVertex == 2u) ? 1.0 : 0.0,
                             (patchVertex >= 2u) ? 1.0 : 0.0);
    return output;
}
