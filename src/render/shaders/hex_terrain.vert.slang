// Hex-terrain vertex stage. One instance per land tile; the base mesh is a hex fan
// (center + 6 corners) in local XZ, reinterpreted as six 3-control-point triangle
// patches. This stage lifts each control point to world XZ and assigns its CONTINUOUS
// base height: the center sits at the tile's own elevation, each corner at the average
// of the three tiles meeting there (own + the two edge-neighbours adjacent to that
// corner). Two tiles sharing a corner therefore agree on its height -> crack free.
// The tessellation evaluation shader adds per-tile detail on top.

struct VertexInput
{
    [[vk::location(0)]] float2 localXZ : TEXCOORD0;       // base-mesh corner (per vertex)
    [[vk::location(1)]] uint cornerIndex : TEXCOORD1;     // 0..5 corner, 6 center (per vertex)
    [[vk::location(2)]] float2 centerXZ : TEXCOORD2;      // tile center world XZ (per instance)
    [[vk::location(3)]] uint classFlags : TEXCOORD3;      // terrain class + flags (per instance)
    [[vk::location(4)]] float4 detailParams : TEXCOORD4;  // {feature, amplitude, windowEnd, noiseFreq}
    [[vk::location(5)]] float4 ownAndNear : TEXCOORD5;    // {ownElevY, neighborElevY[0..2]}
    [[vk::location(6)]] float4 farAndSize : TEXCOORD6;    // {neighborElevY[3..5], hexSize}
};

struct VertexOutput
{
    float3 worldPos : TEXCOORD0;       // (x, continuous base height, z)
    float2 centerXZ : TEXCOORD1;
    float4 detailParams : TEXCOORD2;
    uint classFlags : TEXCOORD3;
    float hexSize : TEXCOORD4;
};

VertexOutput main(VertexInput input)
{
    const float ownY = input.ownAndNear.x;
    float neighborY[6];
    neighborY[0] = input.ownAndNear.y;
    neighborY[1] = input.ownAndNear.z;
    neighborY[2] = input.ownAndNear.w;
    neighborY[3] = input.farAndSize.x;
    neighborY[4] = input.farAndSize.y;
    neighborY[5] = input.farAndSize.z;

    float baseHeight;
    if (input.cornerIndex >= 6u)
    {
        baseHeight = ownY;  // center
    }
    else
    {
        const uint k = input.cornerIndex;
        baseHeight = (ownY + neighborY[(k + 5u) % 6u] + neighborY[k]) / 3.0;
    }

    VertexOutput output;
    output.worldPos = float3(input.centerXZ.x + input.localXZ.x, baseHeight,
                             input.centerXZ.y + input.localXZ.y);
    output.centerXZ = input.centerXZ;
    output.detailParams = input.detailParams;
    output.classFlags = input.classFlags;
    output.hexSize = input.farAndSize.w;
    return output;
}
