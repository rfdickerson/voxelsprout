import camera_uniform;

float3 faceNormal(uint face)
{
    if (face == 0u) return float3(1.0, 0.0, 0.0);
    if (face == 1u) return float3(-1.0, 0.0, 0.0);
    if (face == 2u) return float3(0.0, 1.0, 0.0);
    if (face == 3u) return float3(0.0, -1.0, 0.0);
    if (face == 4u) return float3(0.0, 0.0, 1.0);
    return float3(0.0, 0.0, -1.0);
}

struct FragmentInput
{
    nointerpolation uint inFace : TEXCOORD0;
    nointerpolation uint inMaterial : TEXCOORD1;
    float3 inWorldPosition : TEXCOORD2;
};

float4 main(FragmentInput input) : SV_Target0
{
    const float3 worldNormal = faceNormal(input.inFace);
    const float3 viewNormal = normalize(mul(camera.view, float4(worldNormal, 0.0)).xyz);
    const float viewDepth = max(-mul(camera.view, float4(input.inWorldPosition, 1.0)).z, 0.0);

    const float3 encodedNormal = (viewNormal * 0.5) + 0.5;
    return float4(encodedNormal, viewDepth);
}
