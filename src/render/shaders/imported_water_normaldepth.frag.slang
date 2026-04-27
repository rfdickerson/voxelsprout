import camera_uniform;

struct FragmentInput
{
    float3 inWorldPosition : TEXCOORD0;
    float2 inUv : TEXCOORD1;
};

float4 main(FragmentInput input) : SV_Target0
{
    const float3 worldNormal = float3(0.0, 1.0, 0.0);
    const float3 viewNormal = normalize(mul(camera.view, float4(worldNormal, 0.0)).xyz);
    const float viewDepth = max(-mul(camera.view, float4(input.inWorldPosition, 1.0)).z, 0.0);
    return float4((viewNormal * 0.5) + 0.5, viewDepth);
}
