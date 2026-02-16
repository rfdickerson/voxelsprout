import camera_uniform;

struct FragmentInput
{
    float3 inWorldPosition : TEXCOORD0;
    float3 inWorldNormal : TEXCOORD1;
    float3 inTint : TEXCOORD2;
};

float4 main(FragmentInput input) : SV_Target0
{
    const float3 worldNormal = normalize(input.inWorldNormal);
    const float3 viewNormal = normalize(mul(camera.view, float4(worldNormal, 0.0)).xyz);
    const float viewDepth = max(-mul(camera.view, float4(input.inWorldPosition, 1.0)).z, 0.0);

    const float3 encodedNormal = (viewNormal * 0.5) + 0.5;
    return float4(encodedNormal, viewDepth);
}
