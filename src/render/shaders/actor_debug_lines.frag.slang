struct FragmentInput
{
    float3 inColor : TEXCOORD0;
};

float4 main(FragmentInput input) : SV_Target
{
    return float4(input.inColor, 1.0);
}
