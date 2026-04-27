import camera_uniform;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

struct FragmentInput
{
    float3 inLocalDirection : TEXCOORD0;
    float3 inColor : TEXCOORD1;
    float2 inUv : TEXCOORD2;
    nointerpolation uint inTextureIndex : TEXCOORD3;
};

float colorWhiteness(float3 color)
{
    const float spread = max(max(abs(color.r - color.g), abs(color.g - color.b)), abs(color.b - color.r));
    return saturate(1.0 - (spread * 3.5));
}

float blueSkySuppression(float3 color)
{
    const float blueDominance = color.b - max(color.r, color.g);
    return 1.0 - smoothstep(0.035, 0.20, blueDominance);
}

float4 main(FragmentInput input) : SV_Target
{
    const float4 sampled = bindlessTextures[input.inTextureIndex].Sample(input.inUv);
    const float luminance = dot(sampled.rgb, float3(0.2126, 0.7152, 0.0722));
    const float inferredCloudMask =
        smoothstep(0.50, 0.86, luminance) *
        smoothstep(0.28, 0.78, colorWhiteness(sampled.rgb)) *
        blueSkySuppression(sampled.rgb);

    const float3 localDir = normalize(input.inLocalDirection);
    const float horizonFade = smoothstep(-0.12, 0.22, localDir.y);
    const float3 toSun = -normalize(camera.sunDirectionIntensity.xyz);
    const float sunFacing = pow(saturate(dot(localDir, toSun)), 3.0);

    const float3 warmSun = float3(1.10, 0.90, 0.64);
    const float3 coolShadow = float3(0.70, 0.72, 0.86);
    float3 cloudColor = lerp(float3(luminance, luminance, luminance), sampled.rgb, 0.55) * input.inColor;
    cloudColor = lerp(cloudColor * coolShadow, cloudColor * warmSun, sunFacing * 0.65);
    cloudColor *= lerp(0.82, 1.20, sunFacing);

    const float alpha = saturate(inferredCloudMask * horizonFade * 0.46);
    return float4(cloudColor, alpha);
}
