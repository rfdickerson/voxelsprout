import camera_uniform;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

struct FragmentInput
{
    float3 inLocalDirection : TEXCOORD0;
    float4 inColor : TEXCOORD1;
    float2 inUv : TEXCOORD2;
    nointerpolation uint inTextureIndex : TEXCOORD3;
};

float4 main(FragmentInput input) : SV_Target
{
    const uint layer = min(input.inTextureIndex, 3u);
    const int textureSlot = int(camera.weatherCloudParams[layer].x + 0.5);
    if (textureSlot < 0)
    {
        discard;
    }
    const float4 sampled = bindlessTextures[textureSlot].Sample(input.inUv);
    const float3 localDir = normalize(input.inLocalDirection);
    const float horizonFade = smoothstep(-0.12, 0.22, localDir.y);
    const float3 toSun = -normalize(camera.sunDirectionIntensity.xyz);
    const float sunFacing = pow(saturate(dot(localDir, toSun)), 3.0);

    const float4 authoredTint = camera.weatherCloudTint[layer];
    // The retail DDS already carries the cloud shading in RGB. clouds.nif's
    // vertex RGB is shader control data (not a display tint); multiplying it
    // into the sampled colour turns SkyrimClear clouds magenta. Vertex alpha,
    // below, is the authored coverage channel.
    float3 cloudColor = sampled.rgb * lerp(0.90, 1.10, sunFacing);

    const float vertexFade = lerp(0.35, 1.0, input.inColor.a);
    const float alpha = saturate(
        sampled.a * authoredTint.w * vertexFade * horizonFade);
    return float4(cloudColor, alpha);
}
