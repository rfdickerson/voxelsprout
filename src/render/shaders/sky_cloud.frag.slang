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
    if (input.inTextureIndex >= 4u) {
        const uint index = input.inTextureIndex - 4u;
        if (index >= 6u) discard;
        const int slot = int(camera.nightSkySlots[index / 4u][index % 4u]);
        if (slot < 0) discard;
        const float4 texel = bindlessTextures[slot].Sample(input.inUv);
        const float fade = smoothstep(0.0, 0.15, normalize(input.inLocalDirection).y);
        const float visibility = max(camera.nightSkyTint.x, max(camera.nightSkyTint.y, camera.nightSkyTint.z));
        if (index < 4u) {
            return float4(texel.rgb * texel.a * input.inColor.rgb * input.inColor.a * camera.nightSkyTint.rgb * fade, 0.0);
        }
        const float alpha = texel.a * visibility * fade;
        return float4(texel.rgb * alpha, alpha);
    }
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

    // Authored zero-alpha borders must vanish where the sparse sheets end.
    // A nonzero floor exposes their straight polygon edges against the sky.
    const float vertexFade = saturate(input.inColor.a);
    const float alpha = saturate(
        sampled.a * authoredTint.w * vertexFade * horizonFade);
    return float4(cloudColor * alpha, alpha);
}
