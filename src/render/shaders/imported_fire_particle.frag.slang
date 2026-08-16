struct FireParticlePushConstants
{
    float4 positionRadius;
    float4 colorIntensity;
    float4 simulationParams;
};

[[vk::push_constant]]
ConstantBuffer<FireParticlePushConstants> firePc;

struct FragmentInput
{
    float2 uv : TEXCOORD0;
    float age : TEXCOORD1;
    float variation : TEXCOORD2;
};

float4 main(FragmentInput input) : SV_Target0
{
    const float2 signedUv = (input.uv * 2.0) - 1.0;
    // A tapered soft lobe instead of a round sprite. Several independently
    // phased lobes make the flame silhouette without a sampled texture.
    const float taper = lerp(1.0, 0.30, smoothstep(0.18, 1.0, input.uv.y));
    const float distanceFromCore = length(float2(signedUv.x / taper, signedUv.y));
    float coverage = pow(saturate(1.0 - distanceFromCore), 1.55);
    coverage *= pow(max(1.0 - input.age, 0.0), 0.34);

    const float centerHeat = pow(saturate(1.0 - abs(signedUv.x)), 3.0) *
        pow(saturate(1.0 - input.uv.y), 0.55);
    const float3 amber = firePc.colorIntensity.rgb;
    const float3 hot = float3(1.55, 0.72, 0.10);
    const float3 ember = float3(1.0, 0.055, 0.008);
    float3 color = lerp(hot, amber, smoothstep(0.08, 0.65, input.age));
    color = lerp(color, ember, smoothstep(0.68, 1.0, input.age));
    color = lerp(color, hot, centerHeat * 0.72);
    color *= lerp(0.82, 1.18, input.variation);

    // Premultiplied emission for ONE+ONE additive blending. Values above one
    // intentionally feed bloom before ACES tone mapping.
    const float3 emission = color * (firePc.colorIntensity.w * coverage);
    return float4(emission, coverage);
}
