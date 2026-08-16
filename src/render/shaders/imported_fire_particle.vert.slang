import camera_uniform;

// Same 48-byte range as ChunkPushConstants; the shared main pipeline layout
// therefore needs no new descriptors or layout permutation.
struct FireParticlePushConstants
{
    float4 positionRadius;       // xyz emitter origin, w spawn radius
    float4 colorIntensity;       // linear HDR rgb, w emission intensity
    float4 simulationParams;     // lifetime, upward speed, size, seed
};

[[vk::push_constant]]
ConstantBuffer<FireParticlePushConstants> firePc;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float age : TEXCOORD1;
    float variation : TEXCOORD2;
};

float hash11(float value)
{
    return frac(sin(value * 12.9898) * 43758.5453);
}

float2 quadCorner(uint vertexId)
{
    if (vertexId == 0u) return float2(-1.0, -1.0);
    if (vertexId == 1u) return float2( 1.0, -1.0);
    if (vertexId == 2u) return float2( 1.0,  1.0);
    if (vertexId == 3u) return float2(-1.0, -1.0);
    if (vertexId == 4u) return float2( 1.0,  1.0);
    return float2(-1.0, 1.0);
}

VertexOutput main(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    VertexOutput output;
    const float id = float(instanceId) + firePc.simulationParams.w;
    const float lifetime = max(firePc.simulationParams.x, 0.05);
    const float timeSeconds = camera.skyConfig1.z;
    const float phase = hash11(id + 0.17);
    const float age = frac((timeSeconds / lifetime) + phase);
    const float angle = hash11(id + 3.71) * 6.28318530718;
    const float radialSample = sqrt(hash11(id + 8.13));
    const float narrowing = lerp(1.0, 0.30, age);
    const float radius = firePc.positionRadius.w * radialSample * narrowing;
    const float turbulence = firePc.positionRadius.w * 0.18;
    const float swayX = sin((timeSeconds * 5.1) + (id * 1.91)) * turbulence * age;
    const float swayZ = cos((timeSeconds * 4.3) + (id * 2.37)) * turbulence * age;

    float3 center = firePc.positionRadius.xyz;
    center.x += (cos(angle) * radius) + swayX;
    center.z += (sin(angle) * radius) + swayZ;
    center.y += firePc.simulationParams.y * lifetime * age;

    const float variation = hash11(id + 13.41);
    const float sizeFade = pow(max(1.0 - age, 0.02), 0.62);
    const float particleSize = firePc.simulationParams.z * lerp(0.62, 1.25, variation) * sizeFade;
    const float2 corner = quadCorner(vertexId);
    float4 viewPosition = mul(camera.view, float4(center, 1.0));
    viewPosition.xy += corner * particleSize;
    output.position = mul(camera.proj, viewPosition);
    output.uv = (corner * 0.5) + 0.5;
    output.age = age;
    output.variation = variation;
    return output;
}
