import camera_uniform;
import atmosphere_common;

[[vk::binding(1, 0)]]
RWTexture2D<float4> atmosphereTransmittanceOut;

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint width = 0u;
    uint height = 0u;
    atmosphereTransmittanceOut.GetDimensions(width, height);
    if (dispatchThreadID.x >= width || dispatchThreadID.y >= height)
    {
        return;
    }

    const AtmosphereParameters params = loadAtmosphereParameters();
    const float2 uv = (float2(dispatchThreadID.xy) + 0.5) / float2(float(width), float(height));
    const float mu = (uv.x * 2.0) - 1.0;
    const float h = saturate01(uv.y) * max(params.topRadiusKm - params.bottomRadiusKm, 0.001);
    const float radial = sqrt(max(1.0 - (mu * mu), 0.0));
    const float3 originKm = float3(0.0, params.bottomRadiusKm + h, 0.0);
    const float3 dir = normalize(float3(radial, mu, 0.0));
    const float3 transmittance = sampleTransmittanceToBoundary(originKm, dir, params, 40u);
    atmosphereTransmittanceOut[dispatchThreadID.xy] = float4(transmittance, 1.0);
}
