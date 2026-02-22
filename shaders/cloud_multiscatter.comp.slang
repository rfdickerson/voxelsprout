static const float3 kCloudAabbMin = float3(-10.0f, 0.0f, -10.0f);
static const float3 kCloudAabbMax = float3(10.0f, 10.0f, 10.0f);
static const float kPi = 3.1415926535f;

struct MultiScatterPush {
    float4 sunDirectionIntensity;
    float4 mediumParams; // x = density scale, y = g, z = albedo, w = extinction coeff
    float4 cloudShapeParams; // x = macro scale, y = detail scale
    float4 cloudProfileParams; // x = coverage, y = weather scale, z = cloud base, w = cloud top
    float4 cloudWarpParams; // y = erosion strength
    float4 solveParams; // x = volume dim, y = lambda, z = step distance, w = reserved
};

[[vk::push_constant]]
ConstantBuffer<MultiScatterPush> gPush;

[[vk::binding(0, 0)]]
RWTexture3D<float4> gMultiScatterOut;
[[vk::binding(1, 0)]]
Texture3D<float4> gMultiScatterIn;
[[vk::binding(2, 0)]]
Texture3D<float> gSunTransmittance;
[[vk::binding(3, 0)]]
Texture3D<float> gDensityVolume;

float3 sampleMs(int3 p, uint dim) {
    int3 clamped = clamp(p, int3(0, 0, 0), int3(int(dim) - 1, int(dim) - 1, int(dim) - 1));
    return gMultiScatterIn.Load(int4(clamped, 0)).xyz;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint dim = (uint)max(gPush.solveParams.x, 1.0f);
    if (dispatchThreadID.x >= dim || dispatchThreadID.y >= dim || dispatchThreadID.z >= dim) {
        return;
    }

    float lambda = saturate(gPush.solveParams.y);
    float stepDistance = max(gPush.solveParams.z, 1e-3f);
    float densityScale = max(gPush.mediumParams.x, 0.01f);
    float extinctionCoeff = max(gPush.mediumParams.w, 0.01f);
    float cloudAlbedo = saturate(gPush.mediumParams.z);
    float3 sunRadiance = float3(1.0f, 1.0f, 0.98f) * max(gPush.sunDirectionIntensity.w, 0.01f);

    int3 voxel = int3(dispatchThreadID);
    uint3 msDim = uint3(dim, dim, dim);

    uint3 densityDim;
    gDensityVolume.GetDimensions(densityDim.x, densityDim.y, densityDim.z);
    densityDim = max(densityDim, uint3(1u, 1u, 1u));

    uint3 sunDim;
    gSunTransmittance.GetDimensions(sunDim.x, sunDim.y, sunDim.z);
    sunDim = max(sunDim, uint3(1u, 1u, 1u));

    uint3 densityCoord = min((dispatchThreadID * densityDim) / msDim, densityDim - 1u);
    uint3 sunCoord = min((dispatchThreadID * sunDim) / msDim, sunDim - 1u);
    int4 densityTexCoord = int4(densityCoord, 0);
    int4 sunTexCoord = int4(sunCoord, 0);

    float rawDensity = gDensityVolume.Load(densityTexCoord).r;
    float sunTr = gSunTransmittance.Load(sunTexCoord).r;
    float density = rawDensity * densityScale * extinctionCoeff;
    float3 source = density * sunTr * sunRadiance * (1.0f / (4.0f * kPi));

    float3 prev = sampleMs(voxel, dim);
    float3 msXp = sampleMs(voxel + int3(1, 0, 0), dim);
    float3 msXn = sampleMs(voxel + int3(-1, 0, 0), dim);
    float3 msYp = sampleMs(voxel + int3(0, 1, 0), dim);
    float3 msYn = sampleMs(voxel + int3(0, -1, 0), dim);
    float3 msZp = sampleMs(voxel + int3(0, 0, 1), dim);
    float3 msZn = sampleMs(voxel + int3(0, 0, -1), dim);

    float3 sunDir = normalize(gPush.sunDirectionIntensity.xyz);
    float wXp = 0.5f - 0.5f * sunDir.x;
    float wXn = 0.5f + 0.5f * sunDir.x;
    float wYp = 0.5f - 0.5f * sunDir.y;
    float wYn = 0.5f + 0.5f * sunDir.y;
    float wZp = 0.5f - 0.5f * sunDir.z;
    float wZn = 0.5f + 0.5f * sunDir.z;

    float3 blur =
        (msXp * wXp + msXn * wXn) +
        (msYp * wYp + msYn * wYn) +
        (msZp * wZp + msZn * wZn);
    blur *= (1.0f / 3.0f);

    float3 propagated = lerp(prev, blur, lambda);
    float transmission = exp(-density * stepDistance);
    float scatteringFraction = (1.0f - transmission) * cloudAlbedo;
    float3 outE = source + (propagated * scatteringFraction);
    gMultiScatterOut[dispatchThreadID] = float4(max(outE, 0.0f), 1.0f);
}
