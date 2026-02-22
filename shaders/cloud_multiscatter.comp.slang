static const float3 kCloudAabbMin = float3(-10.0f, 0.0f, -10.0f);
static const float3 kCloudAabbMax = float3(10.0f, 10.0f, 10.0f);
static const float kPi = 3.1415926535f;

struct MultiScatterPush {
    float4 sunDirectionIntensity;
    float4 mediumParams; // x = density scale, y = g, z = albedo
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
[[vk::binding(4, 0)]]
SamplerState gVolumeSampler;

float sampleDensity(float3 p) {
    float3 uvw = saturate((p - kCloudAabbMin) / (kCloudAabbMax - kCloudAabbMin));
    return gDensityVolume.SampleLevel(gVolumeSampler, uvw, 0.0f);
}

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
    float cloudAlbedo = saturate(gPush.mediumParams.z);
    float3 sunRadiance = float3(1.0f, 1.0f, 0.98f) * max(gPush.sunDirectionIntensity.w, 0.01f);

    int3 voxel = int3(dispatchThreadID);
    float3 uvw = (float3(dispatchThreadID) + 0.5f) / float(dim);
    float3 p = lerp(kCloudAabbMin, kCloudAabbMax, uvw);

    float density = sampleDensity(p) * densityScale * 1.5f;
    float sunTr = gSunTransmittance.SampleLevel(gVolumeSampler, uvw, 0.0f);
    float3 source = density * sunTr * sunRadiance * (1.0f / (4.0f * kPi));

    float3 prev = sampleMs(voxel, dim);
    float3 blur = (
        sampleMs(voxel + int3(1, 0, 0), dim) +
        sampleMs(voxel + int3(-1, 0, 0), dim) +
        sampleMs(voxel + int3(0, 1, 0), dim) +
        sampleMs(voxel + int3(0, -1, 0), dim) +
        sampleMs(voxel + int3(0, 0, 1), dim) +
        sampleMs(voxel + int3(0, 0, -1), dim)) * (1.0f / 6.0f);

    float3 propagated = lerp(prev, blur, lambda);
    float damping = cloudAlbedo * exp(-density * stepDistance);
    float3 outE = source + propagated * damping;
    gMultiScatterOut[dispatchThreadID] = float4(max(outE, 0.0f), 1.0f);
}
