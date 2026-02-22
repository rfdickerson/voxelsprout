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
Texture3D<float4> gCloudNoiseVolume;
[[vk::binding(4, 0)]]
SamplerState gVolumeSampler;

float4 sampleCloudNoise(float3 p) {
    return gCloudNoiseVolume.SampleLevel(gVolumeSampler, p, 0.0f);
}

float remap(float value, float originalMin, float originalMax, float newMin, float newMax) {
    float denom = originalMax - originalMin;
    if (abs(denom) < 1e-5f) {
        return newMin;
    }
    float t = (value - originalMin) / denom;
    return lerp(newMin, newMax, t);
}

float densityField(float3 p) {
    const float cloudBase = gPush.cloudProfileParams.z;
    const float cloudTop = max(gPush.cloudProfileParams.w, cloudBase + 0.25f);
    const float macroScale = max(gPush.cloudShapeParams.x, 0.05f);
    const float detailScale = max(gPush.cloudShapeParams.y, 0.08f);
    const float erosionStrength = max(gPush.cloudWarpParams.y, 0.0f);
    const float coverage = saturate(gPush.cloudProfileParams.x);
    const float densityScale = max(gPush.mediumParams.x, 0.01f);

    if (p.y < cloudBase || p.y > cloudTop) {
        return 0.0f;
    }

    float edgeFadeX = smoothstep(-9.5f, -6.0f, p.x) * (1.0f - smoothstep(6.0f, 9.5f, p.x));
    float edgeFadeZ = smoothstep(-9.5f, -6.0f, p.z) * (1.0f - smoothstep(6.0f, 9.5f, p.z));
    float boundsMask = edgeFadeX * edgeFadeZ;

    float2 weatherUvw = (p.xz + float2(15.2f, -8.4f)) * macroScale * 0.06f;
    float weatherNoise = sampleCloudNoise(float3(weatherUvw.x, 0.0f, weatherUvw.y)).r;
    float cellMask = saturate((weatherNoise + (coverage - 0.5f)));

    float h = saturate((p.y - cloudBase) / max((cloudTop - cloudBase), 0.1f));
    float bottomRamp = smoothstep(0.0f, 0.1f, h);
    float localTop = lerp(0.50f, 1.0f, cellMask);
    float topTaper = 1.0f - smoothstep(localTop - 0.15f, localTop + 0.12f, h);
    const float localCoverageBoost = saturate(0.25f + 0.75f * cellMask);

    float structuralMask = boundsMask * max(bottomRamp, 0.18f) * max(topTaper, 0.12f) * localCoverageBoost;
    if (structuralMask <= 0.0f) {
        return 0.0f;
    }

    float3 baseUvw = p * macroScale * 0.12f;
    float baseNoise = sampleCloudNoise(baseUvw).r;
    float remapThreshold = (1.0f - structuralMask) * 0.55f;
    float baseDensity = remap(baseNoise, remapThreshold, 1.0f, 0.0f, 1.0f);
    baseDensity = saturate(pow(saturate(baseDensity), 0.6f));
    if (baseDensity <= 0.0f) {
        return 0.0f;
    }

    float3 detailUvw = p * detailScale * 0.35f;
    float4 detailNoise = sampleCloudNoise(detailUvw);
    float highFreqFbm = detailNoise.g * 0.625f + detailNoise.b * 0.25f + detailNoise.a * 0.125f;
    highFreqFbm += (detailNoise.r - 0.5f) * 0.05f;

    float heightBias = lerp(0.1f, 1.0f, h);
    float erosionModifier = lerp(highFreqFbm, 0.0f, baseDensity) * heightBias;
    float finalDensity = remap(baseDensity, erosionModifier * erosionStrength, 1.0f, 0.0f, 1.0f);
    return saturate(finalDensity) * densityScale;
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

    float density = densityField(p) * densityScale * 1.5f;
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
