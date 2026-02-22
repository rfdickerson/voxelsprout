static const float3 kCloudAabbMin = float3(-10.0f, 0.0f, -10.0f);
static const float3 kCloudAabbMax = float3(10.0f, 10.0f, 10.0f);

struct SunTransmittancePush {
    float4 sunDirectionIntensity;
    float4 mediumParams; // x = density scale, y = g, z = albedo, w = extinction coeff
    float4 cloudShapeParams; // x = macro scale, y = detail scale
    float4 cloudProfileParams; // x = coverage, y = weather scale, z = cloud base, w = cloud top
    float4 cloudWarpParams; // y = erosion strength
    float4 volumeParams; // x = volume dimension, y = march steps
};

[[vk::push_constant]]
ConstantBuffer<SunTransmittancePush> gPush;

[[vk::binding(0, 0)]]
RWTexture3D<float> gSunTransmittance;
[[vk::binding(1, 0)]]
Texture3D<float> gDensityVolume;
[[vk::binding(2, 0)]]
SamplerState gVolumeSampler;

uint hash3(uint3 p) {
    uint h = p.x * 747796405u;
    h ^= p.y * 2891336453u;
    h ^= p.z * 1181783497u;
    h ^= (h >> 16u);
    h *= 2246822519u;
    h ^= (h >> 13u);
    h *= 3266489917u;
    h ^= (h >> 16u);
    return h;
}

float sampleDensity(float3 p) {
    float3 uvw = (p - kCloudAabbMin) / (kCloudAabbMax - kCloudAabbMin);
    if (any(uvw < 0.0f) || any(uvw > 1.0f)) {
        return 0.0f;
    }
    return gDensityVolume.SampleLevel(gVolumeSampler, uvw, 0.0f);
}

bool intersectAabb(float3 rayOrigin, float3 rayDir, float3 bmin, float3 bmax, out float tMin, out float tMax) {
    float3 invD = 1.0f / rayDir;
    float3 t0 = (bmin - rayOrigin) * invD;
    float3 t1 = (bmax - rayOrigin) * invD;

    float3 tSmall = min(t0, t1);
    float3 tLarge = max(t0, t1);

    tMin = max(max(tSmall.x, tSmall.y), max(tSmall.z, 0.0f));
    tMax = min(min(tLarge.x, tLarge.y), tLarge.z);
    return tMax > tMin;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint dim = (uint)max(gPush.volumeParams.x, 1.0f);
    if (dispatchThreadID.x >= dim || dispatchThreadID.y >= dim || dispatchThreadID.z >= dim) {
        return;
    }

    float3 uvw = (float3(dispatchThreadID) + 0.5f) / float(dim);
    float3 p = lerp(kCloudAabbMin, kCloudAabbMax, uvw);
    float3 sunDir = normalize(gPush.sunDirectionIntensity.xyz);

    float tMin = 0.0f;
    float tMax = 0.0f;
    if (!intersectAabb(p + sunDir * 0.05f, sunDir, kCloudAabbMin, kCloudAabbMax, tMin, tMax)) {
        gSunTransmittance[dispatchThreadID] = 1.0f;
        return;
    }

    uint steps = (uint)max(gPush.volumeParams.y, 1.0f);
    float segmentLen = tMax - tMin;
    float stepSize = segmentLen / float(steps);
    float densityScale = max(gPush.mediumParams.x, 0.01f);
    float extinctionCoeff = max(gPush.mediumParams.w, 0.01f);
    if (stepSize <= 0.0f) {
        gSunTransmittance[dispatchThreadID] = 1.0f;
        return;
    }

    const uint h = hash3(dispatchThreadID);
    const float jitter = float(h & 0x00ffffffu) * (1.0f / 16777216.0f); // [0,1)

    float opticalDepth = 0.0f;
    float t = tMin + ((frac(jitter + 0.5f)) * stepSize);
    [loop]
    for (uint i = 0; i < steps; ++i) {
        float3 x = p + sunDir * t;
        float sigmaT = sampleDensity(x) * densityScale * extinctionCoeff;
        opticalDepth += sigmaT * stepSize;
        t += stepSize;
    }

    gSunTransmittance[dispatchThreadID] = exp(-opticalDepth);
}
