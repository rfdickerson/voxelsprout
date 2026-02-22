static const float3 kCloudAabbMin = float3(-10.0f, 0.0f, -10.0f);
static const float3 kCloudAabbMax = float3(10.0f, 10.0f, 10.0f);

struct DensityBakePush {
    float4 mediumParams; // x = density scale, y = g, z = albedo, w = extinction coeff
    float4 cloudShapeParams; // x = macro scale, y = detail scale
    float4 cloudProfileParams; // x = coverage, y = weather scale, z = cloud base, w = cloud top
    float4 cloudWarpParams; // y = erosion strength
    float4 volumeParams; // x = volume dimension
};

[[vk::push_constant]]
ConstantBuffer<DensityBakePush> gPush;

[[vk::binding(0, 0)]]
RWTexture3D<float> gDensityVolume;
[[vk::binding(1, 0)]]
Texture3D<float4> gCloudNoiseVolume;
[[vk::binding(3, 0)]]
SamplerState gVolumeSampler;
[[vk::binding(4, 0)]]
RWTexture3D<float> gDensityEdgeVolume;

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

void densityField(float3 p, out float outDensity, out float outEdge) {
    outDensity = 0.0f;
    outEdge = 0.0f;

    const float cloudBase = gPush.cloudProfileParams.z;
    const float cloudTop = max(gPush.cloudProfileParams.w, cloudBase + 0.25f);
    const float cloudHeight = max(cloudTop - cloudBase, 0.1f);
    const float macroScale = max(gPush.cloudShapeParams.x, 0.05f);
    const float coverage = saturate(gPush.cloudProfileParams.x);

    // Keep a soft fade band below cloudBase to avoid a hard geometric floor.
    const float baseFeather = cloudHeight * 0.10f;
    if (p.y < (cloudBase - baseFeather) || p.y > cloudTop) {
        return;
    }

    float edgeFadeX = smoothstep(-9.5f, -6.0f, p.x) * (1.0f - smoothstep(6.0f, 9.5f, p.x));
    float edgeFadeZ = smoothstep(-9.5f, -6.0f, p.z) * (1.0f - smoothstep(6.0f, 9.5f, p.z));
    float boundsMask = edgeFadeX * edgeFadeZ;

    float3 q = (p - kCloudAabbMin) / (kCloudAabbMax - kCloudAabbMin);
    float h = saturate((p.y - cloudBase) / cloudHeight);

    float3 warp = sampleCloudNoise(q * macroScale * 1.5f).gba - 0.5f;
    float3 qWarped = q + (warp * 0.04f);
    qWarped.x += h * 0.08f;

    float2 weatherUvw = (qWarped.xz + float2(0.31f, -0.17f)) * (macroScale * 0.5f);
    float weatherNoise = sampleCloudNoise(float3(weatherUvw.x, 0.0f, weatherUvw.y)).r;
    float cellMask = saturate((weatherNoise - (1.0f - coverage)) / max(coverage, 0.01f));

    // Perturb vertical profile to avoid flat "sliced" tops/bottoms near bounds.
    float heightWarp = (weatherNoise - 0.5f) * 0.15f;
    float hWarped = saturate(h + heightWarp);
    float bottomRamp = smoothstep(0.0f, 0.05f, hWarped);
    float localTop = lerp(0.3f, 1.0f, cellMask);
    float topTaper = 1.0f - smoothstep(localTop - 0.25f, localTop, hWarped);

    float structuralMask = boundsMask * bottomRamp * topTaper * cellMask;
    if (structuralMask <= 0.0f) {
        return;
    }

    float3 baseUvw = qWarped * (macroScale * 2.5f);
    float baseNoise = sampleCloudNoise(baseUvw).r;
    baseNoise = saturate(baseNoise + (coverage * 0.5f * structuralMask));

    float remapThreshold = (1.0f - structuralMask) * 0.65f;
    float baseDensity = remap(baseNoise, remapThreshold, 1.0f, 0.0f, 1.0f);
    baseDensity *= boundsMask;
    baseDensity = saturate(pow(baseDensity, 0.6f));
    if (baseDensity <= 0.0f) {
        return;
    }

    outDensity = baseDensity;
    outEdge = pow(1.0f - baseDensity, 0.5f);
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
    float density = 0.0f;
    float edge = 0.0f;
    densityField(p, density, edge);
    gDensityVolume[dispatchThreadID] = density;
    gDensityEdgeVolume[dispatchThreadID] = edge;
}
