import camera_uniform;
import noise;

struct HullOutput
{
    float2 patchMin : TEXCOORD0;
    float2 patchMax : TEXCOORD1;
    float2 cornerUv : TEXCOORD2;
};

struct PatchConstants
{
    float edges[4] : SV_TessFactor;
    float inside[2] : SV_InsideTessFactor;
};

struct DomainOutput
{
    float4 position : SV_Position;
    float3 outWorldPosition : TEXCOORD0;
    float3 outWorldNormal : TEXCOORD1;
    float outSlope : TEXCOORD2;
    float outFlow : TEXCOORD3;
};

struct TerrainSample
{
    float height;
    float flow;
    float ridge;
    float cliff;
};

// Noise primitives (hashNoise / valueNoise / ridgedNoise / fbm3) live in the
// shared `noise` module. `fbm` here is the three-octave variant.
float fbm(float2 p)
{
    return fbm3(p);
}

float2 domainWarp(float2 worldXz)
{
    const float2 broadWarp =
        float2(fbm(worldXz * 0.0055 + float2(13.7, -7.1)),
               fbm(worldXz * 0.0055 + float2(-19.4, 29.3))) * 30.0;
    const float2 localWarp =
        float2(valueNoise((worldXz + broadWarp) * 0.022 + float2(41.0, -16.0)),
               valueNoise((worldXz + broadWarp) * 0.022 + float2(-27.0, 52.0))) * 6.5;
    return worldXz + broadWarp + localWarp;
}

float flowAtWarped(float2 warped)
{
    const float primary = ridgedNoise(warped * 0.010 + float2(73.0, -41.0));
    const float branch = ridgedNoise((warped * 0.021) + float2(-26.0, 88.0));
    return pow(saturate((primary * 0.78) + (branch * 0.22)), 4.2);
}

TerrainSample sampleTerrain(float2 worldXz)
{
    const float2 warped = domainWarp(worldXz);
    const float macroShape = fbm(warped * 0.0080);
    const float rolling = fbm(warped * 0.020);
    const float slopeDetail = fbm(warped * 0.072 + float2(11.0, -47.0));
    const float rawRidge =
        (ridgedNoise(warped * 0.016 - float2(24.0, -18.0)) * 0.72) +
        (ridgedNoise(warped * 0.039 + float2(9.0, -35.0)) * 0.28);
    const float ridge = pow(saturate(rawRidge), 2.2);
    const float sharpenedRidge = pow(saturate((ridge - 0.23) / 0.77), 1.45);

    const float basinNoise = (fbm(warped * 0.004 + float2(141.0, 26.0)) * 0.5) + 0.5;
    const float basin = pow(saturate((0.54 - basinNoise) / 0.54), 1.65) * 5.4;
    const float flow = flowAtWarped(warped);
    const float flowChannel = smoothstep(0.48, 0.94, flow);

    float height =
        9.0 +
        (macroShape * 5.8) +
        (rolling * 2.9) +
        (slopeDetail * 0.55) +
        (ridge * 2.8) +
        (sharpenedRidge * 4.4) -
        basin;

    // Lightweight erosion approximation: route water through ridged flow masks,
    // cut channels, then deposit a small amount along the banks.
    const float bankDeposit = smoothstep(0.30, 0.68, flow) * (1.0 - flowChannel) * 0.52;
    height += bankDeposit;
    height -= flowChannel * (0.55 + (sharpenedRidge * 0.85));

    // Cliff bias on steep ridge transitions. This is evaluated again from local
    // height deltas in the domain shader, but this pre-bias creates crisper forms.
    const float cliff = smoothstep(0.58, 0.92, sharpenedRidge) * smoothstep(0.16, 0.62, abs(rolling));
    height = lerp(height, floor(height * 0.42) / 0.42, cliff * 0.18);

    TerrainSample result;
    result.height = height;
    result.flow = flow;
    result.ridge = sharpenedRidge;
    result.cliff = cliff;
    return result;
}

float heightAt(float2 worldXz)
{
    return sampleTerrain(worldXz).height;
}

[domain("quad")]
[partitioning("fractional_odd")]
DomainOutput main(
    PatchConstants patchConstants,
    const OutputPatch<HullOutput, 4> patch,
    float2 uv : SV_DomainLocation)
{
    const float2 patchMin = patch[0].patchMin;
    const float2 patchMax = patch[0].patchMax;
    const float2 worldXz = lerp(patchMin, patchMax, uv);
    const TerrainSample terrain = sampleTerrain(worldXz);
    const float height = terrain.height;
    const float sampleStep = 1.0;
    const float heightWest = heightAt(worldXz - float2(sampleStep, 0.0));
    const float heightEast = heightAt(worldXz + float2(sampleStep, 0.0));
    const float heightNorth = heightAt(worldXz - float2(0.0, sampleStep));
    const float heightSouth = heightAt(worldXz + float2(0.0, sampleStep));
    const float gradientMagnitude =
        max(abs(heightEast - heightWest), abs(heightSouth - heightNorth)) / (sampleStep * 2.0);
    const float cliffBias = saturate(max(terrain.cliff, smoothstep(0.58, 1.65, gradientMagnitude)));
    const float normalYScale = lerp(sampleStep * 2.0, sampleStep * 1.10, cliffBias);
    const float3 normal = normalize(float3(heightWest - heightEast, normalYScale, heightNorth - heightSouth));

    const float3 worldPosition = float3(worldXz.x, height, worldXz.y);
    DomainOutput output;
    output.outWorldPosition = worldPosition;
    output.outWorldNormal = normal;
    output.outSlope = saturate(1.0 - normal.y);
    output.outFlow = terrain.flow;
    output.position = mul(camera.mvp, float4(worldPosition, 1.0));
    return output;
}
