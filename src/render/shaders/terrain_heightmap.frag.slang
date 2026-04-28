import camera_uniform;
import sh_lighting;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

static const uint kBindlessTextureIndexTerrainDetail = 9u;

struct FragmentInput
{
    float3 inWorldPosition : TEXCOORD0;
    float3 inWorldNormal : TEXCOORD1;
    float inSlope : TEXCOORD2;
    float inFlow : TEXCOORD3;
};

float3 terrainAlbedo(float height, float slope, float flow)
{
    const float3 scrubGrass = float3(0.28, 0.42, 0.18);
    const float3 drySoil = float3(0.38, 0.30, 0.20);
    const float3 darkChannel = float3(0.22, 0.18, 0.13);
    const float3 rock = float3(0.42, 0.42, 0.39);
    const float3 cliffRock = float3(0.52, 0.51, 0.47);
    const float rockMask = smoothstep(0.22, 0.62, slope);
    const float cliffMask = smoothstep(0.58, 0.88, slope);
    const float ridgeMask = smoothstep(15.0, 23.0, height);
    const float flowMask = smoothstep(0.42, 0.86, flow);
    float3 albedo = lerp(scrubGrass, drySoil, smoothstep(0.08, 0.34, slope) * 0.58);
    albedo = lerp(albedo, rock, rockMask);
    albedo = lerp(albedo, cliffRock, ridgeMask * cliffMask * 0.46);
    albedo = lerp(albedo, darkChannel, flowMask * (1.0 - cliffMask * 0.35) * 0.58);
    return albedo;
}

void terrainNormalBasis(float3 normal, out float3 tangent, out float3 bitangent)
{
    const float3 up = (abs(normal.y) < 0.96) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    tangent = normalize(cross(up, normal));
    bitangent = normalize(cross(normal, tangent));
}

float3 applyNearTerrainDetail(float3 normal, float3 worldPosition, float slope, out float3 albedoModulation)
{
    const float cameraDistance = distance(camera.shadowVoxelGridSize.xyz, worldPosition);
    const float distanceFade = 1.0 - smoothstep(88.0, 175.0, cameraDistance);
    const float slopeFade = smoothstep(0.02, 0.58, slope);
    const float detailWeight = distanceFade * lerp(0.48, 1.0, slopeFade);

    if (detailWeight <= 0.001)
    {
        albedoModulation = float3(1.0, 1.0, 1.0);
        return normal;
    }

    const float2 detailUv0 = worldPosition.xz * 0.22;
    const float2 detailUv1 = (float2(worldPosition.z, -worldPosition.x) * 0.38) + float2(17.0, -9.0);
    const float3 detailColor0 = bindlessTextures[kBindlessTextureIndexTerrainDetail].Sample(detailUv0).rgb;
    const float3 detailColor1 = bindlessTextures[kBindlessTextureIndexTerrainDetail].Sample(detailUv1).rgb;
    const float detailHeight0 = dot(detailColor0, float3(0.299, 0.587, 0.114));
    const float detailHeight1 = dot(detailColor1, float3(0.299, 0.587, 0.114));
    const float2 detailRg = float2(detailHeight0 - 0.5, detailHeight1 - 0.5) * 2.0;

    float3 tangent;
    float3 bitangent;
    terrainNormalBasis(normal, tangent, bitangent);
    const float detailNormalStrength = 0.45 * detailWeight;
    const float3 detailNormal = normalize(normal + (tangent * detailRg.x + bitangent * detailRg.y) * detailNormalStrength);

    const float3 neutralDetail = saturate((detailColor0 * 1.58) + float3(0.05, 0.04, 0.015));
    const float3 contrastDetail = lerp(float3(1.0, 1.0, 1.0), neutralDetail, 0.90);
    albedoModulation = lerp(float3(1.0, 1.0, 1.0), contrastDetail, detailWeight);
    return detailNormal;
}

float4 main(FragmentInput input) : SV_Target0
{
    float3 albedoModulation = float3(1.0, 1.0, 1.0);
    const float3 normal = applyNearTerrainDetail(
        normalize(input.inWorldNormal),
        input.inWorldPosition,
        input.inSlope,
        albedoModulation
    );
    const float3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const float3 sunColor = camera.sunColorShadow.xyz * float3(1.05, 0.95, 0.88);
    const float ndotl = max(dot(normal, -sunDirection), 0.0);
    const float3 shAmbient = evaluateShHemisphereIrradiance(normal) * float3(0.78, 0.86, 1.08) * 0.72;
    const float3 direct = sunColor * (sunIntensity * ndotl);
    const float3 albedo = terrainAlbedo(input.inWorldPosition.y, input.inSlope, input.inFlow) * albedoModulation;
    const float3 color = albedo * (shAmbient + direct);
    return float4(color, 1.0);
}
