import camera_uniform;
import sh_lighting;
import sdf_scene;

[[vk::binding(4, 0)]]
Sampler2DShadow shadowMap;

[[vk::binding(9, 0)]]
Sampler3D<float4> voxelGiVolume;

[[vk::binding(11, 0)]]
Sampler3D<float4> voxelGiOccupancyVolume;

struct FragmentInput
{
    float2 inUv : TEXCOORD0;
};

struct FragmentOutput
{
    float4 color : SV_Target0;
    float depth : SV_Depth;
};

int chooseShadowCascade(float viewDepth)
{
    if (viewDepth <= camera.shadowCascadeSplits.x) return 0;
    if (viewDepth <= camera.shadowCascadeSplits.y) return 1;
    if (viewDepth <= camera.shadowCascadeSplits.z) return 2;
    return 3;
}

float sampleShadowVisibility(float3 worldPosition, float3 normal, float viewDepth, float ndotl)
{
    const int cascadeIndex = chooseShadowCascade(viewDepth);
    const float cascadeT = float(cascadeIndex) / 3.0;
    const float normalOffset = lerp(camera.shadowConfig0.x, camera.shadowConfig0.y, cascadeT) * (1.0 + (1.0 - ndotl));
    const float3 samplePosition = worldPosition + (normal * normalOffset);
    const float4 shadowClip = mul(camera.lightViewProj[cascadeIndex], float4(samplePosition, 1.0));
    if (abs(shadowClip.w) < 1e-6) return 1.0;

    const float3 shadowNdc = shadowClip.xyz / shadowClip.w;
    const float2 shadowUvLocal = (shadowNdc.xy * 0.5) + 0.5;
    if (shadowUvLocal.x <= 0.0 || shadowUvLocal.x >= 1.0 || shadowUvLocal.y <= 0.0 || shadowUvLocal.y >= 1.0)
    {
        return 1.0;
    }
    if (shadowNdc.z <= 0.0 || shadowNdc.z >= 1.0)
    {
        return 1.0;
    }

    const float4 atlasRect = camera.shadowAtlasUvRects[cascadeIndex];
    const float2 shadowUv = atlasRect.xy + (shadowUvLocal * atlasRect.zw);

    uint texW = 0u;
    uint texH = 0u;
    shadowMap.GetDimensions(texW, texH);
    const float texelScale = 1.0 / max(float(min(texW, texH)), 1.0);
    const float baseBias =
        lerp(camera.shadowConfig0.z, camera.shadowConfig0.w, cascadeT) * texelScale +
        lerp(camera.shadowConfig1.x, camera.shadowConfig1.y, cascadeT) * texelScale * (1.0 - ndotl);
    const int kernelRadius = clamp(int(camera.shadowConfig3.w + 0.5), 1, 3);
    const float2 texelSize = float2(1.0 / max(float(texW), 1.0), 1.0 / max(float(texH), 1.0));

    float visibility = 0.0;
    float weightSum = 0.0;
    for (int y = -kernelRadius; y <= kernelRadius; ++y)
    {
        for (int x = -kernelRadius; x <= kernelRadius; ++x)
        {
            const float2 uv = shadowUv + (float2(float(x), float(y)) * texelSize);
            const float weight = 1.0;
            visibility += shadowMap.SampleCmpLevelZero(uv, shadowNdc.z + baseBias) * weight;
            weightSum += weight;
        }
    }
    return visibility / max(weightSum, 1e-4);
}

float3 sampleVoxelGi(float3 worldPosition, float3 normal)
{
    const float cellSize = max(camera.voxelGiGridOriginCellSize.w, 0.001);
    const float3 gridOrigin = camera.voxelGiGridOriginCellSize.xyz;
    const float3 gridExtent = max(camera.voxelGiGridExtentStrength.xyz, float3(cellSize, cellSize, cellSize));
    const float3 uvwCenter = (worldPosition - gridOrigin) / gridExtent;
    if (any(uvwCenter < float3(0.0, 0.0, 0.0)) || any(uvwCenter > float3(1.0, 1.0, 1.0)))
    {
        return float3(0.0, 0.0, 0.0);
    }

    const float3 uvwForward = (worldPosition + (normal * (cellSize * 0.55)) - gridOrigin) / gridExtent;
    const float3 uvwInterior = (worldPosition - (normal * (cellSize * 0.35)) - gridOrigin) / gridExtent;
    const float3 giCenter = voxelGiVolume.SampleLevel(uvwCenter, 0.0).xyz;
    const float3 giForward =
        (any(uvwForward < float3(0.0, 0.0, 0.0)) || any(uvwForward > float3(1.0, 1.0, 1.0)))
            ? giCenter
            : voxelGiVolume.SampleLevel(uvwForward, 0.0).xyz;

    const float interiorOccupancy =
        (any(uvwInterior < float3(0.0, 0.0, 0.0)) || any(uvwInterior > float3(1.0, 1.0, 1.0)))
            ? 0.0
            : voxelGiOccupancyVolume.SampleLevel(uvwInterior, 0.0).x;
    const float oneSidedBlend = (interiorOccupancy > 0.5) ? 1.0 : 0.0;
    const float3 giRadiance = lerp((giCenter * 0.70) + (giForward * 0.30), (giCenter * 0.20) + (giForward * 0.80), oneSidedBlend);
    return giRadiance * max(camera.voxelGiGridExtentStrength.w, 0.0);
}

float giLumaFromColor(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float3 voxelGiFalseColor(float t)
{
    const float x = clamp(t, 0.0, 1.0);
    if (x < 0.25) return lerp(float3(0.02, 0.02, 0.10), float3(0.00, 0.42, 1.00), x / 0.25);
    if (x < 0.50) return lerp(float3(0.00, 0.42, 1.00), float3(0.00, 0.95, 0.45), (x - 0.25) / 0.25);
    if (x < 0.75) return lerp(float3(0.00, 0.95, 0.45), float3(1.00, 0.92, 0.12), (x - 0.50) / 0.25);
    return lerp(float3(1.00, 0.92, 0.12), float3(1.00, 0.16, 0.12), (x - 0.75) / 0.25);
}

FragmentOutput main(FragmentInput input)
{
    FragmentOutput output;
    output.color = float4(0.0, 0.0, 0.0, 1.0);
    output.depth = 0.0;

    const float blend = sdfBlendFromTime(camera.skyConfig1.z);
    const float3 objectCenter = sdfObjectCenter();
    const float3 rayOrigin = camera.shadowVoxelGridSize.xyz;
    const float3 rayDirection = viewToWorldDirection(reconstructViewRay(input.inUv));

    float tHit = 0.0;
    float3 hitWorld = rayOrigin;
    if (!raymarchScene(rayOrigin, rayDirection, 0.0, kSdfMaxDistance, objectCenter, blend, tHit, hitWorld))
    {
        discard;
    }

    const float3 normal = sdfSceneNormal(hitWorld, objectCenter, blend);
    const float4 viewPosition = mul(camera.view, float4(hitWorld, 1.0));
    const float viewDepth = max(-viewPosition.z, 0.0);
    const float4 clipPosition = mul(camera.proj, viewPosition);
    if (abs(clipPosition.w) < 1e-6)
    {
        discard;
    }
    output.depth = clipPosition.z / clipPosition.w;

    const float3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const float3 sunColor = camera.sunColorShadow.xyz;
    const float ndotl = max(dot(normal, -sunDirection), 0.0);
    const float shadowVisibility = (ndotl > 0.0001) ? sampleShadowVisibility(hitWorld, normal, viewDepth, ndotl) : 1.0;
    const float3 ambientSh = evaluateShHemisphereIrradiance(normal) * 0.78;
    const float giProbeDistance = max(camera.voxelGiGridOriginCellSize.w * 0.90, 0.08);
    const float3 ambientGi = sampleVoxelGi(hitWorld + (normal * giProbeDistance), normal);
    const int voxelGiDebugMode = int(camera.shadowVoxelGridOrigin.z + 0.5);
    if (voxelGiDebugMode == 1)
    {
        const float3 giClamped = max(ambientGi, float3(0.0, 0.0, 0.0));
        const float peak = max(max(giClamped.x, giClamped.y), giClamped.z);
        const float3 hue = (peak > 0.0001) ? (giClamped / peak) : float3(0.0, 0.0, 0.0);
        const float intensity = clamp(log2(1.0 + (peak * 8.0)) / 4.0, 0.0, 1.0);
        output.color = float4(hue * intensity, 1.0);
        return output;
    }
    if (voxelGiDebugMode == 2)
    {
        const float giLuma = giLumaFromColor(ambientGi);
        const float giMapped = clamp(log2(1.0 + (giLuma * 6.0)) / 3.0, 0.0, 1.0);
        output.color = float4(voxelGiFalseColor(giMapped), 1.0);
        return output;
    }
    if (voxelGiDebugMode == 3)
    {
        const float giLuma = giLumaFromColor(ambientGi);
        const float giMapped = clamp(log2(1.0 + (giLuma * 4.0)) / 3.0, 0.0, 1.0);
        output.color = float4(giMapped, giMapped, giMapped, 1.0);
        return output;
    }

    const float ambientRebalanceStrength = max(camera.shadowVoxelGridOrigin.x, 0.0);
    const float ambientFloor = clamp(camera.shadowVoxelGridOrigin.y, 0.0, 1.0);
    const float giLuma = giLumaFromColor(ambientGi);
    const float ambientScale = clamp(1.0 / (1.0 + (giLuma * ambientRebalanceStrength)), ambientFloor, 1.0);
    const float3 direct = sunColor * (sunIntensity * ndotl * shadowVisibility);
    const float3 baseColor = float3(0.82, 0.82, 0.82);
    const float3 viewDir = normalize(rayOrigin - hitWorld);
    const float3 halfVec = normalize((-sunDirection) + viewDir);
    const float specular = pow(max(dot(normal, halfVec), 0.0), 48.0) * 0.18 * shadowVisibility;
    const float3 indirectLit = ((ambientSh * ambientScale * 1.55) + (ambientGi * 2.35)) * 0.98;
    const float3 directLit = direct * 0.90;
    output.color = float4(baseColor * (indirectLit + directLit) + specular, 1.0);
    return output;
}
