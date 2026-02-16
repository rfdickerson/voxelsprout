import camera_uniform;
import sh_lighting;

interface ITransportStyle
{
    float3 ShadeBase(float3 tint, float localAlong, float flowTime, float flowSpeed, float flowHighlight);
};

struct PipeStyle : ITransportStyle
{
    float3 ShadeBase(float3 tint, float localAlong, float flowTime, float flowSpeed, float flowHighlight)
    {
        const float transferStart = 0.26;
        const float transferEnd = 0.74;
        const float transferEdge = 0.03;
        const float liquidMask =
            smoothstep(transferStart, transferStart + transferEdge, localAlong) *
            (1.0 - smoothstep(transferEnd - transferEdge, transferEnd, localAlong));
        const float3 liquidBaseColor = float3(1.0, 0.08, 0.78);
        const float3 liquidColor = liquidBaseColor * (0.82 + (0.48 * flowHighlight));
        return lerp(tint, liquidColor, liquidMask);
    }
};

struct ConveyorStyle : ITransportStyle
{
    float3 ShadeBase(float3 tint, float localAlong, float flowTime, float flowSpeed, float flowHighlight)
    {
        const float conveyorBand = 1.0 - abs((frac((localAlong * 4.0) - (flowTime * flowSpeed * 0.8)) * 2.0) - 1.0);
        const float conveyorHighlight = pow(clamp(conveyorBand, 0.0, 1.0), 2.2);
        return tint * (1.0 + (0.30 * conveyorHighlight));
    }
};

struct TrackStyle : ITransportStyle
{
    float3 ShadeBase(float3 tint, float localAlong, float flowTime, float flowSpeed, float flowHighlight)
    {
        const float tiePulse = step(0.68, frac(localAlong * 6.0));
        const float tieDarken = lerp(1.0, 0.82, tiePulse);
        return tint * tieDarken;
    }
};

float3 applyTransportStyle<TStyle : ITransportStyle>(TStyle style, float3 tint, float localAlong, float flowTime, float flowSpeed, float flowHighlight)
{
    return style.ShadeBase(tint, localAlong, flowTime, flowSpeed, flowHighlight);
}

struct FragmentInput
{
    float3 inWorldPosition : TEXCOORD0;
    float3 inWorldNormal : TEXCOORD1;
    float3 inTint : TEXCOORD2;
    float inVertexAo : TEXCOORD3;
    float inLocalAlong : TEXCOORD4;
    float inStyle : TEXCOORD5;
};

float4 main(FragmentInput input) : SV_Target0
{
    const float3 kSunWarmTint = float3(1.05, 0.95, 0.88);
    const float3 kSkyCoolTint = float3(0.78, 0.86, 1.08);
    const float3 normal = normalize(input.inWorldNormal);
    const float3 sunDirection = normalize(camera.sunDirectionIntensity.xyz);
    const float sunIntensity = max(camera.sunDirectionIntensity.w, 0.0);
    const float3 sunColor = camera.sunColorShadow.xyz * kSunWarmTint;
    const float ndotl = max(dot(normal, -sunDirection), 0.0);

    const float3 shNormalIrradiance = evaluateShIrradiance(normal);
    const float3 shHemisphereIrradiance = evaluateShHemisphereIrradiance(normal);
    const float3 ambientIrradiance = lerp(shNormalIrradiance, shHemisphereIrradiance, 0.70);
    const float vertexAoEnable = clamp(camera.shadowVoxelGridOrigin.w, 0.0, 1.0);
    const float vertexAo = lerp(1.0, clamp(input.inVertexAo, 0.0, 1.0), vertexAoEnable);
    float3 ambient = (ambientIrradiance * kSkyCoolTint) * (0.36 * vertexAo);
    const float ambientVertical = lerp(0.85, 1.1, (normal.y * 0.5) + 0.5);
    ambient *= ambientVertical;
    const float3 directSun = sunColor * (sunIntensity * ndotl);
    const float indirectScale = 0.92;
    const float directScale = 1.10;

    const float flowTime = camera.skyConfig1.z;
    const float flowSpeed = max(camera.skyConfig1.w, 0.0);
    const float flowCoord = (input.inLocalAlong * 6.5) - (flowTime * flowSpeed);
    const float flowBand = 1.0 - abs((frac(flowCoord) * 2.0) - 1.0);
    const float flowHighlight = pow(clamp(flowBand, 0.0, 1.0), 3.0);

    const float stylePipe = 1.0 - step(0.5, input.inStyle);
    const float styleConveyor = step(0.5, input.inStyle) * (1.0 - step(1.5, input.inStyle));
    const float styleTrack = step(1.5, input.inStyle);

    const float3 pipeTint = applyTransportStyle(PipeStyle(), input.inTint, input.inLocalAlong, flowTime, flowSpeed, flowHighlight);
    const float3 conveyorTint = applyTransportStyle(ConveyorStyle(), input.inTint, input.inLocalAlong, flowTime, flowSpeed, flowHighlight);
    const float3 trackTint = applyTransportStyle(TrackStyle(), input.inTint, input.inLocalAlong, flowTime, flowSpeed, flowHighlight);

    float3 surfaceTint = input.inTint;
    surfaceTint = lerp(surfaceTint, pipeTint, stylePipe);
    surfaceTint = lerp(surfaceTint, conveyorTint, styleConveyor);
    surfaceTint = lerp(surfaceTint, trackTint, styleTrack);

    const float transferStart = 0.26;
    const float transferEnd = 0.74;
    const float transferEdge = 0.03;
    const float pipeLiquidMask =
        smoothstep(transferStart, transferStart + transferEdge, input.inLocalAlong) *
        (1.0 - smoothstep(transferEnd - transferEdge, transferEnd, input.inLocalAlong));
    const float liquidMask = pipeLiquidMask * stylePipe;
    const float3 liquidBaseColor = float3(1.0, 0.08, 0.78);
    const float3 liquidColor = liquidBaseColor * (0.82 + (0.48 * flowHighlight));

    const float3 lit =
        ((((ambient * indirectScale) + (directSun * directScale))) * surfaceTint) +
        (liquidColor * ((0.24 + (0.42 * flowHighlight)) * liquidMask));
    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float3 viewDir = normalize(cameraWorld - input.inWorldPosition);
    const float nDotV = saturate(dot(normal, viewDir));
    const float rim = pow(1.0 - nDotV, 2.0);
    const float3 rimTint = float3(1.06, 0.94, 0.84);
    const float3 shaded = lit + (rimTint * (rim * 0.04));

    return float4(shaded, 1.0);
}
