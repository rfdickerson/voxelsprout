import camera_uniform;
import sdf_scene;

struct FragmentInput
{
    float2 inUv : TEXCOORD0;
};

struct FragmentOutput
{
    float4 normalDepth : SV_Target0;
    float depth : SV_Depth;
};

FragmentOutput main(FragmentInput input)
{
    FragmentOutput output;
    output.normalDepth = float4(0.5, 0.5, 0.5, 0.0);
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

    const float3 worldNormal = sdfSceneNormal(hitWorld, objectCenter, blend);
    const float3 viewNormal = normalize(mul((float3x3)camera.view, worldNormal));
    const float4 viewPosition = mul(camera.view, float4(hitWorld, 1.0));
    const float viewDepth = max(-viewPosition.z, 0.0);
    const float4 clipPosition = mul(camera.proj, viewPosition);
    if (abs(clipPosition.w) < 1e-6)
    {
        discard;
    }

    output.normalDepth = float4((viewNormal * 0.5) + 0.5, viewDepth);
    output.depth = clipPosition.z / clipPosition.w;
    return output;
}
