import camera_uniform;
import sdf_scene;

struct FragmentInput
{
    float2 inUv : TEXCOORD0;
    nointerpolation int cascadeIndex : TEXCOORD1;
};

float main(FragmentInput input) : SV_Depth
{
    const int cascadeIndex = clamp(input.cascadeIndex, 0, 3);
    const float2 ndcXY = (input.inUv * 2.0) - 1.0;
    const float4x4 invLightViewProj = inverse4x4(camera.lightViewProj[cascadeIndex]);

    float4 nearWorldH = mul(invLightViewProj, float4(ndcXY.x, ndcXY.y, 1.0, 1.0));
    float4 farWorldH = mul(invLightViewProj, float4(ndcXY.x, ndcXY.y, 0.0, 1.0));
    if (abs(nearWorldH.w) < 1e-6 || abs(farWorldH.w) < 1e-6)
    {
        discard;
    }
    nearWorldH.xyz /= nearWorldH.w;
    farWorldH.xyz /= farWorldH.w;

    const float3 rayOrigin = nearWorldH.xyz;
    const float3 rayVector = farWorldH.xyz - nearWorldH.xyz;
    const float rayLength = length(rayVector);
    if (rayLength <= 1e-6)
    {
        discard;
    }
    const float3 rayDirection = rayVector / rayLength;

    const float blend = sdfBlendFromTime(camera.skyConfig1.z);
    const float3 objectCenter = sdfObjectCenter();
    float tHit = 0.0;
    float3 hitWorld = rayOrigin;
    if (!raymarchScene(rayOrigin, rayDirection, 0.0, rayLength, objectCenter, blend, tHit, hitWorld))
    {
        discard;
    }

    const float4 clipPosition = mul(camera.lightViewProj[cascadeIndex], float4(hitWorld, 1.0));
    if (abs(clipPosition.w) < 1e-6)
    {
        discard;
    }
    return clipPosition.z / clipPosition.w;
}
