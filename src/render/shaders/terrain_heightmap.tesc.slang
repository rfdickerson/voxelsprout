import camera_uniform;

struct HullInput
{
    float2 patchMin : TEXCOORD0;
    float2 patchMax : TEXCOORD1;
    float2 cornerUv : TEXCOORD2;
};

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

float patchTessFactor(float2 patchMin, float2 patchMax)
{
    const float2 centerXz = (patchMin + patchMax) * 0.5;
    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float distance = length(float3(centerXz.x, 0.0, centerXz.y) - cameraWorld);
    const float nearTess = 11.0;
    const float farTess = 2.0;
    return clamp(lerp(nearTess, farTess, smoothstep(48.0, 260.0, distance)), farTess, nearTess);
}

PatchConstants patchConstants(InputPatch<HullInput, 4> patch)
{
    const float tess = patchTessFactor(patch[0].patchMin, patch[0].patchMax);
    PatchConstants constants;
    constants.edges[0] = tess;
    constants.edges[1] = tess;
    constants.edges[2] = tess;
    constants.edges[3] = tess;
    constants.inside[0] = tess;
    constants.inside[1] = tess;
    return constants;
}

[domain("quad")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(4)]
[patchconstantfunc("patchConstants")]
HullOutput main(InputPatch<HullInput, 4> patch, uint controlPointId : SV_OutputControlPointID)
{
    HullOutput output;
    output.patchMin = patch[controlPointId].patchMin;
    output.patchMax = patch[controlPointId].patchMax;
    output.cornerUv = patch[controlPointId].cornerUv;
    return output;
}
