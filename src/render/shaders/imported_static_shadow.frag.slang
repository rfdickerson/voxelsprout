[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

static const uint kImportedMaterialFlagAlphaTest = 1u;
static const uint kInvalidImportedTextureIndex = 0xffffffffu;

struct FragmentInput
{
    float2 inUv : TEXCOORD0;
    nointerpolation uint inTextureIndex : TEXCOORD1;
    nointerpolation uint inFlags : TEXCOORD2;
};

void main(FragmentInput input)
{
    if (input.inTextureIndex == kInvalidImportedTextureIndex)
    {
        return;
    }
    if ((input.inFlags & kImportedMaterialFlagAlphaTest) == 0u)
    {
        return;
    }
    if (bindlessTextures[input.inTextureIndex].Sample(input.inUv).a < 0.5)
    {
        discard;
    }
}
