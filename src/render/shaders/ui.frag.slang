// UI fragment shader. The renderer binds one texture per draw command at
// (set 0, binding 0): an opaque white 1x1 for solid fills, an RGBA atlas for
// images/9-slice, or the R8 glyph atlas for text.
[[vk::binding(0, 0)]]
Sampler2D<float4> uiTexture;

struct FragmentInput
{
    float4 inColor : TEXCOORD0;
    float2 inUv : TEXCOORD1;
    nointerpolation uint inMode : TEXCOORD2;
};

static const uint kModeSolid = 0u;
static const uint kModeTextured = 1u;
static const uint kModeGlyphAlpha = 2u;

float srgbToLinear1(float c)
{
    return c <= 0.04045 ? (c / 12.92) : pow((c + 0.055) / 1.055, 2.4);
}

float3 srgbToLinear(float3 c)
{
    return float3(srgbToLinear1(c.r), srgbToLinear1(c.g), srgbToLinear1(c.b));
}

float4 main(FragmentInput input) : SV_Target0
{
    float4 straight = input.inColor;
    if (input.inMode == kModeTextured)
    {
        straight = uiTexture.Sample(input.inUv) * input.inColor;
    }
    else if (input.inMode == kModeGlyphAlpha)
    {
        const float coverage = uiTexture.Sample(input.inUv).r;
        straight = float4(input.inColor.rgb, input.inColor.a * coverage);
    }

    // Author colors are sRGB hex; the swapchain is *_SRGB and re-encodes on store,
    // so linearize here to keep WYSIWYG. Alpha stays straight for blending.
    return float4(srgbToLinear(straight.rgb), straight.a);
}
