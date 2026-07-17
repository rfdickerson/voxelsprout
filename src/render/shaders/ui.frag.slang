// UI fragment shader. Textures live in one bindless table so clip/scissor state,
// not texture switches, defines draw command boundaries.
[[vk::binding(0, 0)]]
Sampler2D<float4> uiTextures[];

struct FragmentInput
{
    float4 inColor : TEXCOORD0;
    float2 inUv : TEXCOORD1;
    nointerpolation uint inMode : TEXCOORD2;
    nointerpolation float4 inSdf : TEXCOORD3;
    nointerpolation uint inTextureSlot : TEXCOORD4;
};

static const uint kModeSolid = 0u;
static const uint kModeTextured = 1u;
static const uint kModeGlyphAlpha = 2u;
static const uint kModeShadow = 3u;
static const uint kModeRoundRect = 4u;
static const uint kModeRoundRectGlow = 5u;
static const uint kModeGlyphSdf = 6u;

// Signed distance to a rounded box centred at the origin, half-size b, corner
// radius r. Negative inside, positive outside, magnitude in pixels.
float sdRoundBox(float2 p, float2 b, float r)
{
    float2 q = abs(p) - b + r;
    return min(max(q.x, q.y), 0.0) + length(max(q, float2(0.0, 0.0))) - r;
}

// Stem darkening: anti-aliased glyph coverage blends in linear space, which
// makes strokes read thin/faint (esp. light-on-dark). Raising coverage to a
// power <1 thickens AA edges back toward a perceptual gamma-space weight.
// >1 = heavier; tune ~1.25..1.6.
static const float kTextGamma = 1.4;

float srgbToLinear1(float c)
{
    return c <= 0.04045 ? (c / 12.92) : pow((c + 0.055) / 1.055, 2.4);
}

float3 srgbToLinear(float3 c)
{
    return float3(srgbToLinear1(c.r), srgbToLinear1(c.g), srgbToLinear1(c.b));
}

// The swapchain is B8G8R8A8_UNORM: the driver presents the raw bytes to the
// display, which treats them as sRGB-encoded. So the shader must encode
// linear → sRGB before returning. (Same role as the pow(1/2.2) in the
// tonemapper for the 3-D pass.)
float linearToSrgb1(float c)
{
    return c <= 0.0031308 ? (c * 12.92) : (1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

float3 linearToSrgb(float3 c)
{
    return float3(linearToSrgb1(c.r), linearToSrgb1(c.g), linearToSrgb1(c.b));
}

float4 main(FragmentInput input) : SV_Target0
{
    // Composite in LINEAR space. Vertex colors are authored as sRGB hex, so
    // they are linearized here. Color textures use an sRGB image format, so
    // the sampler already returns linear values -- no manual conversion here
    // (which would double-darken). The final linearToSrgb() call re-encodes
    // for the UNORM swapchain before output.
    float3 baseLin = srgbToLinear(input.inColor.rgb);
    float alpha = input.inColor.a;
    float3 rgb = baseLin;

    if (input.inMode == kModeTextured)
    {
        float4 tex = uiTextures[NonUniformResourceIndex(input.inTextureSlot)].Sample(input.inUv);
        rgb = tex.rgb * baseLin;
        alpha *= tex.a;
    }
    else if (input.inMode == kModeGlyphAlpha)
    {
        float coverage = uiTextures[NonUniformResourceIndex(input.inTextureSlot)].Sample(input.inUv).r;
        coverage = pow(coverage, 1.0 / kTextGamma);       // stem darkening
        alpha *= coverage;
    }
    else if (input.inMode == kModeGlyphSdf)
    {
        // R8 sample is a normalized (0..1) signed-distance-field byte baked by
        // stbtt_GetGlyphSDF (Font::loadFrom*(..., sdf=true)). Recover a signed
        // pixel distance the same way Font::sdfDistScale()/sdfDistBias() define
        // it CPU-side: inSdf.x = sdfDistScale, inSdf.y = sdfDistBias.
        // Negative = inside the glyph, matching sdRoundBox's sign convention, so
        // this reuses the exact RoundRect AA formula below instead of the
        // GlyphAlpha gamma/stem-darkening hack -- real analytic AA at any scale.
        float sample = uiTextures[NonUniformResourceIndex(input.inTextureSlot)].Sample(input.inUv).r;
        float d = input.inSdf.y - sample * input.inSdf.x;
        float aa = max(fwidth(d), 1e-4);
        alpha *= clamp(0.5 - d / aa, 0.0, 1.0);
    }
    else if (input.inMode == kModeShadow)
    {
        // inUv = pixel offset from the shadow-rect centre; inSdf = (halfW, halfH,
        // radius, blurSigma). Distance-to-rounded-rect Gaussian falloff: 0 on/inside
        // the rounded box, growing outward; ~1% alpha at 3*blurSigma from the edge.
        float d = sdRoundBox(input.inUv, input.inSdf.xy, input.inSdf.z);
        float sigma = max(input.inSdf.w, 1e-3);
        float dn = max(d, 0.0) / sigma;
        alpha *= exp(-dn * dn * 0.5);
    }
    else if (input.inMode == kModeRoundRect)
    {
        // inUv = pixel offset from the shape centre; inSdf = (halfW, halfH, radius, border).
        float d = sdRoundBox(input.inUv, input.inSdf.xy, input.inSdf.z);
        float border = input.inSdf.w;
        if (border > 0.0)
        {
            d = abs(d) - border * 0.5;  // stroke centred on the rounded edge path
        }
        // ~1px screen-space anti-aliasing from the analytic distance gradient.
        float aa = max(fwidth(d), 1e-4);
        alpha *= clamp(0.5 - d / aa, 0.0, 1.0);
    }
    else if (input.inMode == kModeRoundRectGlow)
    {
        // Soft halo: full inside the rounded edge, exponential falloff outside,
        // reaching ~5% of the colour's alpha at inSdf.w (the glow radius).
        float d = sdRoundBox(input.inUv, input.inSdf.xy, input.inSdf.z);
        float glow = max(input.inSdf.w, 1e-3);
        alpha *= exp(-3.0 * max(d, 0.0) / glow);
    }
    // kModeSolid falls through with rgb = baseLin and the vertex alpha.

    // Encode linear → sRGB for the UNORM swapchain.
    return float4(linearToSrgb(rgb), alpha);
}
