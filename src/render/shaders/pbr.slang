module pbr;

// Metallic-roughness BRDF terms, layered onto the engine's existing baked-GI
// diffuse chain rather than replacing it (docs/ROADMAP.md, Core Rendering).
//
// Deliberately a *specular* library only: the diffuse response stays whatever
// the calling shader already computed (SH ambient + voxel GI + wrapped sun +
// local lights), so opting a surface into PBR adds highlights and metal
// behavior without re-tuning the art direction. The only thing PBR takes away
// from diffuse is the metallic fraction, which the caller applies via
// pbrDiffuseScale().
//
// The material itself is unpacked from ImportedScenePackedVertex::flags — see
// the bit layout in src/import/imported_scene.h, mirrored in
// imported_static.frag.slang.

public struct PbrSurface
{
    public float metallic;
    public float roughness;
};

// Dielectrics reflect ~4% at normal incidence; metals tint their reflection
// with the base color and have no diffuse lobe of their own.
public float3 pbrSpecularF0(float3 albedo, float metallic)
{
    return lerp(float3(0.04, 0.04, 0.04), albedo, saturate(metallic));
}

// Metals lose their diffuse lobe entirely; dielectrics keep all of it.
public float pbrDiffuseScale(float metallic)
{
    return 1.0 - saturate(metallic);
}

// Clamped so a perfectly smooth surface still produces a finite, filterable
// highlight instead of an aliasing single-pixel spark.
public float pbrPerceptualToAlpha(float roughness)
{
    const float clamped = clamp(roughness, 0.045, 1.0);
    return clamped * clamped;
}

// Trowbridge-Reitz / GGX normal distribution.
public float pbrDistributionGgx(float ndoth, float alpha)
{
    const float a2 = alpha * alpha;
    const float d = ((ndoth * ndoth) * (a2 - 1.0)) + 1.0;
    return a2 / max(3.14159265 * d * d, 1e-7);
}

// Smith height-correlated visibility, already divided by the 4*NdotL*NdotV
// denominator of the Cook-Torrance specular term.
public float pbrVisibilitySmithGgx(float ndotv, float ndotl, float alpha)
{
    const float a2 = alpha * alpha;
    const float lambdaV = ndotl * sqrt(((ndotv - (ndotv * a2)) * ndotv) + a2);
    const float lambdaL = ndotv * sqrt(((ndotl - (ndotl * a2)) * ndotl) + a2);
    return 0.5 / max(lambdaV + lambdaL, 1e-7);
}

public float3 pbrFresnelSchlick(float vdoth, float3 f0)
{
    const float f = pow(saturate(1.0 - vdoth), 5.0);
    return f0 + ((float3(1.0, 1.0, 1.0) - f0) * f);
}

// Cook-Torrance specular for one analytic light. Returns the BRDF value
// already multiplied by NdotL, so the caller only multiplies by the light's
// radiance (and its own shadow/attenuation term).
public float3 pbrDirectSpecular(
    PbrSurface surface,
    float3 albedo,
    float3 normal,
    float3 viewDirection,
    float3 lightDirection)
{
    const float ndotl = saturate(dot(normal, lightDirection));
    if (ndotl <= 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }
    const float ndotv = saturate(dot(normal, viewDirection)) + 1e-5;
    const float3 halfVector = normalize(viewDirection + lightDirection);
    const float ndoth = saturate(dot(normal, halfVector));
    const float vdoth = saturate(dot(viewDirection, halfVector));

    const float alpha = pbrPerceptualToAlpha(surface.roughness);
    const float d = pbrDistributionGgx(ndoth, alpha);
    const float vis = pbrVisibilitySmithGgx(ndotv, ndotl, alpha);
    const float3 f = pbrFresnelSchlick(vdoth, pbrSpecularF0(albedo, surface.metallic));
    return d * vis * f * ndotl;
}

// Karis' analytic split-sum environment BRDF ("Mobile HDR PBR", SIGGRAPH 2014).
// Stands in for a prefiltered BRDF LUT, which this engine has no pass for; the
// error is small at the roughness range stylized content actually uses.
public float3 pbrEnvironmentBrdf(float3 f0, float roughness, float ndotv)
{
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    const float4 r = (float4(roughness, roughness, roughness, roughness) * c0) + c1;
    const float a004 = (min(r.x * r.x, exp2(-9.28 * saturate(ndotv))) * r.x) + r.y;
    const float2 ab = (float2(-1.04, 1.04) * a004) + r.zw;
    return (f0 * ab.x) + float3(ab.y, ab.y, ab.y);
}

// Ambient (image-based-ish) specular. There is no prefiltered radiance probe in
// this engine, so the surrounding irradiance doubles as the reflected radiance:
// correct-ish for rough surfaces, and for smooth ones it at least keeps metals
// from reading as black holes in ambient-only lighting. Rougher surfaces get
// less of it, matching how a real prefiltered chain loses energy to the tail.
public float3 pbrAmbientSpecular(
    PbrSurface surface,
    float3 albedo,
    float3 ambientRadiance,
    float ndotv)
{
    const float3 f0 = pbrSpecularF0(albedo, surface.metallic);
    const float3 envBrdf = pbrEnvironmentBrdf(f0, surface.roughness, ndotv);
    const float horizonFade = lerp(1.0, 0.35, saturate(surface.roughness));
    return ambientRadiance * envBrdf * horizonFade;
}
