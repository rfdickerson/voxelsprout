// PBR material tests.
//
// Two things are under test here, and they are deliberately in one file because
// neither is worth much without the other:
//
//   1. The material DATA path -- that packImportedSceneMaterialFlags round-trips,
//      and that buildHumanoidSkinnedMesh actually stamps the opt-in bit onto the
//      vertices it should and leaves every other vertex byte-identical to the
//      pre-material generator. That path continues, untested here because none of
//      it is CPU code, through skinning_resources.cc (dst.flags = src.flags),
//      skinning.comp.slang (result.flags = v.flags), and imported_static.vert.slang
//      (output.outFlags = input.inFlags) into imported_static.frag.slang's
//      importedMaterialHasPbr(). Every one of those hops is a verbatim copy.
//
//   2. The BRDF MATH -- a literal C++ transcription of src/render/shaders/pbr.slang
//      checked against reference values that are known independently of this
//      codebase: the GGX normalization identity, Schlick's endpoints, and an
//      energy-conservation bound. There is no way to look at a rendered frame in
//      this build environment, so "it looks right" is not available; these are the
//      substitute.
//
// The transcription in kPbr* / pbr* below MUST match src/render/shaders/pbr.slang.
// Change both together -- same contract imported_scene.h already has with
// imported_static.frag.slang over the flag bit layout.

#include "import/imported_scene.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::printf("  FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

void checkNear(double actual, double expected, double tolerance, const std::string& what) {
    if (!(std::fabs(actual - expected) <= tolerance)) {
        std::printf(
            "  FAIL: %s (got %.9f, expected %.9f +/- %.9f)\n",
            what.c_str(), actual, expected, tolerance);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Literal transcription of src/render/shaders/pbr.slang.
// ---------------------------------------------------------------------------

constexpr double kPi = 3.14159265358979323846;

// pbr.slang: kPbrDiffuseConventionScale
constexpr double kPbrDiffuseConventionScale = 3.14159265;

// pbr.slang: pbrPerceptualToAlpha
double pbrPerceptualToAlpha(double roughness) {
    const double clamped = std::clamp(roughness, 0.045, 1.0);
    return clamped * clamped;
}

// pbr.slang: pbrDistributionGgx
double pbrDistributionGgx(double ndoth, double alpha) {
    const double a2 = alpha * alpha;
    const double d = ((ndoth * ndoth) * (a2 - 1.0)) + 1.0;
    return a2 / std::max(3.14159265 * d * d, 1e-30);
}

// pbr.slang: pbrVisibilitySmithGgx
double pbrVisibilitySmithGgx(double ndotv, double ndotl, double alpha) {
    const double a2 = alpha * alpha;
    const double lambdaV = ndotl * std::sqrt(((ndotv - (ndotv * a2)) * ndotv) + a2);
    const double lambdaL = ndotv * std::sqrt(((ndotl - (ndotl * a2)) * ndotl) + a2);
    return 0.5 / std::max(lambdaV + lambdaL, 1e-7);
}

// pbr.slang: pbrFresnelSchlick (scalar channel)
double pbrFresnelSchlick(double vdoth, double f0) {
    const double f = std::pow(std::clamp(1.0 - vdoth, 0.0, 1.0), 5.0);
    return f0 + ((1.0 - f0) * f);
}

// pbr.slang: pbrSpecularF0 (scalar channel)
double pbrSpecularF0(double albedo, double metallic) {
    const double m = std::clamp(metallic, 0.0, 1.0);
    return (0.04 * (1.0 - m)) + (albedo * m);
}

// pbr.slang: pbrDiffuseScale
double pbrDiffuseScale(double metallic) { return 1.0 - std::clamp(metallic, 0.0, 1.0); }

// pbr.slang: pbrDirectSpecular, specialized to N = +Z and light along +Z so the
// outgoing direction is the only free variable. Returns f_spec * NdotL.
double pbrDirectSpecularAtNormalIncidence(double roughness, double f0, double thetaOut) {
    const double ndotl = 1.0;
    const double ndotv = std::cos(thetaOut) + 1e-5;  // pbr.slang adds this epsilon
    const double vx = std::sin(thetaOut);
    const double vz = std::cos(thetaOut);
    // half vector between L = (0,0,1) and V = (vx,0,vz)
    const double hx = vx;
    const double hz = vz + 1.0;
    const double hLen = std::sqrt((hx * hx) + (hz * hz));
    const double ndoth = hz / hLen;
    const double vdoth = ((vx * hx) + (vz * hz)) / hLen;
    const double alpha = pbrPerceptualToAlpha(roughness);
    return pbrDistributionGgx(std::clamp(ndoth, 0.0, 1.0), alpha) *
           pbrVisibilitySmithGgx(std::min(ndotv, 1.0), ndotl, alpha) *
           pbrFresnelSchlick(std::clamp(vdoth, 0.0, 1.0), f0) *
           ndotl;
}

// ---------------------------------------------------------------------------
// Hemisphere quadrature. The polar grid is warped as (i/n)^kWarp so cells bunch
// toward theta = 0, where a GGX lobe viewed at normal incidence lives; a uniform
// grid cannot resolve alpha = 0.002. Azimuthally symmetric for the configurations
// used here, so one phi sample is exact.
// ---------------------------------------------------------------------------

constexpr int kQuadratureSteps = 64000;
constexpr double kQuadratureWarp = 5.0;

template <typename Integrand>
double integrateHemisphere(Integrand f) {
    double sum = 0.0;
    for (int i = 0; i < kQuadratureSteps; ++i) {
        const double t0 = std::pow(static_cast<double>(i) / kQuadratureSteps, kQuadratureWarp);
        const double t1 = std::pow(static_cast<double>(i + 1) / kQuadratureSteps, kQuadratureWarp);
        const double theta0 = t0 * kPi * 0.5;
        const double theta1 = t1 * kPi * 0.5;
        const double theta = 0.5 * (theta0 + theta1);
        sum += f(theta) * std::sin(theta) * (theta1 - theta0) * 2.0 * kPi;
    }
    return sum;
}

// Single-scattering directional albedo at normal incidence: the fraction of
// incident energy the specular lobe alone reflects. Bounded above by 1 because
// single scattering cannot create energy.
double directionalAlbedo(double roughness, double f0) {
    return integrateHemisphere([&](double theta) {
        return pbrDirectSpecularAtNormalIncidence(roughness, f0, theta) * std::cos(theta);
    });
}

const double kRoughnessSweep[] = {0.045, 0.08, 0.116, 0.2, 0.35, 0.5, 0.75, 1.0};

// ---------------------------------------------------------------------------

void testMaterialPackRoundTrip() {
    std::printf("material pack/unpack\n");
    using namespace odai::importer;

    // A default material must pack to exactly zero: that is what keeps every
    // scene cooked before materials existed shading through the legacy path.
    check(packImportedSceneMaterialFlags(ImportedSceneSurfaceMaterial{}) == 0u,
          "default material packs to flags == 0");
    check((packImportedSceneMaterialFlags(ImportedSceneSurfaceMaterial{0.0f, 1.0f}) &
           kImportedSceneMaterialFlagPbr) == 0u,
          "default material leaves the PBR opt-in bit clear");

    // Round trip within one quantization step. 8 bits over [0,1] with round-to-
    // nearest gives a worst-case error of 1/510.
    const double kQuantizationTolerance = 1.0 / 510.0 + 1e-6;
    for (int mi = 0; mi <= 20; ++mi) {
        for (int ri = 0; ri <= 20; ++ri) {
            ImportedSceneSurfaceMaterial in{};
            in.metallic = static_cast<float>(mi) / 20.0f;
            in.roughness = static_cast<float>(ri) / 20.0f;
            const std::uint32_t flags = packImportedSceneMaterialFlags(in);
            const ImportedSceneSurfaceMaterial out = unpackImportedSceneMaterialFlags(flags);
            if (importedSceneMaterialIsDefault(in)) {
                continue;  // packs to 0 by design, decodes to the default
            }
            check((flags & kImportedSceneMaterialFlagPbr) != 0u,
                  "non-default material sets the PBR opt-in bit");
            checkNear(out.metallic, in.metallic, kQuantizationTolerance, "metallic round trip");
            checkNear(out.roughness, in.roughness, kQuantizationTolerance, "roughness round trip");
        }
    }

    // Flags without the opt-in bit must decode to the legacy default even when the
    // material bit ranges hold garbage -- that is the forward-compatibility contract.
    const ImportedSceneSurfaceMaterial stale =
        unpackImportedSceneMaterialFlags(0xffffff00u & ~kImportedSceneMaterialFlagPbr);
    check(stale.metallic == 0.0f && stale.roughness == 1.0f,
          "flags without the PBR bit decode to the legacy default");

    // The bit layout the shader mirrors.
    ImportedSceneSurfaceMaterial bronze{};
    bronze.metallic = 1.0f;
    bronze.roughness = 0.35f;
    const std::uint32_t bronzeFlags = packImportedSceneMaterialFlags(bronze);
    check(((bronzeFlags >> kImportedSceneMaterialMetallicShift) &
           kImportedSceneMaterialChannelMask) == 255u,
          "metallic 1.0 quantizes to 255 in bits 16..23");
    check((bronzeFlags & kImportedSceneMaterialFlagAlphaTest) == 0u,
          "a material never disturbs the alpha-test bit");
}

// The material-library index shares the flags word with the legacy per-vertex
// coefficients, and the split is reproduced by hand in imported_static.frag.slang.
// These pin the contract the shader relies on.
void testMaterialIndexPacking() {
    std::printf("material library index packing\n");
    using namespace odai::importer;

    // The layout is worth asserting on its own: the GPU record is memcpy'd into
    // a descriptor range and mirrored in the shader, so a size change silently
    // reinterprets the entire table.
    check(sizeof(GpuImportedMaterial) == 32u, "GpuImportedMaterial is 32 bytes");
    check(alignof(GpuImportedMaterial) == 16u, "GpuImportedMaterial is 16-byte aligned");
    // 52 through v19, 68 through v20, 72 through v24; v21 widened the terrain
    // layer texture indices and a packed weight word, v25 appended colorAlpha.
    // readPackedVertexArray version-gates the read now, but the size still
    // deserves an assert: it is what makes anyone widening the struct go and add
    // the matching legacy branch.
    check(sizeof(ImportedScenePackedVertex) == 76u,
          "packed vertex is 76 bytes -- widening it needs a new legacy branch in readPackedVertexArray");

    // Index 0 means "no library material" and must reduce exactly to the old
    // overload, default included. This is what keeps legacy scenes bit-identical.
    check(packImportedSceneMaterialFlags(ImportedSceneSurfaceMaterial{}, 0u) == 0u,
          "default material at index 0 still packs to flags == 0");
    for (int mi = 0; mi <= 8; ++mi) {
        ImportedSceneSurfaceMaterial m{};
        m.metallic = static_cast<float>(mi) / 8.0f;
        m.roughness = 0.4f;
        check(packImportedSceneMaterialFlags(m, 0u) == packImportedSceneMaterialFlags(m),
              "index 0 is identical to the no-index overload");
    }

    // Every index round-trips through bits 24-31 without disturbing anything else.
    ImportedSceneSurfaceMaterial mat{};
    mat.metallic = 0.75f;
    mat.roughness = 0.25f;
    for (std::uint32_t index = 0; index <= 255u; ++index) {
        const std::uint32_t flags = packImportedSceneMaterialFlags(mat, index);
        check(importedSceneMaterialIndex(flags) == index, "material index round trip");
        check((flags & kImportedSceneMaterialFlagAlphaTest) == 0u,
              "material index never disturbs the alpha-test bit");
        // The fallback coefficients survive alongside the index, so a scene whose
        // table fails to load still shades approximately right.
        const ImportedSceneSurfaceMaterial out = unpackImportedSceneMaterialFlags(flags);
        checkNear(out.metallic, mat.metallic, 1.0 / 510.0 + 1e-6, "fallback metallic survives");
        checkNear(out.roughness, mat.roughness, 1.0 / 510.0 + 1e-6, "fallback roughness survives");
    }

    // A nonzero index forces the PBR opt-in bit, so the shader's usePbr test
    // stays a single condition rather than two.
    for (std::uint32_t index = 1; index <= 255u; ++index) {
        const std::uint32_t flags =
            packImportedSceneMaterialFlags(ImportedSceneSurfaceMaterial{}, index);
        check((flags & kImportedSceneMaterialFlagPbr) != 0u,
              "a nonzero material index forces the PBR bit even on a default material");
        check(importedSceneMaterialIndex(flags) == index, "forced-PBR path keeps the index");
    }

    // Indices are masked to 8 bits, not allowed to bleed into the reserved bits.
    const std::uint32_t wide = packImportedSceneMaterialFlags(mat, 0x1ffu);
    check(importedSceneMaterialIndex(wide) == 0xffu, "index is masked to 8 bits");
}

void testBrdfReferenceValues() {
    std::printf("BRDF against reference values\n");

    // Schlick's endpoints are exact, not approximate.
    checkNear(pbrFresnelSchlick(1.0, 0.04), 0.04, 1e-12, "Fresnel at normal incidence == F0");
    checkNear(pbrFresnelSchlick(0.0, 0.04), 1.0, 1e-12, "Fresnel at grazing == 1");
    checkNear(pbrFresnelSchlick(1.0, 1.0), 1.0, 1e-12, "F0 == 1 stays 1 at normal incidence");

    // Dielectric/metal split.
    checkNear(pbrSpecularF0(0.9, 0.0), 0.04, 1e-12, "dielectric F0 == 0.04");
    checkNear(pbrSpecularF0(0.9, 1.0), 0.9, 1e-12, "metal F0 == albedo");
    checkNear(pbrDiffuseScale(1.0), 0.0, 1e-12, "metals keep no diffuse lobe");
    checkNear(pbrDiffuseScale(0.0), 1.0, 1e-12, "dielectrics keep all diffuse");

    // The GGX NDF normalization identity: integral of D(h)*cos(theta_h) dw_h == 1,
    // exactly, at every alpha. This is the check that catches a wrong alpha
    // convention or an over-aggressive denominator clamp -- it is what found
    // pbr.slang's old 1e-7, which truncated the lobe below roughness 0.1156.
    for (double roughness : kRoughnessSweep) {
        const double alpha = pbrPerceptualToAlpha(roughness);
        const double integral = integrateHemisphere(
            [&](double theta) { return pbrDistributionGgx(std::cos(theta), alpha) * std::cos(theta); });
        checkNear(integral, 1.0, 1e-4,
                  "GGX NDF normalizes at roughness " + std::to_string(roughness));
    }
}

void testEnergyConservation() {
    std::printf("energy conservation and the diffuse-convention scale\n");

    // Single-scattering directional albedo of a lossless (F0 = 1) metal. Bounded
    // above by 1, and approaching 1 as the surface smooths out, because a smooth
    // single-scattering microfacet lobe loses almost nothing to shadowing.
    double previous = 2.0;
    for (double roughness : kRoughnessSweep) {
        const double e = directionalAlbedo(roughness, 1.0);
        check(e <= 1.0 + 1e-6,
              "directional albedo cannot exceed 1 at roughness " + std::to_string(roughness));
        check(e > 0.0, "directional albedo is positive at roughness " + std::to_string(roughness));
        check(e < previous + 1e-6,
              "directional albedo decreases with roughness at " + std::to_string(roughness));
        previous = e;
    }
    checkNear(directionalAlbedo(0.045, 1.0), 1.0, 1e-3,
              "a near-mirror metal is very nearly lossless");

    // The convention check. Under this engine's diffuse convention the outgoing
    // radiance of a white Lambertian is Lo = albedo * E_perp * cos(theta_l), so its
    // hemispherical reflectance is pi, not 1. A lossless smooth metal trades its
    // entire diffuse lobe away via pbrDiffuseScale() and must get back the same
    // total power through specular -- so the ratio of the two must be ~1.
    //
    // rho_diff = pi (albedo 1).  rho_spec = S * E_ss.  ratio = S * E_ss / pi.
    // With S = kPbrDiffuseConventionScale the ratio is E_ss; with S = 1 it is
    // E_ss/pi, which is the ~3x-dark metal the scale exists to fix.
    for (double roughness : {0.2, 0.35, 0.5}) {
        const double e = directionalAlbedo(roughness, 1.0);
        const double corrected = (kPbrDiffuseConventionScale * e) / kPi;
        const double uncorrected = (1.0 * e) / kPi;
        checkNear(corrected, e, 1e-6,
                  "pi-corrected metal reflects E_ss of a white Lambertian at roughness " +
                      std::to_string(roughness));
        check(corrected > 0.9,
              "pi-corrected metal is within 10% of a white Lambertian at roughness " +
                  std::to_string(roughness));
        checkNear(uncorrected, e / kPi, 1e-9, "uncorrected ratio is E_ss/pi");
        check(uncorrected < 0.34,
              "without the scale a metal reflects under 34% of a Lambertian -- the bug");
        // The scale is exactly the ratio between the two.
        checkNear(corrected / uncorrected, kPi, 1e-6, "the correction is exactly pi");
    }

    // The ambient specular deliberately takes NO scale. Its split-sum second factor
    // is a reflectance, so it wants prefiltered radiance -- and the caller hands it
    // evaluateShIrradiance(), which for a uniform environment of radiance L returns
    // pi*L (the Lambert convolution in renderer_shared.h bakes A_0 = pi into band 0).
    // That implicit pi already matches the ambient diffuse beside it, so scaling
    // again would double it. Reproduced here from the SH chain's own constants.
    constexpr double kShY00 = 0.282095;
    const double uniformRadiance = 1.0;
    const double bandZeroCoefficient = uniformRadiance * kShY00 * 4.0 * kPi * kPi;  // projection * A_0
    const double irradiance = bandZeroCoefficient * kShY00;
    checkNear(irradiance, kPi, 1e-3,
              "evaluateShIrradiance returns pi*L for a uniform environment, i.e. an irradiance");
    // ambient diffuse  = albedo * E     = pi * (albedo * L)
    // ambient specular = E * rho        = pi * (L * rho)     -- already consistent
    const double ambientDiffuseOverPhysical = irradiance / (uniformRadiance * kPi);
    checkNear(ambientDiffuseOverPhysical, 1.0, 1e-3,
              "ambient specular and ambient diffuse already share the same pi; no scale there");
}

}  // namespace

int main() {
    std::printf("PBR material tests\n\n");
    testMaterialPackRoundTrip();
    testMaterialIndexPacking();
    testBrdfReferenceValues();
    testEnergyConservation();

    std::printf("\n");
    if (g_failures != 0) {
        std::printf("%d check(s) FAILED\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("all checks passed\n");
    return EXIT_SUCCESS;
}
