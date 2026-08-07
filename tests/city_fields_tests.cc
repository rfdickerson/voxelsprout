// Tests for the citybuilder spatial-field math (citybuilder_fields.h): the
// building influence/service tables, the data-layer descriptor table, the disc
// splat every field is built from, and the population-weighted mean the
// citywide Education/Health/Safety stats are averages of. Headless — this file
// links nothing but citybuilder_fields.cc, which is the reason the math lives
// outside citybuilder_app.cc. Lightweight harness style, matching the other
// test executables.

#include "games/citybuilder/citybuilder_fields.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using odai::games::citybuilder::Building;
using odai::games::citybuilder::DataLayer;
using odai::games::citybuilder::DataLayerDesc;
using odai::games::citybuilder::InfluenceSpec;
using odai::games::citybuilder::Service;
using odai::games::citybuilder::buildingInfluence;
using odai::games::citybuilder::buildingService;
using odai::games::citybuilder::dataLayerClass;
using odai::games::citybuilder::dataLayerDesc;
using odai::games::citybuilder::kDataLayerClasses;
using odai::games::citybuilder::kFireProtectRadius;
using odai::games::citybuilder::nextDataLayer;
using odai::games::citybuilder::populationWeightedMean;
using odai::games::citybuilder::splatDisc;

int g_failures = 0;

void expectTrue(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[city fields test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectNear(float got, float want, float tol, const std::string& message) {
    if (std::fabs(got - want) > tol) {
        std::cerr << "[city fields test] FAIL: " << message << " (got " << got << ", want " << want
                  << ")\n";
        ++g_failures;
    }
}

// CIE relative luminance of a packed 0xRRGGBB colour (sRGB decode + Rec.709
// weights). Monotonic in perceived lightness, which is all the ramp audit below
// needs, and avoids pulling ui/ into a headless test.
float relativeLuminance(std::uint32_t rgb) {
    const auto channel = [](std::uint32_t byte) {
        const float c = static_cast<float>(byte) / 255.0f;
        return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
    };
    return 0.2126f * channel((rgb >> 16) & 0xFFu) + 0.7152f * channel((rgb >> 8) & 0xFFu) +
           0.0722f * channel(rgb & 0xFFu);
}

// Every building that splats influence must do so with a positive radius, and
// the only nuisance source is the power plant. Nothing else in the sim is
// allowed to quietly become a bad neighbour without this test noticing.
void testInfluenceTable() {
    expectTrue(buildingInfluence(Building::None).radius == 0, "None has no influence");
    expectTrue(buildingInfluence(Building::Power).nuisance, "power plant is a nuisance");
    expectTrue(buildingInfluence(Building::Fire).radius == kFireProtectRadius,
               "fire amenity ring is the fire protection radius");

    static constexpr Building kAll[] = {Building::Police, Building::Fire,   Building::Clinic,
                                        Building::School, Building::Park,   Building::Library,
                                        Building::Amphitheater, Building::Power};
    for (const Building b : kAll) {
        const InfluenceSpec inf = buildingInfluence(b);
        expectTrue(inf.radius > 0, "every placeable building has a reach");
        expectTrue(inf.peak > 0.0f, "every placeable building has a peak");
        if (b != Building::Power) {
            expectTrue(!inf.nuisance, "only the power plant is a nuisance");
        }
    }
}

// Service mapping: schools and libraries both educate, police and fire both
// make people safe, and anything with no service maps to Count (the "feeds no
// service" sentinel computeFields skips on).
void testServiceTable() {
    expectTrue(buildingService(Building::School) == Service::Education, "school educates");
    expectTrue(buildingService(Building::Library) == Service::Education, "library educates");
    expectTrue(buildingService(Building::Clinic) == Service::Health, "clinic heals");
    expectTrue(buildingService(Building::Police) == Service::Safety, "police is safety");
    expectTrue(buildingService(Building::Fire) == Service::Safety, "fire is safety");
    expectTrue(buildingService(Building::Park) == Service::Count, "park feeds no service");
    expectTrue(buildingService(Building::Power) == Service::Count, "power feeds no service");
    expectTrue(buildingService(Building::None) == Service::Count, "empty tile feeds no service");

    // Anything that covers a service must have a ring to cover it with.
    static constexpr Building kServiced[] = {Building::School, Building::Library, Building::Clinic,
                                             Building::Police, Building::Fire};
    for (const Building b : kServiced) {
        expectTrue(buildingInfluence(b).radius > 0, "serviced building has a coverage radius");
    }
}

// The descriptor table is indexed by the enum, so an entry added out of order
// (or missed) has to show up as a mismatched name here.
void testDataLayerTable() {
    expectTrue(std::string(dataLayerDesc(DataLayer::LandValue).name) == "Land Value",
               "land value descriptor");
    expectTrue(std::string(dataLayerDesc(DataLayer::Pollution).name) == "Pollution",
               "pollution descriptor");
    expectTrue(std::string(dataLayerDesc(DataLayer::Safety).name) == "Safety", "safety descriptor");
    for (int i = 1; i < static_cast<int>(DataLayer::Count); ++i) {
        const DataLayerDesc& d = dataLayerDesc(static_cast<DataLayer>(i));
        expectTrue(d.name != nullptr && *d.name != '\0', "layer has a name");
        expectTrue(*d.lowLabel != '\0' && *d.highLabel != '\0', "layer legend is labelled");
    }

    // The legibility contract on every ramp: lightness rises with the value, in
    // steps big enough to tell apart. That is what makes the wash readable in
    // greyscale and under red/green colour-vision deficiency, where hue is gone
    // but luminance survives — so it is pinned here, not left to eyeballing.
    for (int i = 1; i < static_cast<int>(DataLayer::Count); ++i) {
        const DataLayerDesc& d = dataLayerDesc(static_cast<DataLayer>(i));
        float prev = -1.0f;
        for (int k = 0; k < kDataLayerClasses; ++k) {
            const float lum = relativeLuminance(d.ramp[k]);
            expectTrue(lum > prev, std::string(d.name) + " ramp lightness is monotonic");
            // ~0.03 relative luminance is far above the just-noticeable
            // difference for a large flat field; the ramps clear ~11 CIE L*.
            if (k > 0) {
                expectTrue(lum - prev > 0.03f,
                           std::string(d.name) + " ramp steps are distinguishable");
            }
            prev = lum;
        }
    }

    // Class bucketing: both ends land inside the ramp, and the wash reaches the
    // top class (a ramp whose brightest swatch never appears is a broken key).
    expectTrue(dataLayerClass(0.0f) == 0, "zero maps to the lowest class");
    expectTrue(dataLayerClass(1.0f) == kDataLayerClasses - 1, "one maps to the top class");
    expectTrue(dataLayerClass(1.7f) == kDataLayerClasses - 1, "over-unity clamps to the top class");
    expectTrue(dataLayerClass(-0.2f) == 0, "negative clamps to the lowest class");

    // L cycles forward and wraps back to Off rather than sticking on the last.
    DataLayer l = DataLayer::None;
    for (int i = 1; i < static_cast<int>(DataLayer::Count); ++i) {
        l = nextDataLayer(l);
        expectTrue(l == static_cast<DataLayer>(i), "cycle visits every layer in order");
    }
    expectTrue(nextDataLayer(l) == DataLayer::None, "cycle wraps back to off");
}

// Splat shape: peak at the centre, linear falloff, exactly zero at and beyond
// the radius, and no square corners (it's a disc).
void testSplatShape() {
    constexpr int kW = 11, kH = 11;
    std::vector<float> f(kW * kH, 0.0f);
    splatDisc(f, kW, kH, 5, 5, 4, 1.0f);

    const auto at = [&](int c, int r) { return f[static_cast<std::size_t>(r) * kW + c]; };
    expectNear(at(5, 5), 1.0f, 1e-5f, "peak at the source tile");
    expectNear(at(7, 5), 0.5f, 1e-5f, "half strength at half the radius");
    expectNear(at(9, 5), 0.0f, 1e-5f, "zero at the radius");
    expectNear(at(10, 5), 0.0f, 1e-5f, "nothing beyond the radius");
    // (3,3) is 2.83 tiles away — inside a radius-4 square but comfortably
    // inside the disc too; (2,2) at 4.24 is inside the square and outside the
    // disc, which is the case a naive box splat gets wrong.
    expectTrue(at(3, 3) > 0.0f, "diagonal inside the disc is covered");
    expectNear(at(2, 2), 0.0f, 1e-5f, "square corner outside the disc is not covered");

    // Radially symmetric.
    expectNear(at(3, 5), at(7, 5), 1e-5f, "symmetric in x");
    expectNear(at(5, 3), at(5, 7), 1e-5f, "symmetric in y");
}

// Splatting near an edge must clip, not wrap onto the far side of the grid or
// run off the end of the span.
void testSplatClipsToGrid() {
    constexpr int kW = 8, kH = 8;
    std::vector<float> f(kW * kH, 0.0f);
    splatDisc(f, kW, kH, 0, 0, 3, 1.0f);

    const auto at = [&](int c, int r) { return f[static_cast<std::size_t>(r) * kW + c]; };
    expectNear(at(0, 0), 1.0f, 1e-5f, "corner source still peaks");
    expectNear(at(7, 0), 0.0f, 1e-5f, "no wrap onto the opposite edge");
    expectNear(at(0, 7), 0.0f, 1e-5f, "no wrap onto the bottom row");
    expectTrue(at(1, 0) > 0.0f && at(0, 1) > 0.0f, "in-bounds neighbours still covered");

    // A zero (or negative) radius is a no-op, not a divide by zero — every
    // building with no influence takes that path.
    std::vector<float> z(kW * kH, 0.0f);
    splatDisc(z, kW, kH, 4, 4, 0, 1.0f);
    float sum = 0.0f;
    for (const float v : z) sum += v;
    expectNear(sum, 0.0f, 1e-6f, "radius 0 splats nothing");
}

// Overlapping sources accumulate — computeFields relies on this (and clamps
// afterward), so it must not be a max or an overwrite.
void testSplatAccumulates() {
    constexpr int kW = 9, kH = 9;
    std::vector<float> f(kW * kH, 0.0f);
    splatDisc(f, kW, kH, 4, 4, 3, 1.0f);
    splatDisc(f, kW, kH, 4, 4, 3, 1.0f);
    expectNear(f[static_cast<std::size_t>(4) * kW + 4], 2.0f, 1e-5f,
               "two sources on one tile add up");
}

// The stat that makes coverage mean something: a service is scored by what the
// population actually gets, not by the raw area covered.
void testPopulationWeightedMean() {
    const std::vector<float> field  = {1.0f, 0.0f, 0.0f, 1.0f};
    const std::vector<float> nobody = {0.0f, 0.0f, 0.0f, 0.0f};
    expectNear(populationWeightedMean(field, nobody), 0.0f, 1e-6f,
               "no population means no average to take");

    // Everyone lives on the covered tiles: full marks, even though half the
    // map is uncovered.
    const std::vector<float> onCovered = {10.0f, 0.0f, 0.0f, 10.0f};
    expectNear(populationWeightedMean(field, onCovered), 1.0f, 1e-6f,
               "coverage where the people are scores 1");

    // Everyone lives on the uncovered tiles: zero, however much land is served.
    const std::vector<float> offCovered = {0.0f, 10.0f, 10.0f, 0.0f};
    expectNear(populationWeightedMean(field, offCovered), 0.0f, 1e-6f,
               "coverage away from the people scores 0");

    // A big district beats a small one: 30 people at 1.0 and 10 at 0.0 -> 0.75.
    const std::vector<float> lopsided = {30.0f, 10.0f, 0.0f, 0.0f};
    expectNear(populationWeightedMean(field, lopsided), 0.75f, 1e-6f,
               "denser district carries the average");

    // Partial coverage averages, it doesn't threshold.
    const std::vector<float> half = {0.5f, 0.5f, 0.5f, 0.5f};
    const std::vector<float> even = {1.0f, 1.0f, 1.0f, 1.0f};
    expectNear(populationWeightedMean(half, even), 0.5f, 1e-6f, "partial coverage averages");
}

}  // namespace

int main() {
    testInfluenceTable();
    testServiceTable();
    testDataLayerTable();
    testSplatShape();
    testSplatClipsToGrid();
    testSplatAccumulates();
    testPopulationWeightedMean();

    if (g_failures != 0) {
        std::cerr << "[city fields test] " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "[city fields test] all checks passed\n";
    return 0;
}
