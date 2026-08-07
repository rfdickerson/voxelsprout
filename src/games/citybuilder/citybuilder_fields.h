#pragma once

#include <cstdint>
#include <span>

// The city's spatial fields: what a tile can be, how far each municipal building
// reaches, and the two pieces of math that turn those two facts into per-tile
// numbers.
//
// Deliberately decoupled from Tile and GameApp — everything here works on flat
// spans, so it compiles and tests headlessly (odai_city_fields_tests), the same
// split citybuilder_citizens.h makes for the roster sim.
namespace odai::games::citybuilder {

enum class Terrain : std::uint8_t { Grass, Water };
enum class Zone : std::uint8_t { None, Residential, Commercial, Industrial };
enum class Building : std::uint8_t {
    None, Police, Fire, Clinic, School, Park, Library, Amphitheater, Power
};

// Fire-dept reach. Named because two callers share it: the amenity splat below
// and the fire-suppression coverage in stepFire().
inline constexpr int kFireProtectRadius = 6;

// Spatial influence each municipal building splats into the land-value fields
// (CityBuilderApp::computeFields) — shared with the placement-time ring preview
// in drawWorldOverlay so what the player is shown is exactly what the sim uses.
// radius 0 = no influence.
struct InfluenceSpec {
    int   radius = 0;
    float peak = 0.0f;
    bool  nuisance = false;
};

constexpr InfluenceSpec buildingInfluence(Building b) {
    switch (b) {
        case Building::Park:         return {5, 0.55f, false};
        case Building::Police:       return {6, 0.28f, false};
        case Building::Fire:         return {kFireProtectRadius, 0.24f, false};
        case Building::Clinic:       return {6, 0.24f, false};
        case Building::School:       return {6, 0.30f, false};
        case Building::Library:      return {5, 0.26f, false};
        case Building::Amphitheater: return {7, 0.50f, false};  // culture draw
        case Building::Power:        return {5, 0.45f, true};   // dirty neighbour
        default:                     return {};
    }
}

// The three services whose reach the citywide stats are derived from. A building
// covers the same radius it splats amenity over — one table, so the ring the
// placement preview draws is the ring the coverage sim actually uses. What
// differs is the peak: amenity peaks are land-value weights, coverage peaks are
// always 1.0 (fully served at the door, nothing at the radius).
enum class Service : int { Education, Health, Safety, Count };

constexpr Service buildingService(Building b) {
    switch (b) {
        case Building::School:
        case Building::Library: return Service::Education;
        case Building::Clinic:  return Service::Health;
        case Building::Police:
        case Building::Fire:    return Service::Safety;
        default:                return Service::Count;  // feeds no service
    }
}

// Data layers the player can wash the map with. LandValue is the original
// overlay; the rest surface fields the sim already computes but never showed.
enum class DataLayer : int {
    None, LandValue, Pollution, Education, Health, Safety, Traffic, Count
};

// Choropleth classes per data layer. Five is the cartographic default for a
// sequential scale: enough steps to rank a neighbourhood, few enough that every
// step is unmistakable and the legend can show one swatch per class. A
// continuous gradient over a tile grid is worse than useless here — 0.62 and
// 0.71 look identical — whereas five classes turn a service's linear falloff
// into visible contour rings, so the player can *see* where coverage stops.
inline constexpr int kDataLayerClasses = 5;

struct DataLayerDesc {
    const char* name;
    const char* lowLabel;
    const char* highLabel;
    // Sequential ramp, packed 0xRRGGBB, class 0 (lowest) → class 4 (highest).
    //
    // Every ramp is monotonic in CIE L* with roughly even ~17 L* steps, and
    // stays monotonic when simulated through deuteranopia and protanopia — so
    // the ordering survives red/green colour-vision deficiency and greyscale
    // screenshots, because lightness (not hue) carries the value. Hue only
    // carries layer *identity*: one hue family per layer, matched to that
    // service's building colour, so the washed map says which layer is up
    // without reading the legend.
    //
    // Direction is fixed and never inverted: lighter always means more of the
    // field named on the chip. Valence lives in the hue (acid smog vs. clean
    // teal), not in a flipped scale — a scale that reverses per layer forces
    // the player to re-learn the map every time they press L.
    std::uint32_t ramp[kDataLayerClasses];
};

// Table-driven so the legend, the tile tint and the toolbar chips all read one
// source, and adding a layer is one entry rather than four edits.
const DataLayerDesc& dataLayerDesc(DataLayer layer);

// Class index for a normalized (0..1) field value. Clamps, so an over-unity
// accumulated field lands in the top class rather than off the end of the ramp.
constexpr int dataLayerClass(float v) {
    const int i = static_cast<int>(v * static_cast<float>(kDataLayerClasses));
    return i < 0 ? 0 : (i >= kDataLayerClasses ? kDataLayerClasses - 1 : i);
}

// Cycle order for the L hotkey, wrapping back to None off the end.
constexpr DataLayer nextDataLayer(DataLayer layer) {
    const int n = static_cast<int>(layer) + 1;
    return n >= static_cast<int>(DataLayer::Count) ? DataLayer::None : static_cast<DataLayer>(n);
}

// Adds `peak` at (c, r) falling linearly to zero at `radius`, clipped to the
// grid. Accumulates — overlapping sources add up.
void splatDisc(std::span<float> field, int width, int height, int c, int r,
               int radius, float peak);

// Mean of `field` weighted by `weight`, both indexed r * width + c. Returns 0
// when the total weight is zero: an empty city has no coverage to average, and
// callers keep their own starting values rather than reading that as 0%.
float populationWeightedMean(std::span<const float> field, std::span<const float> weight);

}  // namespace odai::games::citybuilder
