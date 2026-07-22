#pragma once

#include <cstdint>
#include <utility>
#include <vector>

// Seeded terrain generation for the city builder: a carved river (edge-to-edge
// water connectivity by construction), optional lakes and a coastline, a
// low-frequency forest-density mask, and a scored city-site anchor. Pure CPU,
// engine-agnostic, deterministic per seed.
namespace odai::procgen {

// Tuning knobs; the game overlays values from the Lua Config.terrain table.
struct CityTerrainParams {
    float landMin = 0.55f;      // reject maps with less buildable land than this
    int riverWidthMin = 2;      // carved river half-width varies in this range
    int riverWidthMax = 3;
    int lakeMax = 2;            // 0..N seeded lakes
    float coastChance = 0.25f;  // chance one map edge floods into a coastline
    float forestFreq = 0.09f;   // fbm frequency of the tree-density mask
};

struct CityTerrainDesc {
    int width = 56;
    int height = 56;
    std::uint32_t seed = 0;
    CityTerrainParams params;
};

struct CityTerrain {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> water;                 // 1 = water, row-major
    std::vector<float> forest;                       // 0..1 tree density, row-major
    std::vector<std::pair<short, short>> riverPath;  // ordered centerline (c, r)
    short siteC = 0;                                 // recommended city-centre anchor
    short siteR = 0;
    bool valid = false;                              // playability invariants held
};

// Generate a map. Internally retries with perturbed seeds (up to 8 attempts)
// until the playability invariants hold: enough land, one large buildable
// component, a viable city-site window, and river connectivity across the map.
// If every attempt fails, returns the best-scoring attempt with valid=false —
// the game still runs, just on an awkward map.
CityTerrain generateCityTerrain(const CityTerrainDesc& desc);

}  // namespace odai::procgen
