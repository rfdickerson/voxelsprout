#include "game/game_sim.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace odai::game {

namespace {

// --- tunables specific to the driver (rules tunables live in economy.h) -----
constexpr int kSettlerCost = 72;        // production for a settler
constexpr int kCapitalStartPop = 2;     // founding population of a capital
constexpr int kRushGoldPerShield = 4;   // gold to rush-buy one production point
constexpr int kCityWorkRadius = 2;      // hex radius a city works tiles within
constexpr int kSettleMinSpacing = 3;    // min hex distance between any two cities
constexpr int kSciencePerPopDiv = 3;    // science = base + population / this
constexpr int kMaxGrowthBonus = 80;     // cap on stacked food-box carryover %
constexpr int kCivicUpkeepPerCity = 2;  // gold/turn per city past the first
constexpr int kStarveBuffer = 8;        // food debt a city absorbs before losing pop

// --- map generation noise (compact port of the strategy_map_gen tool) -------
std::uint32_t hashCoords(std::int32_t x, std::int32_t y, std::uint32_t seed) {
    std::uint32_t h = seed + 0x9E3779B9u;
    h ^= static_cast<std::uint32_t>(x) * 0x85EBCA77u;
    h = (h ^ (h >> 15)) * 0xC2B2AE3Du;
    h ^= static_cast<std::uint32_t>(y) * 0x27D4EB2Fu;
    h = (h ^ (h >> 13)) * 0x165667B1u;
    return h ^ (h >> 16);
}
float hashFloat(std::int32_t x, std::int32_t y, std::uint32_t seed) {
    return static_cast<float>(hashCoords(x, y, seed) & 0xFFFFFFu) / static_cast<float>(0x1000000u);
}
float smoothstep(float t) { return t * t * (3.0f - (2.0f * t)); }
float valueNoise(float x, float y, std::uint32_t seed) {
    const float fx = std::floor(x);
    const float fy = std::floor(y);
    const auto ix = static_cast<std::int32_t>(fx);
    const auto iy = static_cast<std::int32_t>(fy);
    const float tx = smoothstep(x - fx);
    const float ty = smoothstep(y - fy);
    const float v00 = hashFloat(ix, iy, seed);
    const float v10 = hashFloat(ix + 1, iy, seed);
    const float v01 = hashFloat(ix, iy + 1, seed);
    const float v11 = hashFloat(ix + 1, iy + 1, seed);
    const float top = v00 + ((v10 - v00) * tx);
    const float bottom = v01 + ((v11 - v01) * tx);
    return top + ((bottom - top) * ty);
}
float fbm(float x, float y, std::uint32_t seed) {
    float sum = 0.0f, amp = 0.5f, freq = 1.0f;
    for (int o = 0; o < 4; ++o) {
        sum += valueNoise(x * freq, y * freq, seed + static_cast<std::uint32_t>(o) * 101u) * amp;
        amp *= 0.5f;
        freq *= 2.0f;
    }
    return sum;
}
TerrainType classifyTerrain(std::int16_t elevation, float latitude01, float moisture) {
    if (elevation <= 0) return TerrainType::Ocean;
    if (elevation == 1) return TerrainType::Coast;
    const float polar = std::abs((latitude01 * 2.0f) - 1.0f);
    if (elevation >= 6) return (polar > 0.45f || elevation >= 7) ? TerrainType::Snow : TerrainType::Mountains;
    if (elevation >= 5) return TerrainType::Mountains;
    if (elevation >= 4) return TerrainType::Hills;
    if (polar > 0.80f) return TerrainType::Snow;
    if (polar > 0.64f) return TerrainType::Tundra;
    if (polar < 0.18f && moisture < 0.42f) return TerrainType::Desert;
    if (polar < 0.28f && moisture > 0.58f) return TerrainType::Jungle;
    if (moisture > 0.62f) return TerrainType::Forest;
    return (moisture > 0.45f) ? TerrainType::Grassland : TerrainType::Plains;
}

float focusScore(CityFocus f, const Yields& y) {
    const float fd = static_cast<float>(y.food);
    const float pr = static_cast<float>(y.production);
    const float gd = static_cast<float>(y.gold);
    const float sc = static_cast<float>(y.science);
    switch (f) {
        case CityFocus::Food:       return 2.2f * fd + 0.5f * pr + 0.3f * gd + 0.2f * sc;
        case CityFocus::Production: return 0.8f * fd + 2.2f * pr + 0.3f * gd + 0.2f * sc;
        case CityFocus::Gold:       return 0.8f * fd + 0.5f * pr + 2.2f * gd + 0.4f * sc;
        case CityFocus::Balanced:
        case CityFocus::Count:      break;
    }
    return 1.0f * fd + 1.0f * pr + 0.7f * gd + 0.6f * sc;
}

int foodToGrow(int population) {
    return balance::kFoodBoxBase + (balance::kFoodBoxPerPop * population);
}

// Wonder fatigue: every wonder an empire already owns raises the cost of its
// next one. A leader can't hog them all -- after a few, rivals can out-race it.
// This is the anti-snowball that keeps every civ in the wonder hunt.
int effectiveCost(const Empire& emp, const BuildingDef& d) {
    if (!d.isWonder) return d.productionCost;
    const int owned = static_cast<int>(emp.wonders.size());
    return d.productionCost + (d.productionCost * 30 * owned) / 100;
}

// Per-wonder personality affinity: how much this wonder's score is amplified for
// a given leader. Religion civs love happiness wonders; culture civs love art
// wonders; science civs love knowledge wonders; merchants love gold wonders.
float wonderAffinityScore(const std::string& id, const Personality& p) {
    if (id == "colosseum" || id == "hanging_gardens" || id == "great_wall" || id == "grand_temple")
        return 0.4f + 0.9f * p.religion;
    if (id == "parthenon" || id == "oracle")
        return 0.4f + 0.9f * p.culture;
    if (id == "great_library" || id == "copernicus")
        return 0.3f + 0.9f * p.science;
    if (id == "colossus" || id == "grand_bazaar")
        return 0.3f + 0.9f * p.gold;
    return 1.0f;  // pyramids: pure builder (gated mainly by wonderLove)
}

void addYields(Yields& acc, const Yields& y) {
    acc.food += y.food;
    acc.production += y.production;
    acc.gold += y.gold;
    acc.science += y.science;
    acc.culture += y.culture;
}

int cityMaintenance(const City& city) {
    int m = 0;
    for (const std::string& b : city.buildings) {
        const BuildingDef* d = findBuildingDef(b);
        if (d != nullptr) m += d->maintenance;
    }
    return m;
}

void logEvent(World& world, std::uint8_t empire, GameEvent::Kind kind, std::string text) {
    world.events.push_back(GameEvent{world.turn, empire, std::move(text), kind});
}

}  // namespace

// --- small struct helpers ---------------------------------------------------

bool City::hasBuilding(const std::string& id) const {
    return std::find(buildings.begin(), buildings.end(), id) != buildings.end();
}
bool Empire::knows(const std::string& techId) const {
    return std::find(researched.begin(), researched.end(), techId) != researched.end();
}
bool Empire::ownsWonder(const World& /*world*/, const std::string& wonderId) const {
    return std::find(wonders.begin(), wonders.end(), wonderId) != wonders.end();
}
bool World::wonderTaken(const std::string& id) const {
    return std::find(builtWonders.begin(), builtWonders.end(), id) != builtWonders.end();
}
Empire* World::empireById(std::uint8_t id) {
    for (Empire& e : empires) {
        if (e.id == id) return &e;
    }
    return nullptr;
}
int World::cityCount(std::uint8_t empireId) const {
    int n = 0;
    for (const City& c : cities) {
        if (c.owner == empireId) ++n;
    }
    return n;
}

// --- territory --------------------------------------------------------------

namespace {
void claimTerritory(World& world, const City& city) {
    StrategyMap& map = world.map;
    const int cc = static_cast<int>(city.col);
    const int cr = static_cast<int>(city.row);
    for (int r = cr - kCityWorkRadius - 1; r <= cr + kCityWorkRadius + 1; ++r) {
        for (int c = cc - kCityWorkRadius - 1; c <= cc + kCityWorkRadius + 1; ++c) {
            if (!map.inBounds(c, r)) continue;
            if (hexDistance(cc, cr, c, r) > kCityWorkRadius) continue;
            MapTile& t = map.at(static_cast<std::uint32_t>(c), static_cast<std::uint32_t>(r));
            if (t.owner == 0) t.owner = city.owner;
        }
    }
}
}  // namespace

// --- yields / happiness -----------------------------------------------------

void computeCityYields(World& world, City& city) {
    StrategyMap& map = world.map;
    Empire* emp = world.empireById(city.owner);

    // 1. Gather workable, owned tiles in range and pick the best `population` of
    //    them under the city's current focus.
    struct Cand { Yields y; float score; };
    std::vector<Cand> cands;
    const int cc = static_cast<int>(city.col);
    const int cr = static_cast<int>(city.row);
    for (int r = cr - kCityWorkRadius - 1; r <= cr + kCityWorkRadius + 1; ++r) {
        for (int c = cc - kCityWorkRadius - 1; c <= cc + kCityWorkRadius + 1; ++c) {
            if (!map.inBounds(c, r)) continue;
            if (c == cc && r == cr) continue;
            if (hexDistance(cc, cr, c, r) > kCityWorkRadius) continue;
            const MapTile& t = map.at(static_cast<std::uint32_t>(c), static_cast<std::uint32_t>(r));
            if (!tileIsWorkable(t.terrain)) continue;
            if (t.owner != city.owner) continue;
            const Yields ty = terrainYields(t.terrain, t.flags);
            cands.push_back({ty, focusScore(city.focus, ty)});
        }
    }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });

    Yields y = kCityCenterYields;
    const int workers = std::min<int>(city.population, static_cast<int>(cands.size()));
    for (int i = 0; i < workers; ++i) addYields(y, cands[static_cast<std::size_t>(i)].y);

    // Population is itself a knowledge engine.
    y.science += city.population / kSciencePerPopDiv;

    // 2. Buildings: flat yields + accumulating percentage bonuses + happiness.
    int prodPct = 0, goldPct = 0, sciPct = 0;
    int happy = balance::kBaseHappyCap;
    int growBonus = 0;
    for (const std::string& b : city.buildings) {
        const BuildingDef* d = findBuildingDef(b);
        if (d == nullptr || d->isWonder) continue;  // wonders handled empire-wide below
        addYields(y, d->flat);
        prodPct += d->prodPct;
        goldPct += d->goldPct;
        sciPct += d->sciencePct;
        happy += d->happiness;
        growBonus += d->growthBonus;
    }

    // 3. Wonders apply to every city in the owning empire.
    if (emp != nullptr) {
        for (const std::string& w : emp->wonders) {
            if (w == "pyramids")              prodPct += 25;
            else if (w == "hanging_gardens") { y.food += 1; happy += 2; }
            else if (w == "colossus")          y.gold += 2;
            else if (w == "oracle")          { y.science += 2; y.culture += 1; }
            else if (w == "great_library")   { y.science += 3; sciPct += 25; }
            else if (w == "parthenon")         y.culture += 3;
            else if (w == "colosseum")         happy += 3;
            else if (w == "great_wall")        happy += 2;
            else if (w == "grand_temple")    { y.culture += 2; happy += 1; }
            else if (w == "copernicus")      { y.science += 3; sciPct += 25; }
            else if (w == "grand_bazaar")    { y.gold += 3; goldPct += 25; }
        }
    }

    // 4. Apply percentage multipliers.
    y.production += (y.production * prodPct) / 100;
    y.gold       += (y.gold * goldPct) / 100;
    y.science    += (y.science * sciPct) / 100;

    // 5. Happiness: an over-large city falls into disorder (no growth, production
    //    penalty) until more happiness infrastructure is built. THE squeeze.
    city.happyCap = happy;
    city.growthBonusPct = std::min(growBonus, kMaxGrowthBonus);
    const int deficit = std::max(0, city.population - happy);
    city.inDisorder = deficit > 0;
    if (deficit > 0) {
        const int penalty = std::min(60, deficit * 25);
        y.production -= (y.production * penalty) / 100;
    }

    // 6. Citizen food upkeep -> net food (drives growth / starvation).
    y.food -= city.population * balance::kCitizenFoodUpkeep;

    city.yields = y;
}

// --- AI: research -----------------------------------------------------------

namespace {
void pickResearch(World& world, Empire& emp) {
    if (!emp.researching.empty()) return;
    const TechDef* best = nullptr;
    float bestScore = -1e9f;
    for (const TechDef& t : techTree()) {
        if (emp.knows(t.id)) continue;
        bool prereqsMet = true;
        for (const std::string& p : t.prereqs) {
            if (!emp.knows(p)) { prereqsMet = false; break; }
        }
        if (!prereqsMet) continue;

        float score = -0.04f * static_cast<float>(t.cost);  // cheaper is sooner
        for (const std::string& u : t.unlocks) {
            const BuildingDef* d = findBuildingDef(u);
            if (d == nullptr) continue;
            if (d->isWonder) {
                if (!world.wonderTaken(u)) score += 3.0f * emp.personality.wonderLove;
            } else {
                score += 2.0f;
                if (d->goldPct > 0 || d->flat.gold > 0)        score += emp.personality.gold;
                if (d->sciencePct > 0 || d->flat.science > 0)  score += emp.personality.science;
                if (d->flat.culture > 0)                        score += emp.personality.culture;
                if (d->happiness > 0)                           score += emp.personality.religion;
            }
        }
        if (score > bestScore) { bestScore = score; best = &t; }
    }
    // Tree exhausted -> repeatable Future Tech keeps research (and score) flowing.
    emp.researching = (best != nullptr) ? best->id : "future_tech";
}

int futureTechCost(int already) { return 240 + 60 * already; }

// Best buildable economic building for a city right now (unlocked, affordable to
// start, not yet built), scored by personality and city needs.
std::string chooseBuilding(const World& world, const Empire& emp, const City& city) {
    const BuildingDef* best = nullptr;
    float bestScore = 0.0f;
    for (const BuildingDef& d : buildingDefs()) {
        if (d.isWonder) continue;
        if (city.hasBuilding(d.id)) continue;
        if (!d.requiredTech.empty() && !emp.knows(d.requiredTech)) continue;
        // Broke: don't take on heavy upkeep (cheap happiness relief still allowed
        // via the relief path). This is what stops the build-then-fire-sale spiral.
        if (emp.treasury < 6 && d.maintenance >= 2) continue;
        // Only suggest a coastal building if the city actually has coast nearby.
        if (d.id == "harbor") {
            bool coastal = false;
            const int cc = static_cast<int>(city.col), cr = static_cast<int>(city.row);
            for (int r = cr - 2; r <= cr + 2 && !coastal; ++r)
                for (int c = cc - 2; c <= cc + 2 && !coastal; ++c)
                    if (world.map.inBounds(c, r) && hexDistance(cc, cr, c, r) <= 2 &&
                        terrainIsWater(world.map.at(static_cast<std::uint32_t>(c), static_cast<std::uint32_t>(r)).terrain))
                        coastal = true;
            if (!coastal) continue;
        }
        float score = 1.0f;
        // Religious leaders build happiness infrastructure proactively (one tier
        // earlier) and score it much higher. Everyone else reacts at the last moment.
        if (d.happiness > 0) {
            const int lead = (emp.personality.religion >= 1.5f) ? 2 : 1;
            if (city.population >= city.happyCap - lead)
                score += 6.0f * emp.personality.religion;
        }
        if (d.flat.culture > 0) score += 2.5f * emp.personality.culture;
        if (d.flat.food > 0 || d.growthBonus > 0) score += 2.0f;
        if (d.sciencePct > 0 || d.flat.science > 0) score += 1.5f * emp.personality.science;
        if (d.goldPct > 0 || d.flat.gold > 0) score += 1.5f * emp.personality.gold;
        if (d.prodPct > 0 || d.flat.production > 0) score += 2.0f;
        score -= 0.01f * static_cast<float>(d.productionCost);
        if (score > bestScore) { bestScore = score; best = &d; }
    }
    return best != nullptr ? best->id : std::string{};
}

std::string chooseWonder(const World& world, const Empire& emp, const City& city) {
    // Don't double up: skip wonders the empire already owns or is building elsewhere.
    const BuildingDef* best = nullptr;
    float bestScore = 0.0f;
    for (const BuildingDef& d : buildingDefs()) {
        if (!d.isWonder) continue;
        if (world.wonderTaken(d.id)) continue;
        if (!d.requiredTech.empty() && !emp.knows(d.requiredTech)) continue;
        bool buildingElsewhere = false;
        for (std::size_t ci : emp.cityIndices) {
            if (world.cities[ci].producing == d.id) { buildingElsewhere = true; break; }
        }
        if (buildingElsewhere) continue;
        float score = static_cast<float>(d.score) * emp.personality.wonderLove
                      * wonderAffinityScore(d.id, emp.personality);
        score -= 0.02f * static_cast<float>(effectiveCost(emp, d));  // fatigue makes the Nth wonder less worth chasing
        // A productive city is a better wonder forge.
        score += 0.4f * static_cast<float>(city.yields.production);
        if (score > bestScore) { bestScore = score; best = &d; }
    }
    return best != nullptr ? best->id : std::string{};
}

int desiredCityCount(const Empire& emp) {
    return 3 + static_cast<int>(std::lround(2.0f * emp.personality.expansion));
}

// Is there a legal place to found a new city near this empire? (also returns it)
bool findSettleSpot(const World& world, const Empire& emp, std::uint32_t& outCol, std::uint32_t& outRow) {
    const StrategyMap& map = world.map;
    float bestScore = 1.0f;  // require a minimally decent site
    bool found = false;
    for (std::uint32_t row = 0; row < map.height; ++row) {
        for (std::uint32_t col = 0; col < map.width; ++col) {
            const MapTile& t = map.at(col, row);
            if (terrainIsWater(t.terrain) || !tileIsWorkable(t.terrain)) continue;
            if (t.terrain == TerrainType::Desert || t.terrain == TerrainType::Tundra) continue;
            if (t.owner != 0 && t.owner != emp.id) continue;  // not into rival land

            // Spacing + proximity: must be >= min spacing from every city, but
            // within reach (<= 6) of one of THIS empire's cities.
            bool spacingOk = true;
            bool nearOwn = false;
            for (const City& c : world.cities) {
                const int dist = hexDistance(static_cast<int>(col), static_cast<int>(row),
                                             static_cast<int>(c.col), static_cast<int>(c.row));
                if (dist < kSettleMinSpacing) { spacingOk = false; break; }
                if (c.owner == emp.id && dist <= 6) nearOwn = true;
            }
            if (!spacingOk || !nearOwn) continue;

            // Score the site by the food/production around it.
            float score = 0.0f;
            const int cc = static_cast<int>(col), cr = static_cast<int>(row);
            for (int r = cr - 2; r <= cr + 2; ++r)
                for (int c = cc - 2; c <= cc + 2; ++c) {
                    if (!map.inBounds(c, r) || hexDistance(cc, cr, c, r) > 2) continue;
                    const MapTile& nt = map.at(static_cast<std::uint32_t>(c), static_cast<std::uint32_t>(r));
                    if (!tileIsWorkable(nt.terrain)) continue;
                    const Yields y = terrainYields(nt.terrain, nt.flags);
                    score += static_cast<float>(y.food) + 0.8f * static_cast<float>(y.production) +
                             0.5f * static_cast<float>(y.gold);
                }
            if (score > bestScore) { bestScore = score; outCol = col; outRow = row; found = true; }
        }
    }
    return found;
}

const std::array<const char*, 16> kCityNames = {
    "Ashford", "Brightwater", "Crowmere", "Dawnvale", "Eldergrove", "Fairhaven", "Goldcrest",
    "Highrock", "Ironhold", "Jadeport", "Kingsreach", "Larkfield", "Mistral", "Northwatch",
    "Oakhollow", "Pinecliff"};

bool isHappinessBuilding(const std::string& id) {
    return id == "temple" || id == "walls" || id == "aqueduct";
}

// A happiness building this city can build right now to relieve disorder, or "".
std::string pickReliefBuilding(const Empire& emp, const City& city) {
    for (const char* id : {"temple", "aqueduct", "cathedral", "walls"}) {
        const BuildingDef* d = findBuildingDef(id);
        if (d != nullptr && !city.hasBuilding(id) &&
            (d->requiredTech.empty() || emp.knows(d->requiredTech))) {
            return id;
        }
    }
    return std::string{};
}

// The full production-priority pick, used whenever a city has a free queue.
std::string pickProduction(World& world, Empire& emp, City& city) {
    // 1. Relieve disorder first.
    if (city.inDisorder || city.population >= city.happyCap) {
        const std::string relief = pickReliefBuilding(emp, city);
        if (!relief.empty()) return relief;
    }
    // 2. Expand while under the desired city count and a site exists.
    if (world.cityCount(emp.id) < desiredCityCount(emp) && city.population >= 2) {
        std::uint32_t sc = 0, sr = 0;
        if (findSettleSpot(world, emp, sc, sr)) return "settler";
    }
    // 3. Reach for a wonder if productive, wonder-loving (low wonderLove civs skip
    //    the race entirely -- Genghis doesn't build monuments).
    if (city.yields.production >= 3 && emp.personality.wonderLove >= 0.5f) {
        const std::string w = chooseWonder(world, emp, city);
        if (!w.empty()) return w;
    }
    // 4. Best economic building.
    const std::string b = chooseBuilding(world, emp, city);
    if (!b.empty()) return b;
    // 5. Nothing worth building: leave empty -> production becomes wealth, and we
    //    re-evaluate next turn (never lock a city out of building).
    return std::string{};
}

void cityChooseFocusAndProduction(World& world, Empire& emp, City& city) {
    // --- focus (the grow-vs-build knob) ---
    // Guard rail: never starve a city. Last turn's net food tells us if we can
    // afford a non-food focus this turn.
    const bool foodSafe = city.yields.food >= 1 || city.foodStored > foodToGrow(city.population) / 2;
    if (city.yields.food < 0) {
        city.focus = CityFocus::Food;                       // emergency: feed the people
    } else if (emp.treasury < 4) {
        city.focus = CityFocus::Gold;                       // dig out of the red
    } else if ((city.producing == "settler" || isWonder(city.producing)) && foodSafe) {
        city.focus = CityFocus::Production;                 // crank the big project
    } else if (city.population < 4) {
        city.focus = CityFocus::Food;                       // grow young cities
    } else {
        city.focus = CityFocus::Balanced;
    }

    // --- production queue ---
    // Re-pick when idle, OR pre-empt the current build for urgent disorder relief
    // (but never interrupt a wonder we've committed to).
    if (city.producing.empty()) {
        city.producing = pickProduction(world, emp, city);
    } else if (city.inDisorder && !isWonder(city.producing) && !isHappinessBuilding(city.producing)) {
        const std::string relief = pickReliefBuilding(emp, city);
        if (!relief.empty()) {
            const BuildingDef* d = findBuildingDef(relief);
            if (d != nullptr) city.accumulated = std::min(city.accumulated, d->productionCost);
            city.producing = relief;
        }
    }
}

// Found a new city from a completed settler. Returns false if no spot (caller
// refunds the production). Mutates world.cities, so callers must not hold City&.
bool foundCityFromSettler(World& world, Empire& emp, std::size_t parentIndex) {
    std::uint32_t col = 0, row = 0;
    if (!findSettleSpot(world, emp, col, row)) return false;

    City c{};
    if (emp.nextCityName < static_cast<int>(emp.cityNames.size())) {
        c.name = emp.cityNames[static_cast<std::size_t>(emp.nextCityName++)];
    } else {
        c.name = kCityNames[world.cities.size() % kCityNames.size()];
    }
    c.col = col;
    c.row = row;
    c.owner = emp.id;
    c.population = balance::kSettlerFoundPop;
    c.focus = CityFocus::Food;
    c.foundedThisGame = true;
    world.cities.push_back(c);
    const std::size_t newIndex = world.cities.size() - 1;
    emp.cityIndices.push_back(newIndex);
    claimTerritory(world, world.cities.back());

    // Settler cost in population is paid by the parent city.
    City& parent = world.cities[parentIndex];
    parent.population = std::max(1, parent.population - balance::kSettlerPopCost);
    logEvent(world, emp.id, GameEvent::Founded,
             emp.name + " founds " + c.name + " (pop " + std::to_string(c.population) + ")");
    return true;
}

void applyProduction(World& world, Empire& emp, std::size_t cityIndex) {
    City& city = world.cities[cityIndex];
    const int prod = std::max(0, city.yields.production);

    if (city.producing.empty()) {
        emp.treasury += prod;  // idle queue: production sold for wealth this turn
        return;
    }

    city.accumulated += prod;

    if (city.producing == "settler") {
        if (city.accumulated < kSettlerCost) return;
        city.accumulated -= kSettlerCost;
        if (!foundCityFromSettler(world, emp, cityIndex)) {
            emp.treasury += kSettlerCost / 3;  // no room: salvage some gold
        }
        world.cities[cityIndex].producing.clear();
        return;
    }

    const BuildingDef* d = findBuildingDef(city.producing);
    if (d == nullptr) {  // unknown id: drop it
        city.producing.clear();
        city.accumulated = 0;
        return;
    }

    const int cost = effectiveCost(emp, *d);

    // Rush-buy: spend gold to finish a project. For wonders this is the snipe
    // that turns a treasury lead into a wonder-race win; for ordinary buildings
    // it is how a cash-rich empire converts an idle hoard into momentum. Each
    // keeps a reserve so an empire never bankrupts itself rushing.
    const bool wonderRush = d->isWonder && !world.wonderTaken(d->id) &&
                            city.accumulated * 2 >= cost && emp.treasury > 120;
    const bool buildingRush = !d->isWonder && emp.treasury > 260;
    if (city.accumulated < cost && (wonderRush || buildingRush)) {
        const int reserve = d->isWonder ? 100 : 200;
        const int need = cost - city.accumulated;
        const int afford = (emp.treasury - reserve) / kRushGoldPerShield;
        const int buy = std::min(need, std::max(0, afford));
        city.accumulated += buy;
        emp.treasury -= buy * kRushGoldPerShield;
    }

    if (city.accumulated < cost) return;

    if (d->isWonder) {
        if (world.wonderTaken(d->id)) {
            // Lost the race: refund half the invested shields as gold.
            emp.treasury += city.accumulated / 2;
            logEvent(world, emp.id, GameEvent::WonderLost,
                     emp.name + " loses the race for the " + d->name + " (refunded)");
            city.accumulated = 0;
            city.producing.clear();
            return;
        }
        city.buildings.push_back(d->id);
        emp.wonders.push_back(d->id);
        world.builtWonders.push_back(d->id);
        logEvent(world, emp.id, GameEvent::Wonder, emp.name + " completes the " + d->name + "!");
    } else {
        if (!city.hasBuilding(d->id)) city.buildings.push_back(d->id);
        logEvent(world, emp.id, GameEvent::Building, city.name + " builds a " + d->name);
    }
    city.accumulated = std::max(0, city.accumulated - cost);
    city.producing.clear();
}

void growCity(World& world, City& city) {
    const int net = city.yields.food;  // already net of upkeep
    // In disorder a city can't grow: a food surplus is wasted, but a real deficit
    // still bleeds the granary.
    if (!city.inDisorder || net < 0) city.foodStored += net;

    if (city.foodStored < 0) {
        // A small buffer means a one-turn dip won't cost a citizen -- the focus
        // guard gets a chance to switch back to Food first. Only a sustained
        // famine actually shrinks the city.
        if (city.foodStored < -kStarveBuffer && city.population > 1) {
            city.population -= 1;
            city.foodStored = 0;
            logEvent(world, city.owner, GameEvent::Starve, city.name + " starves (-1 pop)");
        }
        return;
    }
    if (city.inDisorder) return;  // happy cities grow; unhappy ones stagnate

    const int threshold = foodToGrow(city.population);
    if (city.foodStored >= threshold) {
        city.population += 1;
        const int kept = (threshold * city.growthBonusPct) / 100;
        city.foodStored = (city.foodStored - threshold) + kept;
        logEvent(world, city.owner, GameEvent::Growth,
                 city.name + " grows to pop " + std::to_string(city.population));
    }
}

void goldSqueeze(World& world, Empire& emp) {
    // Spend down debt by selling buildings -- the painful sacrifice. Shed the
    // highest-upkeep building first (most relief per sale), but never dump a
    // happiness building out of a city that is already unhappy, and never sell
    // something a city is mid-way through rebuilding -- that just oscillates.
    while (emp.treasury < 0) {
        City* victimCity = nullptr;
        int victimPos = -1;
        int bestMaint = 0;
        for (std::size_t ci : emp.cityIndices) {
            City& c = world.cities[ci];
            for (std::size_t bi = 0; bi < c.buildings.size(); ++bi) {
                const std::string& id = c.buildings[bi];
                const BuildingDef* d = findBuildingDef(id);
                if (d == nullptr || d->isWonder || d->maintenance == 0) continue;
                if (isHappinessBuilding(id) && c.inDisorder) continue;  // don't worsen disorder
                if (id == c.producing) continue;                        // don't sell what we're rebuilding
                if (d->maintenance > bestMaint) {
                    bestMaint = d->maintenance;
                    victimCity = &c;
                    victimPos = static_cast<int>(bi);
                }
            }
        }
        if (victimCity == nullptr) {  // nothing left to sell
            emp.treasury = 0;
            break;
        }
        const std::string sold = victimCity->buildings[static_cast<std::size_t>(victimPos)];
        const BuildingDef* d = findBuildingDef(sold);
        emp.treasury += (d != nullptr ? d->productionCost / 2 : 0);
        victimCity->buildings.erase(victimCity->buildings.begin() + victimPos);
        logEvent(world, emp.id, GameEvent::FireSale,
                 emp.name + " sells a " + (d ? d->name : sold) + " to stay solvent");
    }
}

}  // namespace

// --- scoring ----------------------------------------------------------------

void recomputeScore(World& world, Empire& emp) {
    int pop = 0, cityN = 0, buildings = 0, wonderScore = 0;
    for (std::size_t ci : emp.cityIndices) {
        const City& c = world.cities[ci];
        pop += c.population;
        ++cityN;
        for (const std::string& b : c.buildings) {
            const BuildingDef* d = findBuildingDef(b);
            if (d != nullptr && d->isWonder) wonderScore += d->score;
            else ++buildings;
        }
    }
    emp.totalPopulation = pop;
    emp.score = pop * 4 + cityN * 6 + buildings * 2 + wonderScore +
                static_cast<int>(emp.researched.size()) * 10 + emp.futureTechs * 10 +
                emp.culturePoints / 8 + emp.treasury / 25;
}

// --- the turn ---------------------------------------------------------------

void stepTurn(World& world, std::vector<TurnSample>& samples) {
    // 1. AI: pick research, then per-city focus + production queue.
    for (Empire& emp : world.empires) {
        if (!emp.alive) continue;
        pickResearch(world, emp);
    }
    // Snapshot the city-index lists (foundCityFromSettler may append new cities;
    // those new cities act starting next turn).
    for (Empire& emp : world.empires) {
        if (!emp.alive) continue;
        const std::vector<std::size_t> indices = emp.cityIndices;
        for (std::size_t ci : indices) {
            cityChooseFocusAndProduction(world, emp, world.cities[ci]);
        }
    }

    // 2. Yields -> production -> science/gold accumulation -> growth.
    for (Empire& emp : world.empires) {
        if (!emp.alive) continue;
        const std::vector<std::size_t> indices = emp.cityIndices;  // stable snapshot
        for (std::size_t ci : indices) {
            computeCityYields(world, world.cities[ci]);
        }
        for (std::size_t ci : indices) {
            City& c = world.cities[ci];
            emp.sciencePool += std::max(0, c.yields.science);
            emp.culturePoints += std::max(0, c.yields.culture);
            emp.treasury += c.yields.gold - cityMaintenance(c);
            applyProduction(world, emp, ci);   // may found new cities
        }
        // Administrative upkeep scales with empire size, so endless expansion
        // runs you into the red -- the pressure that forces fire-sales.
        const int adminCities = std::max(0, world.cityCount(emp.id) - 1);
        emp.treasury -= adminCities * kCivicUpkeepPerCity;
        // Growth after production so a settler's pop cost lands first.
        for (std::size_t ci : indices) {
            growCity(world, world.cities[ci]);
        }
    }

    // 3. Research completion (may resolve several cheap techs in one turn).
    for (Empire& emp : world.empires) {
        if (!emp.alive) continue;
        bool progressed = true;
        while (progressed && !emp.researching.empty()) {
            progressed = false;
            if (emp.researching == "future_tech") {
                const int cost = futureTechCost(emp.futureTechs);
                if (emp.sciencePool >= cost) {
                    emp.sciencePool -= cost;
                    emp.futureTechs += 1;
                    logEvent(world, emp.id, GameEvent::Tech,
                             emp.name + " advances Future Tech " + std::to_string(emp.futureTechs));
                    emp.researching.clear();
                    pickResearch(world, emp);
                    progressed = true;
                }
                continue;
            }
            const TechDef* t = findTech(emp.researching);
            if (t == nullptr) { emp.researching.clear(); break; }
            if (emp.sciencePool >= t->cost) {
                emp.sciencePool -= t->cost;
                emp.researched.push_back(t->id);
                logEvent(world, emp.id, GameEvent::Tech, emp.name + " discovers " + t->name);
                emp.researching.clear();
                pickResearch(world, emp);
                progressed = true;
            }
        }
    }

    // 4. Gold squeeze (sell buildings if bankrupt).
    for (Empire& emp : world.empires) {
        if (!emp.alive) continue;
        goldSqueeze(world, emp);
    }

    // 5. Score + metrics row.
    TurnSample sample{};
    sample.turn = world.turn + 1;
    int topScore = -1;
    std::uint8_t leader = 0;
    for (Empire& emp : world.empires) {
        recomputeScore(world, emp);
        sample.score.push_back(emp.score);
        sample.population.push_back(emp.totalPopulation);
        sample.cities.push_back(world.cityCount(emp.id));
        sample.techs.push_back(static_cast<int>(emp.researched.size()));
        sample.treasury.push_back(emp.treasury);
        sample.wonders.push_back(static_cast<int>(emp.wonders.size()));
        int disorderCount = 0;
        for (std::size_t ci : emp.cityIndices) {
            if (world.cities[ci].inDisorder) ++disorderCount;
        }
        sample.disorder.push_back(disorderCount);
        if (emp.score > topScore) { topScore = emp.score; leader = emp.id; }
    }
    sample.leader = leader;
    samples.push_back(std::move(sample));

    world.turn += 1;
}

// --- world construction -----------------------------------------------------

World makeWorld(const WorldConfig& config) {
    World world{};
    world.rng = config.seed ^ 0xABCDEF01u;
    StrategyMap& map = world.map;
    map.resize(config.width, config.height);
    map.hexSize = 64.0f;
    map.elevationStep = 28.0f;

    const float noiseScale = 0.16f;
    for (std::uint32_t row = 0; row < config.height; ++row) {
        for (std::uint32_t col = 0; col < config.width; ++col) {
            const float nx = static_cast<float>(col) * noiseScale;
            const float nz = static_cast<float>(row) * noiseScale;
            float base = fbm(nx, nz, config.seed);
            const float u = (static_cast<float>(col) / static_cast<float>(config.width - 1)) - 0.5f;
            const float v = (static_cast<float>(row) / static_cast<float>(config.height - 1)) - 0.5f;
            const float falloff = 1.0f - std::min(1.0f, std::sqrt((u * u) + (v * v)) * 1.75f);
            base = (base * 0.62f) + (falloff * 0.6f) - 0.22f;
            const auto elev = static_cast<std::int16_t>(std::lround(base * 9.0f));
            const float lat = static_cast<float>(row) / static_cast<float>(config.height - 1);
            const float moisture = fbm(nx + 31.7f, nz - 12.3f, config.seed + 7u);
            MapTile& tile = map.at(col, row);
            tile.elevation = std::clamp<std::int16_t>(elev, -2, 8);
            tile.terrain = classifyTerrain(tile.elevation, lat, moisture);
            tile.visibility = TileVisibility::Visible;
        }
    }

    // Scatter a few rivers down from the highlands (adds gold to riverside tiles).
    for (std::uint32_t row = 1; row + 1 < config.height; ++row) {
        for (std::uint32_t col = 1; col + 1 < config.width; ++col) {
            const MapTile& tile = map.at(col, row);
            if (tile.elevation < 4 ||
                (hashCoords(static_cast<int>(col), static_cast<int>(row), config.seed + 99u) % 17u) != 0u)
                continue;
            int cc = static_cast<int>(col), cr = static_cast<int>(row);
            for (int step = 0; step < static_cast<int>(config.width + config.height); ++step) {
                map.at(static_cast<std::uint32_t>(cc), static_cast<std::uint32_t>(cr)).flags |= TileFlag_River;
                int bc = cc, br = cr;
                std::int16_t be = map.at(static_cast<std::uint32_t>(cc), static_cast<std::uint32_t>(cr)).elevation;
                for (int dir = 0; dir < 6; ++dir) {
                    int nc = 0, nr = 0;
                    if (!tileNeighbor(map, cc, cr, dir, nc, nr)) continue;
                    const std::int16_t e = map.at(static_cast<std::uint32_t>(nc), static_cast<std::uint32_t>(nr)).elevation;
                    if (e < be) { be = e; bc = nc; br = nr; }
                }
                if (bc == cc && br == cr) break;
                cc = bc; cr = br;
                if (terrainIsWater(map.at(static_cast<std::uint32_t>(cc), static_cast<std::uint32_t>(cr)).terrain)) break;
            }
        }
    }

    // Named leaders, each with a distinct playstyle. The player is always empire 1
    // (Egypt/Ramesses by default -- the builder/religion archetype).
    struct Preset {
        const char* civName;
        const char* leaderName;
        Personality p;
        const char* cityNames[8];
    };
    // Personality fields: name, expansion, wonderLove, science, gold, religion, culture
    const std::array<Preset, 6> presets = {{
        // Egypt / Ramesses: monument builder + temples everywhere; loves happiness wonders.
        {"Egypt",    "Ramesses", {"Builder / Religion",  0.8f, 1.8f, 0.8f, 1.0f, 2.0f, 1.2f},
         {"Memphis","Thebes","Heliopolis","Luxor","Alexandria","Abydos","Karnak","Giza"}},
        // Mongols / Genghis: pure land-grab; skips wonders and temples entirely.
        {"Mongols",  "Genghis",  {"Expansionist",        1.8f, 0.3f, 0.7f, 1.0f, 0.5f, 0.3f},
         {"Karakorum","Samarkand","Bukhara","Merv","Urgench","Otrar","Kashgar","Balkh"}},
        // Greece / Pericles: science + culture; loves Great Library and Parthenon.
        {"Greece",   "Pericles", {"Scientist / Culture", 0.8f, 1.3f, 2.0f, 0.8f, 0.6f, 1.9f},
         {"Athens","Sparta","Corinth","Argos","Olympia","Delphi","Miletos","Eretria"}},
        // Phoenicia / Hiram: gold and trade; harbors, Grand Bazaar, coastal cities.
        {"Phoenicia","Hiram",    {"Merchant",            1.2f, 0.9f, 0.9f, 2.1f, 0.6f, 0.5f},
         {"Tyre","Sidon","Byblos","Carthage","Utica","Kition","Akko","Arwad"}},
        // Rome / Augustus: steady builder-expansionist; Colosseum + good infrastructure.
        {"Rome",     "Augustus", {"Builder",             1.5f, 1.1f, 1.0f, 1.2f, 1.0f, 0.8f},
         {"Rome","Neapolis","Mediolanum","Lugdunum","Londinium","Aquileia","Capua","Ravenna"}},
        // India / Ashoka: religion + culture pacifist; max happiness, Hanging Gardens.
        {"India",    "Ashoka",   {"Religion / Culture",  0.6f, 1.4f, 1.2f, 0.8f, 2.3f, 1.9f},
         {"Pataliputra","Taxila","Ujjain","Varanasi","Mathura","Rajgriha","Nalanda","Sarnath"}},
    }};

    const int empireCount = std::clamp(config.empireCount, 1, 6);

    // Pick capital sites: best-scoring habitable land, spaced well apart.
    auto siteScore = [&](std::uint32_t col, std::uint32_t row) -> float {
        const MapTile& t = map.at(col, row);
        if (terrainIsWater(t.terrain) || !tileIsWorkable(t.terrain)) return -1.0f;
        if (t.terrain == TerrainType::Desert || t.terrain == TerrainType::Tundra ||
            t.terrain == TerrainType::Snow || t.terrain == TerrainType::Mountains)
            return -1.0f;
        float score = 0.0f;
        const int cc = static_cast<int>(col), cr = static_cast<int>(row);
        for (int r = cr - 2; r <= cr + 2; ++r)
            for (int c = cc - 2; c <= cc + 2; ++c) {
                if (!map.inBounds(c, r) || hexDistance(cc, cr, c, r) > 2) continue;
                const MapTile& nt = map.at(static_cast<std::uint32_t>(c), static_cast<std::uint32_t>(r));
                if (!tileIsWorkable(nt.terrain)) continue;
                const Yields y = terrainYields(nt.terrain, nt.flags);
                score += static_cast<float>(y.food) + 0.8f * static_cast<float>(y.production) +
                         0.5f * static_cast<float>(y.gold);
            }
        return score;
    };

    std::vector<std::pair<std::uint32_t, std::uint32_t>> capitals;
    for (int e = 0; e < empireCount; ++e) {
        float best = 0.5f;
        std::uint32_t bcol = 0, brow = 0;
        bool ok = false;
        for (std::uint32_t row = 0; row < config.height; ++row)
            for (std::uint32_t col = 0; col < config.width; ++col) {
                const float s = siteScore(col, row);
                if (s <= best) continue;
                bool farEnough = true;
                for (const auto& cap : capitals) {
                    if (hexDistance(static_cast<int>(col), static_cast<int>(row),
                                    static_cast<int>(cap.first), static_cast<int>(cap.second)) < 6) {
                        farEnough = false;
                        break;
                    }
                }
                if (!farEnough) continue;
                best = s; bcol = col; brow = row; ok = true;
            }
        if (ok) capitals.emplace_back(bcol, brow);
    }

    for (std::size_t e = 0; e < capitals.size(); ++e) {
        Empire emp{};
        emp.id = static_cast<std::uint8_t>(e + 1);
        emp.name = presets[e].civName;
        emp.leaderName = presets[e].leaderName;
        emp.personality = presets[e].p;
        emp.treasury = 20;
        for (int n = 0; n < 8 && presets[e].cityNames[n] != nullptr; ++n)
            emp.cityNames.push_back(presets[e].cityNames[n]);
        emp.nextCityName = 1;  // 0 = capital (already placed below)
        world.empires.push_back(emp);

        City c{};
        c.name = !emp.cityNames.empty() ? emp.cityNames[0]
                                        : (std::string(presets[e].leaderName) + "'s Capital");
        c.col = capitals[e].first;
        c.row = capitals[e].second;
        c.owner = emp.id;
        c.population = kCapitalStartPop;
        c.focus = CityFocus::Balanced;
        c.foundedThisGame = false;
        world.cities.push_back(c);
        world.empires.back().cityIndices.push_back(world.cities.size() - 1);
        claimTerritory(world, world.cities.back());
    }

    for (Empire& emp : world.empires) recomputeScore(world, emp);
    return world;
}

}  // namespace odai::game
