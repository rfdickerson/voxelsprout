#include "games/citybuilder/citybuilder_save.h"

#include <cstdint>
#include <fstream>
#include <istream>
#include <ostream>
#include <type_traits>
#include <vector>

#include "games/citybuilder/citybuilder_app.h"

namespace odai::games::citybuilder {

namespace {

constexpr std::uint32_t kCityMagic = 0x59544943u;  // 'CITY'
constexpr std::uint32_t kCityVersion = 1u;

std::string g_lastError;

bool fail(std::string message) {
    g_lastError = std::move(message);
    return false;
}

template <typename T>
void put(std::ostream& out, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    out.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
}

template <typename T>
bool get(std::istream& in, T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    in.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    return in.good();
}

void putStr(std::ostream& out, const std::string& value) {
    put(out, static_cast<std::uint32_t>(value.size()));
    if (!value.empty()) out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

bool getStr(std::istream& in, std::string& value) {
    std::uint32_t size = 0;
    if (!get(in, size)) return false;
    if (size > (1u << 20)) return false;  // a name is not a megabyte: refuse junk
    value.resize(size);
    if (size == 0) return true;
    in.read(value.data(), static_cast<std::streamsize>(size));
    return in.good();
}

void putFloats(std::ostream& out, const std::vector<float>& v) {
    put(out, static_cast<std::uint32_t>(v.size()));
    for (const float f : v) put(out, f);
}

bool getFloats(std::istream& in, std::vector<float>& v) {
    std::uint32_t n = 0;
    if (!get(in, n)) return false;
    if (n > (1u << 22)) return false;
    v.assign(n, 0.0f);
    for (std::uint32_t i = 0; i < n; ++i) {
        if (!get(in, v[i])) return false;
    }
    return true;
}

}  // namespace

const std::string& lastSaveError() { return g_lastError; }

bool saveCity(const CityBuilderApp& app, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return fail("could not open '" + path + "' for writing");

    put(out, kCityMagic);
    put(out, kCityVersion);
    put(out, static_cast<std::int32_t>(CityBuilderApp::kGridW));
    put(out, static_cast<std::int32_t>(CityBuilderApp::kGridH));

    // World identity first — every procedural name in the city derives from it.
    put(out, app.m_worldSeed);
    put(out, app.m_siteC);
    put(out, app.m_siteR);

    // ── Grid, field by field. See the header on why this is not an fwrite. ──
    for (const Tile& t : app.m_tiles) {
        put(out, static_cast<std::uint8_t>(t.terrain));
        put(out, static_cast<std::uint8_t>(t.zone));
        put(out, static_cast<std::uint8_t>(t.building));
        put(out, static_cast<std::uint8_t>(t.road ? 1 : 0));
        put(out, static_cast<std::uint8_t>(t.bldgOrigin ? 1 : 0));
        put(out, t.footprint);
        put(out, t.bOriginC);
        put(out, t.bOriginR);
        put(out, t.develop);
        put(out, static_cast<std::uint8_t>(t.powered ? 1 : 0));
        put(out, static_cast<std::uint8_t>(t.poweredRoad ? 1 : 0));
        put(out, static_cast<std::uint8_t>(t.nearRoad ? 1 : 0));
        put(out, t.desirability);
        put(out, t.scenicPhase);
        put(out, t.zoneAge);
        put(out, t.trafficLoad);
        put(out, t.fireTicks);
        put(out, t.charTicks);
        put(out, static_cast<std::uint8_t>(t.charred ? 1 : 0));
    }

    // ── Economy and clock. ──────────────────────────────────────────────────
    put(out, app.m_money);
    put(out, static_cast<std::int32_t>(app.m_year));
    put(out, static_cast<std::int32_t>(app.m_month));
    put(out, static_cast<std::int32_t>(app.m_population));
    put(out, static_cast<std::int32_t>(app.m_jobs));
    put(out, app.m_education);
    put(out, app.m_health);
    put(out, app.m_happiness);
    put(out, app.m_powerCoverage);
    put(out, app.m_resDemand);
    put(out, app.m_comDemand);
    put(out, app.m_indDemand);
    put(out, static_cast<std::int32_t>(app.m_burningTiles));
    put(out, static_cast<std::int32_t>(app.m_charredTiles));
    put(out, app.m_cityHeat);
    put(out, app.m_rng);

    // ── Atmosphere: the storm system carries real charge, so a save taken
    // mid-front has to reload mid-front or the weather visibly teleports. ────
    put(out, static_cast<std::uint8_t>(app.m_season));
    put(out, static_cast<std::uint8_t>(app.m_weather));
    put(out, static_cast<std::uint8_t>(app.m_weatherTarget));
    put(out, app.m_weatherIntensity);
    put(out, app.m_atmoHeat);
    put(out, app.m_atmoInstability);
    put(out, app.m_stormSeverity);
    put(out, app.m_weatherRng);

    // ── Charts. ─────────────────────────────────────────────────────────────
    putFloats(out, app.m_histPop);
    putFloats(out, app.m_histMoney);
    putFloats(out, app.m_histEdu);
    putFloats(out, app.m_histHealth);
    putFloats(out, app.m_histHappy);

    // ── Citizens. ───────────────────────────────────────────────────────────
    const CitizenSim::SaveState cs = app.m_citizens.saveState();
    put(out, cs.rngState);
    put(out, cs.citizenCounter);
    put(out, static_cast<std::uint32_t>(cs.roster.size()));
    for (const Citizen& cz : cs.roster) {
        put(out, cz.seed);
        putStr(out, cz.firstName);
        putStr(out, cz.lastName);
        put(out, cz.homeC);
        put(out, cz.homeR);
        put(out, cz.workC);
        put(out, cz.workR);
        put(out, cz.spouse);
        put(out, cz.affair);
        put(out, cz.traits);
        put(out, static_cast<std::uint8_t>(cz.atWork ? 1 : 0));
    }

    if (!out.good()) return fail("write failed part-way through '" + path + "'");
    g_lastError.clear();
    return true;
}

bool loadCity(CityBuilderApp& app, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail("could not open '" + path + "' for reading");

    std::uint32_t magic = 0, version = 0;
    std::int32_t gw = 0, gh = 0;
    if (!get(in, magic) || magic != kCityMagic) return fail("'" + path + "' is not a city save");
    if (!get(in, version) || version > kCityVersion) {
        return fail("save is from a newer build (version " + std::to_string(version) + ")");
    }
    if (!get(in, gw) || !get(in, gh)) return fail("truncated header");
    if (gw != CityBuilderApp::kGridW || gh != CityBuilderApp::kGridH) {
        return fail("save grid is " + std::to_string(gw) + "x" + std::to_string(gh) +
                    ", this build is " + std::to_string(CityBuilderApp::kGridW) + "x" +
                    std::to_string(CityBuilderApp::kGridH));
    }

    // Everything below lands in a scratch copy first, so a truncated or corrupt
    // file cannot leave the player looking at half a city.
    CityBuilderApp::Grid tiles{};
    if (!get(in, app.m_worldSeed) || !get(in, app.m_siteC) || !get(in, app.m_siteR)) {
        return fail("truncated world header");
    }

    const auto getBool = [&in](bool& b) {
        std::uint8_t v = 0;
        if (!get(in, v)) return false;
        b = v != 0;
        return true;
    };
    const auto getEnum = [&in](auto& e) {
        std::uint8_t v = 0;
        if (!get(in, v)) return false;
        e = static_cast<std::remove_reference_t<decltype(e)>>(v);
        return true;
    };

    for (Tile& t : tiles) {
        if (!getEnum(t.terrain) || !getEnum(t.zone) || !getEnum(t.building)) return fail("grid");
        if (!getBool(t.road) || !getBool(t.bldgOrigin)) return fail("grid");
        if (!get(in, t.footprint) || !get(in, t.bOriginC) || !get(in, t.bOriginR)) {
            return fail("grid");
        }
        if (!get(in, t.develop)) return fail("grid");
        if (!getBool(t.powered) || !getBool(t.poweredRoad) || !getBool(t.nearRoad)) {
            return fail("grid");
        }
        if (!get(in, t.desirability) || !get(in, t.scenicPhase) || !get(in, t.zoneAge) ||
            !get(in, t.trafficLoad)) {
            return fail("grid");
        }
        if (!get(in, t.fireTicks) || !get(in, t.charTicks)) return fail("grid");
        if (!getBool(t.charred)) return fail("grid");
    }

    std::int32_t year = 0, month = 0, pop = 0, jobs = 0, burning = 0, charred = 0;
    if (!get(in, app.m_money) || !get(in, year) || !get(in, month) || !get(in, pop) ||
        !get(in, jobs) || !get(in, app.m_education) || !get(in, app.m_health) ||
        !get(in, app.m_happiness) || !get(in, app.m_powerCoverage) || !get(in, app.m_resDemand) ||
        !get(in, app.m_comDemand) || !get(in, app.m_indDemand) || !get(in, burning) ||
        !get(in, charred) || !get(in, app.m_cityHeat) || !get(in, app.m_rng)) {
        return fail("truncated economy block");
    }

    if (!getEnum(app.m_season) || !getEnum(app.m_weather) || !getEnum(app.m_weatherTarget) ||
        !get(in, app.m_weatherIntensity) || !get(in, app.m_atmoHeat) ||
        !get(in, app.m_atmoInstability) || !get(in, app.m_stormSeverity) ||
        !get(in, app.m_weatherRng)) {
        return fail("truncated atmosphere block");
    }

    if (!getFloats(in, app.m_histPop) || !getFloats(in, app.m_histMoney) ||
        !getFloats(in, app.m_histEdu) || !getFloats(in, app.m_histHealth) ||
        !getFloats(in, app.m_histHappy)) {
        return fail("truncated history block");
    }

    CitizenSim::SaveState cs;
    std::uint32_t rosterSize = 0;
    if (!get(in, cs.rngState) || !get(in, cs.citizenCounter) || !get(in, rosterSize)) {
        return fail("truncated citizen header");
    }
    if (rosterSize > 4096u) return fail("implausible roster size");
    cs.roster.resize(rosterSize);
    for (Citizen& cz : cs.roster) {
        std::uint8_t atWork = 0;
        if (!get(in, cz.seed) || !getStr(in, cz.firstName) || !getStr(in, cz.lastName) ||
            !get(in, cz.homeC) || !get(in, cz.homeR) || !get(in, cz.workC) ||
            !get(in, cz.workR) || !get(in, cz.spouse) || !get(in, cz.affair) ||
            !get(in, cz.traits) || !get(in, atWork)) {
            return fail("truncated citizen record");
        }
        cz.atWork = atWork != 0;
    }

    // Commit.
    app.m_tiles = tiles;
    app.m_year = year;
    app.m_month = month;
    app.m_population = pop;
    app.m_jobs = jobs;
    app.m_burningTiles = burning;
    app.m_charredTiles = charred;
    app.m_citizens.restoreState(std::move(cs));

    // Derived state is rebuilt rather than stored: parcels, fields, coverage,
    // building counts and the procedural name caches all fall out of the grid
    // plus the world seed. Regenerating the names is a free determinism check —
    // if the seed round-tripped, every street comes back with the same name.
    app.m_businessNames.clear();
    app.m_blockNames.clear();
    app.m_streetNames.clear();
    app.recomputeStats();
    app.rebuildDestinations();
    app.m_sceneDirty = true;

    g_lastError.clear();
    return true;
}

}  // namespace odai::games::citybuilder
