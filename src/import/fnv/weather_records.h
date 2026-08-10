#pragma once

// WTHR (weather) and CLMT (climate) records, across a load order.
//
// This is the payload half of a weather mod. Nevada Skies ships 387 WTHR and 15
// CLMT records against 9 scripts: the scripts only choose which weather is
// active, and everything that is actually seen -- sky gradient, fog, cloud
// layers, sun and ambient light -- is in these records. Reading them is what
// makes such a mod do something without running a line of its code.
//
// Records are keyed by REMAPPED formID (see plugin_load_order.h), so a later
// plugin overriding an earlier one's weather replaces it, and two plugins that
// both define "their own" record 0x01000ABC do not collide.

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "import/fnv/plugin_load_order.h"

namespace odai::importer::fnv {

// FNV weather colors are authored per time of day. Fallout 3 had four slots;
// New Vegas added noon and midnight, which is why NAM0 is 240 bytes (10 color
// types x 6 slots x RGBA) rather than Fallout 3's 160, and why a WTHR record
// carries six \x00IAD..\x05IAD image-space adapters.
enum class FalloutWeatherTimeSlot : std::uint32_t {
    Sunrise = 0,
    Day = 1,
    Sunset = 2,
    Night = 3,
    Noon = 4,
    Midnight = 5,
    Count = 6,
};

// The ten color channels NAM0 stores, in record order, per the fopdoc WTHR
// spec. Indices 2 and 9 are genuinely unused by the game; they are kept so the
// indices match the file layout rather than being renumbered into something
// that no longer maps onto the bytes.
enum class FalloutWeatherColor : std::uint32_t {
    SkyUpper = 0,
    Fog = 1,
    Unused2 = 2,
    Ambient = 3,
    Sunlight = 4,
    Sun = 5,
    Stars = 6,
    SkyLower = 7,
    Horizon = 8,
    Unused9 = 9,
    Count = 10,
};

struct FalloutColorRgb {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
};

struct FalloutWeatherRecord {
    std::uint32_t formId = 0;  // remapped into the load order's global space
    std::string editorId;

    // NAM0, indexed [color][timeSlot].
    FalloutColorRgb colors[static_cast<std::size_t>(FalloutWeatherColor::Count)]
                          [static_cast<std::size_t>(FalloutWeatherTimeSlot::Count)]{};

    // Cloud layer textures, relative to Data\Textures, in layer order:
    // DNAM, CNAM, ANAM, BNAM. Nevada Skies keeps these in its own BSA, which is
    // why a mod directory has to index archives and not only loose files.
    //
    // An unused layer is not empty -- it points at "sky\\alpha.dds", a fully
    // transparent 1520-byte placeholder. Treating that as a real layer uploads
    // and blends four textures where one or two were meant, so callers should
    // ask isEmptyCloudLayer() rather than testing for an empty string.
    static constexpr std::size_t kCloudLayerCount = 4;
    std::string cloudTextures[kCloudLayerCount];

    // PNAM: per-layer tint, indexed [layer][timeSlot]. These are what make a
    // cloud layer read as sunlit, overcast or nearly black at midnight; the
    // texture itself is the same all day.
    FalloutColorRgb cloudColors[kCloudLayerCount]
                               [static_cast<std::size_t>(FalloutWeatherTimeSlot::Count)]{};

    // FNAM, six floats.
    float fogDayNear = 0.0f;
    float fogDayFar = 0.0f;
    float fogNightNear = 0.0f;
    float fogNightFar = 0.0f;
    float fogDayPower = 0.0f;
    float fogNightPower = 0.0f;

    // DATA, 15 bytes.
    std::uint8_t windSpeed = 0;
    std::uint8_t cloudSpeedLower = 0;
    std::uint8_t cloudSpeedUpper = 0;
    std::uint8_t transDelta = 0;
    std::uint8_t sunGlare = 0;
    std::uint8_t sunDamage = 0;
    // Bit flags: 1 = pleasant, 2 = cloudy, 4 = rainy, 8 = snow.
    std::uint8_t classification = 0;

    [[nodiscard]] bool hasPrecipitation() const { return (classification & 0x0Cu) != 0u; }
};

// CLMT: which weathers a region may run, and when the sun moves.
struct FalloutClimateWeatherEntry {
    std::uint32_t weatherFormId = 0;  // remapped
    std::int32_t chance = 0;          // relative weight within the list
};

struct FalloutClimateRecord {
    std::uint32_t formId = 0;
    std::string editorId;
    std::vector<FalloutClimateWeatherEntry> weathers;  // WLST
    std::string sunTexture;                            // FNAM
    std::string sunGlareTexture;                       // GNAM
    // TNAM, in 10-minute units as stored.
    std::uint8_t sunriseBegin = 0;
    std::uint8_t sunriseEnd = 0;
    std::uint8_t sunsetBegin = 0;
    std::uint8_t sunsetEnd = 0;
};

// Everything a load order says about weather, already merged.
struct FalloutWeatherTables {
    // By remapped formID. A later plugin's record with the same ID replaces the
    // earlier one, which is exactly how an override plugin is meant to work.
    std::unordered_map<std::uint32_t, FalloutWeatherRecord> weathers;
    std::unordered_map<std::uint32_t, FalloutClimateRecord> climates;
    // WRLD -> its CNAM climate, so a worldspace can name the weathers it runs.
    std::unordered_map<std::uint32_t, std::uint32_t> climateByWorldspaceFormId;
    // Lowercased editor ID -> formID, for both weathers and climates. Naming a
    // weather is how a human picks one; formIDs move when the load order does.
    std::unordered_map<std::string, std::uint32_t> weatherFormIdByEditorId;

    [[nodiscard]] const FalloutWeatherRecord* findWeather(std::uint32_t formId) const;
    [[nodiscard]] const FalloutWeatherRecord* findWeatherByEditorId(const std::string& editorId) const;
};

// Walks every plugin in `order`, extracting WTHR/CLMT records and each WRLD's
// climate assignment, remapping formIDs as it goes.
//
// Only the top-level WTHR/CLMT/WRLD groups are entered; the world-children
// groups that hold nearly all of FalloutNV.esm's 234 MB are skipped without
// being read, so this costs a group-header walk rather than a file read.
bool buildFalloutWeatherTables(
    const FalloutLoadOrder& order, FalloutWeatherTables& outTables, std::string& outError);

// True for the transparent placeholder Fallout uses to mean "this layer is
// unused". Matched by name because the file is shared by nearly every weather
// and decoding it to discover it is empty would be per-weather wasted work.
bool isEmptyCloudLayer(const std::string& texturePath);

// PNAM tint for one cloud layer at an hour of the day, interpolated the same
// way sampleFalloutWeatherColor interpolates the sky.
FalloutColorRgb sampleFalloutWeatherCloudTint(
    const FalloutWeatherRecord& weather,
    int layer,
    float hourOfDay,
    float sunriseHour = 6.0f,
    float sunsetHour = 19.0f);

// Samples a weather's color for an hour of the day, blending between the two
// adjacent time slots rather than stepping, so a sunrise does not snap.
//
// `sunriseHour`/`sunsetHour` come from the climate; the defaults are Fallout's
// own rough values and are only used when no climate applies.
FalloutColorRgb sampleFalloutWeatherColor(
    const FalloutWeatherRecord& weather,
    FalloutWeatherColor channel,
    float hourOfDay,
    float sunriseHour = 6.0f,
    float sunsetHour = 19.0f);

}  // namespace odai::importer::fnv
