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
// spec. Indices 2 and 9 are the two the Fallout docs call unused -- but they
// are not unused in OBLIVION, whose Construction Set names that same colour tab
// Clouds-Lower and Clouds-Upper. Fallout 3 moved cloud tints out to PNAM and
// left these two channels behind; Oblivion has no PNAM at all and tints its two
// cloud layers from here. See buildFalloutWeatherTables, which falls back to
// them, and the FalloutColorRgb comment on why a black tint is not benign.
enum class FalloutWeatherColor : std::uint32_t {
    SkyUpper = 0,
    Fog = 1,
    CloudsLower = 2,
    Ambient = 3,
    Sunlight = 4,
    Sun = 5,
    Stars = 6,
    SkyLower = 7,
    Horizon = 8,
    CloudsUpper = 9,
    Count = 10,
};

struct FalloutColorRgb {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
};

// How a cloud layer's texture covers the sky. The two generations do not merely
// scale differently -- they are different projections, and drawing one as the
// other is not a tuning error, it is meaningless.
enum class FalloutCloudMapping : std::uint32_t {
    // Fallout and Oblivion: ONE FISHEYE IMAGE OF THE WHOLE SKY. Zenith at the
    // centre of the image, horizon on the rim of the inscribed circle. Sampled
    // exactly once, so wrap addressing never comes into play, and "scrolling"
    // is a rotation about the zenith.
    DomeFisheye = 0,
    // Skyrim: a SEAMLESSLY TILING cloud sheet drawn on a dome, sampled many
    // times across the sky with wrap addressing. Verified by eye on
    // Sky\SkyrimCloudsLower01.dds, which has no radial structure whatever --
    // it is a tileable cloud field, and its shape is in the alpha channel.
    TilingPlane = 1,
};

// Roughly where in the sky a cloud layer belongs.
//
// SKYRIM'S LAYER NUMBER IS AN ELEVATION, and this is the whole reason its
// twenty-nine layers can be reduced to a handful without the sky collapsing
// into a whiteout. Censused across SkyrimClear, SkyrimCloudy, SkyrimFog and
// SkyrimOvercastRain, the textures a layer names sort cleanly by index:
//
//   0-7    Sky\SkyrimCloudsUpper01/02/04     high, thin, overhead
//   8-14   Sky\SkyrimCloudsLower01/03        the main deck
//   15-27  Sky\SkyrimCloudsHorizon01         banks around the skyline
//   28     Sky\SkyrimCloudsFill              a flat 32x32 colour swatch
//
// SkyrimClear proves the point on its own: it enables 0-7 and 15/19/20 and no
// Lower layer at all -- high wisps and a horizon bank over open blue.
//
// Fallout and Oblivion have four layers and no such structure; both bands
// there mean "the whole sky", which is what keeps their look unchanged.
enum class FalloutCloudBand : std::uint32_t {
    Upper = 0,
    Lower = 1,
    Horizon = 2,
    // A flat colour with no shape in it at all. This engine already draws an
    // authored three-stop sky gradient from NAM0, which is the same job done
    // better, so nothing composites this band -- drawn as a full-sky wash it
    // can only flatten the gradient underneath it.
    Fill = 3,
    // Fallout and Oblivion: no elevation structure, the layer covers the sky.
    WholeSky = 4,
};

// One cloud layer exactly as the record authors it. TEXTURE, TINT, OPACITY AND
// DRIFT TRAVEL TOGETHER because the record indexes them all by the same layer
// number, and pairing a texture from one layer with a tint from another is the
// single most destructive mistake available here: Skyrim authors black tints on
// the layers it has DISABLED, so picking textures and tints independently
// paints the whole sky black and looks like a broken shader.
struct FalloutWeatherCloudLayer {
    // The record's own layer number: 0..3 for Fallout/Oblivion, 0..28 for
    // Skyrim. Kept because it is the only stable identity a layer has, and a
    // log line naming "layer 2 of 4 loaded" is useless for diagnosing which of
    // the twenty-nine that was.
    int index = 0;
    std::string texture;
    FalloutCloudBand band = FalloutCloudBand::WholeSky;
    // Per time-of-day slot, the same six slots the sky colours use.
    FalloutColorRgb tint[static_cast<std::size_t>(FalloutWeatherTimeSlot::Count)]{};
    // JNAM, per slot. 1.0 for the games that author no such field.
    float alpha[static_cast<std::size_t>(FalloutWeatherTimeSlot::Count)]{};
    // Drift, normalized to [-1, 1] with 0 meaning still. Fallout stores two
    // bytes for four layers (128 = still); Skyrim stores one byte per layer per
    // axis in RNAM/QNAM (127 = still). What a unit of drift MEANS is the
    // renderer's business and differs by mapping -- a rotation about the zenith
    // for a fisheye, a UV translation for a tiling sheet.
    float driftX = 0.0f;
    float driftY = 0.0f;
};

struct FalloutWeatherRecord {
    std::uint32_t formId = 0;  // remapped into the load order's global space
    std::string editorId;

    // NAM0, indexed [color][timeSlot].
    FalloutColorRgb colors[static_cast<std::size_t>(FalloutWeatherColor::Count)]
                          [static_cast<std::size_t>(FalloutWeatherTimeSlot::Count)]{};

    // Every cloud layer the record ENABLES, in layer order, with its own tint,
    // opacity and drift attached. This is the only cloud representation; the
    // raw DNAM/CNAM/ANAM/BNAM (Fallout, Oblivion) and x0TX (Skyrim) subrecords
    // are merged with PNAM/JNAM/RNAM/QNAM on the way in.
    //
    // Layers the record does not use are already gone:
    //  - Fallout points an unused layer at "sky\\alpha.dds", a fully
    //    transparent 1520-byte placeholder shared by nearly every weather.
    //  - Skyrim's NAM1 is a per-layer DISABLED bitfield, and its disabled
    //    layers are exactly the ones whose textures do not ship. SkyrimCloudy
    //    names eleven layers of which four are dead leftovers from the Oblivion
    //    and Fallout records it was copied from -- Sky\OblivionCloudCloudyUpper01,
    //    Sky\WastelandCloudHorizon01 and Sky\SkyrimClouds04 exist in no Skyrim
    //    archive at all.
    //
    // A caller composites as many of these as it has slots for, and takes each
    // layer's tint and alpha FROM THE SAME ELEMENT.
    static constexpr std::size_t kCloudLayerCount = 4;
    std::vector<FalloutWeatherCloudLayer> cloudLayers;

    // Which projection those textures are drawn with; see FalloutCloudMapping.
    FalloutCloudMapping cloudMapping = FalloutCloudMapping::DomeFisheye;

    // SKYRIM SPLITS FOG INTO A NEAR AND A FAR COLOUR, and NAM0 channel 1 --
    // the single "Fog" that Oblivion and Fallout author -- is the NEAR one.
    // SkyrimCloudy's near fog is (14,128,156) by day, a saturated cyan meant to
    // tint the air a few metres out; its FAR fog, NAM0 channel 12, is
    // (139,175,194), the pale blue-grey a distant ridge should wash toward.
    // Driving aerial perspective from the near colour renders the whole city
    // cyan, which looks like a grading bug rather than a channel mix-up.
    //
    // Only populated when NAM0 carries the extra rows (17 for Skyrim, 10 for
    // everything earlier), so a Fallout record keeps using channel 1 as before.
    bool hasFogFarColor = false;
    FalloutColorRgb fogFarColors[static_cast<std::size_t>(FalloutWeatherTimeSlot::Count)]{};

    // FNAM. Four floats in Oblivion, six in Fallout (the two powers arrived
    // with Fallout 3), eight in Skyrim (a day/night maximum past the powers,
    // which nothing here reads).
    float fogDayNear = 0.0f;
    float fogDayFar = 0.0f;
    float fogNightNear = 0.0f;
    float fogNightFar = 0.0f;
    float fogDayPower = 0.0f;
    float fogNightPower = 0.0f;

    // DATA, 15 bytes in Fallout and 19 in Skyrim. The two games agree on the
    // first six fields and on byte 11, which is why so little of this branches
    // -- but bytes 1 and 2 are Fallout's two cloud speeds and are RESERVED in
    // Skyrim, whose per-layer speeds live in RNAM/QNAM instead.
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
    // WNAM: worldspace formID -> its parent's. A worldspace that names no
    // climate inherits its parent's, which is how every Skyrim walled city gets
    // a sky at all -- WhiterunWorld's record carries an EDID and this and
    // nothing else. Empty for the earlier games, whose city worldspaces mostly
    // name their own climate.
    std::unordered_map<std::uint32_t, std::uint32_t> parentWorldspaceFormId;
    // Lowercased WRLD EditorID -> its formID. Without this the map above can be
    // read but not ADDRESSED: a caller knows it is streaming "WastelandNV", not
    // which formID that is, and picking any entry gets the wrong climate the
    // moment more than one worldspace is loaded.
    std::unordered_map<std::string, std::uint32_t> worldspaceFormIdByEditorId;
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

// A cloud layer's tint at an hour of the day, interpolated the same way
// sampleFalloutWeatherColor interpolates the sky. Takes the LAYER rather than
// the record and an index, so there is no way to ask for one layer's tint while
// drawing another's texture.
FalloutColorRgb sampleFalloutWeatherCloudTint(
    const FalloutWeatherCloudLayer& layer,
    float hourOfDay,
    float sunriseHour = 6.0f,
    float sunsetHour = 19.0f);

// And its opacity, from the same slots.
float sampleFalloutWeatherCloudAlpha(
    const FalloutWeatherCloudLayer& layer,
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

// The same time-of-day blend over a bare six-slot row, for a channel that is not
// one of the ten FalloutWeatherColor names -- Skyrim's Fog Far.
FalloutColorRgb sampleFalloutWeatherColorRow(
    const FalloutColorRgb* slots,
    float hourOfDay,
    float sunriseHour = 6.0f,
    float sunsetHour = 19.0f);

}  // namespace odai::importer::fnv
