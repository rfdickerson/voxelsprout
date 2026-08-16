#include "import/fnv/weather_records.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>

#include "import/fnv/esm_reader.h"

namespace odai::importer::fnv {

namespace {

constexpr std::size_t kColorCount = static_cast<std::size_t>(FalloutWeatherColor::Count);
constexpr std::size_t kSlotCount = static_cast<std::size_t>(FalloutWeatherTimeSlot::Count);
constexpr std::size_t kCloudLayerCount = FalloutWeatherRecord::kCloudLayerCount;

// Fallout 3 authors FOUR time slots, not six -- so its NAM0 is 160 bytes and
// its PNAM 64, and a `size >= 240` guard silently rejects the whole record.
// The failure is not a parse error, it is an all-zero colour table: every
// Fallout 3 weather rendered a pure black sky at noon while the terrain looked
// correct, which is the same display-referred symptom a genuinely dark weather
// produces. Read whichever layout the bytes are, and fill New Vegas's two extra
// slots from the neighbours Fallout 3 does author.
//
// The first four slots are the same channels in the same order in both games,
// which is what makes the widening a copy rather than an interpolation.
// SKYRIM ADDED ROWS, NOT SLOTS, AND THAT BREAKS A `>=` TEST. Its NAM0 is 272
// bytes -- SEVENTEEN colour channels at FOUR slots -- and its PNAM is 512, being
// thirty-two cloud layers at four. Both sail past a `size >= rowCount * 6 * 4`
// check and were read as New Vegas six-slot tables, which does not fail: it
// returns plausible colours sampled from the wrong channel at the wrong hour.
//
// So the sizes are matched EXACTLY first, and a table that is merely bigger than
// the four-slot layout is four slots with rows this reader does not want. That
// is safe because readWeatherColorRow indexes [row][slot] and never needs to
// know how many rows follow -- and Skyrim's first ten channels are the same ten,
// in the same order, that Fallout's NAM0 declares.
std::size_t weatherSlotsInSubrecord(std::uint32_t size, std::size_t rowCount) {
    if (rowCount == 0u) {
        return 0u;
    }
    if (size == rowCount * kSlotCount * 4u) {
        return kSlotCount;  // New Vegas
    }
    if (size == rowCount * 4u * 4u) {
        return 4u;  // Fallout 3, Oblivion
    }
    // Bigger than either exact layout: extra ROWS at four slots (Skyrim), so
    // long as the byte count is a whole number of four-slot rows.
    if (size > rowCount * 4u * 4u && (size % (4u * 4u)) == 0u) {
        return 4u;
    }
    if (size >= rowCount * kSlotCount * 4u) {
        return kSlotCount;
    }
    if (size >= rowCount * 4u * 4u) {
        return 4u;
    }
    return 0u;
}

// Fills the two slots New Vegas added from the neighbours a four-slot record
// does author: Noon takes Day and Midnight takes Night, which is the value the
// older game was showing at that hour anyway.
template <typename T>
void widenFourSlots(T* out) {
    out[static_cast<std::size_t>(FalloutWeatherTimeSlot::Noon)] =
        out[static_cast<std::size_t>(FalloutWeatherTimeSlot::Day)];
    out[static_cast<std::size_t>(FalloutWeatherTimeSlot::Midnight)] =
        out[static_cast<std::size_t>(FalloutWeatherTimeSlot::Night)];
}

// Reads one row of RGBA colours out of a NAM0/PNAM-shaped block and widens a
// four-slot row to six.
void readWeatherColorRow(const std::uint8_t* data,
                         std::size_t row,
                         std::size_t fileSlots,
                         FalloutColorRgb* out) {
    for (std::size_t slot = 0; slot < fileSlots; ++slot) {
        const std::size_t offset = ((row * fileSlots) + slot) * 4u;
        out[slot] = FalloutColorRgb{data[offset], data[offset + 1u], data[offset + 2u]};
    }
    if (fileSlots == 4u) {
        widenFourSlots(out);
    }
}

// Skyrim's per-layer cloud texture subrecords are chr('0' + layer) then "0TX",
// for layer 0..28 -- so layer 0 is "00TX", layer 10 is ":0TX" and layer 28 is
// "L0TX". Matched by shape rather than by a 29-entry table because that is what
// the shape actually is; the ASCII the layer byte lands on is not otherwise a
// legible name. Returns the layer, or -1 for anything else.
int skyrimCloudLayerSubrecord(const std::string& type) {
    if (type.size() != 4u || type.compare(1u, 3u, "0TX") != 0) {
        return -1;
    }
    const int layer = static_cast<int>(static_cast<unsigned char>(type[0])) - '0';
    return (layer >= 0 && layer <= 28) ? layer : -1;
}

// Skyrim's cloud tables are 32 entries wide (PNAM, JNAM, RNAM, QNAM all agree),
// even though only layers 0..28 are addressable. Fallout's four fit inside it,
// so one scratch width covers both games.
constexpr std::size_t kCloudScratchLayers = 32u;

// Every cloud field, gathered by the record's own layer index, before anything
// decides which layers are actually used.
struct CloudLayerScratch {
    std::string textures[kCloudScratchLayers];
    FalloutColorRgb tints[kCloudScratchLayers][kSlotCount]{};
    float alpha[kCloudScratchLayers][kSlotCount]{};
    float driftX[kCloudScratchLayers]{};
    float driftY[kCloudScratchLayers]{};
    std::uint32_t disabledMask = 0;
    bool hasDisabledMask = false;
    bool hasTints = false;
    bool hasAlpha = false;
    bool isSkyrim = false;
};

std::string toLowerAsciiCopy(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

std::string readZeroTerminated(const EsmSubrecordView& subrecord) {
    if (subrecord.data == nullptr || subrecord.size == 0u) {
        return {};
    }
    std::string text(reinterpret_cast<const char*>(subrecord.data), subrecord.size);
    const std::size_t terminator = text.find('\0');
    if (terminator != std::string::npos) {
        text.resize(terminator);
    }
    return text;
}

float readFloat(const std::uint8_t* bytes) {
    float value = 0.0f;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

// Turns the by-index scratch into the record's list of layers that are actually
// drawn, each carrying its own tint, alpha and drift.
//
// The DEDUPLICATION is Skyrim-only and deliberate. Skyrim authors the same
// sheet on several adjacent layers -- SkyrimCloudy puts SkyrimCloudsLower01 on
// 8, 9, 10 and 11 and SkyrimCloudsHorizon01 on 16, 21 and 22, at slightly
// different drift rates, to build parallax out of a handful of textures. A
// compositor with four slots that took them in order would spend all four on
// one sheet and never reach the horizon band. Fallout is left alone: its four
// layers are already distinct roles, and dropping a repeat there would change
// what every existing weather draws.
// Skyrim's layer number is an elevation; see FalloutCloudBand.
FalloutCloudBand skyrimCloudBand(std::size_t index) {
    if (index <= 7u) {
        return FalloutCloudBand::Upper;
    }
    if (index <= 14u) {
        return FalloutCloudBand::Lower;
    }
    if (index <= 27u) {
        return FalloutCloudBand::Horizon;
    }
    return FalloutCloudBand::Fill;
}

void buildCloudLayers(const CloudLayerScratch& scratch, FalloutWeatherRecord& weather) {
    weather.cloudMapping =
        scratch.isSkyrim ? FalloutCloudMapping::TilingPlane : FalloutCloudMapping::DomeFisheye;

    // Fallout stores two speeds for four layers: the lower pair and the upper
    // pair, 128 being still.
    const float pairDrift[2] = {(static_cast<float>(weather.cloudSpeedLower) - 128.0f) / 128.0f,
                                (static_cast<float>(weather.cloudSpeedUpper) - 128.0f) / 128.0f};

    for (std::size_t index = 0; index < kCloudScratchLayers; ++index) {
        const std::string& path = scratch.textures[index];
        if (path.empty() || isEmptyCloudLayer(path)) {
            continue;
        }
        if (scratch.hasDisabledMask && ((scratch.disabledMask >> index) & 1u) != 0u) {
            continue;
        }
        const FalloutCloudBand band =
            scratch.isSkyrim ? skyrimCloudBand(index) : FalloutCloudBand::WholeSky;
        if (band == FalloutCloudBand::Fill) {
            continue;
        }
        if (scratch.isSkyrim) {
            // Per BAND, not globally: SkyrimOvercastRain puts SkyrimCloudsLower03
            // on layer 8 (the overhead deck) and again on 27 (a horizon bank),
            // and those are two different things drawn two different ways.
            const std::string lowered = toLowerAsciiCopy(path);
            bool seen = false;
            for (const FalloutWeatherCloudLayer& existing : weather.cloudLayers) {
                if (existing.band == band && toLowerAsciiCopy(existing.texture) == lowered) {
                    seen = true;
                    break;
                }
            }
            if (seen) {
                continue;
            }
        }

        FalloutWeatherCloudLayer layer;
        layer.index = static_cast<int>(index);
        layer.texture = path;
        layer.band = band;
        for (std::size_t slot = 0; slot < kSlotCount; ++slot) {
            layer.tint[slot] = scratch.tints[index][slot];
            layer.alpha[slot] = scratch.hasAlpha ? scratch.alpha[index][slot] : 1.0f;
        }
        if (scratch.isSkyrim) {
            layer.driftX = scratch.driftX[index];
            layer.driftY = scratch.driftY[index];
        } else {
            layer.driftX = pairDrift[index < 2u ? 0u : 1u];
        }
        weather.cloudLayers.push_back(std::move(layer));
    }
}

std::uint32_t readU32(const std::uint8_t* bytes) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

std::int32_t readI32(const std::uint8_t* bytes) {
    std::int32_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

}  // namespace

const FalloutWeatherRecord* FalloutWeatherTables::findWeather(std::uint32_t formId) const {
    const auto found = weathers.find(formId);
    return found == weathers.end() ? nullptr : &found->second;
}

const FalloutWeatherRecord* FalloutWeatherTables::findWeatherByEditorId(
    const std::string& editorId) const {
    const auto found = weatherFormIdByEditorId.find(toLowerAsciiCopy(editorId));
    if (found == weatherFormIdByEditorId.end()) {
        return nullptr;
    }
    return findWeather(found->second);
}

bool buildFalloutWeatherTables(
    const FalloutLoadOrder& order, FalloutWeatherTables& outTables, std::string& outError) {
    outError.clear();
    outTables = FalloutWeatherTables{};

    for (std::size_t pluginIndex = 0; pluginIndex < order.size(); ++pluginIndex) {
        const FalloutLoadOrderEntry& plugin = order.entries()[pluginIndex];
        EsmReader reader;
        if (!reader.open(plugin.path)) {
            outError = "cannot read " + plugin.path.string() + ": " + reader.lastError();
            return false;
        }

        EsmReader::Visitor visitor;
        // Only the three top-level groups matter. Refusing every nested group
        // is what keeps this cheap: nearly all of FalloutNV.esm lives under
        // world-children groups, and a WRLD record's climate is on the record
        // itself, not among its cells.
        visitor.onGroupEnter = [](const EsmGroupView& group) {
            if (group.groupType != 0) {
                return false;
            }
            return group.rawLabel == "WTHR" || group.rawLabel == "CLMT" ||
                group.rawLabel == "WRLD";
        };
        visitor.onRecordHeader = [](const EsmRecordHeaderView& header) {
            return header.type == "WTHR" || header.type == "CLMT" || header.type == "WRLD";
        };
        visitor.onRecord = [&](const EsmRecordView& record) {
            const std::uint32_t formId = order.remapFormId(pluginIndex, record.formId);

            if (record.type == "WTHR") {
                FalloutWeatherRecord weather;
                weather.formId = formId;
                // Everything a cloud layer needs, gathered by RECORD LAYER
                // INDEX and assembled once the whole record has been read.
                // Assembling as we go is not an option: NAM1, PNAM, JNAM, RNAM
                // and QNAM all arrive AFTER the texture names they describe.
                CloudLayerScratch clouds;
                for (const EsmSubrecordView& sub : record.subrecords) {
                    if (sub.type == "EDID") {
                        weather.editorId = readZeroTerminated(sub);
                    } else if (sub.type == "DNAM") {
                        clouds.textures[0] = readZeroTerminated(sub);
                    } else if (sub.type == "CNAM") {
                        clouds.textures[1] = readZeroTerminated(sub);
                    } else if (sub.type == "ANAM") {
                        clouds.textures[2] = readZeroTerminated(sub);
                    } else if (sub.type == "BNAM") {
                        clouds.textures[3] = readZeroTerminated(sub);
                    } else if (const int skyrimLayer = skyrimCloudLayerSubrecord(sub.type);
                               skyrimLayer >= 0) {
                        // SKYRIM DOES NOT USE DNAM/CNAM/ANAM/BNAM FOR CLOUDS.
                        // It authors up to 29 dome layers, each in a subrecord
                        // whose type is chr('0' + layer) followed by "0TX" --
                        // so layer 0 is "00TX", layer 10 is ":0TX" and layer 28
                        // is "L0TX". SkyrimCloudy names eleven of them.
                        clouds.textures[static_cast<std::size_t>(skyrimLayer)] =
                            readZeroTerminated(sub);
                        clouds.isSkyrim = true;
                    } else if (sub.type == "NAM1" && sub.size >= 4u) {
                        // Skyrim's per-layer DISABLED bitfield, and the only
                        // honest way to know which layers a record actually
                        // uses. SkyrimCloudy's is 0xEF9EF0FF, leaving 8-11,
                        // 16, 21, 22 and 28 enabled -- exactly the layers whose
                        // textures ship and whose PNAM tints are non-zero.
                        clouds.disabledMask = readU32(sub.data);
                        clouds.hasDisabledMask = true;
                    } else if (sub.type == "JNAM" &&
                               sub.size >= kCloudScratchLayers * 4u * sizeof(float)) {
                        // Per-layer, per-slot alpha. Skyrim's Fill layer is a
                        // 32x32 fully-opaque swatch held at 0.4-0.5 here; drawn
                        // at 1.0 it is an opaque coat of paint over the sky.
                        for (std::size_t layer = 0; layer < kCloudScratchLayers; ++layer) {
                            for (std::size_t slot = 0; slot < 4u; ++slot) {
                                clouds.alpha[layer][slot] =
                                    readFloat(sub.data + (((layer * 4u) + slot) * 4u));
                            }
                            widenFourSlots(clouds.alpha[layer]);
                        }
                        clouds.hasAlpha = true;
                    } else if ((sub.type == "RNAM" || sub.type == "QNAM") &&
                               sub.size >= kCloudScratchLayers) {
                        // Per-layer drift, one byte per layer per axis, 127
                        // still. RNAM is the Y axis and QNAM the X.
                        for (std::size_t layer = 0; layer < kCloudScratchLayers; ++layer) {
                            const float drift =
                                (static_cast<float>(sub.data[layer]) - 127.0f) / 127.0f;
                            if (sub.type == "RNAM") {
                                clouds.driftY[layer] = drift;
                            } else {
                                clouds.driftX[layer] = drift;
                            }
                        }
                    } else if (sub.type == "PNAM") {
                        // Four layers x N slots in Fallout, 32 x 4 in Skyrim.
                        // Both are [layer][slot] tables of RGBA, so one reader
                        // covers them once the row count is known.
                        const std::size_t rows = sub.size / 16u >= kCloudScratchLayers
                                                     ? kCloudScratchLayers
                                                     : kCloudLayerCount;
                        const std::size_t slots = weatherSlotsInSubrecord(sub.size, rows);
                        for (std::size_t layer = 0; slots != 0u && layer < rows; ++layer) {
                            readWeatherColorRow(sub.data, layer, slots, clouds.tints[layer]);
                        }
                        clouds.hasTints = slots != 0u;
                    } else if (sub.type == "NAM0") {
                        const std::size_t slots = weatherSlotsInSubrecord(sub.size, kColorCount);
                        for (std::size_t color = 0; slots != 0u && color < kColorCount; ++color) {
                            readWeatherColorRow(sub.data, color, slots,
                                                weather.colors[color]);
                        }
                        // Skyrim's extra channels. Row 12 is Fog Far; see
                        // FalloutWeatherRecord::hasFogFarColor for why taking
                        // row 1 instead turns a whole city cyan.
                        constexpr std::size_t kSkyrimFogFarRow = 12u;
                        if (slots != 0u && sub.size >= (kSkyrimFogFarRow + 1u) * slots * 4u) {
                            readWeatherColorRow(sub.data, kSkyrimFogFarRow, slots,
                                                weather.fogFarColors);
                            weather.hasFogFarColor = true;
                        }
                    } else if (sub.type == "FNAM" && sub.size >= 16u) {
                        // Oblivion authors FOUR floats here; the day/night fog
                        // POWER pair arrived with Fallout 3, so its FNAM is 24.
                        // Guarding on 24 silently left every Oblivion weather
                        // with fogDayFar = 0 -- and a forced weather publishes
                        // that zero as the aerial-perspective distance, which
                        // fogs the frame flat from the near plane out. It hid
                        // because the no-weather fallback is 160000 and
                        // Oblivion's own Clear authors 170000, so the default
                        // sky looked right and only --weather was broken.
                        weather.fogDayNear = readFloat(sub.data);
                        weather.fogDayFar = readFloat(sub.data + 4);
                        weather.fogNightNear = readFloat(sub.data + 8);
                        weather.fogNightFar = readFloat(sub.data + 12);
                        if (sub.size >= 24u) {
                            weather.fogDayPower = readFloat(sub.data + 16);
                            weather.fogNightPower = readFloat(sub.data + 20);
                        }
                    } else if (sub.type == "DATA" && sub.size >= 12u) {
                        weather.windSpeed = sub.data[0];
                        weather.cloudSpeedLower = sub.data[1];
                        weather.cloudSpeedUpper = sub.data[2];
                        weather.transDelta = sub.data[3];
                        weather.sunGlare = sub.data[4];
                        weather.sunDamage = sub.data[5];
                        weather.classification = sub.data[11];
                    }
                }
                // OBLIVION HAS NO PNAM. Its cloud tints live in the two NAM0
                // channels the Fallout docs write off as unused, which the
                // Oblivion CS names Clouds-Lower and Clouds-Upper -- Fallout 3
                // moved tints out to PNAM and left the channels behind.
                //
                // Without this every Oblivion cloud layer keeps the zero tint,
                // and since the tint IS the layer colour that draws the clouds
                // PURE BLACK: cloud-shaped black smears hanging over an
                // otherwise correct sky, which reads as geometry floating in
                // the air rather than as a missing subrecord.
                //
                // Layer order follows the file names: Overcast's DNAM is
                // CloudsOvercast.dds and its CNAM is CloudsOvercastLower.dds,
                // so DNAM is the upper layer and CNAM the lower.
                if (!clouds.hasTints) {
                    using Color = FalloutWeatherColor;
                    const std::size_t sources[2] = {static_cast<std::size_t>(Color::CloudsUpper),
                                                    static_cast<std::size_t>(Color::CloudsLower)};
                    for (std::size_t layer = 0; layer < 2u; ++layer) {
                        for (std::size_t slot = 0; slot < kSlotCount; ++slot) {
                            clouds.tints[layer][slot] = weather.colors[sources[layer]][slot];
                        }
                    }
                }
                buildCloudLayers(clouds, weather);
                if (!weather.editorId.empty()) {
                    outTables.weatherFormIdByEditorId[toLowerAsciiCopy(weather.editorId)] = formId;
                }
                // Assignment, not insert: a later plugin overriding this record
                // must replace it.
                outTables.weathers[formId] = std::move(weather);
                return;
            }

            if (record.type == "CLMT") {
                FalloutClimateRecord climate;
                climate.formId = formId;
                for (const EsmSubrecordView& sub : record.subrecords) {
                    if (sub.type == "EDID") {
                        climate.editorId = readZeroTerminated(sub);
                    } else if (sub.type == "WLST") {
                        // Triples of (weather formID, chance, global formID).
                        // Fallout 3 stored pairs; New Vegas added the global,
                        // so accept both rather than misreading one as the
                        // other.
                        const std::uint32_t stride = (sub.size % 12u == 0u) ? 12u : 8u;
                        for (std::uint32_t offset = 0; offset + stride <= sub.size;
                             offset += stride) {
                            FalloutClimateWeatherEntry entry;
                            entry.weatherFormId =
                                order.remapFormId(pluginIndex, readU32(sub.data + offset));
                            entry.chance = readI32(sub.data + offset + 4);
                            climate.weathers.push_back(entry);
                        }
                    } else if (sub.type == "FNAM") {
                        climate.sunTexture = readZeroTerminated(sub);
                    } else if (sub.type == "GNAM") {
                        climate.sunGlareTexture = readZeroTerminated(sub);
                    } else if (sub.type == "TNAM" && sub.size >= 4u) {
                        climate.sunriseBegin = sub.data[0];
                        climate.sunriseEnd = sub.data[1];
                        climate.sunsetBegin = sub.data[2];
                        climate.sunsetEnd = sub.data[3];
                    }
                }
                outTables.climates[formId] = std::move(climate);
                return;
            }

            if (record.type == "WRLD") {
                for (const EsmSubrecordView& sub : record.subrecords) {
                    if (sub.type == "EDID") {
                        outTables.worldspaceFormIdByEditorId[toLowerAsciiCopy(
                            readZeroTerminated(sub))] = formId;
                    }
                    // On a WRLD, CNAM is the climate -- not the cloud texture it
                    // means on a WTHR. Same four bytes, different record.
                    if (sub.type == "CNAM" && sub.size >= 4u) {
                        outTables.climateByWorldspaceFormId[formId] =
                            order.remapFormId(pluginIndex, readU32(sub.data));
                    }
                    // WNAM: the parent worldspace. A WALLED CITY NAMES NO
                    // CLIMATE OF ITS OWN and inherits its parent's -- Skyrim's
                    // WhiterunWorld record is an EDID and this field, nothing
                    // else, while Tamriel carries the climate. Without the hop
                    // the city gets no weather at all, which renders as the
                    // procedural sky with no cloud layer over it.
                    if (sub.type == "WNAM" && sub.size >= 4u) {
                        outTables.parentWorldspaceFormId[formId] =
                            order.remapFormId(pluginIndex, readU32(sub.data));
                    }
                }
            }
        };

        if (!reader.walk(visitor)) {
            outError = "malformed plugin " + plugin.path.string() + ": " + reader.lastError();
            return false;
        }
    }
    return true;
}

bool isEmptyCloudLayer(const std::string& texturePath) {
    if (texturePath.empty()) {
        return true;
    }
    std::string normalized = toLowerAsciiCopy(texturePath);
    for (char& c : normalized) {
        if (c == '/') {
            c = '\\';
        }
    }
    // The length guard is load-bearing: size() - 9u is unsigned, so an
    // 8-character path such as "sky1.dds" wraps to SIZE_MAX, which compares
    // equal to the npos a failed rfind returns -- silently reporting a real
    // cloud layer as the empty placeholder and dropping it from the sky.
    constexpr std::size_t kAlphaSuffixLength = 9u;  // "alpha.dds"
    return normalized == "sky\\alpha.dds" ||
        (normalized.size() >= kAlphaSuffixLength &&
         normalized.rfind("alpha.dds") == normalized.size() - kAlphaSuffixLength);
}

namespace {

// The control points both samplers walk. Pulled out so the sky and the cloud
// tints cannot drift onto different curves and disagree about when dusk is.
struct ControlPoint {
    float hour;
    FalloutWeatherTimeSlot slot;
};

std::array<ControlPoint, 8> dayCurve(float sunriseHour, float sunsetHour) {
    const float noonHour = 12.0f;
    return {{
        {0.0f, FalloutWeatherTimeSlot::Midnight},
        {sunriseHour, FalloutWeatherTimeSlot::Sunrise},
        {(sunriseHour + noonHour) * 0.5f, FalloutWeatherTimeSlot::Day},
        {noonHour, FalloutWeatherTimeSlot::Noon},
        {(noonHour + sunsetHour) * 0.5f, FalloutWeatherTimeSlot::Day},
        {sunsetHour, FalloutWeatherTimeSlot::Sunset},
        {(sunsetHour + 24.0f) * 0.5f, FalloutWeatherTimeSlot::Night},
        {24.0f, FalloutWeatherTimeSlot::Midnight},
    }};
}

// Resolves an hour to the two slots either side of it and how far between.
void resolveSlots(
    float hourOfDay, float sunriseHour, float sunsetHour,
    std::size_t& outFrom, std::size_t& outTo, float& outT) {
    const std::array<ControlPoint, 8> points = dayCurve(sunriseHour, sunsetHour);
    float hour = std::fmod(hourOfDay, 24.0f);
    if (hour < 0.0f) {
        hour += 24.0f;
    }
    std::size_t upper = points.size() - 1u;
    for (std::size_t i = 1; i < points.size(); ++i) {
        if (hour <= points[i].hour) {
            upper = i;
            break;
        }
    }
    const ControlPoint& a = points[upper - 1u];
    const ControlPoint& b = points[upper];
    const float span = std::max(b.hour - a.hour, 1e-3f);
    outFrom = static_cast<std::size_t>(a.slot);
    outTo = static_cast<std::size_t>(b.slot);
    outT = std::clamp((hour - a.hour) / span, 0.0f, 1.0f);
}

FalloutColorRgb blendColors(const FalloutColorRgb& from, const FalloutColorRgb& to, float t) {
    const auto blend = [t](std::uint8_t x, std::uint8_t y) {
        return static_cast<std::uint8_t>(std::lround(
            (static_cast<float>(x) * (1.0f - t)) + (static_cast<float>(y) * t)));
    };
    return FalloutColorRgb{blend(from.r, to.r), blend(from.g, to.g), blend(from.b, to.b)};
}

}  // namespace

FalloutColorRgb sampleFalloutWeatherCloudTint(
    const FalloutWeatherCloudLayer& layer,
    float hourOfDay,
    float sunriseHour,
    float sunsetHour) {
    std::size_t from = 0;
    std::size_t to = 0;
    float t = 0.0f;
    resolveSlots(hourOfDay, sunriseHour, sunsetHour, from, to, t);
    return blendColors(layer.tint[from], layer.tint[to], t);
}

float sampleFalloutWeatherCloudAlpha(
    const FalloutWeatherCloudLayer& layer,
    float hourOfDay,
    float sunriseHour,
    float sunsetHour) {
    std::size_t from = 0;
    std::size_t to = 0;
    float t = 0.0f;
    resolveSlots(hourOfDay, sunriseHour, sunsetHour, from, to, t);
    return std::clamp(std::lerp(layer.alpha[from], layer.alpha[to], t), 0.0f, 1.0f);
}

FalloutColorRgb sampleFalloutWeatherColor(
    const FalloutWeatherRecord& weather,
    FalloutWeatherColor channel,
    float hourOfDay,
    float sunriseHour,
    float sunsetHour) {
    const std::size_t colorIndex = static_cast<std::size_t>(channel);
    if (colorIndex >= kColorCount) {
        return {};
    }
    std::size_t from = 0;
    std::size_t to = 0;
    float t = 0.0f;
    resolveSlots(hourOfDay, sunriseHour, sunsetHour, from, to, t);
    return blendColors(weather.colors[colorIndex][from], weather.colors[colorIndex][to], t);
}

FalloutColorRgb sampleFalloutWeatherColorRow(
    const FalloutColorRgb* slots, float hourOfDay, float sunriseHour, float sunsetHour) {
    if (slots == nullptr) {
        return {};
    }
    std::size_t from = 0;
    std::size_t to = 0;
    float t = 0.0f;
    resolveSlots(hourOfDay, sunriseHour, sunsetHour, from, to, t);
    return blendColors(slots[from], slots[to], t);
}

}  // namespace odai::importer::fnv
