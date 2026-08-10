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
// 10 color channels x 6 time slots x RGBA.
constexpr std::size_t kNam0Size = kColorCount * kSlotCount * 4u;
constexpr std::size_t kCloudLayerCount = FalloutWeatherRecord::kCloudLayerCount;
// 4 cloud layers x 6 time slots x RGBA.
constexpr std::size_t kPnamSize = kCloudLayerCount * kSlotCount * 4u;

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
                for (const EsmSubrecordView& sub : record.subrecords) {
                    if (sub.type == "EDID") {
                        weather.editorId = readZeroTerminated(sub);
                    } else if (sub.type == "DNAM") {
                        weather.cloudTextures[0] = readZeroTerminated(sub);
                    } else if (sub.type == "CNAM") {
                        weather.cloudTextures[1] = readZeroTerminated(sub);
                    } else if (sub.type == "ANAM") {
                        weather.cloudTextures[2] = readZeroTerminated(sub);
                    } else if (sub.type == "BNAM") {
                        weather.cloudTextures[3] = readZeroTerminated(sub);
                    } else if (sub.type == "PNAM" && sub.size >= kPnamSize) {
                        for (std::size_t layer = 0; layer < kCloudLayerCount; ++layer) {
                            for (std::size_t slot = 0; slot < kSlotCount; ++slot) {
                                const std::size_t offset = ((layer * kSlotCount) + slot) * 4u;
                                weather.cloudColors[layer][slot] = FalloutColorRgb{
                                    sub.data[offset], sub.data[offset + 1u], sub.data[offset + 2u]};
                            }
                        }
                    } else if (sub.type == "NAM0" && sub.size >= kNam0Size) {
                        for (std::size_t color = 0; color < kColorCount; ++color) {
                            for (std::size_t slot = 0; slot < kSlotCount; ++slot) {
                                const std::size_t offset = ((color * kSlotCount) + slot) * 4u;
                                weather.colors[color][slot] = FalloutColorRgb{
                                    sub.data[offset], sub.data[offset + 1u], sub.data[offset + 2u]};
                            }
                        }
                    } else if (sub.type == "FNAM" && sub.size >= 24u) {
                        weather.fogDayNear = readFloat(sub.data);
                        weather.fogDayFar = readFloat(sub.data + 4);
                        weather.fogNightNear = readFloat(sub.data + 8);
                        weather.fogNightFar = readFloat(sub.data + 12);
                        weather.fogDayPower = readFloat(sub.data + 16);
                        weather.fogNightPower = readFloat(sub.data + 20);
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
                    // On a WRLD, CNAM is the climate -- not the cloud texture it
                    // means on a WTHR. Same four bytes, different record.
                    if (sub.type == "CNAM" && sub.size >= 4u) {
                        outTables.climateByWorldspaceFormId[formId] =
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
    const FalloutWeatherRecord& weather,
    int layer,
    float hourOfDay,
    float sunriseHour,
    float sunsetHour) {
    if (layer < 0 || static_cast<std::size_t>(layer) >= kCloudLayerCount) {
        return {};
    }
    std::size_t from = 0;
    std::size_t to = 0;
    float t = 0.0f;
    resolveSlots(hourOfDay, sunriseHour, sunsetHour, from, to, t);
    return blendColors(
        weather.cloudColors[static_cast<std::size_t>(layer)][from],
        weather.cloudColors[static_cast<std::size_t>(layer)][to], t);
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

}  // namespace odai::importer::fnv
