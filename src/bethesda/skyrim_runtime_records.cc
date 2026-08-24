#include "bethesda/skyrim_runtime_records.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>

namespace odai::bethesda {
namespace {

std::uint32_t u32(const std::uint8_t* bytes) {
    std::uint32_t value = 0u;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

float f32(const std::uint8_t* bytes) {
    float value = 0.0f;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

std::string zstring(const importer::fnv::EsmSubrecordView& subrecord) {
    std::string value(reinterpret_cast<const char*>(subrecord.data), subrecord.size);
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

}  // namespace

bool readSkyrimLocation(
    const importer::fnv::EsmRecordView& record,
    RecordKey stableRecord,
    SkyrimLocationDefinition& out,
    std::string& outError) {
    if (record.type != "LCTN" || !stableRecord.valid()) {
        outError = "Skyrim location requires an LCTN record and stable RecordKey";
        return false;
    }
    SkyrimLocationDefinition parsed;
    parsed.record = std::move(stableRecord);
    for (const importer::fnv::EsmSubrecordView& subrecord : record.subrecords) {
        if (subrecord.type == "EDID") {
            parsed.editorId = zstring(subrecord);
        } else if (subrecord.type == "PNAM") {
            if (subrecord.size < 4u) {
                outError = "LCTN PNAM is shorter than one form ID";
                return false;
            }
            parsed.parentFormId = u32(subrecord.data);
        } else if (subrecord.type == "KWDA") {
            if ((subrecord.size % 4u) != 0u) {
                outError = "LCTN KWDA is not a packed form-ID array";
                return false;
            }
            for (std::uint32_t offset = 0u; offset < subrecord.size; offset += 4u) {
                const std::uint32_t keyword = u32(subrecord.data + offset);
                if (keyword != 0u) parsed.keywordFormIds.push_back(keyword);
            }
        }
    }
    std::sort(parsed.keywordFormIds.begin(), parsed.keywordFormIds.end());
    parsed.keywordFormIds.erase(
        std::unique(parsed.keywordFormIds.begin(), parsed.keywordFormIds.end()),
        parsed.keywordFormIds.end());
    out = std::move(parsed);
    outError.clear();
    return true;
}

bool readSkyrimGlobalVariable(
    const importer::fnv::EsmRecordView& record,
    RecordKey stableRecord,
    SkyrimGlobalVariableDefinition& out,
    std::string& outError) {
    if (record.type != "GLOB" || !stableRecord.valid()) {
        outError = "Skyrim global variable requires a GLOB record and stable RecordKey";
        return false;
    }
    SkyrimGlobalVariableDefinition parsed;
    parsed.record = std::move(stableRecord);
    bool hasValue = false;
    for (const importer::fnv::EsmSubrecordView& subrecord : record.subrecords) {
        if (subrecord.type == "EDID") {
            parsed.editorId = zstring(subrecord);
        } else if (subrecord.type == "FLTV") {
            if (subrecord.size < 4u) {
                outError = "GLOB FLTV is shorter than one float";
                return false;
            }
            parsed.initialValue = f32(subrecord.data);
            hasValue = true;
        }
    }
    if (!hasValue || !std::isfinite(parsed.initialValue)) {
        outError = "GLOB has no finite FLTV value";
        return false;
    }
    out = std::move(parsed);
    outError.clear();
    return true;
}

}  // namespace odai::bethesda
