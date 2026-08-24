#pragma once

#include "bethesda/runtime_ids.h"
#include "import/fnv/esm_reader.h"

#include <cstdint>
#include <string>
#include <vector>

namespace odai::bethesda {

struct SkyrimLocationDefinition {
    RecordKey record;
    std::string editorId;
    std::uint32_t parentFormId = 0u;
    std::vector<std::uint32_t> keywordFormIds;
};

struct SkyrimGlobalVariableDefinition {
    RecordKey record;
    std::string editorId;
    float initialValue = 0.0f;
};

bool readSkyrimLocation(
    const importer::fnv::EsmRecordView& record,
    RecordKey stableRecord,
    SkyrimLocationDefinition& out,
    std::string& outError);

bool readSkyrimGlobalVariable(
    const importer::fnv::EsmRecordView& record,
    RecordKey stableRecord,
    SkyrimGlobalVariableDefinition& out,
    std::string& outError);

}  // namespace odai::bethesda
