#pragma once

#include "bethesda/runtime_ids.h"
#include "import/bethesda/plugin_load_order.h"

#include <cstdint>
#include <string>

namespace odai::bethesda {

bool stableRecordKey(
    const importer::bethesda::FalloutLoadOrder& loadOrder,
    std::uint32_t resolvedFormId,
    RecordKey& out,
    std::string& outError);

bool resolvedFormId(
    const importer::bethesda::FalloutLoadOrder& loadOrder,
    const RecordKey& key,
    std::uint32_t& out,
    std::string& outError);

}  // namespace odai::bethesda
