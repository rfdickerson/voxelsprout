#pragma once

#include "bethesda/runtime_ids.h"
#include "import/fnv/plugin_load_order.h"

#include <cstdint>
#include <string>

namespace odai::bethesda {

bool stableRecordKey(
    const importer::fnv::FalloutLoadOrder& loadOrder,
    std::uint32_t resolvedFormId,
    RecordKey& out,
    std::string& outError);

bool resolvedFormId(
    const importer::fnv::FalloutLoadOrder& loadOrder,
    const RecordKey& key,
    std::uint32_t& out,
    std::string& outError);

}  // namespace odai::bethesda
