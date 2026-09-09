#pragma once
#include "bethesda/runtime_ids.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/plugin_load_order.h"
#include <map>

namespace odai::bethesda {
struct SkyrimItemDefinition {
    RecordKey record;
    std::string name;
    std::string text;
    std::string model;
    float meleeDamage = 0.0f;
    float healing = 0.0f;
    std::string recordType;
};
// Immutable winning-record metadata; no game text or assets are saved in ODAI saves.
bool loadSkyrimItems(const importer::fnv::FalloutLoadOrder &order,
                     const importer::fnv::FalloutAssetSource &assets,
                     std::map<RecordKey, SkyrimItemDefinition> &out, std::string &error);
} // namespace odai::bethesda
